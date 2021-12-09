#    This script is part of skeletor (http://www.github.com/navis-org/skeletor).
#    Copyright (C) 2018 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.

import igraph as ig
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

from tqdm.auto import tqdm

from ..utilities import make_trimesh
from .base import Skeleton
from .utils import make_swc, reindex_swc, edges_to_graph

__all__ = ['by_wavefront']

# This flag determines whether we use inverse radii or edge lengths for the MST.
# Radii make more sense if working with tubular structures like neurons but this
# might not apply for all scenarios
PRESERVE_BACKBONE = True


def by_wavefront(mesh,
                 waves=1,
                 origins=None,
                 step_size=1,
                 radius_agg='mean',
                 progress=True):
    """Skeletonize a mesh using wave fronts.

    The algorithm tries to find rings of vertices and collapse them to
    their center. This is done by propagating a wave across the mesh starting at
    a single seed vertex. As the wave travels across the mesh we keep track of
    which vertices are are encountered at each step. Groups of connected
    vertices that are "hit" by the wave at the same time are considered rings
    and subsequently collapsed. By its nature this works best with tubular meshes.

    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to be skeletonize. Can an object that has
                    ``.vertices`` and ``.faces`` properties  (e.g. a
                    trimesh.Trimesh) or a tuple ``(vertices, faces)`` or a
                    dictionary ``{'vertices': vertices, 'faces': faces}``.
    waves :         int
                    Number of waves to run across the mesh. Each wave is
                    initialized at a different vertex which produces slightly
                    different rings. The final skeleton is produced from a mean
                    across all waves. More waves produce higher resolution
                    skeletons but also introduce more noise.
    origins :       int | list of ints, optional
                    Vertex ID(s) where the wave(s) are initialized. If we run
                    out of origins (either because less `origins` than `waves`
                    or because no origin for one of the connected components)
                    will fall back to semi-random origin.
    step_size :     int
                    Values greater 1 effectively lead to binning of rings. For
                    example a stepsize of 2 means that two adjacent vertex rings
                    will be collapsed to the same center. This can help reduce
                    noise in the skeleton (and as such counteracts a large
                    number of waves).
    radius_agg :    "mean" | "median" | "max" | "min" | "percentile75" | "percentile25"
                    Function used to aggregate radii over sample (i.e. the
                    vertices forming a ring that we collapse to its center).
    progress :      bool
                    If True, will show progress bar.

    Returns
    -------
    skeletor.Skeleton
                    Holds results of the skeletonization and enables quick
                    visualization.

    """
    agg_map = {'mean': np.mean, 'max': np.max, 'min': np.min,
               'median': np.median,
               'percentile75': lambda x: np.percentile(x, 75),
               'percentile25': lambda x: np.percentile(x, 25)}
    assert radius_agg in agg_map, f'Unknown `radius_agg`: "{radius_agg}"'
    rad_agg_func = agg_map[radius_agg]

    mesh = make_trimesh(mesh, validate=False)

    centers_final, radii_final, G = _cast_waves(mesh, waves=waves,
                                                origins=origins,
                                                step_size=step_size,
                                                rad_agg_func=rad_agg_func,
                                                progress=progress)

    # Collapse vertices into nodes
    (node_centers,
     vertex_to_node_map) = np.unique(centers_final,
                                     return_inverse=True, axis=0)

    # Map radii for individual vertices to the collapsed nodes
    # Using pandas is the fastest way here
    node_radii = pd.DataFrame()
    node_radii['node_id'] = vertex_to_node_map
    node_radii['radius'] = radii_final
    node_radii = node_radii.groupby('node_id').radius.apply(rad_agg_func).values

    # Contract vertices
    G.contract_vertices(vertex_to_node_map)

    # Remove self loops and duplicate edges
    G = G.simplify()

    # Generate hierarchical tree
    el = np.array(G.get_edgelist())

    if PRESERVE_BACKBONE:
        # Use the minimum radius between vertices in an edge
        weights_rad = np.vstack((node_radii[el[:, 0]],
                                 node_radii[el[:, 1]])).mean(axis=0)

        # For each node generate a vector based on its immediate neighbors
        vect, alpha = dotprops(node_centers)
        weights_alpha = np.vstack((alpha[el[:, 0]],
                                   alpha[el[:, 1]])).mean(axis=0)

        # Combine both which means we are most likely to cut at small branches
        # outside of the backbone
        weights = weights_rad * weights_alpha

        # MST doesn't like 0 for weights
        weights[weights <= 0] = weights[weights > 0].min() / 2

    else:
        weights = np.linalg.norm(node_centers[el[:, 0]] - node_centers[el[:, 1]], axis=1)
    tree = G.spanning_tree(weights=1 / weights)

    # Create a directed acyclic and hierarchical graph
    G_nx = edges_to_graph(edges=np.array(tree.get_edgelist()),
                          nodes=np.arange(0, len(G.vs)),
                          fix_tree=True,  # this makes sure graph is oriented
                          drop_disconnected=False)

    # Generate the SWC table
    swc = make_swc(G_nx, coords=node_centers, reindex=False, validate=True)
    swc['radius'] = node_radii[swc.node_id.values]
    _, new_ids = reindex_swc(swc, inplace=True)

    # Update vertex to node ID map
    vertex_to_node_map = np.array([new_ids[n] for n in vertex_to_node_map])

    return Skeleton(swc=swc, mesh=mesh, mesh_map=vertex_to_node_map,
                    method='wavefront')


def _cast_waves(mesh, waves=1, origins=None, step_size=1,
                rad_agg_func=np.mean, progress=True):
    """Cast waves across mesh."""
    if not isinstance(origins, type(None)):
        if isinstance(origins, int):
            origins = [origins]
        elif not isinstance(origins, (set, list)):
            raise TypeError('`origins` must be vertex ID (int) or list '
                            f'thereof, got "{type(origins)}"')
        origins = np.asarray(origins).astype(int)
    else:
        origins = np.array([])

    # Wave must be a positive integer >= 1
    waves = int(waves)
    if waves < 1:
        raise ValueError('`waves` must be integer >= 1')

    # Same for step size
    step_size = int(step_size)
    if step_size < 1:
        raise ValueError('`step_size` must be integer >= 1')

    # Generate Graph (must be undirected)
    G = ig.Graph(edges=mesh.edges_unique, directed=False)
    #G.es['weight'] = mesh.edges_unique_length

    # Prepare empty array to fill with centers
    centers = np.full((mesh.vertices.shape[0], 3, waves), fill_value=np.nan)
    radii = np.full((mesh.vertices.shape[0], waves), fill_value=np.nan)

    # Go over each connected component
    with tqdm(desc='Skeletonizing', total=len(G.vs), disable=not progress) as pbar:
        for cc in G.clusters():
            # Make a subgraph for this connected component
            SG = G.subgraph(cc)
            cc = np.array(cc)

            # Select seeds according to the number of waves
            n_waves = min(waves, len(cc))
            pot_seeds = np.arange(len(cc))
            np.random.seed(1985)  # make seeds predictable
            # See if we can use any origins
            if len(origins):
                # Get those origins in this cc
                in_cc = np.isin(origins, cc)
                if any(in_cc):
                    # Map origins into cc
                    cc_map = dict(zip(cc, np.arange(0, len(cc))))
                    seeds = np.array([cc_map[o] for o in origins[in_cc]])
                else:
                    seeds = np.array([])
                if len(seeds) < n_waves:
                    remaining_seeds = pot_seeds[~np.isin(pot_seeds, seeds)]
                    seeds = np.append(seeds,
                                      np.random.choice(remaining_seeds,
                                                       size=n_waves - len(seeds),
                                                       replace=False))
            else:
                seeds = np.random.choice(pot_seeds, size=n_waves, replace=False)
            seeds = seeds.astype(int)

            # Get the distance between the seeds and all other nodes
            dist = np.array(SG.shortest_paths(source=seeds, target=None, mode='all'))

            if step_size > 1:
                mx = dist.flatten()
                mx = mx[mx < float('inf')].max()
                dist = np.digitize(dist, bins=np.arange(0, mx, step_size))

            # Cast the desired number of waves
            for w in range(dist.shape[0]):
                this_wave = dist[w, :]
                # Collect groups
                mx = this_wave[this_wave < float('inf')].max()
                for i in range(0, int(mx) + 1):
                    this_dist = this_wave == i
                    ix = np.where(this_dist)[0]
                    SG2 = SG.subgraph(ix)
                    for cc2 in SG2.clusters():
                        this_verts = cc[ix[cc2]]
                        this_center = mesh.vertices[this_verts].mean(axis=0)
                        this_radius = cdist(this_center.reshape(1, -1), mesh.vertices[this_verts])
                        this_radius = rad_agg_func(this_radius)
                        centers[this_verts, :, w] = this_center
                        radii[this_verts, w] = this_radius

            pbar.update(len(cc))

    # Get mean centers and radii over all the waves we casted
    centers_final = np.nanmean(centers, axis=2)
    radii_final = np.nanmean(radii, axis=1)

    return centers_final, radii_final, G


def dotprops(x, k=20):
    """Generate vectors and alpha from local neighborhood."""
    # Checks and balances
    n_points = x.shape[0]

    # Make sure we don't ask for more nearest neighbors than we have points
    k = min(n_points, k)

    # Create the KDTree and get the k-nearest neighbors for each point
    tree = cKDTree(x)
    dist, ix = tree.query(x, k=k)
    # This makes sure we have (N, k) shaped array even if k = 1
    ix = ix.reshape(x.shape[0], k)

    # Get points: array of (N, k, 3)
    pt = x[ix]

    # Generate centers for each cloud of k nearest neighbors
    centers = np.mean(pt, axis=1)

    # Generate vector from center
    cpt = pt - centers.reshape((pt.shape[0], 1, 3))

    # Get inertia (N, 3, 3)
    inertia = cpt.transpose((0, 2, 1)) @ cpt

    # Extract vector and alpha
    u, s, vh = np.linalg.svd(inertia)
    vect = vh[:, 0, :]
    alpha = (s[:, 0] - s[:, 1]) / np.sum(s, axis=1)

    return vect, alpha
