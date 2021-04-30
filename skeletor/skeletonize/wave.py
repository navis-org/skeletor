#    This script is part of skeletor (http://www.github.com/schlegelp/skeletor).
#    Copyright (C) 2018 Philipp Schlegel
#    Modified from https://github.com/aalavandhaann/Py_BL_MeshSkeletonization
#    by #0K Srinivasan Ramachandran.
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

from scipy.spatial.distance import cdist

from tqdm.auto import tqdm

from ..utilities import make_trimesh
from .base import Skeleton
from .utils import make_swc, reindex_swc, edges_to_graph

__all__ = ['by_wavefront']


def by_wavefront(mesh, waves=1, step_size=1, progress=True):
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
    step_size :     int
                    Values greater 1 effectively lead to binning of rings. For
                    example a stepsize of 2 means that two adjacent vertex rings
                    will be collapsed to the same center. This can help reduce
                    noise in the skeleton (and as such counteracts a large
                    number of waves).
    drop_disconnected : bool
                    If True, will drop disconnected nodes from the skeleton.
                    Note that this might result in empty skeletons.
    progress :      bool
                    If True, will show progress bar.

    Returns
    -------
    skeletor.Skeleton
                    Holds results of the skeletonization and enables quick
                    visualization.

    """
    mesh = make_trimesh(mesh, validate=False)

    centers_final, radii_final, G = _cast_waves(mesh, waves=waves,
                                                step_size=step_size,
                                                progress=progress)

    # Collapse vertices into nodes
    (node_centers,
     vertex_to_node_map) = np.unique(centers_final,
                                     return_inverse=True, axis=0)

    # Map radii for individual vertices to the collapsed nodes
    node_radii = [radii_final[vertex_to_node_map == i].mean() for i in range(vertex_to_node_map.max() + 1)]
    node_radii = np.array(node_radii)

    # Contract vertices
    G.contract_vertices(vertex_to_node_map)

    # Remove self loops and duplicate edges
    G = G.simplify()

    # Generate weights
    el = np.array(G.get_edgelist())
    weights = np.linalg.norm(node_centers[el[:, 0]] - node_centers[el[:, 1]], axis=1)

    # Generate hierarchical tree
    tree = G.spanning_tree(weights=weights)

    # Create a directed acyclic and hierarchical graph
    G_nx = edges_to_graph(edges=np.array(tree.get_edgelist()),
                          nodes=np.arange(0, len(G.vs)),
                          fix_tree=True,
                          drop_disconnected=False)

    # Generate the SWC table
    swc = make_swc(G_nx, coords=node_centers, reindex=False)
    swc['radius'] = node_radii[swc.node_id.values]
    _, new_ids = reindex_swc(swc, inplace=True)

    # Update vertex to node ID map
    vertex_to_node_map = np.array([new_ids[n] for n in vertex_to_node_map])

    return Skeleton(swc=swc, mesh=mesh, mesh_map=vertex_to_node_map,
                    method='wavefront')


def _cast_waves(mesh, waves=1, step_size=1, progress=True):
    """Cast waves across mesh."""
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
            seeds = np.linspace(0, len(cc) - 1,
                                min(waves, len(cc))).astype(int)

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
                        this_radius = cdist(this_center.reshape(1, -1), mesh.vertices[this_verts]).min()
                        centers[this_verts, :, w] = this_center
                        radii[this_verts, w] = this_radius

            pbar.update(len(cc))

    # Get mean centers and radii over all the waves we casted
    centers_final = np.nanmean(centers, axis=2)
    radii_final = np.nanmean(radii, axis=1)

    return centers_final, radii_final, G
