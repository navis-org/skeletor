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
import warnings

import ncollpyde
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
                 progress=True,
                 inside_mode='none',
                 center_mode='mean',
                 max_edge_fix_iter=8,
                 inside_smooth_iters=1):
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
    inside_mode :   "none" | "nodes" | "nodes_edges"
                    Optional inside constraint for the generated skeleton:
                      - "none": current behavior (no explicit inside enforcement)
                      - "nodes": force all skeleton nodes inside the mesh
                      - "nodes_edges": force nodes inside and iteratively split
                        crossing edges until all edges are inside or the
                        iteration limit is reached
                    Note that these checks rely on mesh collision queries and
                    work best for watertight meshes.
    center_mode :   "mean" | "inside_mean"
                    How to place ring centers while casting waves:
                      - "mean": center-of-mass of the ring (current behavior)
                      - "inside_mean": center-of-mass projected inside the mesh
                    Note that this relies on mesh collision queries and works
                    best for watertight meshes.
    max_edge_fix_iter : int
                    Maximum number of iterations for fixing crossing edges when
                    ``inside_mode='nodes_edges'``.
    inside_smooth_iters : int
                    Number of smoothing passes applied to degree-2 chain nodes
                    after edge-fixing in ``inside_mode='nodes_edges'``.

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
    assert inside_mode in {'none', 'nodes', 'nodes_edges'}, (
        'inside_mode must be "none", "nodes" or "nodes_edges"'
    )
    assert center_mode in {'mean', 'inside_mean'}, (
        'center_mode must be "mean" or "inside_mean"'
    )
    max_edge_fix_iter = int(max_edge_fix_iter)
    inside_smooth_iters = int(inside_smooth_iters)
    if max_edge_fix_iter < 0:
        raise ValueError('`max_edge_fix_iter` must be >= 0')
    if inside_smooth_iters < 0:
        raise ValueError('`inside_smooth_iters` must be >= 0')

    mesh = make_trimesh(mesh, validate=False)
    coll = None
    mesh_kdtree = None
    if inside_mode != 'none' or center_mode == 'inside_mean':
        coll = ncollpyde.Volume(mesh.vertices, mesh.faces, validate=False)
        mesh_kdtree = cKDTree(mesh.vertices)
    
    centers_final, radii_final, G = _cast_waves(mesh, waves=waves,
                                                origins=origins,
                                                step_size=step_size,
                                                rad_agg_func=rad_agg_func,
                                                center_mode=center_mode,
                                                coll=coll,
                                                mesh_kdtree=mesh_kdtree,
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

    if inside_mode in {'nodes', 'nodes_edges'}:
        swc[['x', 'y', 'z']] = _project_points_inside(swc[['x', 'y', 'z']].values,
                                                      mesh=mesh,
                                                      coll=coll,
                                                      kdtree=mesh_kdtree)

    if inside_mode == 'nodes_edges':
        swc, remaining_crossings = _enforce_inside_edges(swc=swc,
                                                         coll=coll,
                                                         mesh=mesh,
                                                         max_iter=max_edge_fix_iter,
                                                         kdtree=mesh_kdtree)
        swc = _smooth_inside_nodes(swc=swc,
                                   coll=coll,
                                   mesh=mesh,
                                   iters=inside_smooth_iters,
                                   kdtree=mesh_kdtree)

        remaining_crossings = int(_find_crossing_edges(swc, coll).sum())
        if remaining_crossings > 0:
            warnings.warn(
                f'{remaining_crossings} crossing edges remain after '
                f'{max_edge_fix_iter} fix iteration(s); returning best effort '
                'result. Mesh might be non-watertight.',
                RuntimeWarning
            )

    _, new_ids = reindex_swc(swc, inplace=True)

    # Update vertex to node ID map
    vertex_to_node_map = np.array([new_ids[n] for n in vertex_to_node_map])

    return Skeleton(swc=swc, mesh=mesh, mesh_map=vertex_to_node_map,
                    method='wavefront')


def _cast_waves(mesh, waves=1, origins=None, step_size=1,
                rad_agg_func=np.mean, center_mode='mean',
                coll=None, mesh_kdtree=None, progress=True):
    """Cast waves across mesh."""
    assert center_mode in {'mean', 'inside_mean'}

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

    if center_mode == 'inside_mean':
        if coll is None:
            coll = ncollpyde.Volume(mesh.vertices, mesh.faces, validate=False)
        if mesh_kdtree is None:
            mesh_kdtree = cKDTree(mesh.vertices)

    # Go over each connected component
    with tqdm(desc='Skeletonizing', total=len(G.vs), disable=not progress) as pbar:
        for cc in G.connected_components():
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
            dist = np.array(SG.distances(source=seeds, target=None, mode='all'))

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
                    for cc2 in SG2.connected_components():
                        this_verts = cc[ix[cc2]]
                        this_center = mesh.vertices[this_verts].mean(axis=0)
                        if center_mode == 'inside_mean':
                            this_center = _project_points_inside(this_center.reshape(1, 3),
                                                                 mesh=mesh,
                                                                 coll=coll,
                                                                 kdtree=mesh_kdtree)[0]
                        this_radius = cdist(this_center.reshape(1, -1), mesh.vertices[this_verts])
                        this_radius = rad_agg_func(this_radius)
                        centers[this_verts, :, w] = this_center
                        radii[this_verts, w] = this_radius

            pbar.update(len(cc))

    # Get mean centers and radii over all the waves we casted
    centers_final = np.nanmean(centers, axis=2)
    radii_final = np.nanmean(radii, axis=1)

    return centers_final, radii_final, G

def _project_points_inside(points, mesh, coll, kdtree=None):
    """Project points into the mesh volume using nearest-vertex ray heuristics."""
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points
    if points.ndim == 1:
        points = points.reshape(1, 3)

    if kdtree is None:
        kdtree = cKDTree(mesh.vertices)

    projected = points.copy()
    inside = coll.contains(projected)
    if inside.all():
        return projected

    outside_ix = np.where(~inside)[0]
    _, nearest = kdtree.query(projected[outside_ix])

    closest_vertex = mesh.vertices[nearest]
    vnormals = mesh.vertex_normals[nearest]
    sources = closest_vertex - vnormals
    targets = sources - vnormals * 1e4

    ix, loc, _ = coll.intersections(sources, targets)

    final_pos = closest_vertex.copy()
    if len(ix):
        halfvec = np.zeros(sources.shape)
        halfvec[ix] = (loc - closest_vertex[ix]) / 2
        candidate = closest_vertex + halfvec
        now_inside = coll.contains(candidate)
        final_pos[now_inside] = candidate[now_inside]

    still_outside = ~coll.contains(final_pos)
    if still_outside.any():
        push = mesh.edges_unique_length.mean() / 100 if mesh.edges_unique_length.size else 1e-4
        push = max(float(push), 1e-6)

        candidate_in = closest_vertex - vnormals * push
        candidate_out = closest_vertex + vnormals * push
        in_ok = coll.contains(candidate_in)
        out_ok = coll.contains(candidate_out)

        use_in = still_outside & in_ok
        use_out = still_outside & ~in_ok & out_ok
        final_pos[use_in] = candidate_in[use_in]
        final_pos[use_out] = candidate_out[use_out]

    projected[outside_ix] = final_pos

    # Last fallback: snap to nearest mesh vertex.
    still_outside_global = ~coll.contains(projected)
    if still_outside_global.any():
        _, fallback_ix = kdtree.query(projected[still_outside_global])
        projected[still_outside_global] = mesh.vertices[fallback_ix]

    return projected


def _find_crossing_edges(swc, coll, eps=1e-6, return_edge_rows=False):
    """Find edges that cross the mesh boundary."""
    edge_rows = np.where(swc.parent_id.values >= 0)[0]
    crossing = np.zeros(edge_rows.shape[0], dtype=bool)

    if edge_rows.size == 0:
        if return_edge_rows:
            return crossing, edge_rows
        return crossing

    sources = swc.loc[edge_rows, ['x', 'y', 'z']].values
    parent_ids = swc.loc[edge_rows, 'parent_id'].values
    targets = swc.set_index('node_id').loc[parent_ids, ['x', 'y', 'z']].values

    ix, loc, _ = coll.intersections(sources, targets)
    if len(ix):
        d_src = np.linalg.norm(loc - sources[ix], axis=1)
        d_tgt = np.linalg.norm(loc - targets[ix], axis=1)
        real_crossings = (d_src > eps) & (d_tgt > eps)
        if real_crossings.any():
            crossing[np.unique(ix[real_crossings])] = True

    if return_edge_rows:
        return crossing, edge_rows
    return crossing


def _enforce_inside_edges(swc, coll, mesh, max_iter=8, eps=1e-6, kdtree=None):
    """Iteratively split edges that cross outside the mesh."""
    max_iter = int(max_iter)
    if max_iter < 0:
        raise ValueError('`max_iter` must be >= 0')

    has_radius = 'radius' in swc.columns
    swc = swc.copy()

    if kdtree is None:
        kdtree = cKDTree(mesh.vertices)

    for _ in range(max_iter):
        crossing, edge_rows = _find_crossing_edges(swc, coll, eps=eps, return_edge_rows=True)
        if not crossing.any():
            return swc, 0

        to_split = edge_rows[crossing]
        next_node_id = int(swc.node_id.max()) + 1 if not swc.empty else 0
        new_rows = []
        nodes = swc.set_index('node_id')

        for edge_row in to_split:
            child_id = int(swc.iloc[edge_row].node_id)
            parent_id = int(swc.iloc[edge_row].parent_id)

            if parent_id < 0:
                continue

            child_co = nodes.loc[child_id, ['x', 'y', 'z']].values.astype(float)
            parent_co = nodes.loc[parent_id, ['x', 'y', 'z']].values.astype(float)
            midpoint = ((child_co + parent_co) / 2).reshape(1, 3)
            midpoint = _project_points_inside(midpoint, mesh=mesh, coll=coll, kdtree=kdtree)[0]

            swc.loc[swc.node_id == child_id, 'parent_id'] = next_node_id

            row = {col: np.nan for col in swc.columns}
            row['node_id'] = next_node_id
            row['parent_id'] = parent_id
            row['x'], row['y'], row['z'] = midpoint

            if has_radius:
                child_r = pd.to_numeric(pd.Series([nodes.loc[child_id, 'radius']]),
                                        errors='coerce').iloc[0]
                parent_r = pd.to_numeric(pd.Series([nodes.loc[parent_id, 'radius']]),
                                         errors='coerce').iloc[0]
                row['radius'] = np.nanmean(np.array([child_r, parent_r], dtype=float))

            new_rows.append(row)
            next_node_id += 1

        if new_rows:
            swc = pd.concat([swc, pd.DataFrame(new_rows, columns=swc.columns)],
                            ignore_index=True)
        else:
            break

    remaining_crossings = int(_find_crossing_edges(swc, coll, eps=eps).sum())
    return swc, remaining_crossings


def _smooth_inside_nodes(swc, coll, mesh, iters=1, kdtree=None):
    """Smooth degree-2 chain nodes and keep them inside the mesh."""
    iters = int(iters)
    if iters <= 0 or swc.empty:
        return swc

    swc = swc.copy()
    if kdtree is None:
        kdtree = cKDTree(mesh.vertices)

    for _ in range(iters):
        child_counts = swc[swc.parent_id >= 0].groupby('parent_id').size()
        is_chain = (swc.parent_id >= 0) & (swc.node_id.map(child_counts).fillna(0).astype(int) == 1)
        chain_nodes = swc.loc[is_chain, 'node_id'].values.astype(int)

        if chain_nodes.size == 0:
            break

        only_child = swc[swc.parent_id >= 0].groupby('parent_id').node_id.first().to_dict()
        nodes = swc.set_index('node_id')
        parent_ids = nodes.loc[chain_nodes, 'parent_id'].values.astype(int)
        child_ids = np.array([only_child[n] for n in chain_nodes]).astype(int)

        parent_co = nodes.loc[parent_ids, ['x', 'y', 'z']].values.astype(float)
        child_co = nodes.loc[child_ids, ['x', 'y', 'z']].values.astype(float)

        smoothed = (parent_co + child_co) / 2
        smoothed = _project_points_inside(smoothed, mesh=mesh, coll=coll, kdtree=kdtree)

        for node_id, xyz in zip(chain_nodes, smoothed):
            swc.loc[swc.node_id == node_id, ['x', 'y', 'z']] = xyz

    return swc

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
