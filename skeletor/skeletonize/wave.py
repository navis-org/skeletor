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

from .. import _fastcore
from ..utilities import make_trimesh, get_edges_unique
from .base import Skeleton
from .utils import forest_to_swc

try:
    from fastremap import unique
except ModuleNotFoundError:
    from numpy import unique
except BaseException:
    raise
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

    This is conceptually equivalent to constructing the Reeb graph of the
    geodesic distance function on the mesh: the "rings" are connected level sets
    of that distance field and collapsing them to their centroid yields the
    skeleton nodes. The same construction was described by Verroust & Lazarus
    (see References below).

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

    See Also
    --------
    `skeletor.skeletonize.by_wavefront_exact()`
                    This is a version of the `by_wavefront` function in which the wave front
                    moves _exactly_ the given distance along the mesh (see `step_size` parameter)
                    instead of hopping from vertex to vertex. This can give better results on
                    meshes with a low vertex density but is computationally more expensive.

    Returns
    -------
    skeletor.Skeleton
                    Holds results of the skeletonization and enables quick
                    visualization.

    References
    ----------
    The wavefront approach is a Reeb graph of the geodesic distance function.
    The core construction (geodesic level sets collapsed to their centroids,
    organized into a tree) was described by:

    Verroust A, Lazarus F. Extracting skeletal curves from 3D scattered data.
    The Visual Computer. 2000;16(1):15-25.

    For the Reeb graph framing more generally see e.g. Hilaga et al. (SIGGRAPH
    2001) and Ge et al. ("Data Skeletonization via Reeb Graphs", NeurIPS 2011).
    This implementation adds, among other things, the use of multiple
    wavefronts from different seeds (averaged to reduce seed-dependence) and a
    radius-/tangent-weighted MST to preserve the backbone when breaking cycles.

    """
    if not callable(radius_agg):
        agg_map = {'mean': np.mean, 'max': np.max, 'min': np.min,
                'median': np.median,
                'percentile75': lambda x: np.percentile(x, 75),
                'percentile25': lambda x: np.percentile(x, 25)}
        assert radius_agg in agg_map, f'Unknown `radius_agg`: "{radius_agg}"'
        rad_agg_func = agg_map[radius_agg]
    else:
        rad_agg_func = radius_agg

    mesh = make_trimesh(mesh, validate=False)

    edges = get_edges_unique(mesh)
    n_verts = mesh.vertices.shape[0]

    centers_final, radii_final = _cast_waves(mesh, edges,
                                             waves=waves,
                                             origins=origins,
                                             step_size=step_size,
                                             radius_agg=radius_agg,
                                             rad_agg_func=rad_agg_func,
                                             progress=progress)

    # Collapse vertices into nodes
    (node_centers,
     vertex_to_node_map) = unique(centers_final, return_inverse=True, axis=0)
    n_nodes = len(node_centers)

    # Map radii for individual vertices to the collapsed nodes
    # (node IDs from unique() are contiguous 0..M-1, so the sorted groupby
    # output aligns positionally with node IDs)
    node_radii = _grouped_agg(radii_final, vertex_to_node_map,
                              radius_agg, rad_agg_func)

    # Contract vertices, dropping self loops and duplicate edges
    if _fastcore.has('contract_vertices'):
        el = _fastcore.fastcore.contract_vertices(edges, vertex_to_node_map)
    else:
        G = ig.Graph(n=n_verts, edges=edges, directed=False)
        G.contract_vertices(vertex_to_node_map)
        el = np.array(G.simplify().get_edgelist())
    el = np.asarray(el).reshape(-1, 2)  # reshape catches the no-edges case

    if PRESERVE_BACKBONE:
        # Use the mean radius between vertices in an edge
        weights_rad = np.vstack((node_radii[el[:, 0]],
                                 node_radii[el[:, 1]])).mean(axis=0)

        # For each edge generate a vector based on its immediate neighbors
        vect, alpha = dotprops(node_centers)
        weights_alpha = np.vstack((alpha[el[:, 0]],
                                   alpha[el[:, 1]])).mean(axis=0)

        # Combine both which means we are most likely to cut at small branches
        # outside of the backbone
        # Lower weights = more likely to be cut
        weights = weights_rad * weights_alpha

        # MST doesn't like 0 for weights
        weights[weights <= 0] = weights[weights > 0].min() / 2

    else:
        weights = np.linalg.norm(node_centers[el[:, 0]] - node_centers[el[:, 1]], axis=1)

    # We need this to orient all edges towards the root. Note we want to *keep*
    # the heavy edges, i.e. the maximum spanning tree.
    # Heads-up: with several waves plenty of edges end up on exactly the same
    # weight, and a maximum spanning tree is then not unique. The two branches
    # below break those ties differently and can return trees that differ in a
    # few percent of their edges - at identical total weight, so both are
    # equally valid. Don't expect the skeletons to match edge for edge.
    if _fastcore.has('minimum_spanning_tree'):
        keep = _fastcore.fastcore.minimum_spanning_tree(el, n_nodes,
                                                        weights=weights,
                                                        maximize=True)
        tree_edges = el[keep]
    else:
        G = ig.Graph(n=n_nodes, edges=el, directed=False)
        tree_edges = np.array(G.spanning_tree(weights=1 / weights).get_edgelist())

    # Generate the SWC table (this orients the tree and re-indexes nodes such
    # that parents always have lower IDs than their children)
    swc, new_ids = forest_to_swc(tree_edges,
                                 coords=node_centers,
                                 radii=node_radii,
                                 n_nodes=n_nodes)

    # Update vertex to node ID map
    vertex_to_node_map = new_ids[vertex_to_node_map]

    return Skeleton(swc=swc, mesh=mesh, mesh_map=vertex_to_node_map,
                    method='wavefront')


def _grouped_agg(values, ids, radius_agg, rad_agg_func):
    """Aggregate `values` by group.

    `ids` must be contiguous group IDs in ``[0, n_groups)`` with every ID
    present, so the (sorted) groupby output aligns positionally with the IDs.
    """
    grouped = pd.Series(values).groupby(ids)

    if callable(radius_agg):
        return grouped.apply(rad_agg_func).values

    # Use pandas' cython-accelerated aggregations - these are much faster than
    # .apply() which calls the aggregation function once per group
    return {'mean': grouped.mean, 'max': grouped.max,
            'min': grouped.min, 'median': grouped.median,
            'percentile75': lambda: grouped.quantile(0.75),
            'percentile25': lambda: grouped.quantile(0.25),
            }[radius_agg]().values


def _pick_seeds(cc, waves, origins):
    """Pick the wave origins for one connected component.

    Returns seeds as positions *within* ``cc``.
    """
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

    return seeds.astype(int)


def _cast_waves(mesh, edges, waves=1, origins=None, step_size=1,
                radius_agg='mean', rad_agg_func=np.mean, progress=True):
    """Cast waves across mesh.

    Returns the per-vertex ring centers and radii, averaged over all waves.
    """
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

    if _fastcore.has('connected_components_graph', 'geodesic_matrix_graph',
                     'level_set_components'):
        return _cast_waves_fastcore(mesh, edges, waves, origins, step_size,
                                    radius_agg, rad_agg_func, progress)

    return _cast_waves_igraph(mesh, edges, waves, origins, step_size,
                              rad_agg_func, progress)


def _cast_waves_fastcore(mesh, edges, waves, origins, step_size,
                         radius_agg, rad_agg_func, progress):
    """`_cast_waves` backend using navis-fastcore. No graph object involved."""
    fc = _fastcore.fastcore
    verts = np.asarray(mesh.vertices)
    n_verts = verts.shape[0]

    # Prepare empty array to fill with centers
    centers = np.full((n_verts, 3, waves), fill_value=np.nan)
    radii = np.full((n_verts, waves), fill_value=np.nan)

    # Level (i.e. binned distance from the seed) of every vertex, one row per
    # wave. -1 marks vertices a wave never reached, which is exactly the value
    # `level_set_components` treats as "excluded" rather than as a level of its
    # own - so unreachable vertices can't fuse into a phantom ring.
    labels = np.full((waves, n_verts), fill_value=-1, dtype=np.int64)

    # Split into connected components. `connected_components_graph` labels each
    # vertex with the lowest vertex ID in its component, so a stable sort on
    # that label groups components in ascending ID order with their members
    # sorted ascending too - the same order igraph produces, which keeps the
    # seed selection below identical between the two backends.
    comp = fc.connected_components_graph(edges, n_verts)
    srt = np.argsort(comp, kind='stable')
    comp_srt = comp[srt]
    starts = np.flatnonzero(np.r_[True, comp_srt[1:] != comp_srt[:-1]])
    ccs = np.split(srt, starts[1:])

    # Go over each connected component
    with tqdm(desc='Skeletonizing', total=n_verts, disable=not progress,
              leave=False) as pbar:
        for cc in ccs:
            # Select seeds according to the number of waves (positions within cc)
            seeds = _pick_seeds(cc, waves, origins)

            # Get the distance between the seeds and all nodes in this
            # component. Restricting `targets` to `cc` means only those columns
            # are ever allocated, so this stays component-sized - no subgraph
            # copy needed to keep the scratch space local.
            dist = fc.geodesic_matrix_graph(edges, n_verts,
                                            sources=cc[seeds], targets=cc)

            # Unreachable comes back as -1. Within a component there shouldn't
            # be any, but mask them before binning: `np.digitize` would happily
            # sort -1 into the first bin alongside the seed itself.
            unreachable = dist < 0

            if step_size > 1:
                mx = dist[~unreachable].max()
                dist = np.digitize(dist, bins=np.arange(0, mx, step_size))

            lvl = dist.astype(np.int64)
            lvl[unreachable] = -1

            # Components with fewer vertices than `waves` produce fewer rows;
            # the remaining waves stay at -1 and are skipped below.
            for w in range(lvl.shape[0]):
                labels[w, cc] = lvl[w]

            pbar.update(len(cc))

    # Now collect the rings: for each wave, the connected components of every
    # level set. Since mesh components are disjoint they can't be joined by an
    # edge, so this runs across the whole mesh at once - one O(E) sweep per wave
    # instead of a subgraph construction plus component search per level.
    for w in range(waves):
        ids, n_rings = fc.level_set_components(edges, n_verts, labels[w])

        ix = np.flatnonzero(ids >= 0)
        if not len(ix):
            continue
        ring = ids[ix]

        # Ring centers: with contiguous ring IDs a bincount is the grouped mean
        counts = np.bincount(ring, minlength=n_rings)
        ring_centers = np.empty((n_rings, 3))
        for k in range(3):
            ring_centers[:, k] = np.bincount(ring, weights=verts[ix, k],
                                             minlength=n_rings)
        ring_centers /= counts[:, None]

        # Ring radii: aggregate each vertex's distance to its ring's center
        to_center = np.linalg.norm(verts[ix] - ring_centers[ring], axis=1)
        ring_radii = _grouped_agg(to_center, ring, radius_agg, rad_agg_func)

        centers[ix, :, w] = ring_centers[ring]
        radii[ix, w] = ring_radii[ring]

    # Get mean centers and radii over all the waves we casted
    return np.nanmean(centers, axis=2), np.nanmean(radii, axis=1)


def _cast_waves_igraph(mesh, edges, waves, origins, step_size,
                       rad_agg_func, progress):
    """`_cast_waves` backend using igraph."""
    # Generate Graph (must be undirected)
    G = ig.Graph(n=mesh.vertices.shape[0], edges=edges, directed=False)

    # Prepare empty array to fill with centers
    centers = np.full((mesh.vertices.shape[0], 3, waves), fill_value=np.nan)
    radii = np.full((mesh.vertices.shape[0], waves), fill_value=np.nan)

    # Go over each connected component
    n_total = G.vcount()
    with tqdm(desc='Skeletonizing', total=len(G.vs), disable=not progress, leave=False) as pbar:
        for cc in G.connected_components():
            cc = np.array(cc)

            # Select seeds according to the number of waves (positions within cc)
            seeds = _pick_seeds(cc, waves, origins)

            # Get the distance between the seeds and all nodes in this component.
            # A component spanning most of the graph would make G.subgraph fall back
            # to copy-and-delete (a multi-GB copy of the whole graph -> swap on
            # memory-limited nodes). For that case run BFS on G directly, restricted
            # to cc (BFS scratch ~ component size anyway, no copy). Smaller
            # components are subgraphed so the BFS scratch stays component-local.
            dominant = len(cc) > 0.5 * n_total
            if dominant:
                dist = np.array(G.distances(source=list(cc[seeds]),
                                            target=list(cc), mode='all'))
            else:
                SG = G.subgraph(cc, implementation="create_from_scratch")
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
                    # ix indexes positions within cc. Build the (small) ring
                    # subgraph from the per-component SG so the cost stays local;
                    # for the dominant component there is no SG so use G directly
                    # (G ~ component there anyway). Induced connectivity of a ring
                    # is identical either way.
                    if dominant:
                        SG2 = G.subgraph(cc[ix], implementation="create_from_scratch")
                    else:
                        SG2 = SG.subgraph(ix, implementation="create_from_scratch")
                    for cc2 in SG2.connected_components():
                        this_verts = cc[ix[cc2]]
                        this_center = mesh.vertices[this_verts].mean(axis=0)
                        this_radius = cdist(this_center.reshape(1, -1), mesh.vertices[this_verts])
                        this_radius = rad_agg_func(this_radius)
                        centers[this_verts, :, w] = this_center
                        radii[this_verts, w] = this_radius

            pbar.update(len(cc))

    # Get mean centers and radii over all the waves we casted
    return np.nanmean(centers, axis=2), np.nanmean(radii, axis=1)


def dotprops(x, k=20):
    """Generate vectors and alpha from local neighborhood."""
    if _fastcore.has('dotprops'):
        # Same k convention (the point itself counts) and the same
        # un-normalised scatter matrix, so `alpha` matches to ~1e-16. Note it
        # also returns alpha=0 for a degenerate neighborhood where the SVD
        # below divides by zero and yields NaN.
        return _fastcore.fastcore.dotprops(x, k=k)

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