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
import networkx as nx
import numpy as np

from scipy.sparse.csgraph import dijkstra
from tqdm.auto import tqdm

from .. import _fastcore
from ..utilities import make_trimesh, get_edges_unique
from .base import Skeleton
from .utils import make_swc, edges_to_graph, group_by_label, split_by_label

try:
    from fastremap import unique
except ImportError:
    from numpy import unique
except BaseException:
    raise

__all__ = ['by_teasar']


def by_teasar(mesh, inv_dist, min_length=None, root=None, progress=True):
    """Skeletonize a mesh mesh using the TEASAR algorithm [1].

    This algorithm finds the longest path from a root vertex, invalidates all
    vertices that are within `inv_dist`. Then picks the second longest (and
    still valid) path and does the same. Rinse & repeat until all vertices have
    been invalidated. It's fast + works very well with tubular meshes, and with
    `inv_dist` you have control over the level of detail. Note that by its
    nature the skeleton will be exactly on the surface of the mesh.

    Based on the implementation by Sven Dorkenwald, Casey Schneider-Mizell and
    Forrest Collman in `meshparty` (https://github.com/sdorkenw/MeshParty).

    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to be skeletonize. Can an object that has
                    ``.vertices`` and ``.faces`` properties  (e.g. a
                    trimesh.Trimesh) or a tuple ``(vertices, faces)`` or a
                    dictionary ``{'vertices': vertices, 'faces': faces}``.
    inv_dist :      int | float
                    Distance along the mesh used for invalidation of vertices.
                    This controls how detailed (or noisy) the skeleton will be.
    min_length :    float, optional
                    If provided, will skip any branch that is shorter than
                    `min_length`. Use this to get rid of noise but note that
                    it will lead to vertices not being mapped to skeleton nodes.
                    Such vertices will show up with index -1 in
                    `Skeleton.mesh_map`.
    root :          int, optional
                    Vertex ID of a root. If not provided will use ``0``.
    progress :      bool, optional
                    If True, will show progress bar.

    Returns
    -------
    skeletor.Skeleton
                    Holds results of the skeletonization and enables quick
                    visualization.

    References
    ----------
    [1] Sato, M., Bitter, I., Bender, M. A., Kaufman, A. E., & Nakajima, M.
        (n.d.). TEASAR: tree-structure extraction algorithm for accurate and
        robust skeletons. In Proceedings the Eighth Pacific Conference on
        Computer Graphics and Applications. IEEE Comput. Soc.
        https://doi.org/10.1109/pccga.2000.883951

    """
    mesh = make_trimesh(mesh, validate=False)

    edges_unique, edges_unique_length = get_edges_unique(mesh, lengths=True)

    if root is None:
        root = 0

    if _fastcore.has('connected_components_graph',
                     'GeodesicGraph.path', 'GeodesicGraph.distances',
                     'GeodesicGraph.ball', 'GeodesicGraph.set_weights'):
        edges, mesh_map = _invalidate_fastcore(mesh, edges_unique,
                                               edges_unique_length, inv_dist,
                                               min_length, root, progress)
    else:
        edges, mesh_map = _invalidate_igraph(mesh, edges_unique,
                                             edges_unique_length, inv_dist,
                                             min_length, root, progress)

    # Bail out before the empty edge list reaches the graph construction below,
    # where it surfaces as an unhelpful IndexError
    if not len(edges):
        if not len(edges_unique):
            raise ValueError('Unable to skeletonize: mesh has no edges.')
        raise ValueError(
            f'Unable to skeletonize: `min_length` ({min_length}) is longer than '
            'every path found in the mesh, so no skeleton nodes are left. Try a '
            'smaller value.')

    # Make unique edges (paths will have overlapped!)
    edges = unique(edges, axis=0)

    # Create a directed acyclic and hierarchical graph
    G_nx = edges_to_graph(edges=edges[:, [1, 0]],
                          fix_tree=True, fix_edges=False,
                          weight=False)

    # Generate the SWC table
    swc, new_ids = make_swc(G_nx, coords=mesh.vertices, reindex=True)

    # Update vertex to node ID map
    mesh_map = np.array([new_ids.get(n, -1) for n in mesh_map])

    return Skeleton(swc=swc, mesh=mesh, mesh_map=mesh_map, method='teasar')


def _invalidate_igraph(mesh, edges_unique, edges_unique_length, inv_dist,
                       min_length, root, progress):
    """TEASAR path extraction + invalidation, using igraph."""
    # Generate Graph (must be undirected)
    G = ig.Graph(n=mesh.vertices.shape[0], edges=edges_unique, directed=False)
    G.es['weight'] = edges_unique_length

    edges = np.array([], dtype=np.int64)
    mesh_map = np.full(mesh.vertices.shape[0], fill_value=-1)

    with tqdm(desc='Invalidating', total=len(G.vs),
              disable=not progress, leave=False) as pbar:
        for cc in sorted(G.connected_components(), key=len, reverse=True):
            # An isolated vertex has no edges, so it contributes nothing to the
            # skeleton - and igraph can't hand out a 'weight' adjacency for an
            # empty edge sequence
            if len(cc) < 2:
                pbar.update(len(cc))
                continue

            # Make a subgraph for this connected component
            SG = G.subgraph(cc)
            cc = np.array(cc)

            # Find root within subgraph
            if root in cc:
                this_root = np.where(cc == root)[0][0]
            else:
                this_root = 0

            # Get the sparse adjacency matrix of the subgraph
            sp = SG.get_adjacency_sparse('weight')

            # Get lengths of paths to all nodes from root
            paths = SG.distances(this_root, target=None, weights='weight',
                                 mode='ALL')[0]
            paths = np.array(paths)

            # Prep array for invalidation
            valid = ~np.zeros(paths.shape).astype(bool)
            invalidated = 0

            while np.any(valid):
                # Find the farthest point
                farthest = np.argmax(paths)

                # Get path from root to farthest point
                path = SG.get_shortest_paths(this_root, farthest,
                                             weights='weight', mode='ALL')[0]

                # Get IDs of edges along the path
                ig_vers = getattr(ig, '__version_info__', (0, 0, 0))
                if ig_vers[0] > 0 or ig_vers[1] >= 10:
                    pairs = zip(path[:-1], path[1:])
                    eids = SG.get_eids(pairs, directed=False)
                else:
                    eids = SG.get_eids(path=path, directed=False)

                # Stop if farthest point is closer than min_length
                add = True
                if min_length:
                    # This should only be distance to the first branchpoint
                    # from the tip since we set other weights to zero
                    le = sum(SG.es[eids].get_attribute_values('weight'))
                    if le < min_length:
                        add = False

                if add:
                    # Add these new edges
                    new_edges = np.vstack((cc[path[:-1]], cc[path[1:]])).T
                    edges = np.append(edges, new_edges).reshape(-1, 2)

                # Invalidate points in the path
                valid[path] = False
                paths[path] = 0

                # Must set weights along path to 0 so that this path is
                # taken again in future iterations
                SG.es[eids]['weight'] = 0

                # Get all nodes within `inv_dist` to this path
                # Note: can we somehow only include still valid nodes to speed
                # things up?
                dist, _, sources = dijkstra(sp, directed=False, indices=path,
                                            limit=inv_dist, min_only=True,
                                            return_predecessors=True)

                # Invalidate
                in_dist = dist <= inv_dist
                to_invalidate = np.where(in_dist)[0]
                valid[to_invalidate] = False
                paths[to_invalidate] = 0

                # Update mesh vertex to skeleton node map
                mesh_map[cc[in_dist]] = cc[sources[in_dist]]

                pbar.update((~valid).sum() - invalidated)
                invalidated = (~valid).sum()

    return edges, mesh_map


def _invalidate_fastcore(mesh, edges_unique, edges_unique_length, inv_dist,
                         min_length, root, progress):
    """TEASAR path extraction + invalidation, using navis-fastcore.

    Same algorithm as :func:`_invalidate_igraph`, driven off two prepared
    `GeodesicGraph`s per connected component. Both matter, and they are not
    interchangeable:

    - the *penalised* graph, which the loop re-weights as it goes. Each
      extracted path is zeroed so later ones may re-traverse it for free, which
      is what makes TEASAR pick branches rather than re-walk the trunk.
      `set_weights` edits those few hundred arcs in place, where rebuilding the
      graph to change them would cost O(E) per iteration - the very thing a
      prepared graph is for.
    - the *pristine* graph, over the original lengths, for the invalidation
      radius. The igraph backend snapshots its adjacency for the same reason:
      the ball around a path is measured on the mesh, not on the penalised
      weights the path search sees.

    The one behavioural difference from :func:`_invalidate_igraph` is which
    equidistant path node a vertex is mapped to - see the `ball` call in the
    loop below.
    """
    fastcore = _fastcore.fastcore
    n_verts = mesh.vertices.shape[0]
    edges_unique = np.asarray(edges_unique)
    edges_unique_length = np.asarray(edges_unique_length, dtype=np.float64)

    all_edges = []
    mesh_map = np.full(n_verts, fill_value=-1)

    # Split into connected components, largest first. `connected_components_graph`
    # labels each vertex with the lowest vertex ID in its component, so the
    # grouping reproduces igraph's ordering before we re-sort by size (which is
    # stable too, so equal-sized components keep that order).
    comp = fastcore.connected_components_graph(edges_unique, n_verts)
    ccs = sorted(split_by_label(comp), key=len, reverse=True)

    # Group the edge list by component up front, so the loop below can take each
    # component's edges as a slice.
    #
    # Selecting them per component instead - the obvious reading - means one
    # pass over the *whole* edge list per component, i.e. O(components * E). On
    # a mesh that shatters into thousands of specks that dwarfs the actual work:
    # it measured ~29s of a 33s run on a 447k-vertex, 12463-component mesh,
    # against 4s for every fastcore call in this function put together.
    #
    # An edge never spans components, so either endpoint labels it. The grouping
    # is stable, which keeps each component's edges in their original relative
    # order, so the extracted paths - and the `min_length` edge index below, when
    # there is one - are unchanged.
    #
    # The graphs below are built per component rather than once for the whole
    # mesh for the same reason the loop is per component: every array it carries
    # (`paths`, `valid`) is indexed by local vertex, so a whole-mesh graph would
    # make each of them n_verts wide and each sweep over them O(components *
    # n_verts). Building per component is O(V + E) in total, since the components
    # partition the mesh.
    ecomp = comp[edges_unique[:, 0]]
    eorder, estarts, esizes = group_by_label(ecomp)
    edges_by_cc = edges_unique[eorder]
    lengths_by_cc = edges_unique_length[eorder]
    espan = dict(zip(ecomp[eorder[estarts]].tolist(),
                     zip(estarts.tolist(), (estarts + esizes).tolist())))

    # Reused across components; only the `cc` entries are ever set (and reset)
    glob2loc = np.full(n_verts, -1, dtype=np.int64)

    with tqdm(desc='Invalidating', total=n_verts,
              disable=not progress, leave=False) as pbar:
        for cc in ccs:
            # Isolated vertices contribute nothing to the skeleton (see the
            # igraph version for why this is a hard skip there)
            if len(cc) < 2:
                pbar.update(len(cc))
                continue

            n_local = len(cc)
            glob2loc[cc] = np.arange(n_local)

            # This component's edges, localised. `comp` labels every vertex with
            # the lowest vertex ID in its component and `cc` is ascending, so
            # `cc[0]` is that label and keys straight into `espan`.
            lo, hi = espan[int(cc[0])]
            sub_edges = glob2loc[edges_by_cc[lo:hi]].astype(np.uint32)
            w0 = lengths_by_cc[lo:hi]

            # Find root within component
            this_root = int(glob2loc[root]) if glob2loc[root] >= 0 else 0

            # The graph the loop re-weights, and a pristine one for the
            # invalidation radius - see this function's docstring.
            G = fastcore.GeodesicGraph(sub_edges, n_local, weights=w0)
            G_orig = fastcore.GeodesicGraph(sub_edges, n_local, weights=w0)

            # `min_length` is the only thing that needs the *length* of a path
            # rather than the path itself, and the graph does not hand its
            # weights back - so that one case shadows them here. Skip the whole
            # apparatus when there is no `min_length` to test, which is the
            # default.
            if min_length:
                w = w0.copy()  # kept in step with `G` alongside `set_weights`
                # Sorted (min, max) keys, so a path's edge IDs are one searchsorted
                keys = (np.minimum(sub_edges[:, 0], sub_edges[:, 1]).astype(np.int64)
                        * n_local + np.maximum(sub_edges[:, 0], sub_edges[:, 1]))
                key_order = np.argsort(keys)
                key_sorted = keys[key_order]

            # Get lengths of paths to all nodes from root. Left as the float32
            # `distances` returns: `paths` is only ever argmaxed and zeroed, so
            # widening it would double the bytes scanned to no end.
            paths = G.distances(sources=[this_root])[0]

            # Prep array for invalidation. Anything unreachable is retired up
            # front: it can never be picked as `farthest`, so leaving it valid
            # would spin the loop forever.
            valid = paths >= 0
            paths[~valid] = 0
            invalidated = n_local - int(np.count_nonzero(valid))

            while invalidated < n_local:
                # Find the farthest point
                farthest = int(np.argmax(paths))

                # Get path from root to farthest point, over the *current*
                # weights - already-extracted stretches are free to re-traverse
                path = G.path(this_root, [farthest])[0]

                if not len(path):
                    valid[farthest] = False
                    paths[farthest] = 0
                    invalidated += 1
                    pbar.update(1)
                    continue

                path_edges = np.column_stack((path[:-1], path[1:]))

                # Stop if farthest point is closer than min_length
                add = True
                if min_length:
                    # Get IDs of edges along the path. This should only be
                    # distance to the first branchpoint from the tip since we
                    # set other weights to zero
                    q = (np.minimum(path_edges[:, 0], path_edges[:, 1]).astype(np.int64)
                         * n_local + np.maximum(path_edges[:, 0], path_edges[:, 1]))
                    eids = key_order[np.searchsorted(key_sorted, q)]
                    add = w[eids].sum() >= min_length

                if add:
                    # Add these new edges
                    all_edges.append(cc[path_edges])

                # Must set weights along path to 0 so that this path is
                # taken again in future iterations
                G.set_weights(path_edges, 0.0)
                if min_length:
                    w[eids] = 0  # the shadow copy, kept in step right here

                # Get all nodes within `inv_dist` to this path, and which node of
                # it each is nearest. Ties among equidistant path nodes break
                # differently here than in `_invalidate_igraph`, which lets scipy
                # decide - so a vertex sitting exactly between two of them can map
                # to either. The pick is arbitrary in both.
                #
                # The path itself needs no separate retiring: every source comes
                # back in `near` at distance 0.
                near, _, sources = G_orig.ball(path, inv_dist)

                # Invalidate. Counting the still-valid ones *before* clearing them
                # keeps the tally O(ball) rather than an O(n_local) rescan per
                # iteration - which on a fine `inv_dist` is tens of thousands of
                # sweeps over the whole component.
                newly = int(np.count_nonzero(valid[near]))
                invalidated += newly
                valid[near] = False
                paths[near] = 0

                # Update mesh vertex to skeleton node map
                mesh_map[cc[near]] = cc[sources]

                pbar.update(newly)

            glob2loc[cc] = -1

    edges = (np.vstack(all_edges) if all_edges
             else np.zeros((0, 2), dtype=np.int64))

    return edges, mesh_map


def find_far_points(G):
    """Find the most distal points in the graph."""
    # Use the first node in graph as seed
    source = target = list(G.nodes)[0]
    dist = 0
    while True:
        # Turn into a depth-first
        tree = nx.bfs_tree(G, source=target)
        lp = nx.dag_longest_path(tree, weight=None)
        if len(lp) <= dist:
            break
        source, target = lp[0], lp[-1]
        dist = len(lp)

    return source, target
