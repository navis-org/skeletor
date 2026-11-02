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

import numpy as np

from scipy.sparse.csgraph import dijkstra
from tqdm.auto import tqdm

from .. import _fastcore
from ..utilities import make_trimesh, get_edges_unique
from .base import Skeleton
from .utils import edges_to_swc, group_edges_by_component

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
                    If provided, terminal branches shorter than this are pruned
                    off the finished skeleton - a branch being a tip plus the
                    unbranched run above it, up to the first fork or the root.
                    Pruning one can leave its parent a tip in turn, so this
                    repeats until nothing short is left; the result therefore
                    depends only on the skeleton, not on the order the branches
                    happened to be found in. Note that a component with no fork
                    in it is one branch all the way down, so a short enough
                    fragment drops out whole. Use this to get rid of noise, but
                    note that the vertices of a pruned branch are left unmapped
                    and show up with index -1 in `Skeleton.mesh_map`.
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

    # The primitives `_invalidate_fastcore` itself calls - not a performance
    # gate. `group_edges_by_component` needs `connected_components_graph` but
    # falls back to scipy on its own, so naming it here would only refuse the
    # fast path over something this function never touches.
    if _fastcore.has('GeodesicGraph.path', 'GeodesicGraph.distances',
                     'GeodesicGraph.ball', 'GeodesicGraph.nearest'):
        edges, mesh_map = _invalidate_fastcore(mesh, edges_unique,
                                               edges_unique_length, inv_dist,
                                               root, progress)
    else:
        edges, mesh_map = _invalidate_igraph(mesh, edges_unique,
                                             edges_unique_length, inv_dist,
                                             root, progress)

    # Bail out before the empty edge list reaches the graph construction below,
    # where it surfaces as an unhelpful IndexError
    if not len(edges):
        raise ValueError('Unable to skeletonize: mesh has no edges.')

    # `min_length` is applied here rather than inside the backends, so that it
    # means one thing. Tested during extraction it could only ask "is this
    # branch short *as found*", which depends on what had been extracted so far
    # - a branch rejected as noise came back whole the moment a later path ran
    # through it. Asked of the finished skeleton it is a property instead.
    if min_length:
        edges = _prune_twigs(edges, mesh.vertices, min_length)
        if not len(edges):
            raise ValueError(
                f'Unable to skeletonize: `min_length` ({min_length}) is longer '
                'than every branch in the mesh, so no skeleton nodes are left. '
                'Try a smaller value.')

    # Generate the SWC table. Both backends hand back each skeleton edge once,
    # but the extracted paths can still close cycles, so this has to break those
    # as well as orient and re-index - which is what it does. Reciprocal pairs
    # are tolerated too.
    swc, new_ids = edges_to_swc(edges, coords=mesh.vertices)

    # Update vertex to node ID map. A vertex whose node did not survive - one
    # `min_length` pruned away - indexes the sentinel slot and stays -1, which
    # is what this function's docstring promises.
    mesh_map = new_ids[mesh_map]

    return Skeleton(swc=swc, mesh=mesh, mesh_map=mesh_map, method='teasar')


def _prune_twigs(edges, coords, min_length):
    """Drop terminal branches shorter than ``min_length``.

    A terminal branch is a tip - a node with a single neighbour - plus the
    unbranched run leading away from it, up to and including the edge into the
    first node that forks. A component with no fork in it is one branch end to
    end. Dropping a branch can leave its neighbour a tip in turn, so this runs
    to a fixed point: what comes back has no terminal branch shorter than
    `min_length`, whatever order the branches arrived in.

    Degree is measured on the *undirected* skeleton rather than on a rooted
    tree, which matters more than it looks. Rooting puts one node of each
    component at the top holding a child instead of a parent, so if that node
    happens to be a tip its branch is never a leaf and never gets tested - and
    which node that is comes from the rooting, not from the shape. Done over
    `orient_forest`, which roots at the lowest node ID, a mesh skeletonized from
    the default `root=0` kept a 60-long twig at `min_length=1000` that the same
    mesh rooted anywhere else dropped.

    Parameters
    ----------
    edges :         (N, 2) array
                    Skeleton edges, over node IDs indexing `coords`.
    coords :        (M, 3) array
                    Node locations.
    min_length :    float

    Returns
    -------
    (K, 2) array
                    The edges with both ends still standing, in input order.

    """
    ids = unique(edges)
    local = np.searchsorted(ids, edges)
    pos = np.asarray(coords)[ids]

    # Adjacency over the distinct undirected edges. Self-loops and reciprocal
    # pairs are tolerated on the way in (`edges_to_swc` tolerates them too), and
    # would otherwise show up as degree and stop a run short of a real tip.
    arcs = np.sort(local, axis=1)
    arcs = unique(arcs[arcs[:, 0] != arcs[:, 1]], axis=0)
    src = np.concatenate((arcs[:, 0], arcs[:, 1]))
    dst = np.concatenate((arcs[:, 1], arcs[:, 0]))
    arc_len = np.linalg.norm(pos[src] - pos[dst], axis=1)

    order = np.argsort(src, kind='stable')
    src, dst, arc_len = src[order], dst[order], arc_len[order]
    offs = np.zeros(len(ids) + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=len(ids)), out=offs[1:])
    offs_l, dst_l, len_l = offs.tolist(), dst.tolist(), arc_len.tolist()

    keep = np.ones(len(ids), dtype=bool)
    while True:
        # Surviving degree: an edge counts only while both its ends stand
        deg = np.bincount(src[keep[src] & keep[dst]], minlength=len(ids))
        keep_l = keep.tolist()

        dropped = False
        for tip in np.flatnonzero(keep & (deg == 1)).tolist():
            # A run between two tips is reached from both ends; whichever got
            # there first took the whole thing
            if not keep_l[tip]:
                continue

            run, node, prev, length = [], tip, -1, 0.0
            while True:
                run.append(node)
                step = nxt = -1
                for k in range(offs_l[node], offs_l[node + 1]):
                    cand = dst_l[k]
                    if cand != prev and keep_l[cand]:
                        nxt, step = cand, len_l[k]
                        break
                if nxt < 0:
                    break                      # nothing left to walk into
                length += step                 # the edge out of the run counts
                if deg[nxt] != 2:
                    # A fork ends the run and stays standing; another tip is the
                    # far end of an unbranched component and goes with it.
                    if deg[nxt] == 1:
                        run.append(nxt)
                    break
                prev, node = node, nxt

            if length < min_length:
                keep[run] = False
                for v in run:
                    keep_l[v] = False
                dropped = True

        if not dropped:
            break

    return edges[keep[local].all(axis=1)]


def _invalidate_igraph(mesh, edges_unique, edges_unique_length, inv_dist,
                       root, progress):
    """TEASAR path extraction + invalidation, using igraph."""
    # Generate Graph (must be undirected)
    import igraph as ig
    G = ig.Graph(n=mesh.vertices.shape[0], edges=edges_unique, directed=False)
    G.es['weight'] = edges_unique_length

    all_edges = []
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

            # Distance to each vertex's mapped node, so the nearest one wins
            # rather than the last ball to reach it - see `_invalidate_fastcore`
            best = np.full(len(cc), np.inf)

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

                # Add these new edges
                g = cc[path]
                all_edges.append(np.column_stack((g[:-1], g[1:])))

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
                to_invalidate = dist <= inv_dist
                valid[to_invalidate] = False
                paths[to_invalidate] = 0

                # Update mesh vertex to skeleton node map, nearest node
                # winning. This path is a superset of the ones before it, so the
                # last ball is also the widest and plain assignment would nearly
                # agree - but only nearly, and `_invalidate_fastcore` balls the
                # new stretch alone, so both are pinned to the same rule.
                closer = dist < best
                best[closer] = dist[closer]
                mesh_map[cc[closer]] = cc[sources[closer]]

                pbar.update((~valid).sum() - invalidated)
                invalidated = (~valid).sum()

    edges = (unique(np.vstack(all_edges), axis=0) if all_edges
             else np.zeros((0, 2), dtype=np.int64))

    return edges, mesh_map


def _invalidate_fastcore(mesh, edges_unique, edges_unique_length, inv_dist,
                         root, progress):
    """TEASAR path extraction + invalidation, using navis-fastcore.

    Same algorithm as :func:`_invalidate_igraph` - the same skeleton out of it
    wherever the mesh has no exactly tied routes, and an equally valid one where
    it does (see :mod:`skeletor._fastcore`) - but it never re-weights the graph,
    which is what lets it scale.

    The igraph backend zeroes every path it extracts so that later searches may
    re-traverse it for free, then asks for the shortest path from the root to
    the next tip. Because the zeroed stretches cost nothing, what that search
    returns is always the skeleton so far, then the shortest path from the
    skeleton to the tip. Only that last stretch is new, and it can be had
    directly:

    - :meth:`GeodesicGraph.nearest` finds the skeleton node closest to the tip,
      stopping at the first one it settles;
    - :meth:`GeodesicGraph.path` then walks the tip to that node.

    Both searches are bounded by the length of the new branch, where the search
    they replace settles the whole component before it reaches its target. One
    term does still scale with the skeleton rather than the branch - `nearest`
    is handed the whole target list and charges for it - but it is a scan, not a
    search, and removing it would need an API that lets the caller keep the
    target set between calls. Dropping the re-weighting also leaves one pristine
    graph where there were two (a penalised one and a copy for the invalidation
    radius), since nothing is mutated any more.

    The stretch found this way meets the skeleton only at that one node: had it
    touched another, that node would have settled first and been the junction
    instead. So its edges are new every time, which is what lets this emit each
    of them once and hand back a list that needs no de-duplication - where
    igraph appends the whole root-to-tip path and dedupes on the way out.

    One thing needs care to keep the two in step: a vertex is mapped to the
    *nearest* node that invalidates it rather than to whichever ball covered it
    last. igraph re-balls the whole path every iteration, so its last ball is
    also its widest; balling only the new stretch is cheaper but would otherwise
    leave a vertex mapped to the first ball that happened to reach it. Both
    backends keep `best` for this, which makes the mapping independent of
    iteration order on both sides.

    Ties are still ties: a vertex equidistant from two nodes of the path that
    invalidated it, or two equally short routes to the same junction, go
    whichever way the underlying search settled first, and the backends decide
    that differently.
    """
    fastcore = _fastcore.fastcore
    n_verts = mesh.vertices.shape[0]
    edges_unique = np.asarray(edges_unique)
    edges_unique_length = np.asarray(edges_unique_length, dtype=np.float64)

    all_edges = []
    mesh_map = np.full(n_verts, fill_value=-1)

    # Split into connected components, largest first. Components are labelled by
    # their lowest vertex ID, so the grouping reproduces igraph's ordering before
    # we re-sort by size (which is stable too, so equal-sized components keep
    # that order).
    #
    # The graph below is built per component rather than once for the whole mesh
    # for the same reason the loop is per component: every array it carries
    # (`order`, `valid`) is indexed by local vertex, so a whole-mesh graph would
    # make each of them n_verts wide and each sweep over them O(components *
    # n_verts). Building per component is O(V + E) in total, since the components
    # partition the mesh.
    _, ccs, eorder, espan = group_edges_by_component(edges_unique, n_verts)
    ccs = sorted(ccs, key=len, reverse=True)

    # Reused across components; only the `cc` entries are ever set (and reset)
    glob2loc = np.full(n_verts, -1, dtype=np.int32)

    # Per-component scratch, indexed by *local* vertex, so a component only ever
    # touches the first `n_local` entries. Allocated once at full width rather
    # than per component, which on a fragmented mesh would hand the allocator
    # thousands of arrays. `best` is reset per component and those resets sum to
    # a single pass over the mesh; `skel` needs none, being written from index 0
    # outwards and only ever read below `n_skel`.
    best = np.empty(n_verts, dtype=np.float32)        # distance to mapped node
    skel = np.empty(n_verts, dtype=np.uint32)         # skeleton nodes, as found

    with tqdm(desc='Invalidating', total=n_verts,
              disable=not progress, leave=False) as pbar:
        for cc in ccs:
            # Isolated vertices contribute nothing to the skeleton (see the
            # igraph version for why this is a hard skip there)
            if len(cc) < 2:
                pbar.update(len(cc))
                continue

            n_local = len(cc)
            glob2loc[cc] = np.arange(n_local, dtype=np.int32)

            # This component's edges, localised. `comp` labels every vertex with
            # the lowest vertex ID in its component and `cc` is ascending, so
            # `cc[0]` is that label and keys straight into `espan`.
            # `eorder` is the edge list sorted by component, so this slice of
            # it indexes straight into the originals - no need to carry a
            # reordered copy of the whole edge list (two of them, at 24 bytes an
            # edge) alongside them for the length of the run.
            lo, hi = espan[int(cc[0])]
            esl = eorder[lo:hi]
            sub_edges = glob2loc[edges_unique[esl]].astype(np.uint32)

            # Find root within component
            this_root = int(glob2loc[root]) if glob2loc[root] >= 0 else 0

            # One graph, never mutated - both the branch searches and the
            # invalidation radius measure over the mesh's own lengths. The
            # weights are not kept: the graph holds them from here, and on a big
            # component that array is tens of MB to carry through the loop for
            # nothing.
            G = fastcore.GeodesicGraph(sub_edges, n_local,
                                       weights=edges_unique_length[esl])

            # Get lengths of paths to all nodes from root. Left as the float32
            # `distances` returns: it is only sorted and compared, so widening
            # it would double the bytes scanned to no end.
            paths = G.distances(sources=[this_root])[0]

            # Prep array for invalidation. Anything unreachable is retired up
            # front: it can never be picked as the tip, so leaving it valid
            # would spin the loop forever.
            valid = paths >= 0

            # Visit candidate tips farthest-first. A vertex's distance from the
            # root never changes and a vertex only ever goes valid -> invalid,
            # so one pointer walking this order replaces an O(n_local) argmax
            # per iteration - which on a fine `inv_dist` was half the run. The
            # sort is stable over `-paths`, so equally distant vertices come in
            # ascending ID, which is the one `argmax` would have returned.
            # Unreachable vertices have distance -1 and so sort past every
            # reachable one, then get skipped like any other invalid vertex.
            order = np.argsort(-paths, kind='stable')
            del paths                     # `order` and `valid` carry it from here
            ptr = 0

            # The skeleton starts as the root alone - which is where igraph's
            # first, unpenalised path starts from too.
            best[:n_local] = np.inf
            skel[0] = this_root
            n_skel = 1

            stride = 64
            while ptr < n_local:
                # Find the farthest still-valid tip. Over a component this walks
                # every vertex once, so it is cheap in total - but a block at a
                # time, since a scalar numpy read per step costs more than
                # gathering 64 of them does. `argmax` alone answers both "is any
                # of this block valid" and "which", for one dispatch instead of
                # two. The stride widens over a run of invalid vertices and
                # resets on a hit: a coarse `inv_dist` invalidates most of the
                # component per tip, leaving runs that would otherwise take
                # thousands of 64-wide gathers to cross.
                block = valid[order[ptr:ptr + stride]]
                hit = int(block.argmax())
                if not block[hit]:
                    ptr += stride
                    stride = min(stride * 2, 65536)
                    continue
                ptr += hit
                stride = 64
                tip = int(order[ptr])

                # The skeleton node this tip hangs off, and the stretch back to
                # it: the part of igraph's root-to-tip path that is actually new
                _, nearest = G.nearest(sources=[tip], targets=skel[:n_skel])
                junction = int(nearest[0])

                if junction < 0:
                    # Reachable only on a component's first pass, when the root
                    # is also its farthest vertex - every edge weight zero, say.
                    # igraph walks a one-node path here, which contributes no
                    # edges and balls the root alone; so does this. `nearest`
                    # never returns the source itself, so this is also the only
                    # way `seg` can come back shorter than an edge.
                    seg = np.asarray([tip], dtype=np.uint32)
                else:
                    seg = G.path(tip, [junction])[0]

                    # Extend the skeleton tree. Every node of the stretch bar
                    # the junction is new to it - see the docstring - so these
                    # edges have not been emitted before and will not be again.
                    skel[n_skel:n_skel + len(seg) - 1] = seg[:-1]
                    n_skel += len(seg) - 1

                    g = cc[seg]
                    all_edges.append(np.column_stack((g[:-1], g[1:])))

                # Get all nodes within `inv_dist` to this stretch, and which node
                # of it each is nearest.
                #
                # The stretch itself needs no separate retiring: every source
                # comes back in `near` at distance 0.
                near, near_dist, sources = G.ball(seg, inv_dist)

                # Invalidate. Counting the still-valid ones *before* clearing them
                # keeps the tally O(ball) rather than an O(n_local) rescan per
                # iteration - which on a fine `inv_dist` is tens of thousands of
                # sweeps over the whole component.
                if progress:
                    pbar.update(int(np.count_nonzero(valid[near])))
                valid[near] = False

                # Update vertex to skeleton node map, nearest node winning - see
                # the docstring for why this is not simply the last ball's say.
                closer = near_dist < best[near]
                closest = near[closer]
                best[closest] = near_dist[closer]
                mesh_map[cc[closest]] = cc[sources[closer]]

            glob2loc[cc] = -1

    edges = (np.vstack(all_edges) if all_edges
             else np.zeros((0, 2), dtype=np.int64))

    return edges, mesh_map
