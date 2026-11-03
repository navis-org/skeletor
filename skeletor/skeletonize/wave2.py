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
import pandas as pd
import scipy.sparse
import scipy.sparse.csgraph

from tqdm.auto import tqdm
from scipy.spatial import cKDTree

from .. import _fastcore
from ..utilities import make_trimesh, get_edges_unique
from .base import Skeleton
from .utils import (adjacency, connected_components, group_edges_by_component,
                    prep_origins)
from .wave import _grouped_agg

__all__ = ["by_wavefront_exact"]


def by_wavefront_exact(mesh, step_size, origins=None, radius_agg="mean", progress=True):
    """Skeletonize a mesh using wave fronts with an exact step size.

    This is a version of the `by_wavefront` function in which the wave front
    moves _exactly_ the given distance along the mesh (see `step_size` parameter) instead
    of hopping from vertex to vertex. This can give better results on meshes with a
    low vertex density but is computationally more expensive: the difference between
    the two methods depends heavily on the mesh and the step size, but as a rule of thumb,
    the exact version is about 3-4x slower than the hop version.

    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to be skeletonize. Can an object that has
                    ``.vertices`` and ``.faces`` properties  (e.g. a
                    trimesh.Trimesh) or a tuple ``(vertices, faces)`` or a
                    dictionary ``{'vertices': vertices, 'faces': faces}``.
    step_size :     float | int (>0)
                    The distance each the wave fronts move along the mesh
                    in each step. To get a feel for what a good value might be,
                    you can check the average edge length of the mesh
                    (see example).
    origins :       int | list of ints, optional
                    Vertex ID(s) where the wave(s) are initialized. If there
                    is no origin for a given connected component, will fall
                    back to semi-random origin.
    radius_agg :    "mean" | "median" | "max" | "min" | "percentile75" | "percentile25"
                    Function used to aggregate radii over sample (i.e. the
                    vertices forming a ring that we collapse to its center).
    progress :      bool
                    If True, will show progress bar.

    See Also
    --------
    `skeletor.skeletonize.by_wavefront()`
                    This is the original `by_wavefront` function in which the wave front
                    jumps from vertex to vertex. This is much faster but can give bad
                    results on low-resolution meshes. See its docstring for
                    references on the underlying (Reeb graph / geodesic level
                    set) approach.

    Returns
    -------
    skeletor.Skeleton
                    Holds results of the skeletonization and enables quick
                    visualization.

    Example
    -------
    >>> import skeletor as sk
    >>> mesh = sk.example_mesh()
    >>> # Check the average edge length
    >>> mesh.edges_unique_length.mean()
    119.325
    >>> # A good step size is usually a good bit smaller than the average edge length
    >>> skeleton = sk.skeletonize.by_wavefront_exact(mesh, step_size=50)

    """
    agg_map = {
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "median": np.median,
        "percentile75": lambda x: np.percentile(x, 75),
        "percentile25": lambda x: np.percentile(x, 25),
    }
    assert radius_agg in agg_map, f'Unknown `radius_agg`: "{radius_agg}"'
    rad_agg_func = agg_map[radius_agg]

    mesh = make_trimesh(mesh, validate=False)

    centers_final, radii_final, parents, mesh_map = _cast_waves(
        mesh,
        step_size=step_size,
        origins=origins,
        radius_agg=radius_agg,
        rad_agg_func=rad_agg_func,
        progress=progress,
    )

    # Initialize the SWC dataframe
    swc = pd.DataFrame()
    swc["node_id"] = np.arange(0, len(centers_final))
    swc["parent_id"] = parents
    swc["x"] = centers_final[:, 0]
    swc["y"] = centers_final[:, 1]
    swc["z"] = centers_final[:, 2]
    swc["radius"] = radii_final

    return Skeleton(swc=swc, mesh=mesh, mesh_map=mesh_map, method="wavefront_exact")


# -----------------------------------------------------------------------------
# Graph primitives.
#
# `_cast_waves` below carries its graph as a plain edge list plus parallel
# per-vertex arrays rather than as a graph object. That is what lets the
# bookkeeping around it vectorize - it used to be the bulk of the runtime - and
# it leaves only a handful of genuine graph questions, which these answer.
#
# Each has a navis-fastcore implementation and a fallback, and the two agree:
# same partition, same tree, same distances. Only the primitives differ, not the
# algorithm, so there is one implementation of `_cast_waves` to keep correct.
# -----------------------------------------------------------------------------


def _geodesic(edges, n_nodes, sources, weights=None):
    """Geodesic distance from the nearest of `sources`, in float64.

    The width is not incidental. `round(dist / step_size)` decides which
    wavefront a vertex belongs to, and float32 Dijkstra drifts by ~5e-7 relative
    over a few thousand hops - enough to put vertices in a neighbouring front.
    That is a topology change, not a rounding wobble. At float64 fastcore and
    scipy agree bit for bit, so the two backends give the same skeleton.

    `weights=None` gives hop counts, i.e. breadth-first layers.
    """
    if _fastcore.has('geodesic_matrix_graph(dtype)'):
        dist = _fastcore.fastcore.geodesic_matrix_graph(
            edges, n_nodes, weights=weights, sources=sources, dtype=np.float64)
        dist[dist < 0] = np.inf          # fastcore marks unreachable with -1
        return dist.min(axis=0)

    return scipy.sparse.csgraph.dijkstra(adjacency(edges, n_nodes, weights).tocsr(),
                                         directed=False, indices=sources,
                                         min_only=True)


def _level_set_components(edges, n_nodes, labels):
    """Connected components of every level set at once.

    I.e. the components of the graph with all edges spanning two different labels
    removed - which is what a wavefront is. Negative labels are excluded rather
    than grouped (`_cast_waves` never produces one, but the two backends agree on
    it either way).

    The two number the components differently; only the partition is used.
    """
    if _fastcore.has('level_set_components'):
        ids, n = _fastcore.fastcore.level_set_components(edges, n_nodes, labels)
        return np.asarray(ids, dtype=np.int64), int(n)

    ends = labels[edges]
    within = (ends[:, 0] == ends[:, 1]) & (ends[:, 0] >= 0)
    n, ids = scipy.sparse.csgraph.connected_components(
        adjacency(edges[within], n_nodes), directed=False)
    ids = ids.astype(np.int64)
    ids[labels < 0] = -1
    return ids, n


def _contract(edges, mapping):
    """Collapse nodes onto `mapping` and return the simplified edge list.

    Self-loops (both ends in one group) and duplicates are dropped. The result is
    sorted and deduplicated in *both* branches on purpose: `orient_forest` picks
    among equal-depth parents by the order it meets them, so an edge list in a
    different order would give a different - equally valid, but different - tree,
    and the two backends would drift apart.
    """
    # fastcore's version is parallel, and dispatching to the pool costs ~50 us
    # before it has looked at an edge - longer than numpy takes to contract a
    # few thousand of them (the two cross over at ~6k). On a mesh shattered into
    # thousands of specks that fixed cost was a quarter of the whole run.
    if _fastcore.has('contract_vertices') and len(edges) > 6_000:
        el = np.asarray(_fastcore.fastcore.contract_vertices(
            edges, mapping.astype(np.uint32)))
    else:
        el = mapping[edges]
    el = el.reshape(-1, 2).astype(np.int64)
    el = el[el[:, 0] != el[:, 1]]
    if not len(el):
        return el
    # Pack each (lo, hi) pair into one int64 and dedupe that. `unique(axis=0)`
    # takes a void-dtype view and lexsorts it - 538 ms against 26 ms here at 1M
    # rows, which is a quarter of the whole fallback. `np.unique` returns sorted
    # keys, so unpacking gives the same lexicographically ordered rows.
    lo = np.minimum(el[:, 0], el[:, 1])
    hi = np.maximum(el[:, 0], el[:, 1])
    keys = np.unique(lo * (int(el.max()) + 1) + hi)
    return np.column_stack(np.divmod(keys, int(el.max()) + 1))


def _bfs_tree(edges, n_nodes, root):
    """Breadth-first tree, breaking ties towards the lowest node ID.

    Breadth-first because a depth-first walk mangles branch points, turning
    ``0-1-2-3`` with ``4`` hanging off ``1`` into ``0-1-4-2-3``.

    :func:`~skeletor.skeletonize.utils.orient_forest` gives *a* breadth-first
    tree, but which one depends on the order its search happens to settle
    equal-depth nodes in, and the two backends do not agree on that. The
    contracted graph still holds cycles - only non-bridge edges were dropped - so
    that choice decides which cycle edge gets broken, and the skeletons come out
    differing by a few tenths of a percent of cable. Both are valid; neither is
    more correct. Fixing the rule here is what makes the two interchangeable,
    which matters because whether fastcore is installed is not something a user
    should be able to see in their skeleton.

    Depths are the part both backends already agree on: they are the breadth-first
    layers, which are unique. Only the choice of parent *within* a layer is free,
    and taking the lowest ID settles it.
    """
    if not len(edges):
        return np.full(n_nodes, -1, dtype=np.int64)

    # Named root, plus the lowest node of any component it cannot reach.
    # `_components` labels each node with its component's lowest, so the labels
    # *are* those fallback roots - no traversal needed to find them.
    labels = connected_components(edges, n_nodes)
    roots = np.unique(labels)
    if root >= 0:
        roots = np.append(roots[roots != labels[root]], root)
    depth = _geodesic(edges, n_nodes, roots)

    d_ends = depth[edges]
    closer = d_ends[:, 0] < d_ends[:, 1]
    child = np.where(closer, edges[:, 1], edges[:, 0])
    parent = np.where(closer, edges[:, 0], edges[:, 1])
    # Only edges between adjacent layers can be tree edges
    step = np.abs(d_ends[:, 0] - d_ends[:, 1]) == 1
    child, parent = child[step], parent[step]

    parents = np.full(n_nodes, -1, dtype=np.int64)
    if len(child):
        # Lowest parent per child: sort, then take the first of each run
        srt = np.lexsort((parent, child))
        child, parent = child[srt], parent[srt]
        first = np.flatnonzero(np.r_[True, child[1:] != child[:-1]])
        parents[child[first]] = parent[first]
    return parents


def _bridges(edges, n_nodes):
    """Mask of edges whose removal would disconnect their component."""
    if _fastcore.has('bridges'):
        return np.asarray(_fastcore.fastcore.bridges(edges, n_nodes), dtype=bool)

    # No scipy equivalent, so igraph - which skeletor requires anyway - does it.
    import igraph as ig
    G = ig.Graph(n=n_nodes, edges=edges.tolist(), directed=False)
    mask = np.zeros(len(edges), dtype=bool)
    mask[G.bridges()] = True
    return mask


def _group_by_label_csr(labels, n_labels):
    """Item indices bucketed by label, as CSR ``(indptr, indices)``.

    The same grouping as :func:`group_by_label` for labels already dense in
    ``[0, n_labels)``, but as a counting sort rather than an ``argsort``: the
    bucket offsets are its first half anyway, and scipy's ``coo_tocsr`` is the
    second half in C - 14 ms against 113 ms for the comparison sort at 2.6M keys.
    Within a bucket the items stay in ascending order.
    """
    indptr = np.zeros(n_labels + 1, dtype=np.int64)
    np.cumsum(np.bincount(labels, minlength=n_labels), out=indptr[1:])
    indices = scipy.sparse.coo_matrix(
        (np.arange(1, len(labels) + 1), (labels, np.arange(len(labels)))),
        shape=(n_labels, len(labels))).tocsr().data - 1
    return indptr, indices


def _stitch_pairs(face_edges, first_step, n_ins, offs, n_orig):
    """Join synthetic vertices that sit on the same face at the same step.

    A front crossing a face enters through one edge and leaves through another,
    so the two synthetic vertices it put on those edges are part of the same
    ring. Along an edge the synthetic vertices sit at consecutive steps
    ``first_step[e], first_step[e] + 1, ...`` and carry consecutive IDs, so for
    each pair of edges of a face the vertices to join are simply those on the
    overlap of the two step ranges - no need to tabulate every (vertex, face,
    step) and group it, which is a sort over twice as many rows as there are
    synthetic vertices.

    Parameters
    ----------
    face_edges :    (F, 3) array
                    Local edge IDs of each face's edges.
    first_step :    (E, ) array
                    Step of the first synthetic vertex on each edge (anything
                    for edges without one).
    n_ins :         (E, ) array
                    Number of synthetic vertices on each edge.
    offs :          (E + 1, ) array
                    Cumulative ``n_ins``, so edge ``e``'s vertices are
                    ``n_orig + offs[e] + j``.
    n_orig :        int

    Returns
    -------
    (P, 2) array of local vertex IDs to join. A face with a repeated edge
    (degenerate) yields self-pairs, which the contraction drops.

    """
    # The three edge pairs of every face, as two long columns
    a = face_edges[:, [0, 0, 1]].ravel()
    b = face_edges[:, [1, 2, 2]].ravel()
    lo = np.maximum(first_step[a], first_step[b])
    hi = np.minimum(first_step[a] + n_ins[a], first_step[b] + n_ins[b])
    cnt = np.maximum(hi - lo, 0)
    which = np.flatnonzero(cnt)
    if not len(which):
        return np.zeros((0, 2), dtype=np.int64)
    cnt = cnt[which]
    owner = np.repeat(which, cnt)
    starts = np.zeros(len(cnt) + 1, dtype=np.int64)
    np.cumsum(cnt, out=starts[1:])
    step = lo[owner] + (np.arange(len(owner)) - np.repeat(starts[:-1], cnt))
    a, b = a[owner], b[owner]
    return np.column_stack((n_orig + offs[a] + (step - first_step[a]),
                            n_orig + offs[b] + (step - first_step[b])))


def _cast_waves(mesh, step_size, origins=None, radius_agg='mean',
                rad_agg_func=np.mean, progress=True):
    """Cast waves across mesh.

    Unlike :func:`skeletor.skeletonize.wave._cast_waves`, which hops from vertex
    to vertex, the front here advances exactly ``step_size`` at a time: synthetic
    vertices are inserted along mesh edges at every multiple of it, those sitting
    on the same face at the same multiple are joined, and each resulting ring
    collapses to its centre.

    Returns
    -------
    centers :   (N, 3) array
    radii :     (N, ) array
    parents :   (N, ) array
                Child -> parent, ``-1`` at the roots.
    mesh_map :  (n_vertices, ) array
                Skeleton node for each mesh vertex.

    """
    origins = prep_origins(origins)

    if step_size <= 0:
        raise ValueError('`step_size` must be > 0')

    # Plain ndarray view of the vertices. Indexing `mesh.vertices` directly hands
    # back trimesh TrackedArrays, and every arithmetic op on one of those pays
    # for a `__array_wrap__` call.
    verts = np.asarray(mesh.vertices)
    n_verts = len(verts)

    edges_unique, edges_unique_length = get_edges_unique(mesh, lengths=True)
    edges_unique = np.asarray(edges_unique, dtype=np.int64)
    edges_unique_length = np.asarray(edges_unique_length, dtype=np.float64)

    comp, ccs, eorder, espan = group_edges_by_component(edges_unique, n_verts)

    # The faces of each component (for stitching the fronts) and each edge's
    # position within its component's edge slice, both set up once here rather
    # than searched for per component. A face belongs to the component of its
    # first vertex; `ccs` is in ascending label order, so the position of a
    # component's label in that list is its index.
    fue = np.asarray(mesh.faces_unique_edges, dtype=np.int64)
    cc_labels = np.array([cc[0] for cc in ccs], dtype=np.int64)
    face_cc = np.searchsorted(cc_labels, comp[np.asarray(mesh.faces)[:, 0]])
    f_indptr, f_indices = _group_by_label_csr(face_cc, len(ccs))
    ge2le = np.empty(len(edges_unique), dtype=np.int64)
    lo_of_edge = np.zeros(len(edges_unique), dtype=np.int64)
    for lo, hi in espan.values():
        lo_of_edge[lo:hi] = lo
    ge2le[eorder] = np.arange(len(eorder)) - lo_of_edge

    # Accumulated per component and concatenated at the end, rather than one
    # Python object appended per skeleton node
    centers, parents = [], []
    n_nodes = 0
    mesh_map = np.full(n_verts, -1, dtype=np.int64)

    # The radius aggregation is deferred: a grouped reduction costs ~60 us
    # before it has looked at a value, so it is run once over every front of
    # every component after the loop rather than once per component. Groups
    # are independent, so that is the same result. `node_front` holds each
    # node's group in that global numbering, `-1` for nodes without one.
    front_dist, front_group, node_front = [], [], []
    front_radius = []
    n_groups = n_pending = 0

    def flush_radii():
        """Aggregate the pending fronts' radii (bounds the buffer)."""
        nonlocal n_pending
        if front_dist:
            done = sum(len(r) for r in front_radius)
            front_radius.append(_grouped_agg(np.concatenate(front_dist),
                                             np.concatenate(front_group) - done,
                                             radius_agg, rad_agg_func))
            front_dist.clear()
            front_group.clear()
        n_pending = 0

    def emit_single(cc, vertex):
        """Collapse a whole component onto one node at `vertex`."""
        nonlocal n_nodes
        mesh_map[cc] = n_nodes
        centers.append(verts[vertex][None, :])
        node_front.append(np.full(1, -1, dtype=np.int64))
        parents.append(np.full(1, -1, dtype=np.int64))
        n_nodes += 1

    glob2loc = np.full(n_verts, -1, dtype=np.int64)

    with tqdm(desc='Skeletonizing', total=n_verts,
              disable=not progress, leave=False) as pbar:
        for ci, cc in enumerate(ccs):
            pbar.update(len(cc))
            n_orig = len(cc)
            glob2loc[cc] = np.arange(n_orig)

            # Components are labelled by their lowest vertex and `cc` is
            # ascending, so `cc[0]` is this component's label
            span = espan.get(int(cc[0]))
            if span is None:                  # isolated vertex, no edges at all
                emit_single(cc, cc[0])
                continue

            lo, hi = span
            gedge = eorder[lo:hi]             # global edge ids, for face lookup
            sub_edges = glob2loc[edges_unique[gedge]]
            weights = edges_unique_length[gedge]

            # ---- seed -----------------------------------------------------
            here = origins[np.isin(origins, cc)] if len(origins) else origins
            if len(here):
                seed = int(glob2loc[here[0]])
            elif len(origins):
                np.random.seed(1985)          # make seeds predictable
                seed = int(np.random.choice(np.arange(n_orig)))
            else:
                # The vertex with the "peakiest" vertex degree
                seed = int(np.argmin(np.bincount(sub_edges.ravel(),
                                                 minlength=n_orig)))

            dist = _geodesic(sub_edges, n_orig, [seed], weights)

            finite = dist[np.isfinite(dist)]
            mx = finite.max() if len(finite) else 0.0
            if mx < step_size:
                # Nothing is a full step away, so the component is one node
                emit_single(cc, cc[seed])
                continue

            # `np.isclose(dist % step_size, 0)`, spelled out: the same test
            # (|x| <= atol, rtol has nothing to scale) without its overhead
            dist_is_step = np.abs(dist % step_size) <= 1e-8

            # ---- orient every edge proximal -> distal wrt the seed ---------
            d_ends = dist[sub_edges]
            flip = d_ends[:, 0] > d_ends[:, 1]
            prox = np.where(flip, sub_edges[:, 1], sub_edges[:, 0])
            distal = np.where(flip, sub_edges[:, 0], sub_edges[:, 1])

            cc_pos = verts[cc]
            prox_pos = cc_pos[prox]
            vect = cc_pos[distal] - prox_pos
            with np.errstate(invalid='ignore', divide='ignore'):
                # Zero-length edges give NaN here, harmlessly: nothing is
                # ever inserted on them.
                vect /= np.linalg.norm(vect, axis=1)[:, None]

            # ---- how many synthetic vertices does each edge get? -----------
            # A vertex already sitting on a multiple of `step_size` is itself the
            # front there, so insertion starts one multiple further out.
            d_prox, d_dist = dist[prox], dist[distal]
            start_ins = np.where(dist_is_step[prox],
                                 (np.ceil(d_prox / step_size) + 1) * step_size,
                                 np.ceil(d_prox / step_size) * step_size)
            # `np.arange(start, stop, step)` yields ceil((stop - start) / step)
            n_ins = np.where(np.isfinite(d_dist) & (d_dist > start_ins),
                             np.ceil((d_dist - start_ins) / step_size),
                             0).astype(np.int64)
            total_new = int(n_ins.sum())

            # ---- create them, all at once ---------------------------------
            owner = np.repeat(np.arange(len(sub_edges)), n_ins)  # edge of each
            offs = np.zeros(len(sub_edges) + 1, dtype=np.int64)
            np.cumsum(n_ins, out=offs[1:])
            j = np.arange(total_new) - offs[owner]      # index along its edge

            d_new = start_ins[owner] + j * step_size
            new_ids = n_orig + np.arange(total_new)
            new_pos = vect[owner]
            new_pos *= (d_new - d_prox[owner])[:, None]
            new_pos += prox_pos[owner]

            n_total = n_orig + total_new
            all_vertices = np.vstack([cc_pos, new_pos])

            # ---- rebuild the edge list ------------------------------------
            # A split edge becomes prox -> v0 -> v1 -> ... -> distal. The
            # predecessor of the j-th insertion is `prox` when j == 0 and the
            # previous insertion otherwise.
            split = n_ins > 0
            chain = np.column_stack((np.where(j == 0, prox[owner], new_ids - 1),
                                     new_ids))
            closing = np.column_stack((new_ids[offs[1:][split] - 1], distal[split]))
            untouched = np.column_stack((prox[~split], distal[~split]))
            edges = np.vstack([untouched, chain, closing]).astype(np.int64)
            # Consecutive synthetic vertices on one edge are a step apart by
            # construction, so of the chain only its first link can ever lie
            # within a front. The level sets are searched over this shorter list;
            # the contraction, which needs every link, gets the full one.
            first_link = np.column_stack((prox[split], new_ids[offs[:-1][split]]))
            edges_lvl = [untouched, first_link, closing]

            # ---- which step is each vertex on? -----------------------------
            step_of = np.zeros(n_total, dtype=np.int64)
            is_step = np.zeros(n_total, dtype=bool)

            step_of[new_ids] = np.round(d_new / step_size).astype(np.int64)
            is_step[new_ids] = True
            is_step[:n_orig] |= dist_is_step & np.isfinite(dist)
            is_step[seed] = True
            # Original vertices take the rounded distance, whether or not they
            # landed on a step
            step_of[:n_orig] = np.round(
                np.where(np.isfinite(dist), dist, 0) / step_size).astype(np.int64)

            # ---- stitch each front across the faces it crosses --------------
            # A synthetic vertex sits on a mesh edge, so it belongs to that edge's
            # faces; two of them on the same face at the same step are part of the
            # same ring and get joined.
            if total_new:
                first_step = np.zeros(len(sub_edges), dtype=np.int64)
                first_step[split] = step_of[new_ids[offs[:-1][split]]]
                face_edges = ge2le[fue[f_indices[f_indptr[ci]:f_indptr[ci + 1]]]]
                cross = _stitch_pairs(face_edges, first_step, n_ins, offs, n_orig)
            else:
                cross = np.zeros((0, 2), dtype=np.int64)

            # ---- the fronts themselves --------------------------------------
            # Note we could additionally check for weakly connected components,
            # i.e. splitting  0-1  5-7  where 3-4 bridges two rings. In practice
            #                 | |  | |
            #                 2-3--4-6
            # that leads to lots of side branches, so we don't.
            # The stitches only matter here: both ends of one are on the same
            # step by construction, so they always land in one front and would
            # only ever contract to a self-loop.
            front_ids, n_fronts = _level_set_components(
                np.vstack(edges_lvl + [cross]).astype(np.int64), n_total, step_of)
            del cross, edges_lvl, first_link, untouched, chain, closing

            # ---- collapse each front to its centre --------------------------
            # Every step here is a grouped reduction. Doing it per front is a
            # Python loop over as many fronts as the skeleton has nodes.
            #
            # A front collapses onto its lowest-numbered member, which is what
            # `order[starts]` gives - `group_by_label` is stable, so members come
            # out in ascending order.
            # Lowest-numbered member of each front. A reverse scatter, not
            # `group_by_label`: that argsorts all `n_total` nodes to extract
            # `n_fronts` minima (39 ms against 1 ms at 700k nodes). Walking
            # backwards means the smallest index is written last and wins.
            rep = np.empty(n_fronts, dtype=np.int64)
            rep[front_ids[::-1]] = np.arange(n_total)[::-1]
            mapping = rep[front_ids]

            # Centre and radius are measured over a front's *step* nodes only. A
            # front with none of them is not a ring at all - it is a stray tip
            # beyond the last step - and gets detached rather than collapsed.
            step_nodes = np.flatnonzero(is_step)
            fid = front_ids[step_nodes]
            n_step = np.bincount(fid, minlength=n_fronts)

            # `_grouped_agg` wants contiguous ids covering every group, so
            # renumber over the fronts that actually have step nodes
            live = np.flatnonzero(n_step > 0)
            dense = np.full(n_fronts, -1, dtype=np.int64)
            dense[live] = np.arange(len(live))

            step_pos = all_vertices[step_nodes]
            sums = np.stack([np.bincount(fid, weights=step_pos[:, k],
                                         minlength=n_fronts) for k in range(3)],
                            axis=1)
            front_center = np.zeros((n_fronts, 3))
            front_center[live] = sums[live] / n_step[live, None]
            # Squared distance to the centre, in place: `step_pos` is as long as
            # there are synthetic vertices and a chain of temporaries that size
            # is what this step used to spend its time on.
            step_pos -= front_center[fid]
            np.square(step_pos, out=step_pos)
            d = step_pos.sum(axis=1)
            np.sqrt(d, out=d)
            del step_pos
            front_dist.append(d)
            front_group.append(dense[fid] + n_groups)
            n_pending += len(d)

            has_center = np.zeros(n_total, dtype=bool)
            this_center = np.zeros((n_total, 3))
            has_center[rep[live]] = True
            this_center[rep[live]] = front_center[live]

            # ---- contract, then drop what is left isolated ------------------
            el = _contract(edges, mapping)

            # Edges into a dead front go. Doing this after the contraction
            # rather than before is the same set of edges - an edge is dropped
            # iff either end's front is dead, which contracting does not change
            # - over a few hundred thousand rows instead of tens of millions.
            if len(live) != n_fronts:
                dead = np.zeros(n_total, dtype=bool)
                dead[rep[n_step == 0]] = True
                el = el[~(dead[el[:, 0]] | dead[el[:, 1]])]

            deg = np.bincount(el.ravel(), minlength=n_total)
            keep = np.flatnonzero((deg > 0) & has_center)
            if not len(keep):
                continue
            renum = np.full(n_total, -1, dtype=np.int64)
            renum[keep] = np.arange(len(keep))
            el = renum[el]
            el = el[(el >= 0).all(axis=1)]

            # ---- drop edges that skip a front, but never disconnect ---------
            # `step_of` is an integer step index, so neighbouring fronts differ
            # by exactly 1.
            if len(el):
                sd = step_of[keep][el]
                bad = np.abs(sd[:, 0] - sd[:, 1]) != 1
                bad &= ~_bridges(el, len(keep))
                el = el[~bad]

            # ---- orient the tree from the seed -------------------------------
            root = renum[mapping[seed]]
            this_parents = _bfs_tree(el, len(keep), root)

            offset = n_nodes
            parents.append(np.where(this_parents >= 0, this_parents + offset, -1))
            centers.append(this_center[keep])
            node_front.append(dense[front_ids[keep]] + n_groups)
            n_groups += len(live)
            n_nodes += len(keep)
            if n_pending > 2_000_000:
                flush_radii()

            # ---- mesh vertex -> skeleton node --------------------------------
            node_of_local = renum[mapping[:n_orig]]
            got = node_of_local >= 0
            mesh_map[cc[got]] = offset + node_of_local[got]

            # A handful of tip vertices lose their node (the detached fronts
            # above); each goes to the nearest surviving centre
            orphans = cc[~got]
            if len(orphans):
                mesh_map[orphans] = offset + cKDTree(this_center[keep]).query(
                    verts[orphans])[1]


    centers = np.vstack(centers)

    # Radii of the fronts still pending, then onto their nodes
    flush_radii()
    node_front = np.concatenate(node_front)
    radii = np.zeros(n_nodes)
    if n_groups:
        front_radius = np.concatenate(front_radius)
        has = node_front >= 0
        radii[has] = front_radius[node_front[has]]

    # Catch any mesh vertices that never made it into the graph. Normally a no-op.
    still = np.flatnonzero(mesh_map == -1)
    if len(still):
        mesh_map[still] = cKDTree(centers).query(verts[still])[1]

    return centers, radii, np.concatenate(parents), mesh_map
