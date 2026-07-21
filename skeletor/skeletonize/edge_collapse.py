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

import heapq

import numpy as np
import scipy.sparse
import scipy.spatial

from tqdm.auto import tqdm

from ..utilities import make_trimesh, get_edges_unique
from .base import Skeleton
from .utils import mst_over_mesh, edges_to_graph, make_swc

__all__ = ['by_edge_collapse']


def by_edge_collapse(mesh, shape_weight=1, sample_weight=0.1,
                     link_condition=False, progress=True):
    """Skeletonize a (contracted) mesh by iteratively collapsing edges.

    This algorithm (described in [1]) iteratively collapses edges that are part
    of a face until no more faces are left. Edges are chosen based on a cost
    function that penalizes collapses that would change the shape of the object
    or would introduce long edges.

    This is somewhat sensitive to the dimensions of the input mesh: too large
    and you might experience slow-downs or numpy OverflowErrors; too low and
    you might get skeletons that don't quite match the mesh (e.g. too few nodes).
    If you experience either, try down- or up-scaling your mesh, respectively.

    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to be skeletonize. Can an object that has
                    ``.vertices`` and ``.faces`` properties  (e.g. a
                    trimesh.Trimesh) or a tuple ``(vertices, faces)`` or a
                    dictionary ``{'vertices': vertices, 'faces': faces}``.
    shape_weight :  float, optional
                    Weight for shape costs which penalize collapsing edges that
                    would drastically change the shape of the object.
    sample_weight : float, optional
                    Weight for sampling costs which penalize collapses that
                    would generate prohibitively long edges.
    link_condition : bool, optional
                    If True, enforce the topological link condition from the
                    paper: an edge ``(u, v)`` is only collapsed if every vertex
                    adjacent to *both* ``u`` and ``v`` also forms a triangle
                    with them. This keeps collapses manifold but is **off by
                    default** because the (often non-watertight, non-manifold)
                    contracted meshes this pipeline deals with violate it for a
                    large fraction of edges - blocking those collapses produces
                    a sparser, worse skeleton. Since the final skeleton is
                    rebuilt via a minimum spanning tree over the surviving
                    vertices, manifold-preserving collapses are not required for
                    a good result. Enable it only for clean, manifold meshes.
    progress :      bool
                    If True, will show progress bar.

    Returns
    -------
    skeletor.Skeleton
                    Holds results of the skeletonization and enables quick
                    visualization.

    References
    ----------
    [1] Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh
        contraction. ACM Transactions on Graphics (TOG). 2008 Aug 1;27(3):44.

    """
    mesh = make_trimesh(mesh, validate=False)

    # Shorthand faces and edges
    # We convert to arrays to (a) make a copy and (b) remove potential overhead
    # from these originally being trimesh TrackedArrays
    edges = np.array(get_edges_unique(mesh))
    verts = np.array(mesh.vertices)

    # For cost calculations we will normalise coordinates
    # This prevents getting ridiculuously large cost values ?e300
    # verts = (verts - verts.min()) / (verts.max() - verts.min())
    edge_lengths = np.sqrt(np.sum((verts[edges[:, 0]] - verts[edges[:, 1]])**2, axis=1))

    # Get a list of faces: [(edge1, edge2, edge3), ...]
    face_edges = np.array(mesh.faces_unique_edges)
    # Make sure these faces are unique, i.e. no [(e1, e2, e3), (e3, e2, e1)]
    face_edges = np.unique(np.sort(face_edges, axis=1), axis=0)

    # Shape cost initialisation:
    # Each vertex has a matrix Q which is used to determine the shape cost
    # of collapsing each node. We need to generate a matrix (Q) for each
    # vertex, then when we collapse two nodes, we can update using
    # Qj <- Qi + Qj, so the edges previously associated with vertex i
    # are now associated with vertex j.

    # For each edge, generate a matrix (K). K is made up of two sets of
    # coordinates in 3D space, a and b. a is the normalised edge vector
    # of edge(i,j) and b = a x <coordinates of vertex i> (cross product).
    #
    # The matrix K takes the form:
    #
    #        Kij = 0, -az, ay, -bx
    #              az, 0, -ax, -by
    #             -ay, ax, 0,  -bz
    #
    # The first three columns are the skew-symmetric (cross-product) matrix of
    # `a`. Together with b = a x u this makes ``p^T (K^T K) p`` equal to the
    # squared distance of point p to the line through the edge - i.e. a proper
    # quadric error metric (and hence always >= 0).

    edge_co0, edge_co1 = verts[edges[:, 0]], verts[edges[:, 1]]
    # Unit vector along each edge. Guard against zero-length (degenerate) edges
    # which would otherwise produce NaNs that poison the cost matrices.
    safe_len = edge_lengths.reshape(edges.shape[0], 1).copy()
    safe_len[safe_len == 0] = 1
    a = (edge_co1 - edge_co0) / safe_len
    b = np.cross(a, edge_co0)

    # Bunch of zeros
    zero = np.zeros(a.shape[0])

    # Generate matrix K
    K = [[zero,    -a[:, 2], a[:, 1],    -b[:, 0]],
         [a[:, 2],  zero,    -a[:, 0],   -b[:, 1]],
         [-a[:, 1], a[:, 0], zero,       -b[:, 2]]]
    K = np.array(K)

    # Q for vertex i is then the sum of the products of (kT,k) for ALL edges
    # connected to vertex i.

    # Generate (kT, K)
    kT = np.transpose(K, axes=(1, 0, 2))

    # To get the sum of the products in the correct format we have to
    # do some annoying transposes to get to (4, 4, len(edges))
    K_dot = np.matmul(K.T, kT.T).T

    # Each edge contributes its K_dot matrix to BOTH of its endpoints. Rather
    # than scanning all edges once per vertex (O(V*E)) we scatter-add each
    # edge's contribution onto its two endpoints in a single O(E) pass.
    # Directionality of the edge is intentionally ignored (matches the original
    # implementation).
    K_dot_e = np.moveaxis(K_dot, 2, 0)  # (E, 4, 4)
    Q_vfirst = np.zeros((verts.shape[0], 4, 4), dtype=np.float64)
    np.add.at(Q_vfirst, edges[:, 0], K_dot_e)
    np.add.at(Q_vfirst, edges[:, 1], K_dot_e)
    Q_array = np.moveaxis(Q_vfirst, 0, 2)  # (4, 4, V)

    # Edge collapse:
    # Determining which edge to collapse is a weighted sum of the shape and
    # sampling cost. The shape cost of vertex i is Fa(p) = pT Qi p where p is
    # the coordinates of point p (vertex i here) in homogeneous representation.
    # The variable w from above is the value for the homogeneous 4th dimension.
    # T denotes transpose of matrix.
    # The shape cost of collapsing the edge Fa(i,j) = Fi(vj) + Fj(vj).
    # vi and vj being the coordinates of the vertex in homogeneous representation
    # (p in equation before)
    # The sampling cost penalises edge collapses that generate overly long edges,
    # based on the distance traveled by all edges to vi, when vi is merged with
    # vj. (Eq. 7 in paper)
    # You cannot collapse an edge (i -> j) if k is a common adjacent vertex of
    # both i and j, but (i/j/k) is not a face.
    # We will set the cost of these edges to infinity.

    # Now work out the shape cost of collapsing each edge (eq. 7).
    # Collapsing edge (u, v) merges u into v (see the loop below), so we
    # evaluate the combined error quadric (Q_u + Q_v) at the survivor's
    # homogeneous position v. This is the quadratic form ``p^T Q p`` and -
    # because Q is a sum of K^T K terms and hence positive semi-definite - it
    # is always >= 0.
    verts_h = np.append(verts, np.ones((verts.shape[0], 1)), axis=1)

    # Sum of lengths of all edges incident to each vertex (for the sample cost).
    # Easiest via a sparse matrix from the edges, summed by row. Symmetrize so
    # that a->b == a<-b.
    adj = scipy.sparse.coo_matrix((edge_lengths,
                                   (edges[:, 0], edges[:, 1])),
                                  shape=(verts.shape[0], verts.shape[0]))
    adj = adj + adj.T
    verts_lengths = np.array(adj.sum(axis=1)).flatten()

    def collapse_costs(idx):
        """Cost of collapsing the edges at indices `idx` (eq. 8).

        Both collapse directions are considered and the cheaper one is kept;
        the edge is re-oriented in place so that column 0 is always the vertex
        that gets *removed* and column 1 the *survivor* (matching the collapse
        loop below). Returns the minimum total cost per edge.
        """
        e = edges[idx]
        el = edge_lengths[idx]
        # Combined error quadric of both endpoints (symmetric in u, v)
        Quv = Q_array[:, :, e[:, 0]] + Q_array[:, :, e[:, 1]]
        p0, p1 = verts_h[e[:, 0]], verts_h[e[:, 1]]
        # Shape cost = combined quadric evaluated at the survivor's position
        shape_keep0 = np.einsum('ei,ije,ej->e', p0, Quv, p0)  # survive col 0
        shape_keep1 = np.einsum('ei,ije,ej->e', p1, Quv, p1)  # survive col 1
        # Sample cost depends on the *removed* vertex's incident lengths
        samp_rm0 = el * (verts_lengths[e[:, 0]] - el)  # remove col 0
        samp_rm1 = el * (verts_lengths[e[:, 1]] - el)  # remove col 1
        # Direction A: remove col 0, survive col 1
        costA = shape_keep1 * shape_weight + samp_rm0 * sample_weight
        # Direction B: remove col 1, survive col 0
        costB = shape_keep0 * shape_weight + samp_rm1 * sample_weight
        # Re-orient edges where reversing is cheaper (col 0 = removed vertex)
        flip = costB < costA
        if flip.any():
            e[flip] = e[flip][:, ::-1]
            edges[idx] = e
        return np.minimum(costA, costB)

    # Total cost - weighted sum of shape and sample cost, equation 8 in paper
    F_T = collapse_costs(np.arange(edges.shape[0]))

    # Incidence structures so that each collapse only has to touch the survivor's
    # local neighbourhood instead of rescanning the whole edge/face arrays. The
    # previous implementation did several O(edges) / O(faces) passes per collapse
    # (=> O(faces^2) overall); maintaining adjacency incrementally brings this
    # down to roughly O(faces log faces).
    #
    #   vert2edges[w]  : set of alive edge indices incident to vertex w
    #   edge2faces[e]  : set of alive face indices that still contain edge e
    #
    # `face_edges` stores indices into the (length-stable) `edges` array; an edge
    # collapse never deletes rows, it only relabels vertices, so these indices
    # stay valid throughout.
    n_edges = edges.shape[0]
    vert2edges = [set() for _ in range(verts.shape[0])]
    for e in range(n_edges):
        vert2edges[int(edges[e, 0])].add(e)
        vert2edges[int(edges[e, 1])].add(e)
    edge2faces = [set() for _ in range(n_edges)]
    for f in range(face_edges.shape[0]):
        for x in face_edges[f]:
            edge2faces[int(x)].add(f)

    alive_edge = np.ones(n_edges, dtype=bool)
    alive_face = np.ones(face_edges.shape[0], dtype=bool)
    keep = np.zeros(n_edges, dtype=bool)
    n_faces_alive = int(face_edges.shape[0])

    # Lazy-deletion min-heap of (cost, edge index). We never remove entries when
    # a cost changes; instead we push the new value and discard outdated entries
    # on pop by checking them against the authoritative `F_T` plus the alive/keep
    # flags. This replaces the old O(edges) `np.argmin` per iteration.
    heap = [(float(F_T[e]), e) for e in range(n_edges) if np.isfinite(F_T[e])]
    heapq.heapify(heap)

    def push(idx):
        for e in idx:
            e = int(e)
            c = F_T[e]
            if np.isfinite(c):
                heapq.heappush(heap, (float(c), e))

    with tqdm(desc='Collapsing edges', total=n_faces_alive,
              disable=progress is False) as pbar:
        last = n_faces_alive
        while heap:
            cost, e = heapq.heappop(heap)
            # Skip stale / dead / kept heap entries. If the cheapest live entry
            # has infinite cost there is nothing left we can safely collapse.
            if not alive_edge[e] or keep[e] or cost != F_T[e]:
                continue

            pbar.update(last - n_faces_alive)
            last = n_faces_alive

            # Get the vertices this edge connects. Collapsing (u, v) merges u
            # into v (u is removed, v survives), see the re-orientation in
            # `collapse_costs`.
            u, v = int(edges[e, 0]), int(edges[e, 1])
            if u == v:
                alive_edge[e] = False
                F_T[e] = np.inf
                continue

            # All alive edges that connect u and v (parallel edges), excluding
            # self-loops. An edge incident to *both* u and v necessarily has
            # endpoints exactly {u, v}.
            clps = {e2 for e2 in (vert2edges[u] & vert2edges[v])
                    if edges[e2, 0] != edges[e2, 1]}
            if not clps:
                F_T[e] = np.inf
                continue
            clps_edges = np.fromiter(clps, dtype=np.int64)

            # Alive faces incident to any of the collapsing edges
            faces = set()
            for e2 in clps:
                faces |= edge2faces[e2]
            faces = [f for f in faces if alive_face[f]]

            # If these edges do not have adjacent faces anymore they become
            # skeleton segments - track them as keepers.
            if not faces:
                keep[clps_edges] = True
                F_T[clps_edges] = np.inf
                continue

            # Link condition (optional): we may only collapse edge (u, v) if
            # every vertex adjacent to BOTH u and v also forms a triangle
            # (u, v, k) with it. A "blind" common neighbour (one that does not
            # share a face) would turn into a non-manifold fold on collapse. If
            # we find one, defer this edge (set its cost to infinity) and pick
            # another. Off by default - see the docstring for why.
            if link_condition:
                nbr_u = {(int(edges[e2, 0]) if int(edges[e2, 1]) == u
                          else int(edges[e2, 1])) for e2 in vert2edges[u]}
                nbr_v = {(int(edges[e2, 0]) if int(edges[e2, 1]) == v
                          else int(edges[e2, 1])) for e2 in vert2edges[v]}
                common = (nbr_u & nbr_v) - {u, v}
                # Vertices that legitimately share a face with the edge (u, v)
                face_k = set()
                for f in faces:
                    for x in face_edges[f]:
                        face_k.add(int(edges[int(x), 0]))
                        face_k.add(int(edges[int(x), 1]))
                if common - face_k:
                    F_T[clps_edges] = np.inf
                    continue

            # Adjacent (non-collapsing) edges of each collapsing face as
            # (win, loose) pairs, in face-column order. Each collapsing triangle
            # has exactly one of its edges between u and v, so the two survivors
            # collapse onto each other (loose -> win).
            adj_pairs = []
            for f in sorted(faces):
                non = [int(x) for x in face_edges[f] if int(x) not in clps]
                if len(non) == 2:
                    adj_pairs.append((non[0], non[1]))

            # Remove the collapsing faces
            for f in faces:
                alive_face[f] = False
                for x in face_edges[f]:
                    edge2faces[int(x)].discard(f)
            n_faces_alive -= len(faces)

            # Track the collapsing edges as collapsed
            for e2 in clps:
                alive_edge[e2] = False
                vert2edges[int(edges[e2, 0])].discard(e2)
                vert2edges[int(edges[e2, 1])].discard(e2)
                F_T[e2] = np.inf

            # Replace occurrences of the loosing edge with the winning edge in
            # every face that still references it. Crucially we do NOT skip
            # already-dead loose edges - leaving them behind would create stale
            # degenerate faces that get mistaken for collapsible later on.
            for win, loose in adj_pairs:
                if win == loose:
                    continue
                for f in list(edge2faces[loose]):
                    face_edges[f] = [win if int(x) == loose else int(x)
                                     for x in face_edges[f]]
                    edge2faces[win].add(f)
                    edge2faces[loose].discard(f)
                alive_edge[loose] = False
                vert2edges[int(edges[loose, 0])].discard(loose)
                vert2edges[int(edges[loose, 1])].discard(loose)
                F_T[loose] = np.inf

            # Replace occurrences of first node u with second node v on all edges
            # still incident to u (local thanks to vert2edges).
            for e2 in list(vert2edges[u]):
                if edges[e2, 0] == u:
                    edges[e2, 0] = v
                if edges[e2, 1] == u:
                    edges[e2, 1] = v
                vert2edges[v].add(e2)
            vert2edges[u] = set()

            # Add shape cost of u to shape costs of v
            Q_array[:, :, v] += Q_array[:, :, u]

            # Update sum of incident edge lengths for the survivor.
            # Technically we would have to recalculate lengths of adjacent edges
            # every time but we take the cheap way out and simply add them up.
            verts_lengths[v] += verts_lengths[u]

            # Recompute costs for edges now incident to v (re-orienting them to
            # their cheaper collapse direction) and (re-)push them onto the heap.
            # Both Q_array and verts_lengths must already be updated above.
            inc_v = np.fromiter(vert2edges[v], dtype=np.int64)
            if inc_v.size:
                costs = collapse_costs(inc_v)
                # Kept edges incident to v must stay at infinity.
                costs[keep[inc_v]] = np.inf
                F_T[inc_v] = costs
                push(inc_v)

    # If no edges survived the collapse there is nothing to build a skeleton
    # from. This typically happens for closed, blob-like meshes (e.g. a sphere)
    # that collapse symmetrically without ever leaving a face-less edge - such
    # meshes should be contracted (see `skeletor.pre.contract`) first.
    if not keep.any():
        raise ValueError("Edge collapse did not leave any skeleton edges. This "
                         "usually means the mesh is closed/blob-like rather "
                         "than tubular - try contracting it first with "
                         "`skeletor.pre.contract`.")

    # After the edge collapse, the edges are garbled - I have yet to figure out
    # why and whether that can be prevented. However the vertices in those
    # edges are correct and so we just need to reconstruct their connectivity
    # by extracting a minimum spanning tree over the mesh.
    corrected_edges = mst_over_mesh(mesh, edges[keep].flatten())

    # Generate graph
    G = edges_to_graph(corrected_edges, vertices=mesh.vertices, fix_tree=True,
                       weight=False, drop_disconnected=False)

    swc, new_ids = make_swc(G, mesh, reindex=True)

    return Skeleton(swc=swc, mesh=mesh, mesh_map=None, method='edge_collapse')
