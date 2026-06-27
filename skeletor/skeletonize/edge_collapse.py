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

try:
    import fastremap
except ImportError:
    fastremap = None
except BaseException:
    raise

import numpy as np
import scipy.sparse
import scipy.spatial

from tqdm.auto import tqdm

from ..utilities import make_trimesh
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
    edges = np.array(mesh.edges_unique)
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
    # connected to vertex i:
    # Initialize matrix of correct shape
    Q_array = np.zeros((4, 4, verts.shape[0]), dtype=np.float64)

    # Generate (kT, K)
    kT = np.transpose(K, axes=(1, 0, 2))

    # To get the sum of the products in the correct format we have to
    # do some annoying transposes to get to (4, 4, len(edges))
    K_dot = np.matmul(K.T, kT.T).T

    # Iterate over all vertices
    for v in range(len(verts)):
        # Find edges that contain this vertex
        cond1 = edges[:, 0] == v
        cond2 = edges[:, 1] == v
        # Note that this does not take directionality of edges into account
        # Not sure if that's intended?

        # Get indices of these edges
        indices = np.where(cond1 | cond2)[0]

        # Get the products for all edges adjacent to mesh
        Q = K_dot[:, :, indices]
        # Sum over all edges
        Q = Q.sum(axis=2)
        # Add to Q array
        Q_array[:, :, v] = Q

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

    def collapse_costs(mask):
        """Cost of collapsing the edges selected by `mask` (eq. 8).

        Both collapse directions are considered and the cheaper one is kept;
        the edge is re-oriented in place so that column 0 is always the vertex
        that gets *removed* and column 1 the *survivor* (matching the collapse
        loop below). Returns the minimum total cost per edge.
        """
        e = edges[mask]
        el = edge_lengths[mask]
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
            edges[mask] = e
        return np.minimum(costA, costB)

    # Total cost - weighted sum of shape and sample cost, equation 8 in paper
    F_T = collapse_costs(np.ones(edges.shape[0], dtype=bool))

    # Now start collapsing edges one at a time
    face_count = face_edges.shape[0]  # keep track of face counts for progress bar
    is_collapsed = np.full(edges.shape[0], False)
    keep = np.full(edges.shape[0], False)
    with tqdm(desc='Collapsing edges', total=face_count, disable=progress is False) as pbar:
        while face_edges.size:
            # Uncomment to get a more-or-less random edge collapse
            # F_T[:] = 0

            # Update progress bar
            pbar.update(face_count - face_edges.shape[0])
            face_count = face_edges.shape[0]

            # This has to come at the beginning of the loop
            # Set cost of collapsing edges without faces to infinite
            F_T[keep] = np.inf
            F_T[is_collapsed] = np.inf

            # Get the edge that we want to collapse
            collapse_ix = np.argmin(F_T)
            # If even the cheapest edge has infinite cost there is nothing left
            # we can safely collapse (everything is collapsed, kept, or deferred
            # by the link condition below). Stop - the MST reconstruction at the
            # end will reconnect whatever vertices remain.
            if not np.isfinite(F_T[collapse_ix]):
                break
            # Get the vertices this edge connects
            u, v = edges[collapse_ix]
            # Get all edges that contain these vertices:
            # First, edges that are (uv, x)
            connects_uv = np.isin(edges[:, 0], [u, v])
            # Second, check if any (uv, x) edges are (uv, uv)
            connects_uv[connects_uv] = np.isin(edges[connects_uv, 1], [u, v])

            # Remove uu and vv edges
            uuvv = edges[:, 0] == edges[:, 1]
            connects_uv = connects_uv & ~uuvv
            # Get the edge's indices
            clps_edges = np.where(connects_uv)[0]

            # Now find find the faces the collapsed edge is part of
            # Note: splitting this into three conditions is marginally faster than
            # np.any(np.isin(face_edges, clps_edges), axis=1)
            uv0 = np.isin(face_edges[:, 0], clps_edges)
            uv1 = np.isin(face_edges[:, 1], clps_edges)
            uv2 = np.isin(face_edges[:, 2], clps_edges)
            has_uv = uv0 | uv1 | uv2

            # If these edges do not have adjacent faces anymore
            if not np.any(has_uv):
                # Track this edge as a keeper
                keep[clps_edges] = True
                continue

            # Get the collapsed faces [(e1, e2, e3), ...] for this edge
            clps_faces = face_edges[has_uv]

            # Link condition (optional): we may only collapse edge (u, v) if
            # every vertex adjacent to BOTH u and v also forms a triangle
            # (u, v, k) with it. A "blind" common neighbour (one that does not
            # share a face) would turn into a non-manifold fold on collapse. If
            # we find one, defer this edge (set its cost to infinity) and pick
            # another. Off by default - see the docstring for why.
            if link_condition:
                inc = ~is_collapsed & ~uuvv
                inc_u = inc & ((edges[:, 0] == u) | (edges[:, 1] == u))
                inc_v = inc & ((edges[:, 0] == v) | (edges[:, 1] == v))
                nbr_u = np.where(edges[inc_u, 0] == u, edges[inc_u, 1], edges[inc_u, 0])
                nbr_v = np.where(edges[inc_v, 0] == v, edges[inc_v, 1], edges[inc_v, 0])
                common = np.intersect1d(nbr_u, nbr_v)
                common = common[(common != u) & (common != v)]
                # Vertices that legitimately share a face with the edge (u, v)
                face_k = np.unique(edges[clps_faces])
                if np.setdiff1d(common, face_k).size:
                    F_T[clps_edges] = np.inf
                    continue

            # Remove the collapsed faces
            face_edges = face_edges[~has_uv]

            # Track these edges as collapsed
            is_collapsed[clps_edges] = True

            # Get the adjacent edges (i.e. non-uv edges)
            adj_edges = clps_faces[~np.isin(clps_faces, clps_edges)].reshape(clps_faces.shape[0], 2)

            # We have to do some sorting and finding unique edges to make sure
            # remapping is done correctly further down
            # NOTE: Not sure we really need this, so leaving it out for now
            # adj_edges = np.unique(np.sort(adj_edges, axis=1), axis=0)

            # We need to keep track of changes to the adjacent faces
            # Basically each face in (i, j, k) will be reduced to one edge
            # which points from u -> v
            # -> replace occurrences of loosing edge with winning edge
            for win, loose in adj_edges:
                if fastremap:
                    face_edges = fastremap.remap(face_edges, {loose: win},
                                                 preserve_missing_labels=True,
                                                 in_place=True)
                else:
                    face_edges[face_edges == loose] = win
                is_collapsed[loose] = True

            # Replace occurrences of first node u with second node v
            if fastremap:
                edges = fastremap.remap(edges, {u: v},
                                        preserve_missing_labels=True,
                                        in_place=True)
            else:
                edges[edges == u] = v

            # Add shape cost of u to shape costs of v
            Q_array[:, :, v] += Q_array[:, :, u]

            # Update sum of incident edge lengths for the survivor.
            # Technically we would have to recalculate lengths of adjacent edges
            # every time but we take the cheap way out and simply add them up.
            verts_lengths[v] += verts_lengths[u]

            # Recompute costs for edges now incident to v (re-orienting them to
            # their cheaper collapse direction). Both Q_array and verts_lengths
            # must already be updated above.
            has_v = (edges[:, 0] == v) | (edges[:, 1] == v)
            F_T[has_v] = collapse_costs(has_v)

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
