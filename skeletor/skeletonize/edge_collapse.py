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


def by_edge_collapse(mesh, shape_weight=1, sample_weight=0.1, progress=True):
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
    # of edge(i,j) and b = a * <x/y/z coordinates of vertex i>
    #
    # The matrix K takes the form:
    #
    #        Kij = 0, -az, ay, -bx
    #              az, 0, -ax, -by
    #             -ay, ax, 0,  -bz

    edge_co0, edge_co1 = verts[edges[:, 0]], verts[edges[:, 1]]
    a = (edge_co1 - edge_co0) / edge_lengths.reshape(edges.shape[0], 1)
    # Note: It's a bit unclear to me whether the normalised edge vector should
    # be allowed to have negative values but I seem to be getting better
    # results if I use absolute values
    a = np.fabs(a)
    b = a * edge_co0

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

    # Not sure if we are doing something wrong when calculating the Q array but
    # we end up having negative values which translate into negative scores.
    # This in turn is bad because we propagate that negative score when
    # collapsing edges which leads to a "zipper-effect" where nodes collapse
    # in sequence a->b->c->d until they hit some node with really high cost
    # Q_array -= Q_array.min()

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

    # Now work out the shape cost of collapsing each node (eq. 7)
    # First get coordinates of the first node of each edge
    # Note that in Nik's implementation this was the second node
    p = verts[edges[:, 0]]

    # Append weight factor
    w = 1
    p = np.append(p, np.full((p.shape[0], 1), w), axis=1)

    this_Q1 = Q_array[:, :, edges[:, 0]]
    this_Q2 = Q_array[:, :, edges[:, 1]]

    F1 = np.einsum('ij,kji->ij', p, this_Q1)[:, [0, 1]]
    F2 = np.einsum('ij,kji->ij', p, this_Q2)[:, [0, 1]]

    # Calculate and append shape cost
    F = np.append(F1, F2, axis=1)
    shape_cost = np.sum(F, axis=1)

    # Sum lengths of all edges associated with a given vertex
    # This is easiest by generating a sparse matrix from the edges
    # and then summing by row
    adj = scipy.sparse.coo_matrix((edge_lengths,
                                   (edges[:, 0], edges[:, 1])),
                                  shape=(verts.shape[0], verts.shape[0]))

    # This makes sure the matrix is symmetrical, i.e. a->b == a<-b
    # Note that I'm not sure whether this is strictly necessary but it really
    # can't hurt
    adj = adj + adj.T

    # Get the lengths associated with each vertex
    verts_lengths = adj.sum(axis=1)

    # We need to flatten this (something funny with summing sparse matrices)
    verts_lengths = np.array(verts_lengths).flatten()

    # Map the sum of vertex lengths onto edges (as per first vertex in edge)
    ik_edge = verts_lengths[edges[:, 0]]

    # Calculate sampling cost
    sample_cost = edge_lengths * (ik_edge - edge_lengths)

    # Determine which edge to collapse and collapse it
    # Total Cost - weighted sum of shape and sample cost, equation 8 in paper
    F_T = shape_cost * shape_weight + sample_cost * sample_weight

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

            # Determine which edges require update of costs:
            # In theory we only need to update costs for edges that are
            # associated with vertices v and u (which now also v)
            has_v = (edges[:, 0] == v) | (edges[:, 1] == v)

            # Uncomment to temporarily force updating costs for all edges
            # has_v[:] = True

            # Update shape costs
            this_Q1 = Q_array[:, :, edges[has_v, 0]]
            this_Q2 = Q_array[:, :, edges[has_v, 1]]

            F1 = np.einsum('ij,kji->ij', p[edges[has_v, 0]], this_Q1)[:, [0, 1]]
            F2 = np.einsum('ij,kji->ij', p[edges[has_v, 1]], this_Q2)[:, [0, 1]]

            F = np.append(F1, F2, axis=1)
            new_shape_cost = np.sum(F, axis=1)

            # Update sum of incoming edge lengths
            # Technically we would have to recalculate lengths of adjacent edges
            # every time but we will take the cheap way out and simply add them up
            verts_lengths[v] += verts_lengths[u]
            # Update sample costs for edges associated with v
            ik_edge = verts_lengths[edges[has_v, 0]]
            new_sample_cost = edge_lengths[has_v] * (ik_edge - edge_lengths[has_v])

            F_T[has_v] = new_shape_cost * shape_weight + new_sample_cost * sample_weight

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
