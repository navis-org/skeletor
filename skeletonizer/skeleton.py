#    This script is part of skeletonizer (http://www.github.com/schlegelp/skeletonizer).
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


import numpy as np
import trimesh as tm
import scipy.sparse as sparse

from tqdm.auto import tqdm


def skeletonize(mesh, shape_weight=1, sample_weight=0.1, progress=True):
    """Skeletonize a (contracted) mesh.

    Parameters
    ----------
    mesh :          trimesh.Trimesh
                    Contracted mesh to be skeletonized.
    shape_weight :  float, optional
                    Weight for shape costs which represent impact of merging
                    two nodes on the shape of the object.
    sample_weight : float, optional
                    Weight for sampling costs which increase if a merge would
                    generate prohibitively long edges.
    progress :      bool
                    If True, will show progress bar.


    Example
    -------
    >>> import trimesh as tm
    >>> m = tm.primitives.Cylinder()

    Returns
    -------
    edge list
                    List of contracted edges.

    """
    assert isinstance(mesh, tm.Trimesh)

    # Shorthand faces and edges
    # We convert to arrays to (a) make a copy and (b) remove potential overhead
    # from these originally being trimesh TrackedArrays
    faces = np.array(mesh.faces)
    edges = np.array(mesh.edges_unique)
    # frozen_edges = np.array([frozenset(e) for e in edges])
    edge_lengths = np.array(mesh.edges_unique_length)
    verts = np.array(mesh.vertices)

    # Some weight - ask Nik about it
    w = 1

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
    a = (edge_co1 - edge_co0) / edge_lengths.reshape(edge_lengths.shape[0], 1)
    # Note: Nik's implementation allows negative vectors but shouldn't this be positive vectors only?
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
    Q_array = np.zeros((4, 4, verts.shape[0]))

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

    # Now work out the shape cost of collapsing each node (eq. 7)
    # First get coordinates of the first node of each edge
    # Note that in Nik's implementation this was the second node
    p = verts[edges[:, 0]]

    # Append weight factor
    p = np.append(p, np.full((p.shape[0], 1), w), axis=1)

    this_Q1 = Q_array[:, :, edges[:, 0]]
    this_Q2 = Q_array[:, :, edges[:, 1]]

    F1 = np.einsum('ij,kji->ij', p, this_Q1)[:, [0, 1]]
    F2 = np.einsum('ij,kji->ij', p, this_Q2)[:, [0, 1]]

    # Calculate and append shape cost
    F = np.append(F1, F2, axis=1)
    F = np.sum(F, axis=1)
    shape_cost = F

    # Sum lengths of all edges associated with a given vertex
    # This is easiest by generating a sparse matrix from the edges
    # and then summing by row
    adj = sparse.coo_matrix((edge_lengths,
                             (edges[:, 0], edges[:, 1])),
                            shape=(verts.shape[0], verts.shape[0]))
    adj = adj + adj.T

    # Get the lengths associated with each vertex
    # Note that edges are directional here (which I believe is intended?)
    # i.e. i->k might be 100 but k->i might be 0 (= not set)
    # If not, we could symmetrize the sparse matrix above
    verts_lengths = adj.sum(axis=1)

    # We need to flatten this (something funny with summing sparse matrices)
    verts_lengths = np.array(verts_lengths).flatten()

    # Map the sum of vertex lengths onto edges (as per first vertex in edge)
    ik_edge = verts_lengths[edges[:, 0]]

    # Calculate sampling cost
    sample_cost = edge_lengths * (ik_edge - edge_lengths)

    # Find edges that are not part of any face
    has_face = edge_in_face(edges, faces)
    sample_cost[~has_face] = np.inf
    shape_cost[~has_face] = np.inf

    # Now start collapsing edges one at a time
    face_count = faces.shape[0]  # keep track of face counts for progress bar
    with tqdm(desc='Collapsing faces', total=face_count, disable=progress is False) as pbar:
        while True:
            # Determine which edge to collapse and collapse it
            # Total Cost - weighted sum of shape and sample cost, equation 8 in paper
            F_T = shape_cost * shape_weight + sample_cost * sample_weight

            # Get the edge that we want to collapse
            collapse_ix = np.argmin(F_T)
            collapse_edge = edges[collapse_ix]
            u, v = collapse_edge

            # Replace occurrences of first node u with second node v
            edges[np.where(edges == u)] = v

            # Add shape cost of u to shape costs of v
            Q_array[:, :, v] += Q_array[:, :, u]

            # Determine which edges require update of costs:
            # In theory we only need to update costs for for edges that are
            # associated with vertices v and u (now also v)
            has_v = (np.sum(edges == v, axis=1) >= 1) & (F_T < np.inf)

            # Uncomment to temporarily force update for all edges
            # has_v[:] = True

            # Update shape costs
            this_Q1 = Q_array[:, :, edges[has_v, 0]]
            this_Q2 = Q_array[:, :, edges[has_v, 1]]

            F1 = np.einsum('ij,kji->ij', p[edges[has_v, 0]], this_Q1)[:, [0, 1]]
            F2 = np.einsum('ij,kji->ij', p[edges[has_v, 1]], this_Q2)[:, [0, 1]]

            F = np.append(F1, F2, axis=1)
            F = np.sum(F, axis=1)
            shape_cost[has_v] = F

            # Update sum of incoming edge lengths
            # -> @Nik not sure if this is necessary?
            verts_lengths[v] += verts_lengths[u]
            # Update sample costs for edges associated with v
            ik_edge = verts_lengths[edges[has_v, 0]]
            sample_cost[has_v] = edge_lengths[has_v] * (ik_edge - edge_lengths[has_v])

            # Take care of faces: Remap u to v
            faces[np.where(faces == u)] = v
            # Drop faces that have more than 1 occurrence of v now (e.g. [1, 2, 1])
            faces = faces[np.sum(faces == v, axis=1) <= 1]

            # If no more faces, break
            if faces.shape[0] == 0:
                break

            # Drop non-unique faces
            faces = np.unique(faces, axis=0)

            # Find edge that are not part of any face anymore
            # This also removes self-loops and ...
            # ... therefore also the collapsed edge as (u, v) is now (v, v)
            has_face[has_v] = edge_in_face(edges[has_v], faces)

            # Set sampling cost of edges without faces to infinite
            shape_cost[~has_face] = np.inf
            sample_cost[~has_face] = np.inf

            # Update progress bar
            pbar.update(face_count - faces.shape[0])
            face_count = faces.shape[0]

    # We need one final round of self-loop removal
    edges = edges[edges[:, 0] != edges[:, 1]]

    # Return unique edges
    kept = np.unique(edges, axis=0)

    return kept


def to_graph(edges, vertices):
    """Create networkx Graph from edge list."""
    edges = edges[edges[:, 0] != edges[:, 1]]

    import networkx as nx

    G = nx.Graph()
    nodes = np.unique(edges.flatten())
    coords = vertices[nodes]
    G.add_nodes_from([(e, {'x': co[0], 'y': co[1], 'z': co[2]}) for e, co in zip(nodes, coords)])
    G.add_edges_from([(e[0], e[1]) for e in edges])

    return G


def get_sampling_cost_nik(verts, edges, faces, edge_lengths):
    """Get sampling costs like in Nik's implemention"""
    # Sum lengths of all edges associated with a given vertex
    # This is easiest by generating a sparse matrix from the edges
    # and then summing by row
    adj = sparse.coo_matrix((edge_lengths,
                             (edges[:, 0], edges[:, 1])),
                            shape=(verts.shape[0], verts.shape[0]))

    # Get the lengths associated with each vertex
    # Note that edges are directional here (which I believe is intended?)
    # i.e. i->k might be 100 but k->i might be 0 (= not set)
    # If not, we could symmetrize the sparse matrix above
    verts_lengths = adj.sum(axis=1)

    # We need to flatten this (something funny with summing sparse matrices)
    verts_lengths = np.array(verts_lengths).flatten()

    # Initialise some things
    # matrix of k matricies
    all_K = np.zeros((3, 4, len(edges)))
    # matrix k for each edge
    matrix_kij = np.zeros((3, 4)) # rows/columns

    for k, e in enumerate(edges):
        current_edge = edges[k]
        i = verts[e[0]]
        j = verts[e[1]]
        a = (j - i) / edge_lengths[k]
        b = a * i

        # This can be one line
        matrix_kij = np.array([[0       , 0 - a[2], a[1]    , 0 - b[0]],
                               [a[2]    , 0       , 0 - a[0], 0 - b[1]],
                               [0 - a[1], a[0]    , 0       , 0 - b[2]]])
        all_K[:, :, k] = matrix_kij

    as_array = np.zeros((4, 4, len(verts)))
    for q in range(len(verts)):  # loop over verices
        # get all edges connected to a vertex (get the index of that edge in the list 'edges'
        index = []
        for i, e in enumerate(edges):
            if (e[0] == q or e[1] == q):
                index.append(i)
        # sum
        Q = np.zeros((4, 4, len(index)))
        for i in range(len(index)):
            matrix = all_K[:, :, i]
            Q[:, :, i] = np.dot(matrix.T, matrix)  # product for this edge
        Q = Q.sum(axis=2)
        as_array[:, :, q] = Q
    all_Q = {}
    for index in range(len(verts)):
        all_Q[index] = as_array[:, :, index]

    Shape_cost = []
    Sample_cost = []

    w = 1

    faces = [set(f) for f in faces]

    # These determine if an edge cannot be collapsed - either becuase it
    # And determine the shape and sample cost of collapsing each edge
    for i, e in enumerate(edges):
        # set loops to cost inf
        if e[0] == e[1]:
            # set the cost of collapsing loops to infinity
            Shape_cost.append(np.inf)
            Sample_cost.append(np.inf)
            continue
        #
        test = 0
        for f in faces:
            if set(e) <= f:
                test = 1
                break
        if test == 0:
            Shape_cost.append(np.inf)
            Sample_cost.append(np.inf)
            continue

        # Work out the shape cost of collapsing each node (eq. 7)
        p = np.append(verts[e[1]], w)
        F = sum((np.dot(p.T, all_Q[e[0]], p)) + (np.dot(p.T, all_Q[e[1]], p)))
        Shape_cost.append(F)
        # get the length of edge i->j
        len_ij = edge_lengths[i]
        # Get the sum of lengths of all edges going from i to k, not including i j.
        len_ik = verts_lengths[e[0]]
        Sample_cost.append(len_ij * (len_ik - len_ij))

    return Sample_cost, Shape_cost


def edge_in_face(edges, faces):
    """Test if edges are associated with a face. Returns boolean array."""
    # Concatenate edges of all faces (us)
    edges_in_faces = np.concatenate((faces[:,  [0, 1]],
                                     faces[:,  [1, 2]],
                                     faces[:,  [2, 0]]))
    # Since we don't care about the orientation of edges, we just make it so
    # that the lower index is always in the first column
    edges_in_faces = np.sort(edges_in_faces, axis=1)
    edges = np.sort(edges, axis=1)

    # Make unique edges (low ms)
    # - we don't actually need this and it is costly
    # edges_in_faces = np.unique(edges_in_faces, axis=0)

    # Turn face edges into structured array (few us)
    sorted = np.ascontiguousarray(edges_in_faces).view([('', edges_in_faces.dtype)] * edges_in_faces.shape[-1]).ravel()
    # Sort (low ms) -> this is the most costly step at the moment
    sorted.sort(kind='stable')

    # Turn edges into continuous array (few us)
    comp = np.ascontiguousarray(edges).view(sorted.dtype)

    # This asks where elements of "comp" should be inserted which basically
    # tries to align edges and edges_in_faces (tens of ms)
    ind = sorted.searchsorted(comp)

    # If edges are "out of bounds" of the sorted array of face edges the will
    # have "ind = sorted.shape[0] + 1"
    in_bounds = ind < sorted.shape[0]

    # Prepare results (default = False)
    has_face = np.full(edges.shape[0], False, dtype=bool)

    # Check if actually the same for those indices that are within bounds
    has_face[in_bounds.flatten()] = sorted[ind[in_bounds]] == comp[in_bounds]

    return has_face
