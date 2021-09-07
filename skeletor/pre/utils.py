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
import warnings

import numpy as np
import scipy.sparse as spsp
import scipy.spatial as spspat
import trimesh as tm

from ..utilities import make_trimesh


def getBBox(verts):
    """Return bounding box of vertices."""
    min_coords = np.min(verts, axis=0)
    max_coords = np.max(verts, axis=0)
    diameter = np.linalg.norm(max_coords - min_coords)
    return min_coords, max_coords, diameter


def laplacian_umbrella(mesh):
    """Compute Laplace weights using the uniform weighting.

    Uniform weighting (aka "umbrella operator") only describes the topological
    properties of the mesh but not the geometrical ones. This also makes it
    more robust if the mesh is imperfect.


    Parameters
    ----------
    mesh :          trimesh.Trimesh

    Returns
    -------
    CSR sparse matrix

    References
    ----------
    [1] Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh
        contraction. ACM Transactions on Graphics (TOG). 2008 Aug 1;27(3):44.

    """
    # We're piggy backing off trimesh's laplace operator
    L = tm.smoothing.laplacian_calculation(mesh, equal_weight=False)

    # At this point, rows/cols in L sum up to 1 and the diagonal is zero
    # We have to set the diagonal to -1 to set the weights properly
    L.setdiag(-1)

    return L.tocsr()


def laplacian_cotangent(mesh, normalized=False):
    """Compute Laplace weights from cotangents.

    This produces a n x n curvature-flow Laplace operator with elements::

        L_ij =      w_ij = cot(a_ij) + cot(b_ij)        if (i,j) in edges,
                    -sum(w_ik)                          if i = j
                    0                                   otherwise,

    With a and b being the opposite angles in the faces that share the edge ij.


    Parameters
    ----------
    mesh :          trimesh.Trimesh
    normalized :    bool
                    If True will (sort of) normalize the weights. This requires
                    ``scikit-learn`` to be installed.

    Returns
    -------
    CSR sparse matrix

    References
    ----------
    [1] Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh
        contraction. ACM Transactions on Graphics (TOG). 2008 Aug 1;27(3):44.

    """
    # Get pairs of faces that share an edge
    face_pairs = mesh.face_adjacency

    # Find out which vertices of each face pair are opposite
    # (i.e. not part of the shared edge)
    opposite_verts = mesh.face_adjacency_unshared
    a_ix = np.where(mesh.faces[face_pairs[:, 0]] == opposite_verts[:, [0]])[1]
    b_ix = np.where(mesh.faces[face_pairs[:, 1]] == opposite_verts[:, [1]])[1]

    # Get the angles of these opposite vertices within the shared faces
    a = mesh.face_angles[face_pairs[:, 0], a_ix]
    b = mesh.face_angles[face_pairs[:, 1], b_ix]

    # Clip angles to prevent extreme weights
    #a_min, a_max = np.deg2rad(1), np.deg2rad(179)
    #a = np.clip(a, a_min=a_min, a_max=a_max)
    #b = np.clip(b, a_min=a_min, a_max=a_max)

    # Produce the cotangents
    # The warning filter is for catching division by 0 warnings (see below)
    with warnings.catch_warnings():
        # Sharp angles give high weights:
        # 1 / np.tan(np.deg2rad(1)) = 57
        # Flat angles give low weights:
        # 1 / np.tan(np.deg2rad(179)) = -57
        warnings.simplefilter("ignore")
        cota = 1 / np.tan(a)
        cotb = 1 / np.tan(b)
        data = cota + cotb

        # If a face is fully collapsed angles a or b can be 0 -> in this case
        # we have to make sure not to introduces infinite values
        # -8165619676597685 is the value of 1 / np.tan(180) -> so the opposite
        # of collapsed face
        data[data == np.inf] = 8165619676597685

    # Generate rows and cols
    i = mesh.face_adjacency_edges[:, 0]
    j = mesh.face_adjacency_edges[:, 1]

    # Stack so that we cover i->j and i<-j
    data = np.concatenate((data, data))
    rows = np.concatenate((i, j))
    cols = np.concatenate((j, i))

    # Generate sparse matrix
    n = len(mesh.vertices)
    W = spsp.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Catch some scipy runtime warnings about changing sparse matrix structure
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        W.setdiag(0)

        # Set diagonal to -w(k)
        diag = - np.array(W.sum(axis=0))
        W.setdiag(diag.flatten())

    if normalized:
        from sklearn.preprocessing import normalize
        W = normalize(W)

    return W


def _laplacian_cotangent_legacy(mesh, symmetric=False, normalized=False):
    """Original implemenation (kept for reference)."""
    n = len(mesh.vertices)

    v1v2 = mesh.triangles[:, 0] - mesh.triangles[:, 1]
    v1v3 = mesh.triangles[:, 0] - mesh.triangles[:, 2]

    v2v1 = mesh.triangles[:, 1] - mesh.triangles[:, 0]
    v2v3 = mesh.triangles[:, 1] - mesh.triangles[:, 2]

    v3v1 = mesh.triangles[:, 2] - mesh.triangles[:, 0]
    v3v2 = mesh.triangles[:, 2] - mesh.triangles[:, 1]

    cot1 = np.sum(v2v1 * v3v1, axis=1) / np.linalg.norm(np.cross(v2v1, v3v1),
                                                        axis=1).clip(min=1e-06)
    cot2 = np.sum(v3v2 * v1v2, axis=1) / np.linalg.norm(np.cross(v3v2, v1v2),
                                                        axis=1).clip(min=1e-06)
    cot3 = np.sum(v1v3 * v2v3, axis=1) / np.linalg.norm(np.cross(v1v3, v2v3),
                                                        axis=1).clip(min=1e-06)

    data = np.concatenate((cot1, cot1,
                           cot2, cot2,
                           cot3, cot3),
                          axis=0)
    rows = np.concatenate((mesh.faces[:, 1], mesh.faces[:, 2],
                           mesh.faces[:, 2], mesh.faces[:, 0],
                           mesh.faces[:, 0], mesh.faces[:, 1]),
                          axis=0)
    cols = np.concatenate((mesh.faces[:, 2], mesh.faces[:, 1],
                           mesh.faces[:, 0], mesh.faces[:, 2],
                           mesh.faces[:, 1], mesh.faces[:, 0]),
                          axis=0)

    W = spsp.csr_matrix((data, (rows, cols)), shape=(n, n))

    if (symmetric and not normalized):
        sum_vector = W.sum(axis=0)
        d = spsp.dia_matrix((sum_vector, [0]), shape=(n, n))
        L = d - W
    elif (symmetric and normalized):
        sum_vector = W.sum(axis=0)
        sum_vector_powered = np.power(sum_vector, -0.5)
        d = spsp.dia_matrix((sum_vector_powered, [0]), shape=(n, n))
        eye = spsp.identity(n)
        L = eye - d * W * d
    elif (not symmetric and normalized):
        sum_vector = W.sum(axis=0)
        sum_vector_powered = np.power(sum_vector, -1.0) # this is the same as 1 / sum_vector
        d = spsp.dia_matrix((sum_vector_powered, [0]), shape=(n, n))  # this is a matrix with sum_vector_powered as diagonal
        eye = spsp.identity(n)  # this is a matrix with ones (1) as diagonal
        L = eye - d * W

        # I think what this basically does is:
        # 1. Normalize by each column by its sum
        # 2. Set the diagonal to 1
    else:
        L = W

    return L


def getMeshVPos(mesh, extra_points=[]):
    if any(extra_points):
        return np.append(mesh.vertices, extra_points, axis=0)
    else:
        return mesh.vertices


def averageFaceArea(mesh):
    return np.mean(mesh.area_faces)


def getOneRingAreas(mesh):
    # Collect areas of faces adjacent to each vertex
    vertex_areas = mesh.area_faces[mesh.vertex_faces.flatten()].reshape(mesh.vertex_faces.shape)
    vertex_areas[np.where(mesh.vertex_faces == -1)] = 0

    return np.sum(vertex_areas, axis=1)


def buildKDTree(mesh):
    return spspat.cKDTree(mesh.vertices)


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


def visualizeLaplaceWeights(mesh, quantile=.01, weights=None, cmap='seismic', viewer=None, **kwargs):
    """Visualize Laplacian weights.

    Requires ``navis`` to be installed.

    Parameters
    ----------
    mesh :      trimesh.Trimesh
                Mesh to plot the weights for.
    quantile :  float [0-1]
                The vast majority of weights will be close to the mean while the
                interesting outliers will be very few. By default we are showing
                the top and bottom 0.1 quantile (i.e. the 10% highest and
                lowest values).
    weights :   np.ndarray, optional
                Laplacian weights. If not provided, will be computed.

    """
    mesh = make_trimesh(mesh, validate=False)

    try:
        import navis
        import vispy as vp
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('This function requires navis to be installed:\n'
                          '  pip3 install navis')

    if not isinstance(weights, np.ndarray):
        weights = laplacian_cotangent(mesh,
                                      #symmetric=False,
                                      normalized=True)

    if not isinstance(weights, spsp.coo_matrix):
        weights = spsp.coo_matrix(weights)

    # Get data (upper triangle only -> is supposed to be symmetrical)
    # Also removes diagonal (k=1)
    triu = spsp.triu(weights, k=1)
    row, col, data = triu.row, triu.col, triu.data

    if quantile:
        top = data >= np.quantile(data, 1-quantile)
        bottom = data <= np.quantile(data, quantile)
        row = row[top | bottom]
        col = col[top | bottom]
        data = data[top | bottom]

    # Weights are computed per edge
    co1, co2 = mesh.vertices[row], mesh.vertices[col]
    segments = np.hstack((co1, co2)).reshape(co1.shape[0] * 2, 3)

    # Generate colors
    cmap = plt.get_cmap(cmap)
    weights_norm = (data - data.min()) / (data.max() - data.min())

    colors = cmap(weights_norm)
    alpha = np.clip(np.fabs(weights_norm - .5) * 2, a_min=0.01, a_max=1)

    # We need to provide one color per vertex
    colors = np.hstack((colors, colors)).reshape(colors.shape[0] * 2, 4)
    #alpha = np.hstack((alpha, alpha)).reshape(alpha.shape[0] * 2, 1)

    # Combine color with alpha
    #colors = np.hstack((colors[:, :3], alpha))

    t = vp.scene.visuals.Line(pos=segments,
                              color=colors,
                              # Can only be used with method 'agg'
                              width=kwargs.get('linewidth', 1),
                              connect='segments',
                              antialias=kwargs.get('antialias', True),
                              method=kwargs.get('method', 'gl'))

    if not viewer:
        viewer = navis.get_viewer()
    if not viewer:
        viewer = navis.Viewer()

    viewer.add(t)

    return t
