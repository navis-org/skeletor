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

from collections import OrderedDict
import trimesh

import numpy as np
import scipy.sparse as spsp
import scipy.spatial as spspat


def _make_trimesh(mesh):
    """ Makes sure we work with a trimesh.Trimesh.

    Parameters
    ----------
    meshdata :  tuple | dict | mesh-like object
                Tuple: (vertices, faces)
                dict: {'vertices': [], 'faces': []}
                mesh-like object: mesh.vertices, mesh.faces

    Returns
    -------
    vertices
    faces
    """

    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    elif isinstance(mesh, (tuple, list)):
        if len(mesh) == 2:
            return trimesh.Trimesh(vertices=mesh[0],
                                   faces=mesh[1])
    elif isinstance(mesh, dict):
        return trimesh.Trimesh(vertices=mesh['vertices'],
                               faces=mesh['faces'])
    else:
        try:
            return trimesh.Trimesh(vertices=mesh.vertices,
                                   faces=mesh.faces)
        except BaseException:
            pass

    raise TypeError('Unable to extract vertices/faces from object of type '
                    '"{}"'.format(type(mesh)))


def getBBox(verts):
    """ Returns bounding box of vertices. """
    min_coords = np.min(verts, axis=0)
    max_coords = np.max(verts, axis=0)
    diameter = np.linalg.norm(max_coords - min_coords)
    return min_coords, max_coords, diameter


def meanCurvatureLaplaceWeights(mesh, symmetric=False, normalized=False):
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

    if(symmetric and not normalized):
        sum_vector = W.sum(axis=0)
        d = spsp.dia_matrix((sum_vector, [0]), shape=(n, n))
        L = d - W
    elif(symmetric and normalized):
        sum_vector = W.sum(axis=0)
        sum_vector_powered = np.power(sum_vector, -0.5)
        d = spsp.dia_matrix((sum_vector_powered, [0]), shape=(n, n))
        eye = spsp.identity(n)
        L = eye - d * W * d
    elif (not symmetric and normalized):
        sum_vector = W.sum(axis=0)
        sum_vector_powered = np.power(sum_vector, -1.0)
        d = spsp.dia_matrix((sum_vector_powered, [0]), shape=(n, n))
        eye = spsp.identity(n)
        L = eye - d * W
    else:
        L = W

    return L


def getMeshVPos(mesh, extra_points=[]):
    if any(extra_points):
        return np.append(mesh.vertices, extra_points, axis=0)
    else:
        return mesh.vertices


def averageFaceArea(mesh):
    return 1.0 / (10.0 * np.sqrt(np.mean(mesh.area_faces)))


def getOneRingAreas(mesh):
    # Collect areas of faces adjacent to each vertex
    vertex_areas = mesh.area_faces[mesh.vertex_faces.flatten()].reshape(mesh.vertex_faces.shape)
    vertex_areas[np.where(mesh.vertex_faces == -1)] = 0

    return np.sum(vertex_areas, axis=1)


def buildKDTree(mesh):
    return spspat.cKDTree(mesh.vertices)
