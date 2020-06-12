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

import logging
import time

import numpy as np
import scipy as sp
from scipy.sparse.linalg import lsqr
from tqdm.auto import trange

from .utilities import (meanCurvatureLaplaceWeights, getMeshVPos,
                        averageFaceArea, getOneRingAreas, _make_trimesh)

logger = logging.getLogger('skeletonizer')

if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

def contract_mesh(mesh, iterations=10, SL=10, WC=2):
    """Contract mesh.

    Parameters
    ----------
    mesh :          mesh obj | dict
                    The mesh to be contracted. Can any object (e.g.
                    a trimesh.Trimesh) that has ``.vertices`` and ``.faces``
                    properties or a tuple ``(vertices, faces)`` or a dictionary
                    ``{'vertices': vertices, 'faces': faces}``.
                    Vertices and faces must be (N, 3) numpy arrays.
    iterations :    int, optional
                    Total rounds of contractions.
    SL :            float, optional
                    Factor by which the contraction matrix is multiplied
                    for each iteration.
    WC :            float, optional
                    Weight factor that affects the attraction constraint.

    Returns
    -------
    trimesh.Trimesh
                    Contracted copy of original mesh.

    """
    # Force into trimesh
    m = _make_trimesh(mesh)

    n = len(m.vertices)
    initialFaceWeight = averageFaceArea(m)
    originalOneRing = getOneRingAreas(m)
    zeros = np.zeros((n, 3))

    full_start = time.time()

    WH0_diag = np.zeros(n)
    WH0_diag.fill(WC)
    WH0 = sp.sparse.spdiags(WH0_diag, 0, WH0_diag.size, WH0_diag.size)

    # Make a copy but keep original values
    WH = sp.sparse.dia_matrix(WH0)

    WL_diag = np.zeros(n)
    WL_diag.fill(initialFaceWeight)
    WL = sp.sparse.spdiags(WL_diag, 0, WL_diag.size, WL_diag.size)

    # Copy mesh
    dm = m.copy()

    L = -meanCurvatureLaplaceWeights(dm, normalized=True)

    area_ratios = []
    area_ratios.append(1.0)
    originalFaceAreaSum = np.sum(originalOneRing)
    goodvertices = [[]]
    timetracker = []

    for i in trange(iterations, desc='Contracting'):
        start = time.time()
        vpos = getMeshVPos(dm)
        A = sp.sparse.vstack([L.dot(WL), WH])
        b = np.vstack((zeros, WH.dot(vpos)))
        cpts = np.zeros((n, 3))

        for j in range(3):
            cpts[:, j] = lsqr(A, b[:, j])[0]

        dm.vertices = cpts

        end = time.time()
        logger.debug('TOTAL TIME FOR SOLVING LEAST SQUARES: {:.3f}s'.format(end - start))
        newringareas = getOneRingAreas(dm)
        changeinarea = np.power(newringareas, -0.5)
        area_ratios.append(np.sum(newringareas) / originalFaceAreaSum)

        if(area_ratios[-1] > area_ratios[-2]):
            logger.debug('FACE AREA INCREASED FROM PREVIOUS: {:.4f} {:.4f}'.format(area_ratios[-1], area_ratios[-2]))
            logger.debug('ITERATION TERMINATED AT: {}'.format(i))
            logger.debug('RESTORE TO PREVIOUS GOOD POSITIONS FROM ITERATION: {}'.format(i - 1))

            cpts = goodvertices[0]
            dm.vertices = cpts
            break

        goodvertices[0] = cpts
        logger.debug('RATIO OF CHANGE IN FACE AREA: {:.4f}'.format(area_ratios[-1]))
        WL = sp.sparse.dia_matrix(WL.multiply(SL))
        WH = sp.sparse.dia_matrix(WH0.multiply(changeinarea))
        L = -meanCurvatureLaplaceWeights(dm, normalized=True)
        full_end = time.time()

        timetracker.append(full_end - full_start)
        full_start = time.time()

    logger.debug('TOTAL TIME FOR MESH CONTRACTION ::: ', np.sum(timetracker), ' FOR VERTEX COUNT ::: #', n)
    return dm
