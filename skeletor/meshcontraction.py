#    This script is part of skeletor (http://www.github.com/schlegelp/skeletor).
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
import trimesh as tm

from scipy.sparse.linalg import lsqr
from tqdm.auto import tqdm

from .utilities import (meanCurvatureLaplaceWeights, getMeshVPos,
                        averageFaceArea, getOneRingAreas, make_trimesh)

logger = logging.getLogger('skeletor')

if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def contract(mesh, epsilon=1e-06, iter_lim=10, precision=1e-07, SL=10, WH0=1,
             progress=True, validate=True):
    """Contract mesh.

    In a nutshell: this function contracts the mesh by applying rounds of
    constraint Laplacian smoothing. This function can be fairly expensive
    and I highly recommend you play around with ``epsilon`` and ``precision``:
    the contraction doesn't have to be perfect for good skeletonization results.

    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to be contracted. Can be any object (e.g.
                    a trimesh.Trimesh) that has ``.vertices`` and ``.faces``
                    properties or a tuple ``(vertices, faces)`` or a dictionary
                    ``{'vertices': vertices, 'faces': faces}``.
                    Vertices and faces must be (N, 3) numpy arrays.
    epsilon :       float (0-1), optional
                    Target contraction rate as measured by the sum of all face
                    areas in the contracted versus the original mesh. Algorithm
                    will stop once mesh is contracted below this threshold.
                    Depending on your mesh (number of faces, shape) reaching a
                    strong contraction can be extremely costly with comparatively
                    little benefit for the subsequent skeletonization. Note that
                    the algorithm might stop short of this target if ``iter_lim``
                    is reached first or if the sum of face areas is increasing
                    from one iteration to the next instead of decreasing.
    iter_lim :      int (>1), optional
                    Maximum rounds of contractions.
    precision :     float, optional
                    Sets the precision for finding the least-square solution.
                    This is the main determinant for speed vs quality: lower
                    values will take (much) longer but will get you closer to an
                    optimally contracted mesh. Higher values will be faster but
                    the iterative contractions might stop early.
    SL :            float, optional
                    Factor by which the contraction matrix is multiplied for
                    each iteration. In theory, lower values are more likely to
                    get you an optimal contraction at the cost of needing more
                    iterations.
    WH0 :           float, optional
                    Weight factor that affects the attraction constraints.
                    Increase this value if you're experiencing
    validate :      bool
                    If True, will try to fix potential issues with the mesh
                    (e.g. infinite values, duplicate vertices, degenerate faces)
                    before collapsing. Degenerate meshes can lead to effectively
                    infinite runtime for this function!

    Returns
    -------
    trimesh.Trimesh
                    Contracted copy of original mesh.

    """
    # Force into trimesh
    m = make_trimesh(mesh, validate=validate)

    n = len(m.vertices)
    WL0 = 10.0**-3 * np.sqrt(averageFaceArea(m))
    #initialFaceWeight = 1.0 / (10.0 * np.sqrt(averageFaceArea(m)))
    originalOneRing = getOneRingAreas(m)
    zeros = np.zeros((n, 3))

    full_start = time.time()

    WH0_diag = np.zeros(n)
    WH0_diag.fill(WH0)
    WH0 = sp.sparse.spdiags(WH0_diag, 0, WH0_diag.size, WH0_diag.size)

    # Make a copy but keep original values
    WH = sp.sparse.dia_matrix(WH0)

    WL_diag = np.zeros(n)
    WL_diag.fill(WL0)
    WL = sp.sparse.spdiags(WL_diag, 0, WL_diag.size, WL_diag.size)

    # Copy mesh
    dm = m.copy()

    L = -meanCurvatureLaplaceWeights(dm, normalized=True)

    area_ratios = []
    area_ratios.append(1.0)
    originalFaceAreaSum = np.sum(originalOneRing)
    goodvertices = [[]]
    timetracker = []

    with tqdm(total=iter_lim, desc='Contracting', disable=progress is False) as pbar:
        for i in range(iter_lim):
            start = time.time()
            vpos = getMeshVPos(dm)
            A = sp.sparse.vstack([L.dot(WL), WH])
            b = np.vstack((zeros, WH.dot(vpos)))
            cpts = np.zeros((n, 3))

            for j in range(3):
                # Solve A*x = b
                # Note that we force scipy's lsqr to use current vertex
                # positions as start points - this speeds things up and
                # without we get suboptimal solutions that lead to early
                # termination
                cpts[:, j] = lsqr(A, b[:, j],
                                  atol=precision, btol=precision,
                                  x0=dm.vertices[:, j])[0]

            dm.vertices = cpts

            end = time.time()
            logger.debug('TOTAL TIME FOR SOLVING LEAST SQUARES: {:.3f}s'.format(end - start))
            newringareas = getOneRingAreas(dm)
            changeinarea = np.power(newringareas, -0.5)
            area_ratios.append(np.sum(newringareas) / originalFaceAreaSum)
            pbar.update()

            if (area_ratios[-1] > area_ratios[-2]):
                logger.debug('FACE AREA INCREASED FROM PREVIOUS: {:.4f} {:.4f}'.format(area_ratios[-1], area_ratios[-2]))
                logger.debug('ITERATION TERMINATED AT: {}'.format(i))
                logger.debug('RESTORE TO PREVIOUS GOOD POSITIONS FROM ITERATION: {}'.format(i - 1))
                dm.vertices = goodvertices[0]
                break

            goodvertices[0] = cpts
            logger.debug('RATIO OF CHANGE IN FACE AREA: {:.2g}'.format(area_ratios[-1]))
            WL = sp.sparse.dia_matrix(WL.multiply(SL))
            WH = sp.sparse.dia_matrix(WH0.multiply(changeinarea))
            L = -meanCurvatureLaplaceWeights(dm, normalized=True)
            full_end = time.time()

            timetracker.append(full_end - full_start)
            full_start = time.time()
            pbar.set_postfix({'contr_rate': round(area_ratios[-1], 3)})

            if (area_ratios[-1] <= epsilon):
                break

        logger.debug('TOTAL TIME FOR MESH CONTRACTION ::: {:.2f}s FOR VERTEX COUNT ::: #{}'.format(np.sum(timetracker), n))
        return dm
