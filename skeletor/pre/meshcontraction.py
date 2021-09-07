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

import logging
import time

import numpy as np
import scipy as sp

from scipy.sparse.linalg import lsqr
from tqdm.auto import tqdm

from ..utilities import make_trimesh
from .utils import (laplacian_cotangent, getMeshVPos, laplacian_umbrella,
                    averageFaceArea, getOneRingAreas)

logger = logging.getLogger('skeletor')

if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def contract(mesh, epsilon=1e-06, iter_lim=100, time_lim=None, precision=1e-07,
             SL=2, WH0=1, WL0='auto', operator='cotangent', progress=True,
             validate=True):
    """Contract mesh.

    In a nutshell: this function contracts the mesh by applying rounds of
    _constraint_ Laplacian smoothing. This function can be fairly expensive
    and I highly recommend you play around with ``SL`` to contract the
    mesh in as few steps as possible. The contraction doesn't have to be perfect
    for good skeletonization results (<10%, i.e. `epsilon<0.1`).

    Also: parameterization matters a lot! Default parameters will get you there
    but playing around with `SL` and `WH0` might speed things up by an order of
    magnitude.

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
                    or ``time_lim`` is reached first or if the sum of face areas
                    is increasing from one iteration to the next instead of
                    decreasing.
    iter_lim :      int (>1), optional
                    Maximum rounds of contractions.
    time_lim :      int, optional
                    Maximum run time in seconds. Note that this limit is not
                    checked during but after each round of contraction. Hence,
                    the actual total time will likely overshoot ``time_lim``.
    precision :     float, optional
                    Sets the precision for finding the least-square solution.
                    This is the main determinant for speed vs quality: lower
                    values will take (much) longer but will get you closer to an
                    optimally contracted mesh. Higher values will be faster but
                    the iterative contractions might stop early.
    SL :            float, optional
                    Factor by which the contraction matrix is multiplied for
                    each iteration. Higher values = quicker contraction, lower
                    values = more likely to get you an optimal contraction.
    WH0 :           float, optional
                    Initial weight factor for the attraction constraints.
                    The ratio of the initial weights ``WL0`` and ``WH0``
                    controls the smoothness and the degree of contraction of the
                    first iteration result, thus it determines the amount of
                    details retained in subsequent and final contracted meshes:
                    higher ``WH0`` = more details retained.
    WL0 :           "auto" | float
                    Initial weight factor for the contraction constraints. By
                    default ("auto"), this will be set to ``1e-3 * sqrt(A)``
                    with ``A`` being the average face area. This ensures that
                    contraction forces scale with the coarseness of the mesh.
    operator :      "cotangent" | "umbrella"
                    Which Laplacian operator to use:

                      - The "cotangent" operator (default) takes both topology
                        and geometry of the mesh into account and is hence a
                        better descriptor of the curvature flow. This is the
                        operator used in the original paper.
                      - The "umbrella" operator (aka "uniform weighting") uses
                        only topological features of the mesh. This also makes
                        it more robust against flaws in the mesh! Use it when
                        the cotangent operator produces oddly contracted meshes.

    progress :      bool
                    Whether or not to show a progress bar.
    validate :      bool
                    If True, will try to fix potential issues with the mesh
                    (e.g. infinite values, duplicate vertices, degenerate faces)
                    before collapsing. Degenerate meshes can lead to effectively
                    infinite runtime for this function!

    Returns
    -------
    trimesh.Trimesh
                    Contracted copy of original mesh. The final contraction rate
                    is attached to the mesh as ``.epsilon`` property.

    References
    ----------
    [1] Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh
        contraction. ACM Transactions on Graphics (TOG). 2008 Aug 1;27(3):44.

    """
    assert operator in ('cotangent', 'umbrella')
    start = time.time()

    # Force into trimesh
    m = make_trimesh(mesh, validate=validate)
    n = len(m.vertices)

    # Initialize attraction weights
    zeros = np.zeros((n, 3))
    WH0_diag = np.zeros(n)
    WH0_diag.fill(WH0)
    WH0 = sp.sparse.spdiags(WH0_diag, 0, WH0_diag.size, WH0_diag.size)
    # Make a copy but keep original values
    WH = sp.sparse.dia_matrix(WH0)

    # Initialize contraction weights
    if WL0 == 'auto':
        WL0 = 1e-03 * np.sqrt(averageFaceArea(m))
        #WL0 = 1.0 / 10.0 * np.sqrt(averageFaceArea(m))
    WL_diag = np.zeros(n)
    WL_diag.fill(WL0)
    WL = sp.sparse.spdiags(WL_diag, 0, WL_diag.size, WL_diag.size)

    # Copy mesh
    dm = m.copy()

    area_ratios = [1.0]
    originalRingAreas = getOneRingAreas(dm)
    goodvertices = dm.vertices
    bar_format = ("{l_bar}{bar}| [{elapsed}<{remaining}, "
                  "{postfix[0]}/{postfix[1]}it, "
                  "{rate_fmt}, epsilon {postfix[2]:.2g}")
    with tqdm(total=100,
              bar_format=bar_format,
              disable=progress is False,
              postfix=[1, iter_lim, 1]) as pbar:
        for i in range(iter_lim):
            # Get Laplace weights
            if operator == 'cotangent':
                L = laplacian_cotangent(dm, normalized=True)
            else:
                L = laplacian_umbrella(dm)

            V = getMeshVPos(dm)
            A = sp.sparse.vstack([WL.dot(L), WH])
            b = np.vstack((zeros, WH.dot(V)))

            cpts = np.zeros((n, 3))
            for j in range(3):
                """
                # Solve A*x = b
                # Note that we force scipy's lsqr() to use current vertex
                # positions as start points - this speeds things up and
                # without it we get suboptimal solutions that lead to early
                # termination
                cpts[:, j] = lsqr(A, b[:, j],
                                  atol=precision, btol=precision,
                                  damp=1,
                                  x0=dm.vertices[:, j])[0]
                """
                # The solution below is recommended in scipy's lsqr docstring
                # for when we have an initial estimate
                # Gives use the same results as above but is slightly faster

                # Initial estimate (i.e. our current positions)
                x0 = dm.vertices[:, j]
                # Compute residual vector
                r0 = b[:, j] - A * x0
                # Use LSQR to solve the system
                dx = lsqr(A, r0,
                          atol=precision, btol=precision,
                          damp=1)[0]
                # Add the correction dx to obtain a final solution
                cpts[:, j] = x0 + dx

            # Update mesh with new vertex position
            dm.vertices = cpts

            # Update iteration in progress bar
            if progress:
                pbar.postfix[0] = i + 1

            # Break if face area has increased compared to the last iteration
            new_eps = dm.area / m.area
            if (new_eps > area_ratios[-1]):
                dm.vertices = goodvertices
                if progress:
                    tqdm.write("Total face area increased from last iteration."
                               f" Contraction stopped prematurely after {i} "
                               f"iterations at epsilon {area_ratios[-1]:.2g}.")
                break
            area_ratios.append(new_eps)

            # Update progress bar
            if progress:
                pbar.postfix[2] = area_ratios[-1]
                prog = round((area_ratios[-2] - area_ratios[-1]) / (1 - epsilon) * 100)
                pbar.update(min(prog, 100-pbar.n))

            goodvertices = cpts

            # Update contraction weights -> at each iteration the contraction
            # forces increase to counteract the increased attraction forces
            WL = sp.sparse.dia_matrix(WL.multiply(SL))

            # Update attraction weights -> the smaller the one ring areas
            # the higher the attraction forces
            changeinarea = np.sqrt(originalRingAreas / getOneRingAreas(dm))
            WH = sp.sparse.dia_matrix(WH0.multiply(changeinarea))

            # Stop if we reached our target contraction rate
            if (area_ratios[-1] <= epsilon):
                break

            # Stop if time limit is reached
            if not isinstance(time_lim, (bool, type(None))):
                if (time.time() - start) >= time_lim:
                    break

        # Keep track of final epsilon
        dm.epsilon = area_ratios[-1]

        return dm
