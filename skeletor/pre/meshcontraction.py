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
import warnings

import numpy as np
import scipy as sp

from scipy.sparse.linalg import factorized
from tqdm.auto import tqdm

from ..utilities import make_trimesh
from .utils import (laplacian_cotangent, getMeshVPos, laplacian_umbrella,
                    averageFaceArea, getOneRingAreas)

logger = logging.getLogger('skeletor')

if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def _has_robust_laplacian():
    """Check whether the optional ``robust_laplacian`` package is available."""
    try:
        import robust_laplacian  # noqa: F401
        return True
    except ImportError:
        return False


def contract(mesh, epsilon=0.1, iter_lim=100, time_lim=None, precision=None,
             SL=2, WH0=1, WL0='auto', operator='auto', progress=True,
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
                    or ``time_lim`` is reached first or if the contraction
                    becomes unstable (the sum of face areas increases or the
                    bounding box grows from one iteration to the next).

                    .. warning::

                       Avoid pushing ``epsilon`` too low (the default ``0.1`` is
                       plenty for skeletonization). The contraction reduces the
                       total face area regardless of whether it is faithfully
                       thinning the mesh or merely collapsing (often
                       disconnected) regions onto their centroids. Very small
                       ``epsilon`` therefore tends to *over-contract*: vertices
                       get dragged past the medial axis and away from the
                       original surface, which hurts rather than helps the
                       subsequent skeletonization.
    iter_lim :      int (>1), optional
                    Maximum rounds of contractions.
    time_lim :      int, optional
                    Maximum run time in seconds. Note that this limit is not
                    checked during but after each round of contraction. Hence,
                    the actual total time will likely overshoot ``time_lim``.
    precision :     float, optional
                    DEPRECATED and ignored. The contraction now solves the
                    normal equations directly (exact sparse factorization) so
                    there is no iterative-solver tolerance to set. Passing a
                    value will emit a deprecation warning.
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
    operator :      "auto" | "robust" | "cotangent" | "umbrella"
                    Which Laplacian operator to use:

                      - "auto" (default) uses "robust" if the optional
                        ``robust_laplacian`` package is installed and falls back
                        to "cotangent" otherwise.
                      - The "robust" operator uses the intrinsic mollified
                        Laplacian of Sharp & Crane (via ``robust_laplacian``).
                        It is well-defined even on non-manifold and degenerate
                        meshes and is the recommended choice for the messy
                        meshes this package typically deals with.
                      - The "cotangent" operator takes both topology and
                        geometry of the mesh into account and is hence a good
                        descriptor of the curvature flow. This is the operator
                        used in the original paper.
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
    # Resolve the operator (and check optional dependency for "robust")
    if operator == 'auto':
        operator = 'robust' if _has_robust_laplacian() else 'cotangent'
        if operator == 'cotangent':
            logger.info("`robust_laplacian` not installed - falling back to "
                        "the 'cotangent' operator. Install it with "
                        "`pip install robust_laplacian` for better results on "
                        "imperfect meshes.")
    elif operator == 'robust' and not _has_robust_laplacian():
        raise ImportError("operator='robust' requires the `robust_laplacian` "
                          "package:\n  pip install robust_laplacian")
    assert operator in ('robust', 'cotangent', 'umbrella')

    if precision is not None:
        warnings.warn("`precision` is deprecated and ignored: contract() now "
                      "solves the normal equations directly instead of using an "
                      "iterative least-squares solver.",
                      DeprecationWarning, stacklevel=2)

    if operator == 'robust':
        import robust_laplacian

    start = time.time()

    # Force into trimesh
    m = make_trimesh(mesh, validate=validate)
    n = len(m.vertices)

    # Initial per-vertex attraction weight (W_H) and the scalar contraction
    # weight (W_L0). W_H anchors vertices to their current positions; W_L grows
    # by SL each iteration and drives the curvature-flow contraction.
    WH0 = np.full(n, float(WH0))

    if WL0 == 'auto':
        WL0 = 1e-03 * np.sqrt(averageFaceArea(m))
    WL0 = float(WL0)

    # Copy mesh
    dm = m.copy()

    area_ratios = [1.0]
    ring0 = None  # original one-ring areas; set on the first iteration
    goodvertices = dm.vertices
    # The contraction flow is contractive, so the bounding box must shrink
    # monotonically. A growing bbox (or non-finite vertices) signals numerical
    # breakdown - which happens once `wl` grows so large that ``wl**2 * L.T@L``
    # swamps ``diag(W_H**2)`` below float precision and the solve diverges.
    prev_diag = np.linalg.norm(m.bounds[1] - m.bounds[0])
    bar_format = ("{l_bar}{bar}| [{elapsed}<{remaining}, "
                  "{postfix[0]}/{postfix[1]}it, "
                  "{rate_fmt}, epsilon {postfix[2]:.2g}")
    with tqdm(total=100,
              bar_format=bar_format,
              disable=progress is False,
              postfix=[1, iter_lim, 1]) as pbar:
        for i in range(iter_lim):
            # Compute the Laplace operator and per-vertex areas on the current
            # geometry (faces are constant throughout)
            if operator == 'robust':
                # Intrinsic mollified Laplacian (Sharp & Crane). L is positive
                # semi-definite; its sign is irrelevant here as the energy
                # squares L. M is the (diagonal) lumped mass matrix.
                L, M = robust_laplacian.mesh_laplacian(np.asarray(dm.vertices),
                                                       np.asarray(dm.faces),
                                                       mollify_factor=1e-3)
                ring = M.diagonal()
            elif operator == 'cotangent':
                L = laplacian_cotangent(dm, normalized=False)
                ring = getOneRingAreas(dm)
            else:
                L = laplacian_umbrella(dm)
                ring = getOneRingAreas(dm)

            ring = np.clip(ring, a_min=1e-12, a_max=None)
            # The first iteration runs on the original geometry - keep its areas
            # as the reference for the attraction-weight update
            if ring0 is None:
                ring0 = ring

            # Contraction weight grows by SL each iteration
            wl = WL0 * SL ** i

            # Attraction weights grow where the one-ring area has shrunk. Floor
            # to keep the system positive-definite for collapsed vertices.
            WH = np.maximum(WH0 * np.sqrt(ring0 / ring), 1e-6)

            V = getMeshVPos(dm)

            # Solve the constrained least-squares problem
            #   min  || wl * L * V' ||^2  +  || W_H * (V' - V) ||^2
            # via its (symmetric positive-definite) normal equations
            #   (wl^2 * L^T L + diag(W_H^2)) V' = diag(W_H^2) V
            # One sparse factorization + three back-substitutions per iteration.
            WH2 = WH ** 2
            N = (wl ** 2) * (L.T @ L) + sp.sparse.diags(WH2)
            solve = factorized(N.tocsc())
            rhs = WH2[:, None] * V
            cpts = np.column_stack([solve(np.ascontiguousarray(rhs[:, j]))
                                    for j in range(3)])

            # Update mesh with new vertex position
            dm.vertices = cpts

            # Update iteration in progress bar
            if progress:
                pbar.postfix[0] = i + 1

            # Break if the contraction became unstable (bounding box grew or
            # vertices went non-finite) or if the total face area increased
            # compared to the last iteration. In all cases revert to the last
            # known-good vertices.
            new_eps = dm.area / m.area
            new_diag = np.linalg.norm(dm.bounds[1] - dm.bounds[0])
            unstable = (not np.isfinite(cpts).all()
                        or new_diag > prev_diag * 1.001)
            if unstable or (new_eps > area_ratios[-1]):
                dm.vertices = goodvertices
                reason = ("Mesh became numerically unstable" if unstable
                          else "Total face area increased from last iteration")
                msg = (f"{reason}. Contraction stopped after {i} iterations at "
                       f"epsilon {area_ratios[-1]:.2g}.")
                if unstable:
                    logger.info(msg)
                if progress:
                    tqdm.write(msg)
                break
            area_ratios.append(new_eps)
            prev_diag = new_diag

            # Update progress bar
            if progress:
                pbar.postfix[2] = area_ratios[-1]
                prog = round((area_ratios[-2] - area_ratios[-1]) / (1 - epsilon) * 100)
                pbar.update(min(prog, 100-pbar.n))

            goodvertices = cpts

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
