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
import trimesh as tm

from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from ..utilities import make_trimesh
from ..pre.meshcontraction import _has_robust_laplacian
from ..pre.utils import (laplacian_cotangent, laplacian_umbrella,
                         averageFaceArea, getOneRingAreas)
from .edge_collapse import by_edge_collapse

__all__ = ['by_mean_curvature']

logger = logging.getLogger('skeletor')

if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def by_mean_curvature(mesh, epsilon=0.05, iter_lim=100, time_lim=None,
                      SL=2, WH0=1, WL0='auto', operator='auto',
                      collapse_factor=0.5, progress=True, validate=True):
    """Skeletonize a mesh by mean curvature flow with local remeshing.

    This implements a curve-skeleton extraction in the spirit of the "Mean
    Curvature Skeletons" of Tagliasacchi et al. [1] (the algorithm behind
    CGAL's mesh skeletonization). In a nutshell: the mesh surface is contracted
    by repeatedly solving a constrained Laplacian (mean-curvature) flow - very
    much like `skeletor.pre.contract` - but in between iterations short edges are
    *collapsed* (local remeshing). This is the key difference to plain
    contraction: by removing the cross-sectional edges that the flow squeezes
    together, the surface keeps collapsing onto its medial axis instead of
    grinding to a halt once the triangles degenerate. The thinned "meso-skeleton"
    is then reduced to a 1-D skeleton via the quadric edge-collapse of [2] (see
    `skeletor.skeletonize.by_edge_collapse`).

    .. note::

       Compared to the reference algorithm this implementation performs edge
       *collapses* but no edge *splits* and does not include the optional medial
       centering term. The collapses are guarded by the topological *link
       condition* [2] so that remeshing never pinches the surface into a
       non-manifold fold - without this guard the contraction shreds the mesh
       and introduces spurious breaks into the skeleton.

    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to be skeletonized. Can be any object (e.g. a
                    trimesh.Trimesh) that has ``.vertices`` and ``.faces``
                    properties or a tuple ``(vertices, faces)`` or a dictionary
                    ``{'vertices': vertices, 'faces': faces}``.
    epsilon :       float (0-1), optional
                    Target contraction rate as measured by the sum of all face
                    areas of the contracted versus the original mesh. The
                    contraction stops once the mesh is contracted below this
                    threshold (or ``iter_lim``/``time_lim`` is reached). Because
                    of the remeshing this can be pushed lower than for plain
                    `contract()` without numerical breakdown.
    iter_lim :      int (>1), optional
                    Maximum number of contraction iterations.
    time_lim :      int, optional
                    Maximum run time in seconds (checked after each iteration).
    SL :            float, optional
                    Factor by which the contraction weight is multiplied each
                    iteration. Higher = quicker (but coarser) contraction.
    WH0 :           float, optional
                    Initial attraction weight anchoring vertices to their current
                    position. See `skeletor.pre.contract` for details.
    WL0 :           "auto" | float, optional
                    Initial contraction weight. By default scales with the mean
                    face area of the input mesh.
    operator :      "auto" | "robust" | "cotangent" | "umbrella", optional
                    Which Laplacian operator to use for the flow - see
                    `skeletor.pre.contract` for an explanation of the options.
                    "auto" uses "robust" if the optional ``robust_laplacian``
                    package is installed and falls back to "cotangent".
    collapse_factor : float, optional
                    Controls the local remeshing: after each contraction step,
                    edges shorter than ``collapse_factor`` times the *mean edge
                    length of the original mesh* are collapsed. Larger values
                    collapse more aggressively (faster, coarser skeleton).
    progress :      bool, optional
                    Whether to show progress bars.
    validate :      bool, optional
                    If True, try to fix potential issues with the mesh (e.g.
                    degenerate faces) before skeletonizing.

    Returns
    -------
    skeletor.Skeleton
                    Holds the results of the skeletonization. Includes a
                    ``mesh_map`` (mapping each mesh vertex to a skeleton node)
                    derived from each vertex's contracted position.

    References
    ----------
    [1] Tagliasacchi A, Alhashim I, Olson M, Zhang H. Mean Curvature Skeletons.
        Computer Graphics Forum (SGP). 2012;31(5):1735-1744.
    [2] Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh
        contraction. ACM Transactions on Graphics (TOG). 2008;27(3):44.

    """
    # Resolve the Laplacian operator (mirrors skeletor.pre.contract)
    if operator == 'auto':
        operator = 'robust' if _has_robust_laplacian() else 'cotangent'
    elif operator == 'robust' and not _has_robust_laplacian():
        raise ImportError("operator='robust' requires the `robust_laplacian` "
                          "package:\n  pip install robust_laplacian")
    assert operator in ('robust', 'cotangent', 'umbrella')

    m = make_trimesh(mesh, validate=validate)

    # Run the mean-curvature flow with interleaved edge collapses. This returns
    # the contracted ("meso-skeleton") mesh plus, for every *original* vertex,
    # the position it contracted to.
    cm, orig_pos = _contract_and_remesh(m, epsilon=epsilon, iter_lim=iter_lim,
                                        time_lim=time_lim, SL=SL, WH0=WH0,
                                        WL0=WL0, operator=operator,
                                        collapse_factor=collapse_factor,
                                        progress=progress)

    # Reduce the contracted surface to a 1-D skeleton via quadric edge collapse.
    # The contracted mesh is exactly the kind of input by_edge_collapse works
    # well on (see its docstring).
    skel = by_edge_collapse(cm, progress=progress)

    # Build the mesh -> skeleton node map: assign every original vertex to the
    # skeleton node closest to its *contracted* position. Using the contracted
    # (rather than the original) position respects the flow and avoids assigning
    # vertices across bends/branches.
    tree = cKDTree(skel.vertices)
    _, nn = tree.query(orig_pos)
    # Reindexed node IDs are 0..N-1 and equal to the row index, so `nn` already
    # are valid skeleton node IDs.
    skel.mesh_map = skel.swc.node_id.values[nn]
    skel.mesh = m
    skel.method = 'mean_curvature'

    return skel


def _resolve(parent):
    """Resolve a union-find ``parent`` array to the root of every element."""
    root = parent
    while True:
        nxt = root[root]
        if np.array_equal(nxt, root):
            return nxt
        root = nxt


def _contract_and_remesh(m, epsilon, iter_lim, time_lim, SL, WH0, WL0,
                         operator, collapse_factor, progress):
    """Mean-curvature contraction with interleaved short-edge collapses.

    Returns
    -------
    cm :        trimesh.Trimesh
                The contracted mesh.
    orig_pos :  (N, 3) array
                For each *original* vertex, the position it contracted to.

    """
    if operator == 'robust':
        import robust_laplacian

    start = time.time()

    n0 = len(m.vertices)
    # Global, stable index space = the original vertices. We only ever *collapse*
    # vertices (never split), so this space never grows. `parent` is a union-find
    # over it; `pos` holds the current position of every (root) vertex.
    parent = np.arange(n0)
    pos = np.array(m.vertices, dtype=np.float64)
    # Faces in global indices; shrinks as edges collapse.
    faces = np.array(m.faces)

    WH0 = float(WH0)
    if WL0 == 'auto':
        WL0 = 1e-03 * np.sqrt(averageFaceArea(m))
    WL0 = float(WL0)

    # Scalar reference one-ring area and the edge-collapse length threshold are
    # both derived from the *original* mesh so they stay on a fixed scale.
    ring0_mean = float(np.mean(getOneRingAreas(m)))
    ring0_mean = max(ring0_mean, 1e-12)
    collapse_thresh = collapse_factor * float(np.mean(m.edges_unique_length))

    full_area = m.area
    prev_ratio = 1.0

    with tqdm(total=100, desc='MCF contraction', disable=progress is False) as pbar:
        for i in range(iter_lim):
            # --- Build a compact mesh from the vertices still bounding a face ---
            used = np.unique(faces)
            if len(used) < 4 or len(faces) < 4:
                # Fully (or nearly) collapsed - nothing left to contract.
                break
            remap = np.full(n0, -1)
            remap[used] = np.arange(len(used))
            faces_c = remap[faces]
            verts_c = pos[used]
            cm = tm.Trimesh(verts_c, faces_c, process=False)

            # --- Constrained Laplacian (mean-curvature) flow solve ---
            if operator == 'robust':
                L, M = robust_laplacian.mesh_laplacian(np.asarray(verts_c),
                                                       np.asarray(faces_c),
                                                       mollify_factor=1e-3)
                ring = M.diagonal()
            elif operator == 'cotangent':
                L = laplacian_cotangent(cm, normalized=False)
                ring = getOneRingAreas(cm)
            else:
                L = laplacian_umbrella(cm)
                ring = getOneRingAreas(cm)
            ring = np.clip(ring, a_min=1e-12, a_max=None)

            wl = WL0 * SL ** i
            # Freeze vertices that already sit in thin (small one-ring) regions.
            WH = np.maximum(WH0 * np.sqrt(ring0_mean / ring), 1e-6)

            WH2 = WH ** 2
            N = (wl ** 2) * (L.T @ L) + sp.sparse.diags(WH2)
            solve = factorized(N.tocsc())
            rhs = WH2[:, None] * verts_c
            new = np.column_stack([solve(np.ascontiguousarray(rhs[:, j]))
                                   for j in range(3)])

            if not np.isfinite(new).all():
                logger.info(f"Mean-curvature flow became unstable at iteration "
                            f"{i}. Stopping.")
                break

            # Scatter the new positions back into the global space.
            pos[used] = new

            # --- Local remeshing: collapse short edges ---
            faces = _collapse_short_edges(cm, used, parent, pos, faces,
                                          collapse_thresh)

            # --- Progress / stopping based on remaining surface area ---
            used = np.unique(faces)
            if len(used) < 4 or len(faces) < 4:
                break
            cm_area = tm.Trimesh(pos[used],
                                 np.searchsorted(used, faces),
                                 process=False).area
            ratio = cm_area / full_area

            if progress:
                prog = round((prev_ratio - ratio) / (1 - epsilon) * 100)
                pbar.update(min(max(prog, 0), 100 - pbar.n))
            prev_ratio = ratio

            if ratio <= epsilon:
                break
            if not isinstance(time_lim, (bool, type(None))):
                if (time.time() - start) >= time_lim:
                    break

    # Final contracted mesh.
    used = np.unique(faces)
    if len(used) < 4:
        raise ValueError("Mean-curvature contraction collapsed the mesh "
                         "entirely. This usually means the mesh is "
                         "closed/blob-like rather than tubular.")
    cm = tm.Trimesh(pos[used], np.searchsorted(used, faces), process=False)

    # Contracted position of every original vertex = position of its root.
    roots = _resolve(parent)
    orig_pos = pos[roots]

    return cm, orig_pos


def _collapse_short_edges(cm, used, parent, pos, faces, thresh):
    """Collapse edges shorter than ``thresh`` (one independent round).

    Mutates ``parent`` and ``pos`` in place and returns the updated global-index
    ``faces`` array (degenerate faces removed).

    Parameters
    ----------
    cm :        trimesh.Trimesh
                Compact mesh of the current iteration.
    used :      (K, ) array
                Maps compact vertex index -> global vertex index.
    parent :    (N, ) array
                Union-find over the global vertex space (modified in place).
    pos :       (N, 3) array
                Global vertex positions (modified in place).
    faces :     (F, 3) array
                Faces in global indices.
    thresh :    float
                Edges shorter than this are collapsed.

    """
    edges = cm.edges_unique
    lengths = cm.edges_unique_length

    short = np.where(lengths < thresh)[0]
    if not len(short):
        return faces

    # Only collapse edges that satisfy the topological *link condition* [1]: an
    # edge (u, v) may be collapsed only if every vertex adjacent to both u and v
    # also forms a triangle with them. Collapsing an edge with a "blind" common
    # neighbour (one that shares no face) pinches the surface into a non-manifold
    # fold; repeated over the contraction this shreds the mesh into vertex-only-
    # connected triangle soup and introduces spurious skeleton breaks. The check
    # is equivalent to: #(common neighbours of u, v) == #(faces on edge u-v).
    n = cm.vertices.shape[0]
    e2 = np.concatenate([edges[:, 0], edges[:, 1]])
    e2b = np.concatenate([edges[:, 1], edges[:, 0]])
    A = sp.sparse.csr_matrix((np.ones(len(e2)), (e2, e2b)), shape=(n, n))
    A.data[:] = 1
    A2 = (A @ A).tocsr()
    # Number of faces incident to each (undirected) edge.
    fe = np.sort(np.concatenate([cm.faces[:, [0, 1]],
                                 cm.faces[:, [1, 2]],
                                 cm.faces[:, [0, 2]]]), axis=0)
    ue, fcnt = np.unique(fe, axis=0, return_counts=True)
    FC = sp.sparse.csr_matrix((np.concatenate([fcnt, fcnt]),
                              (np.concatenate([ue[:, 0], ue[:, 1]]),
                               np.concatenate([ue[:, 1], ue[:, 0]]))),
                              shape=(n, n))
    sa, sb = edges[short, 0], edges[short, 1]
    common = np.asarray(A2[sa, sb]).ravel()
    fcount = np.asarray(FC[sa, sb]).ravel()
    short = short[common == fcount]
    if not len(short):
        return faces

    # Collapse short edges, shortest first, as an independent set: skip any edge
    # that touches a vertex already involved in a collapse this round (keeps the
    # operation conflict-free; remaining edges collapse in later iterations).
    order = short[np.argsort(lengths[short])]
    locked = np.zeros(cm.vertices.shape[0], dtype=bool)
    for e in order:
        a, b = edges[e]
        if locked[a] or locked[b]:
            continue
        locked[a] = locked[b] = True
        ga, gb = used[a], used[b]
        # Keep the lower global index as the survivor; merge the other into it
        # and move the survivor to the midpoint (centering).
        if gb < ga:
            ga, gb = gb, ga
        parent[gb] = ga
        pos[ga] = (pos[ga] + pos[gb]) / 2.0

    # Resolve the union-find and rewrite faces, dropping degenerate ones.
    roots = _resolve(parent)
    faces = roots[faces]
    nondegen = ((faces[:, 0] != faces[:, 1])
                & (faces[:, 1] != faces[:, 2])
                & (faces[:, 0] != faces[:, 2]))
    return faces[nondegen]
