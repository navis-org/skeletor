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

import numpy as np
import pandas as pd
import trimesh as tm

from . import _fastcore


def make_trimesh(mesh, validate=True, **kwargs):
    """Construct ``trimesh.Trimesh`` from input data.

    Parameters
    ----------
    meshdata :      tuple | dict | mesh-like object
                    Tuple: (vertices, faces)
                    dict: {'vertices': [], 'faces': []}
                    mesh-like object: mesh.vertices, mesh.faces
    validate :      bool
                    If True, will try to fix potential issues with the mesh
                    (e.g. infinite values, duplicate vertices, degenerate faces).
    **kwargs
                    Keyword arguments are passed through to
                    `skeletor.pre.fix_mesh` if `validate=True`.

    Returns
    -------
    trimesh.Trimesh

    """
    from .pre import fix_mesh

    if isinstance(mesh, tm.Trimesh):
        pass
    elif isinstance(mesh, (tuple, list)):
        if len(mesh) == 2:
            mesh = tm.Trimesh(vertices=mesh[0],
                              faces=mesh[1],
                              process=validate)
    elif isinstance(mesh, dict):
        mesh = tm.Trimesh(vertices=mesh['vertices'],
                          faces=mesh['faces'],
                          process=validate)
    elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
        mesh = tm.Trimesh(vertices=mesh.vertices,
                          faces=mesh.faces,
                          process=validate)
    else:
        raise TypeError('Unable to construct a trimesh.Trimesh from object of '
                        f'type "{type(mesh)}"')

    if validate:
        mesh = fix_mesh(mesh, inplace=True, **kwargs)

    return mesh


def get_edges_unique(mesh, lengths=False, cache=True):
    """Fast drop-in for ``mesh.edges_unique`` / ``mesh.edges_unique_length``.

    trimesh derives ``edges_unique`` by running ``np.unique`` over the packed
    edge keys, which sorts ~3x as many uint64 values as there are faces. That
    sort dominates the runtime (and peak memory) of the mesh -> graph conversion
    for large meshes. This helper avoids that sort: if ``navis-fastcore`` is
    installed we hand the faces to its (multi-threaded, Rust) ``unique_edges``,
    otherwise we deduplicate with a pandas hashtable (``pd.factorize``) and sort
    only the far smaller set of unique edges - typically 1.3-2x faster than
    trimesh and at or below its memory footprint.

    Either way the result is bit-identical to trimesh: the same edges in the same
    row order, with matching ``edges_unique_inverse`` / ``edges_unique_idx`` (and
    hence ``mesh.faces_unique_edges``). Values already present in the trimesh
    cache are reused as-is; freshly computed values are (by default) written back
    into ``mesh._cache`` so repeated accesses and trimesh-derived properties
    reuse them.

    Parameters
    ----------
    mesh :      trimesh.Trimesh
    lengths :   bool
                If True, additionally return the Euclidean length of each unique
                edge (equivalent to ``mesh.edges_unique_length``).
    cache :     bool
                If True (default), write freshly computed values back into
                ``mesh._cache`` so trimesh reuses them.

    Returns
    -------
    edges_unique :          (N, 2) int array
    edges_unique_length :   (N,) float array
                            Only returned if ``lengths=True``.

    """
    # Reuse whatever trimesh (or a previous call) has already computed. We verify
    # first so a stale cache (e.g. after the mesh was mutated) is dropped, then
    # read the raw dict directly to avoid triggering the slow property compute.
    cache_obj = getattr(mesh, '_cache', None)
    cdict = None
    if cache_obj is not None:
        try:
            cache_obj.verify()
            cdict = cache_obj.cache
        except Exception:
            cdict = None

    eu = cdict.get('edges_unique') if cdict is not None else None
    if eu is not None:
        if not lengths:
            return eu
        el = cdict.get('edges_unique_length')
        if el is None:
            el = _edge_lengths(mesh, eu)
            if cache:
                _cache_update(mesh, {'edges_unique_length': el})
        return eu, el

    # Not cached - compute from scratch.
    faces = np.asarray(mesh.faces)
    if faces.size == 0:
        eu = np.zeros((0, 2), dtype=np.int64)
        return (eu, np.zeros(0, dtype=np.float64)) if lengths else eu

    # Both backends bit-pack vertex indices into 32 bits. Fall back to trimesh
    # for the (practically impossible) >2**32 vertices case.
    if int(faces.max()) >= 2 ** 32:
        eu = np.asarray(mesh.edges_unique)
        return (eu, np.asarray(mesh.edges_unique_length)) if lengths else eu

    if _fastcore.has('unique_edges'):
        eu, el, idx, inverse = _unique_edges_fastcore(mesh, faces, lengths, cache)
    else:
        eu, el, idx, inverse = _unique_edges_numpy(mesh, faces, lengths, cache)

    if cache:
        payload = {'edges_unique': eu,
                   'edges_unique_idx': idx,
                   'edges_unique_inverse': inverse}
        if el is not None:
            payload['edges_unique_length'] = el
        _cache_update(mesh, payload)

    return (eu, el) if lengths else eu


def _unique_edges_fastcore(mesh, faces, lengths, extras):
    """``get_edges_unique`` backend using ``navis_fastcore.unique_edges``.

    Returns ``(edges, lengths, index, inverse)`` where all but ``edges`` are
    ``None`` if they weren't asked for. ``unique_edges`` is documented to match
    trimesh's output order, dtype and first-occurrence semantics exactly, so the
    values are interchangeable with the numpy backend's.
    """
    verts = np.asarray(mesh.vertices) if lengths else None

    out = _fastcore.fastcore.unique_edges(faces,
                                          vertices=verts,
                                          return_index=extras,
                                          return_inverse=extras)

    # Only the bare edge array comes back when nothing extra was requested
    if not extras and verts is None:
        return out, None, None, None

    out = list(out)
    eu = out.pop(0)
    idx = out.pop(0) if extras else None
    inverse = out.pop(0) if extras else None
    el = out.pop(0) if verts is not None else None

    return eu, el, idx, inverse


def _unique_edges_numpy(mesh, faces, lengths, extras):
    """``get_edges_unique`` backend using a pandas hashtable dedup (no sort).

    Same return signature as :func:`_unique_edges_fastcore`.
    """
    # `faces_to_edges`: identical row order to trimesh's ``mesh.edges`` so the
    # inverse mapping we cache lines up with ``mesh.faces_unique_edges``.
    e = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    lo = np.minimum(e[:, 0], e[:, 1])
    hi = np.maximum(e[:, 0], e[:, 1])

    key = (lo.astype(np.uint64) << np.uint64(32)) | hi.astype(np.uint64)

    # Hashtable dedup (no sort over all ~3*n_faces edges). `codes` maps each raw
    # edge to its unique index in order of first appearance.
    codes, uniques = pd.factorize(key, sort=False)
    mn = uniques >> np.uint64(32)
    mx = uniques & np.uint64(0xFFFFFFFF)

    # Reproduce trimesh's row order exactly: it sorts unique edges by the ascending
    # hash of ``edges_sorted``, which (after its bit-packing) is ascending
    # ``(max, min)``. Sorting only the ~n_unique deduplicated edges is far cheaper
    # than np.unique sorting every raw edge, and keeps outputs bit-identical for
    # edge-order-sensitive methods (edge collapse, mean-curvature, clustering, ...).
    order = np.argsort((mx << np.uint64(32)) | mn)
    eu = np.empty((len(uniques), 2), dtype=e.dtype)
    eu[:, 0] = mn[order].astype(e.dtype)
    eu[:, 1] = mx[order].astype(e.dtype)

    el = _edge_lengths(mesh, eu) if lengths else None

    if not extras:
        return eu, el, None, None

    # Remap the inverse into the sorted order, and the first-occurrence index.
    new_pos = np.empty(len(order), dtype=np.int64)
    new_pos[order] = np.arange(len(order))
    inverse = new_pos[codes]
    idx_factorize = np.full(len(uniques), len(codes), dtype=np.int64)
    np.minimum.at(idx_factorize, codes, np.arange(len(codes)))
    idx = idx_factorize[order]

    return eu, el, idx, inverse


def _edge_lengths(mesh, edges):
    """Euclidean length of each edge (matches ``mesh.edges_unique_length``).

    Uses the exact expression from ``trimesh.util.row_norm`` rather than an
    equivalent one (e.g. ``np.einsum``): those agree only to within an ulp,
    which is enough to perturb downstream shortest-path distances and make
    results depend on whether this or trimesh computed the lengths.
    """
    verts = np.asarray(mesh.vertices)
    vec = verts[edges[:, 0]] - verts[edges[:, 1]]
    return np.sqrt(np.dot(vec ** 2, [1] * vec.shape[1]))


def _cache_update(mesh, payload):
    """Best-effort write into the trimesh cache; skipped on any incompatibility.

    Only ever called when ``edges_unique`` was absent from the cache, so
    ``faces_unique_edges`` (which cannot be cached without it) can't already
    exist with a different ordering.
    """
    try:
        mesh._cache.update(payload)
    except Exception:
        pass
