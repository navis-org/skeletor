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

import math
import ncollpyde
import numbers
import random

import numpy as np
import scipy.spatial

from ..utilities import make_trimesh


def radii(s, mesh=None, method='knn', aggregate='mean', validate=False, **kwargs):
    """Extract radii for given skeleton table.

    Important
    ---------
    This function really only produces useful radii if the skeleton is centered
    inside the mesh. `by_wavefront` does that by default whereas all other
    skeletonization methods don't. Your best bet to get centered skeletons is
    to contract the mesh first (`sk.pre.contract`).

    Parameters
    ----------
    s :         skeletor.Skeleton
                Skeleton to clean up.
    mesh :      trimesh.Trimesh, optional
                Original mesh (e.g. before contraction). If not provided will
                use the mesh associated with ``s``.
    method :    "knn" | "ray"
                Whether and how to add radius information to each node::

                    - "knn" uses k-nearest-neighbors to get radii: fast but
                      potential for being very wrong
                    - "ray" uses ray-casting to get radii: slower but sometimes
                      less wrong

    aggregate : "mean" | "median" | "max" | "min" | "percentile75"
                Function used to aggregate radii over sample (i.e. across
                k nearest-neighbors or ray intersections)
    validate :  bool
                If True, will try to fix potential issues with the mesh
                (e.g. infinite values, duplicate vertices, degenerate faces)
                before skeletonization. Note that this might make changes to
                your mesh inplace!
    **kwargs
                Keyword arguments are passed to the respective method:

                For method "knn"::

                    n :             int (default 5)
                                    Radius will be the mean over n nearest-neighbors.

                For method "ray"::

                    n_rays :        int (default 20)
                                    Number of rays to cast for each node.
                    projection :    "sphere" (default) | "tangents"
                                    Whether to cast rays in a sphere around each node or in a
                                    circle orthogonally to the node's tangent vector.
                    fallback :      "knn" (default) | None | number
                                    If a point is outside or right on the surface of the mesh
                                    the raycasting will return nonesense results. We can either
                                    ignore those cases (``None``), assign a arbitrary number or
                                    we can fall back to radii from k-nearest-neighbors (``knn``).

    Returns
    -------
    None
                    But attaches `radius` to the skeleton's SWC table. Existing
                    values are replaced!

    """
    if isinstance(mesh, type(None)):
        mesh = s.mesh

    mesh = make_trimesh(mesh, validate=True)

    if method == 'knn':
        radius = get_radius_knn(s.swc[['x', 'y', 'z']].values,
                                aggregate=aggregate,
                                mesh=mesh, **kwargs)
    elif method == 'ray':
        radius = get_radius_ray(s.swc,
                                mesh=mesh,
                                aggregate=aggregate,
                                **kwargs)
    else:
        raise ValueError(f'Unknown method "{method}"')

    s.swc['radius'] = radius

    return


def get_radius_knn(coords, mesh, n=5, aggregate='mean'):
    """Extract radii using k-nearest-neighbors.

    Parameters
    ----------
    coords :    numpy.ndarray
    mesh :      trimesh.Trimesh
    n :         int
                Radius will be the mean over n nearest-neighbors.
    aggregate : "mean" | "median" | "max" | "min" | "percentile75"
                Function used to aggregate radii for `n` nearest neighbors.

    Returns
    -------
    radii :     np.ndarray
                Corresponds to input coords.

    """
    agg_map = {'mean': np.mean, 'max': np.max, 'min': np.min,
               'median': np.median, 'percentile75': lambda x: np.percentile(x, 75)}
    assert aggregate in agg_map
    agg_func = agg_map[aggregate]

    # Generate kdTree
    tree = scipy.spatial.cKDTree(mesh.vertices)

    # Query for coordinates
    dist, ix = tree.query(coords, k=5)

    # Aggregate
    return agg_func(dist, axis=1)


def get_radius_ray(swc, mesh, n_rays=20, aggregate='mean', projection='sphere',
                   fallback='knn'):
    """Extract radii using ray casting.

    Parameters
    ----------
    swc :           pandas.DataFrame
                    SWC table
    mesh :          trimesh.Trimesh
    n_rays :        int
                    Number of rays to cast for each node.
    aggregate :     "mean" | "median" | "max" | "min" | "percentile75"
                    Function used to aggregate radii for over all intersections
                    for a given node.
    projection :    "sphere" | "tangents"
                    Whether to cast rays in a sphere around each node or in a
                    circle orthogonally to the node's tangent vector.
    fallback :      "knn" | None | number
                    If a point is outside or right on the surface of the mesh
                    the raycasting will return nonesense results. We can either
                    ignore those cases (``None``), assign a arbitrary number or
                    we can fall back to radii from k-nearest-neighbors (``knn``).

    Returns
    -------
    radii :     np.ndarray
                Corresponds to input coords.

    """
    agg_map = {'mean': np.mean, 'max': np.max, 'min': np.min,
               'median': np.median, 'percentile75': lambda x: np.percentile(x, 75)}
    assert aggregate in agg_map
    agg_func = agg_map[aggregate]

    assert projection in ['sphere', 'tangents']
    assert (fallback == 'knn') or isinstance(fallback, numbers.Number) or isinstance(fallback, type(None))

    # Get max dimension of mesh
    dim = (swc[['x', 'y', 'z']].max() - swc[['x', 'y', 'z']].min()).values
    radius = max(dim)

    # Vertices for each point on the circle
    points = swc[['x', 'y', 'z']].values
    sources = np.repeat(points, n_rays, axis=0)

    if projection == 'sphere':
        # Repeat points n_rays times
        sources = np.repeat(points, n_rays, axis=0)

        # Get (random) points on a sphere and scale by radius
        targets = fibonacci_sphere(n_rays, randomize=True) * radius
        # Reshape to match sources
        targets = np.tile(targets,
                          (points.shape[0], 1))
        # Offset onto sources
        targets += sources
    else:
        tangents, normals, binormals = frenet_frames(swc)

        v = np.arange(n_rays,
                      dtype=np.float) / n_rays * 2 * np.pi

        all_cx = (radius * -1. * np.tile(np.cos(v), points.shape[0]).reshape((n_rays, points.shape[0]), order='F')).T
        cx_norm = (all_cx[:, :, np.newaxis] * normals[:, np.newaxis, :]).reshape(sources.shape)

        all_cy = (radius * np.tile(np.sin(v), points.shape[0]).reshape((n_rays, points.shape[0]), order='F')).T
        cy_norm = (all_cy[:, :, np.newaxis] * binormals[:, np.newaxis, :]).reshape(sources.shape)

        targets = sources + cx_norm + cy_norm

    # Initialize ncollpyde Volume
    coll = ncollpyde.Volume(mesh.vertices, mesh.faces, validate=False)

    # Get intersections: `ix` points to index of line segment; `loc` is the
    #  x/y/z coordinate of the intersection and `is_backface` is True if
    # intersection happened at the inside of a mesh
    ix, loc, is_backface = coll.intersections(sources, targets)

    # Remove intersections with front faces
    # For some reason this reduces the number of intersections to 0 for many
    # points
    #ix = ix[~is_backface]
    #loc = loc[~is_backface]

    # Calculate intersection distances
    dist = np.sqrt(np.sum((sources[ix] - loc)**2, axis=1))

    # Map from `ix` back to index of original point
    org_ix = (ix / n_rays).astype(int)

    # Split by original index
    split_ix = np.where(org_ix[:-1] - org_ix[1:])[0]
    split = np.split(dist, split_ix)

    # Aggregate over each original ix
    final_dist = np.zeros(points.shape[0])
    for l, i in zip(split, np.unique(org_ix)):
        final_dist[i] = agg_func(l)

    if not isinstance(fallback, type(None)):
        # See if any needs fixing
        inside = coll.contains(points)
        is_zero = final_dist == 0
        needs_fix = ~inside | is_zero

        if any(needs_fix):
            if isinstance(fallback, numbers.Number):
                final_dist[needs_fix] = fallback
            elif fallback == 'knn':
                final_dist[needs_fix] = get_radius_knn(points[needs_fix], mesh, aggregate=aggregate)

    return final_dist


def frenet_frames(swc):
    """Calculate tangents, normals and binormals for each parent->child segment."""
    # Get node locations
    points = swc[['x', 'y', 'z']].values

    # Get the parent of each node
    parents = swc.set_index('node_id').parent_id.to_dict()

    # For roots just use their first child
    roots = swc[swc.parent_id < 0].node_id.values
    root_childs = swc[swc.parent_id.isin(roots)].set_index('parent_id').node_id.to_dict()
    parents.update(root_childs)

    # Get the second point for each node
    parent_co = swc.set_index('node_id')[['x', 'y', 'z']].loc[swc.node_id.map(parents)].values

    # Produce the tangents
    tangents = (points - parent_co)

    normals = np.zeros((len(points), 3))

    epsilon = 0.0001

    mags = np.sqrt(np.sum(tangents * tangents, axis=1))
    tangents /= mags[:, np.newaxis]

    # Get initial normal and binormal
    t = np.abs(tangents[0])

    smallest = np.argmin(t)
    normal = np.zeros(3)
    normal[smallest] = 1.

    vec = np.cross(tangents[0], normal)
    normals[0] = np.cross(tangents[0], vec)

    all_vec = np.cross(tangents[:-1], tangents[1:])
    all_vec_norm = np.linalg.norm(all_vec, axis=1)

    # Normalise vectors if necessary
    where = all_vec_norm > epsilon
    all_vec[where, :] /= all_vec_norm[where].reshape((sum(where), 1))

    # Precompute inner dot product
    dp = np.sum(tangents[:-1] * tangents[1:], axis=1)
    # Clip
    cl = np.clip(dp, -1, 1)
    # Get theta
    th = np.arccos(cl)

    # Compute normal and binormal vectors along the path
    for i in range(1, len(points)):
        normals[i] = normals[i-1]

        vec_norm = all_vec_norm[i-1]
        vec = all_vec[i-1]
        if vec_norm > epsilon:
            normals[i] = rotate(-np.degrees(th[i-1]),
                                vec)[:3, :3].dot(normals[i])

    binormals = np.cross(tangents, normals)

    return tangents, normals, binormals


def rotate(angle, axis):
    """Construct 3x3 rotation matrix for rotation about a vector.

    Parameters
    ----------
    angle : float
            The angle of rotation, in degrees.
    axis :  ndarray
            The x, y, z coordinates of the axis direction vector.
    Returns
    -------
    M :     ndarray
            Transformation matrix describing the rotation.

    """
    angle = np.radians(angle)
    assert len(axis) == 3
    x, y, z = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    M = np.array([[cx * x + c, cy * x - z * s, cz * x + y * s, .0],
                  [cx * y + z * s, cy * y + c, cz * y - x * s, 0.],
                  [cx * z - y * s, cy * z + x * s, cz * z + c, 0.],
                  [0., 0., 0., 1.]]).T
    return M


def fibonacci_sphere(samples: int = 1,
                     randomize: bool = True) -> list:
    """Generate (random) points on a sphere."""
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2. / samples
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    return np.array(points)
