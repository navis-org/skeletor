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

import numpy as np
import trimesh as tm
import scipy.sparse
import scipy.spatial

from ..utilities import make_trimesh


def by_tangent_ball(mesh, step_size='auto', iter_lim=np.inf, output='swc', progress=True):
    """Skeletonize a contracted mesh by finding the maximal tangent ball.

    Notes
    -----
    Based on the algorithm presented in Ma et al. (2012) (see references).

    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to be skeletonize. Can an object that has
                    ``.vertices`` and ``.faces`` properties  (e.g. a
                    trimesh.Trimesh) or a tuple ``(vertices, faces)`` or a
                    dictionary ``{'vertices': vertices, 'faces': faces}``.
    step_size :     'auto'
    output :        "swc" | "graph" | "both"
                    Determines the function's output. See ``Returns``.
    progress :      bool
                    If True, will show progress bar.

    Returns
    -------
    "swc" :         pandas.DataFrame
                    SWC representation of the skeleton.
    "graph" :       networkx.Graph
                    Graph representation of the skeleton.
    "both" :        tuple
                    Both of the above: ``(swc, graph)``.

    References
    ----------
    Ma, J., Bae, S.W. & Choi, S. 3D medial axis point approximation using
    nearest neighbors and the normal field. Vis Comput 28, 7–19 (2012).
    https://doi.org/10.1007/s00371-011-0594-7

    """
    assert output in ['swc', 'graph', 'both']

    mesh = make_trimesh(mesh, validate=False)

    # Generate the KD tree
    tree = scipy.spatial.cKDTree(mesh.vertices)

    init_radius = 3000 #max(mesh.bounds[1] - mesh.bounds[0])

    n_verts = mesh.vertices.shape[0]
    centers = np.zeros(mesh.vertices.shape)
    for i in trange(n_verts, disable=progress is False):
        # Get this vertex
        p = mesh.vertices[i]
        # Get this vertex' normals
        L = mesh.vertex_normals[i]
        # Radius
        b = init_radius
        # Define the initial ball center
        c = p - L * b

        # Now refine
        prev_ix = -1
        while True:
            # Get closest neighbor
            dist, ix = tree.query(c.reshape(1, -1), k=1)

            # If the closest neighbor is farther away than the current radius,
            # the ball is maximal and we can stop
            if dist[0] >= b or ix == i or ix == prev_ix:
                break

            # This is to prevent endless loops
            prev_ix = ix

            # If not, find a new radius that has both p and pi on its surface:
            # i.e. find a new center + radius along the vertex' normal that
            # produces a ball which is tangential to both p and pi

            # This is the new vertex
            pi = mesh.vertices[ix[0]]
            # This is the vector between p and pi
            vec_ppi = p - pi
            # Length of the vector
            d_ppi = np.linalg.norm(vec_ppi)
            # The unit vector
            # u_ppi = vec_ppi / d_ppi
            # Cosine of angle between the normal and this vector
            cos = np.dot(L, vec_ppi) / d_ppi

            # The new ball center
            bi = d_ppi / (2 * cos)
            c = p - L * bi

            # Track the new radius
            b = bi

        # Save new center
        centers[i, :] = c

    """
    # To converge quickly on a solution it is important that the initial
    # radius is already approximately correct
    # Lets use the average distance between vertices
    step_size = 10
    #radius = step_size
    radius = 1500

    # Generate the initial medial ball points from the normals
    n_verts = mesh.vertices.shape[0]
    centers = mesh.vertices - mesh.vertex_normals * radius
    not_max = np.repeat(True, n_verts)

    bar_format = ("{l_bar}{bar}| [{elapsed}<{remaining}, "
                  "{postfix[0]}/{postfix[1]}it, "
                  "{rate_fmt}, r {postfix[2]:.2g}")
    #colors = sns.color_palette('muted', iter_lim)
    with tqdm(total=n_verts,
              bar_format=bar_format,
              disable=progress is False,
              postfix=[1, iter_lim, 1]) as pbar:
        it = 0
        while it < iter_lim and np.any(not_max):
            #v.add(centers[200:201], scatter_kws=dict(color=colors[it]))
            # For each not-yet-maxed ball, check how many neighbors are within
            # its volume
            d, i = tree.query(centers[not_max],
                              k=3,  # we don't care about anything past third neighbors
                              distance_upper_bound=radius
                              )
            # Check if any hits below the current radius
            #hit = np.sum(d <= radius, axis=1) >= 2
            hit = np.sum(d <= radius, axis=1) <= 2

            # Update progress bar
            if progress:
                pbar.postfix[0] = it
                pbar.postfix[2] = radius
                pbar.update(hit.sum())

            not_max_hit = np.where(not_max)[0][hit]
            not_max[not_max_hit] = False

            #radius += step_size
            radius -= step_size
            centers[not_max] = mesh.vertices[not_max] - mesh.vertex_normals[not_max] * radius

            it += 1
    """
    #return centers#, radius
    return tm.Trimesh(centers, mesh.faces)
