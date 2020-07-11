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


import numpy as np
import pandas as pd
import trimesh as tm
import scipy.sparse as sparse
import scipy.spatial

from tqdm.auto import tqdm


def calculate_radii(swc, mesh, method='knn', **kwargs):
    """Add/update radius to SWC table.

    Parameters
    ----------
    swc :       pandas.DataFrame
                SWC table.
    mesh :      trimesh.Trimesh
                Mesh to use for radius generation.
    method :    "knn" | "ray" (TODO)
                Whether and how to add radius information to each node::

                    - "knn" uses k-nearest-neighbors to get radii: fast but potential for being very wrong
                    - "ray" uses ray-casting to get radii: slow but "more right"
    **kwargs
                Keyword arguments are passed to the respective method.

    Returns
    -------
    Nothing
                Just adds/updates 'radius' column.

    """
    if method == 'knn':
        swc['radius'] = _get_radius_kkn(swc[['x', 'y', 'z']].values,
                                        mesh=mesh, **kwargs)
    else:
        raise ValueError(f'Unknown method "{method}"')


def _get_radius_kkn(coords, mesh, n=5):
    """Produce radii using k-nearest-neighbors.

    Parameters
    ----------
    coords :    numpy.ndarray
    mesh :      trimesh.Trimesh
    n :         int
                Radius will be the mean over n nearest-neighbors.
    """

    # Generate kdTree
    tree = scipy.spatial.cKDTree(mesh.vertices)

    # Query for coordinates
    dist, ix = tree.query(coords, k=5)

    # We will use the mean but note that outliers might really mess this up
    return np.mean(dist, axis=1)
