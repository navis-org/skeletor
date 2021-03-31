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


from ..utilities import make_trimesh

from .edge_collapse import by_edge_collapse
from .vertex_cluster import by_vertex_clusters
from .wave import by_wavefront

__all__ = ['skeletonize']


class Skeleton:
    """Class representing a skeleton."""
    def __init__(self, vertices, edges, mesh_map=None, radius=None):
        self.vertices = vertices
        self.edges = edges
        self.mesh_map = mesh_map
        self.radius = radius


def skeletonize(mesh, method, output='swc', progress=True, validate=False,
                drop_disconnected=False, **kwargs):
    """Skeletonize a (contracted) mesh.

    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to be skeletonize. Can an object that has
                    ``.vertices`` and ``.faces`` properties  (e.g. a
                    trimesh.Trimesh) or a tuple ``(vertices, faces)`` or a
                    dictionary ``{'vertices': vertices, 'faces': faces}``.
    method :        "vertex_clusters" | "edge_collapse" | "wavefront"
                    Skeletonization comes in two flavours with different Pros
                    and Cons::

                     - ``vertex_clusters`` groups and collapses vertices based
                       on their geodesic distance along the mesh's surface. It's
                       fast and scales well but can lead to oversimplification.
                       Good for quick & dirty skeletonizations. See
                       ``skeletor.skeletonizers.by_vertex_clusters`` for details.
                     - ``edge_collapse`` implements skeleton extraction by edge
                       collapse described in [1]. It's rather slow and doesn't
                       scale well but is really good at preserving topology.
                       See ``skeletor.skeletonizers.by_edge_collapse`` for
                       details.
    output :        "swc" | "graph" | "both"
                    Determines the function's output. See ``Returns``.
    progress :      bool
                    If True, will show progress bar.
    validate :      bool
                    If True, will try to fix potential issues with the mesh
                    (e.g. infinite values, duplicate vertices, degenerate faces)
                    before skeletonization. Note that this might change your
                    mesh inplace.
    drop_disconnected : bool
                    If True, will drop disconnected nodes from the skeleton.
                    Note that this might result in empty skeletons.

    **kwargs
                    Keyword arguments are passed to the above mentioned
                    functions:

    For method "vertex_clusters":

    sampling_dist : float | int, required
                    Maximal distance at which vertices are clustered. This
                    parameter should be tuned based on the resolution of your
                    mesh.
    cluster_pos :   "median" | "center"
                    How to determine the x/y/z coordinates of the collapsed
                    vertex clusters (i.e. the skeleton's nodes)::

                      - "median": Use the vertex closest to cluster's center of
                        mass.
                      - "center": Use the center of mass. This makes for smoother
                        skeletons but can lead to nodes outside the mesh.

    For method "edge_collapse":

    shape_weight :  float, default = 1
                    Weight for shape costs which penalize collapsing edges that
                    would drastically change the shape of the object.
    sample_weight : float, default = 0.1
                    Weight for sampling costs which penalize collapses that
                    would generate prohibitively long edges.

    For method "wavefront":

    waves :         int, default = 1
                    Number of waves to run across the mesh. Each wave is
                    initialized at a different vertex which produces slightly
                    different rings. The final skeleton is produced from a mean
                    across all waves. More waves produce higher resolution
                    skeletons but also introduce more noise.
    step_size :     float, default = 1
                    Values greater 1 effectively lead to binning of rings. For
                    example a stepsize of 2 means that two adjacent vertex rings
                    will be collapsed to the same center. This can help reduce
                    noise in the skeleton (and as such counteracts a large
                    number of waves).

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
    [1] Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh
        contraction. ACM Transactions on Graphics (TOG). 2008 Aug 1;27(3):44.

    """
    mesh = make_trimesh(mesh, validate=validate)

    assert method in ['vertex_clusters', 'edge_collapse', 'wavefront']
    required_param = {'vertex_clusters': ['sampling_dist'],
                      'edge_collapse': [],
                      'wavefront': []}

    for kw in required_param[method]:
        if kw not in kwargs:
            raise ValueError(f'Method "{method}" requires parameter "{kw}"'
                             ' - see help(skeletor.skeletonize)')

    if method == 'vertex_clusters':
        return by_vertex_clusters(mesh, output=output, progress=progress,
                                  drop_disconnected=drop_disconnected, **kwargs)

    if method == 'edge_collapse':
        return by_edge_collapse(mesh, output=output, progress=progress,
                                drop_disconnected=drop_disconnected, **kwargs)

    if method == 'wavefront':
        return by_wavefront(mesh, output=output, progress=progress,
                            drop_disconnected=drop_disconnected, **kwargs)
