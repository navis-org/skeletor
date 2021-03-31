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

try:
    import fastremap
except ImportError:
    fastremap = None
except BaseException:
    raise

import networkx as nx
import numpy as np
import scipy.sparse
import scipy.spatial

from tqdm.auto import tqdm

from ..utilities import make_trimesh
from .utils import edges_to_graph, make_swc


def by_teasar(mesh, inv_dist, smooth_verts=False, progress=True):
    """Skeletonize using mesh TEASAR.

    Based on the implementation by Sven Dorkenwald, Casey Schneider-Mizell and
    Forrest Collman in `meshparty`.

    Notes
    -----


    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to be skeletonize. Can an object that has
                    ``.vertices`` and ``.faces`` properties  (e.g. a
                    trimesh.Trimesh) or a tuple ``(vertices, faces)`` or a
                    dictionary ``{'vertices': vertices, 'faces': faces}``.
    inv_dist :      int | float
                    Distance along the mesh used for invalidation of vertices.
                    This controls how detailed (or noisy) the skeleton will be.
    smooth_verts :  bool
                    Whether to smooth skeleton vertices.
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

    """
    pass


def paths_on_mesh(mesh, inv_d):
    """Calculate all paths along the mesh."""

    # Turn mesh into graph
    mesh = make_trimesh(mesh, validate=False)

    # Produce weighted edges
    edges = np.concatenate((mesh.edges_unique,
                            mesh.edges_unique_length.reshape(mesh.edges_unique.shape[0], 1)),
                           axis=1)

    # Generate Graph (must be undirected)
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    # Go over all connected components
    for cc in nx.connected_components(G):
        # Get a subgraph
        SG = nx.subgraph(G, cc)

        # Find a root for this subgraph using a far point heuristic
        root = find_far_points(SG)[0]

        # Setup
        valid = np.ones(len(mesh.vertices), np.bool)
        mesh_to_skeleton_map = np.full(len(mesh.vertices), np.nan)

        total_to_visit = np.sum(valid)

def find_far_points(G):
    """Find the most distal points in the graph."""
    # Use the first node in graph as seed
    source = target = list(G.nodes)[0]
    dist = 0
    while True:
        # Turn into a depth-first
        tree = nx.bfs_tree(G, source=target)
        lp = nx.dag_longest_path(tree, weight=None)
        if len(lp) <= dist:
            break
        source, target = lp[0], lp[-1]
        dist = len(lp)

    return source, target
