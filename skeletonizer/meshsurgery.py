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


import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

def face_collapse_sphere(mesh, radius):
    """ Collapse faces by radius.
    """


def face_collapse2(mesh):
    """ Collapses faces by .
    """

    # Turn mesh into graph
    g = nx.Graph()

    # Add edges and set the eucledian distances between vertices as weights
    g.add_edges_from(mesh.edges_unique)
    nx.set_edge_attributes(g, {tuple(e): w for e, w in zip(mesh.edges_unique,
                                                           mesh.edges_unique_length)},
                           name='weight')

    # Turn into hiearchical tree
    tree = nx.algorithms.tree.maximum_branching(g)

    # Trimesh does not support edge only, so we will return a sort of SWC file
    swc = pd.DataFrame(np.array(tree.edges))
    coords = mesh.vertices[np.array(tree.edges)[:, 0]]
    swc['x'], swc['y'], swc['z'] = coords[:, 0], coords[:, 2], coords[:, 2]
    swc.columns = ['node_id', 'parent_id', 'x', 'y', 'z']


def face_collapse(mesh):
    """ Collapses faces by .

    1. Sort edges by length, short ones first
    2. Iterate over all sorted edges:
        - check if edge is part of a face (triangle): nodes connected by the
          edge have at least one two hop path between them
        - collapse nodes
    """

    # Turn mesh into graph
    g = nx.Graph()

    # Add edges and set the eucledian distances between vertices as weights
    g.add_edges_from(mesh.edges_unique)
    nx.set_edge_attributes(g, {tuple(e): w for e, w in zip(mesh.edges_unique,
                                                           mesh.edges_unique_length)},
                           name='weight')

    # Sort edges by length
    edge_order = np.argsort(mesh.edges_unique_length)
    edges = mesh.edges_unique[edge_order]

    for e in tqdm(edges):
        # Check if edge still exists
        if e not in g.edges:
            continue

        # Check if edge is part of a face:
        # Count paths
        paths = nx.all_simple_paths(g, e[0], e[1], cutoff=2)
        for i, p in enumerate(paths):
            # We only need to know that there are more than two paths
            if i >= 1:
                break

        # If less than two paths, don't remove this edge
        if i < 1:
            continue

        # Collapse
        g = nx.contracted_nodes(g, e[0], e[1], self_loops=False)


    # Trimesh does not support edge only, so we will return a sort of SWC file
    swc = pd.DataFrame(np.array(g.edges))
    coords = mesh.vertices[np.array(g.edges)[:, 0]]
    swc['x'], swc['y'], swc['z'] = coords[:, 0], coords[:, 2], coords[:, 2]
    swc.columns = ['node_id', 'parent_id', 'x', 'y', 'z']

    return g, swc

