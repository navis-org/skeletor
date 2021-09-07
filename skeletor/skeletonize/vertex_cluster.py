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

from .base import Skeleton
from .utils import edges_to_graph, make_swc, dfs

__all__ = ['by_vertex_clusters']


def by_vertex_clusters(mesh, sampling_dist, cluster_pos='median', progress=True):
    """Skeletonize a (contracted) mesh by clustering vertices.

    The algorithm traverses the mesh graph and groups vertices together that
    are within a given distance to each other. This uses the geodesic
    (along-the-mesh) distance, not simply the Eucledian distance. Subsequently
    these groups of vertices are collapsed and re-connected respecting the
    topology of the input mesh.

    The graph traversal is fast and scales well, so this method is well suited
    for meshes with lots of vertices. On the downside: this implementation is
    not very clever and you might have to play around with the parameters
    (mostly ``sampling_dist``) to get decent results.

    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to be skeletonize. Can an object that has
                    ``.vertices`` and ``.faces`` properties  (e.g. a
                    trimesh.Trimesh) or a tuple ``(vertices, faces)`` or a
                    dictionary ``{'vertices': vertices, 'faces': faces}``.
    sampling_dist : float | int
                    Maximal distance at which vertices are clustered. This
                    parameter should be tuned based on the resolution of your
                    mesh (see Examples).
    cluster_pos :   "median" | "center"
                    How to determine the x/y/z coordinates of the collapsed
                    vertex clusters (i.e. the skeleton's nodes)::

                      - "median": Use the vertex closest to cluster's center of
                        mass.
                      - "center": Use the center of mass. This makes for smoother
                        skeletons but can lead to nodes outside the mesh.
    progress :      bool
                    If True, will show progress bar.

    Examples
    --------
    >>> import skeletor as sk
    >>> mesh = sk.example_mesh()
    >>> cont = sk.pre.contract(mesh, epsilon=0.1)
    >>> skel = sk.skeletonize.vertex_cluster(cont)
    >>> skel.mesh = mesh

    Returns
    -------
    skeletor.Skeleton
                    Holds results of the skeletonization and enables quick
                    visualization.

    """
    assert cluster_pos in ['center', 'median']

    mesh = make_trimesh(mesh, validate=False)

    # Produce weighted edges
    edges = np.concatenate((mesh.edges_unique,
                            mesh.edges_unique_length.reshape(mesh.edges_unique.shape[0], 1)),
                           axis=1)

    # Generate Graph (must be undirected)
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    # Run the graph traversal that groups vertices into spatial clusters
    not_visited = set(G.nodes)
    seen = set()
    clusters = []
    to_visit = len(not_visited)
    with tqdm(desc='Clustering', total=len(not_visited), disable=progress is False) as pbar:
        while not_visited:
            # Pick a random node
            start = not_visited.pop()
            # Get all nodes in the geodesic vicinity
            cl, seen = dfs(G, n=start, dist_traveled=0,
                           max_dist=sampling_dist, seen=seen)
            cl = set(cl)

            # Append this cluster and track visited/not-visited nodes
            clusters.append(cl)
            not_visited = not_visited - cl

            # Update  progress bar
            pbar.update(to_visit - len(not_visited))
            to_visit = len(not_visited)

    # `clusters` is a list of sets -> let's turn it into list of arrays
    clusters = [np.array(list(c)).astype(int) for c in clusters]

    # Get positions of clusters
    if cluster_pos == 'center':
        # Get the center of each cluster
        cl_coords = np.array([np.mean(mesh.vertices[c], axis=0) for c in clusters])
    elif cluster_pos == 'median':
        # Get the node that's closest to to the clusters center
        cl_coords = []
        for c in clusters:
            cnt = np.mean(mesh.vertices[c], axis=0)
            cnt_dist = np.sum(np.fabs(mesh.vertices[c] - cnt), axis=1)
            median = mesh.vertices[c][np.argmin(cnt_dist)]
            cl_coords.append(median)
        cl_coords = np.array(cl_coords)

    # Generate edges
    cl_edges = np.array(mesh.edges_unique)
    if fastremap:
        mapping = {n: i for i, l in enumerate(clusters) for n in l}
        cl_edges = fastremap.remap(cl_edges, mapping, preserve_missing_labels=False, in_place=True)
    else:
        for i, c in enumerate(clusters):
            cl_edges[np.isin(cl_edges, c)] = i

    # Remove directionality from cluster edges
    cl_edges = np.sort(cl_edges, axis=1)

    # Get unique edges
    cl_edges = np.unique(cl_edges, axis=0)

    # Calculate edge lengths
    co1 = cl_coords[cl_edges[:, 0]]
    co2 = cl_coords[cl_edges[:, 1]]
    cl_edge_lengths = np.sqrt(np.sum((co1 - co2)**2, axis=1))

    # Produce adjacency matrix from edges and edge lengths
    n_clusters = len(clusters)
    adj = scipy.sparse.coo_matrix((cl_edge_lengths,
                                   (cl_edges[:, 0], cl_edges[:, 1])),
                                  shape=(n_clusters, n_clusters))

    # The cluster graph likely still contain cycles, let's get rid of them using
    # a minimum spanning tree
    mst = scipy.sparse.csgraph.minimum_spanning_tree(adj,
                                                     overwrite=True)

    # Turn into COO matrix
    coo = mst.tocoo()

    # Extract edge list
    edges = np.array([coo.row, coo.col]).T

    # Produce final graph - this also takes care of some fixing
    G = edges_to_graph(edges, nodes=np.unique(cl_edges.flatten()),
                       drop_disconnected=False, fix_tree=True)

    # Generate a mesh vertex -> skeleton vertex map
    # Note that nodes are labeled by index of the cluster
    vertex_to_node_map = [i for i, cl in enumerate(clusters) for n in cl]

    # Generate SWC
    swc, new_ids = make_swc(G, cl_coords, reindex=True, validate=False)

    # Update mesh map
    vertex_to_node_map = np.array([new_ids[n] for n in vertex_to_node_map])

    return Skeleton(swc=swc, mesh=mesh, mesh_map=vertex_to_node_map,
                    method='vertex_clusters')
