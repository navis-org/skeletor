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

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.spatial

from tqdm.auto import tqdm

from .. import _fastcore
from ..utilities import make_trimesh, get_edges_unique

from .base import Skeleton
from .utils import edges_to_swc, split_by_label

__all__ = ['by_vertex_clusters']


def _geodesic_clusters_scipy(edges, weights, n_nodes, max_dist, progress=True):
    """Fallback for ``navis_fastcore.geodesic_clusters``.

    Produces bit-identical labels, just one bounded Dijkstra per cluster
    instead of a single sweep - so expect it to be ~100x slower.
    """
    adj = scipy.sparse.coo_matrix((weights, (edges[:, 0], edges[:, 1])),
                                  shape=(n_nodes, n_nodes)).tocsr()

    labels = np.full(n_nodes, -1, dtype=np.int64)
    n_clusters = 0
    with tqdm(desc='Clustering', total=n_nodes, disable=progress is False) as pbar:
        for seed in range(n_nodes):
            if labels[seed] >= 0:
                continue
            # Note the ball is grown through vertices earlier clusters already
            # claimed, so each cluster stays a true ball around its seed
            dist = scipy.sparse.csgraph.dijkstra(adj, directed=False,
                                                 indices=seed, limit=max_dist)
            take = (dist <= max_dist) & (labels < 0)
            labels[take] = n_clusters
            n_clusters += 1
            pbar.update(int(take.sum()))

    return labels, n_clusters


def by_vertex_clusters(mesh, sampling_dist, cluster_pos='median', progress=True):
    """Skeletonize a (contracted) mesh by clustering vertices.

    The algorithm traverses the mesh graph and groups vertices together that
    are within a given distance to each other. This uses the geodesic
    (along-the-mesh) distance, not simply the Euclidean distance. Subsequently
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
    edges_unique, edges_unique_length = get_edges_unique(mesh, lengths=True)
    n_verts = mesh.vertices.shape[0]

    # Group vertices into spatial clusters: walk the vertices in order, and for
    # each one not yet claimed grow a cluster containing every vertex within
    # `sampling_dist` geodesic distance of it that no earlier cluster took.
    if _fastcore.has('geodesic_clusters'):
        labels, n_clusters = _fastcore.fastcore.geodesic_clusters(
            edges_unique, n_verts, sampling_dist, weights=edges_unique_length)
    else:
        labels, n_clusters = _geodesic_clusters_scipy(
            edges_unique, edges_unique_length, n_verts, sampling_dist, progress)

    # Turn the labels into a list of vertex IDs per cluster
    clusters = split_by_label(labels)

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
    cl_edges = np.array(get_edges_unique(mesh))
    if fastremap:
        mapping = {n: i for i, l in enumerate(clusters) for n in l}
        cl_edges = fastremap.remap(cl_edges, mapping, preserve_missing_labels=False, in_place=True)
    else:
        for i, c in enumerate(clusters):
            cl_edges[np.isin(cl_edges, c)] = i

    # Remove directionality from cluster edges, then dedupe. Packing each pair
    # into one int64 rather than `unique(axis=0)`, which takes a void-dtype view
    # and lexsorts it - 140 ms of this function's 210 ms on a 105k-vertex mesh.
    # `np.unique` returns sorted keys, so unpacking gives the same rows.
    lo = np.minimum(cl_edges[:, 0], cl_edges[:, 1]).astype(np.int64)
    hi = np.maximum(cl_edges[:, 0], cl_edges[:, 1]).astype(np.int64)
    stride = int(cl_edges.max()) + 1
    cl_edges = np.column_stack(np.divmod(np.unique(lo * stride + hi), stride))

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

    # Generate a mesh vertex -> skeleton vertex map
    # Note that nodes are labeled by index of the cluster. This has to be
    # indexed *by vertex ID*: the clusters are discovered in traversal order, so
    # concatenating them does not walk the vertices in order.
    vertex_to_node_map = np.full(n_verts, -1, dtype=int)
    for i, cl in enumerate(clusters):
        vertex_to_node_map[cl] = i

    # Generate SWC. This orients the tree (the MST above only picked which edges
    # survive, not which way they point) and re-indexes so that parents always
    # have lower IDs than their children. `nodes` keeps clusters that no surviving
    # edge names, which would otherwise drop out of the skeleton entirely.
    swc, new_ids = edges_to_swc(edges, cl_coords,
                                nodes=np.unique(cl_edges.flatten()))

    # Update mesh map
    vertex_to_node_map = np.array([new_ids.get(n, -1) for n in vertex_to_node_map])

    return Skeleton(swc=swc, mesh=mesh, mesh_map=vertex_to_node_map,
                    method='vertex_clusters')
