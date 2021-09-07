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

import igraph as ig
import networkx as nx
import numpy as np

from scipy.sparse.csgraph import dijkstra
from tqdm.auto import tqdm

from ..utilities import make_trimesh
from .base import Skeleton
from .utils import make_swc, edges_to_graph

try:
    from fastremap import unique
except ImportError:
    from numpy import unique
except BaseException:
    raise

__all__ = ['by_teasar']


def by_teasar(mesh, inv_dist, min_length=None, root=None, progress=True):
    """Skeletonize a mesh mesh using the TEASAR algorithm [1].

    This algorithm finds the longest path from a root vertex, invalidates all
    vertices that are within `inv_dist`. Then picks the second longest (and
    still valid) path and does the same. Rinse & repeat until all vertices have
    been invalidated. It's fast + works very well with tubular meshes, and with
    `inv_dist` you have control over the level of detail. Note that by its
    nature the skeleton will be exactly on the surface of the mesh.

    Based on the implementation by Sven Dorkenwald, Casey Schneider-Mizell and
    Forrest Collman in `meshparty` (https://github.com/sdorkenw/MeshParty).

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
    min_length :    float, optional
                    If provided, will skip any branch that is shorter than
                    `min_length`. Use this to get rid of noise but note that
                    it will lead to vertices not being mapped to skeleton nodes.
                    Such vertices will show up with index -1 in
                    `Skeleton.mesh_map`.
    root :          int, optional
                    Vertex ID of a root. If not provided will use ``0``.
    progress :      bool, optional
                    If True, will show progress bar.

    Returns
    -------
    skeletor.Skeleton
                    Holds results of the skeletonization and enables quick
                    visualization.

    References
    ----------
    [1] Sato, M., Bitter, I., Bender, M. A., Kaufman, A. E., & Nakajima, M.
        (n.d.). TEASAR: tree-structure extraction algorithm for accurate and
        robust skeletons. In Proceedings the Eighth Pacific Conference on
        Computer Graphics and Applications. IEEE Comput. Soc.
        https://doi.org/10.1109/pccga.2000.883951

    """
    mesh = make_trimesh(mesh, validate=False)

    # Generate Graph (must be undirected)
    G = ig.Graph(edges=mesh.edges_unique, directed=False)
    G.es['weight'] = mesh.edges_unique_length

    if not root:
        root = 0

    edges = np.array([], dtype=np.int64)
    mesh_map = np.full(mesh.vertices.shape[0], fill_value=-1)

    with tqdm(desc='Invalidating', total=len(G.vs),
              disable=not progress, leave=False) as pbar:
        for cc in sorted(G.clusters(), key=len, reverse=True):
            # Make a subgraph for this connected component
            SG = G.subgraph(cc)
            cc = np.array(cc)

            # Find root within subgraph
            if root in cc:
                this_root = np.where(cc == root)[0][0]
            else:
                this_root = 0

            # Get the sparse adjacency matrix of the subgraph
            sp = SG.get_adjacency_sparse('weight')

            # Get lengths of paths to all nodes from root
            paths = SG.shortest_paths(this_root, target=None, weights='weight',
                                      mode='ALL')[0]
            paths = np.array(paths)

            # Prep array for invalidation
            valid = ~np.zeros(paths.shape).astype(bool)
            invalidated = 0

            while np.any(valid):
                # Find the farthest point
                farthest = np.argmax(paths)

                # Get path from root to farthest point
                path = SG.get_shortest_paths(this_root, farthest,
                                             weights='weight', mode='ALL')[0]

                # Get IDs of edges along the path
                eids = SG.get_eids(path=path, directed=False)

                # Stop if farthest point is closer than min_length
                add = True
                if min_length:
                    # This should only be distance to the first branchpoint
                    # from the tip since we set other weights to zero
                    le = sum(SG.es[eids].get_attribute_values('weight'))
                    if le < min_length:
                        add = False

                if add:
                    # Add these new edges
                    new_edges = np.vstack((cc[path[:-1]], cc[path[1:]])).T
                    edges = np.append(edges, new_edges).reshape(-1, 2)

                # Invalidate points in the path
                valid[path] = False
                paths[path] = 0

                # Must set weights along path to 0 so that this path is
                # taken again in future iterations
                SG.es[eids]['weight'] = 0

                # Get all nodes within `inv_dist` to this path
                # Note: can we somehow only include still valid nodes to speed
                # things up?
                dist, _, sources = dijkstra(sp, directed=False, indices=path,
                                            limit=inv_dist, min_only=True,
                                            return_predecessors=True)

                # Invalidate
                in_dist = dist <= inv_dist
                to_invalidate = np.where(in_dist)[0]
                valid[to_invalidate] = False
                paths[to_invalidate] = 0

                # Update mesh vertex to skeleton node map
                mesh_map[cc[in_dist]] = cc[sources[in_dist]]

                pbar.update((~valid).sum() - invalidated)
                invalidated = (~valid).sum()

    # Make unique edges (paths will have overlapped!)
    edges = unique(edges, axis=0)

    # Create a directed acyclic and hierarchical graph
    G_nx = edges_to_graph(edges=edges[:, [1, 0]],
                          fix_tree=True, fix_edges=False,
                          weight=False)

    # Generate the SWC table
    swc, new_ids = make_swc(G_nx, coords=mesh.vertices, reindex=True)

    # Update vertex to node ID map
    mesh_map = np.array([new_ids.get(n, -1) for n in mesh_map])

    return Skeleton(swc=swc, mesh=mesh, mesh_map=mesh_map, method='teasar')


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
