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
    from fastremap import unique
except ImportError:
    fastremap = None
    from numpy import unique
except BaseException:
    raise

import networkx as nx
import numpy as np
import pandas as pd
import trimesh as tm
import scipy.sparse
import scipy.spatial


def mst_over_mesh(mesh, verts, limit='auto'):
    """Generate minimum spanning tree by subsetting mesh to given vertices.

    Will (re-)connect vertices based on geodesic distance in original mesh
    using a minimum spanning tree.

    Parameters
    ----------
    mesh :      trimesh.Trimesh
                Mesh to subset.
    verst :     iterable
                Vertex indices to keep for the tree.
    limit :     float | np.inf | "auto"
                Use this to limit the distance for shortest path search
                (``scipy.sparse.csgraph.dijkstra``). Can greatly speed up this
                function at the risk of producing disconnected components. By
                default (auto), we are using 3x the max observed Eucledian
                distance between ``verts``.

    Returns
    -------
    edges :     np.ndarray
                List of `node` -> `parent` edges. Note that these edges are
                already hiearchical, i.e. each node has at exactly 1 parent
                except for the root node(s) which has parent ``-1``.

    """
    # Make sure vertices to keep are unique
    keep = unique(verts)

    # Get some shorthands
    verts = mesh.vertices
    edges = mesh.edges_unique
    edge_lengths = mesh.edges_unique_length

    # Produce adjacency matrix from edges and edge lengths
    adj = scipy.sparse.coo_matrix((edge_lengths,
                                   (edges[:, 0], edges[:, 1])),
                                  shape=(verts.shape[0], verts.shape[0]))

    if limit == 'auto':
        distances = scipy.spatial.distance.pdist(verts[keep])
        limit = np.max(distances) * 3

    # Get geodesic distances between vertices
    dist_matrix = scipy.sparse.csgraph.dijkstra(csgraph=adj, directed=False,
                                                indices=keep, limit=limit)

    # Subset along second axis
    dist_matrix = dist_matrix[:, keep]

    # Get minimum spanning tree
    mst = scipy.sparse.csgraph.minimum_spanning_tree(dist_matrix, overwrite=True)

    # Turn into COO matrix
    coo = mst.tocoo()

    # Extract edge list
    edges = np.array([keep[coo.row], keep[coo.col]]).T

    # Last but not least we have to run a depth first search to turn this
    # into a hierarchical tree, i.e. make edges are orientated in a way that
    # each node only has a single parent (turn a<-b->c into a->b->c)

    # Generate and populate undirected graph
    G = nx.Graph()
    G.add_edges_from(edges)

    # Generate list of parents
    edges = []
    # Go over all connected components
    for c in nx.connected_components(G):
        # Get subgraph of this connected component
        SG = nx.subgraph(G, c)

        # Use first node as root
        r = list(SG.nodes)[0]

        # List of parents: {node: [parent], root: []}
        this_lop = nx.predecessor(SG, r)

        # Note that we assign -1 as root's parent
        edges += [(k, v[0] if v else -1) for k, v in this_lop.items()]

    return np.array(edges).astype(int)


def dfs(G, n, dist_traveled, max_dist, seen):
    """Depth first graph traversal that stops at a given max distance."""
    visited = [n]
    seen.add(n)
    if dist_traveled <= max_dist:
        for nn in nx.neighbors(G, n):
            if nn not in seen:
                new_dist = dist_traveled + G.edges[(nn, n)]['weight']
                traversed, seen = dfs(G=G, n=nn,
                                      dist_traveled=new_dist,
                                      max_dist=max_dist,
                                      seen=seen)
                visited += traversed
    return visited, seen


def make_swc(x, coords, reindex=True, validate=True):
    """Generate SWC table.

    Parameters
    ----------
    x :         numpy.ndarray | networkx.Graph | networkx.DiGraph
                Data to generate SWC from. Can be::

                    - (N, 2) array of child->parent edges
                    - networkX graph

    coords :    trimesh.Trimesh | np.ndarray of vertices
                Coordinates of nodes in ``x``.
    reindex :   bool
                If True, will re-index node IDs such that parent nodes always
                have a lower node ID than their childs. This is a requirement
                for the SWC format. Will also return a dictionary mapping
                original to re-indexed node IDs.
    validate :  bool
                If True, will check check if SWC table is valid and raise an
                exception if issues are found.

    Returns
    -------
    swc :       pandas.DataFrame
    new_ids :   dict
                If ``reindex=True`` will also return a map for original to
                re-indexed node IDs.

    """
    assert isinstance(coords, (tm.Trimesh, np.ndarray))

    if isinstance(x, np.ndarray):
        edges = x
    elif isinstance(x, (nx.Graph, nx.DiGraph)):
        edges = np.array(x.edges)
    else:
        raise TypeError(f'Expected array or Graph, got "{type(x)}"')

    # Make sure edges are unique
    edges = unique(edges, axis=0)

    # Need to convert to None if empty - otherwise DataFrame creation acts up
    if len(edges) == 0:
        edges = None

    # Generate node table (do NOT remove the explicit dtype)
    swc = pd.DataFrame(edges, columns=['node_id', 'parent_id'], dtype=int)

    # See if we need to add manually add rows for root node(s)
    miss = swc.parent_id.unique()
    miss = miss[~np.isin(miss, swc.node_id.values)]
    miss = miss[miss > -1]

    # Must not use any() here because we might get miss=[0]
    if len(miss):
        roots = pd.DataFrame([[n, -1] for n in miss], columns=swc.columns)
        swc = pd.concat([swc, roots], axis=0)

    # See if we need to add any disconnected nodes
    if isinstance(x, (nx.Graph, nx.DiGraph)):
        miss = set(x.nodes) - set(swc.node_id.values) - set([-1])
        if miss:
            disc = pd.DataFrame([[n, -1] for n in miss], columns=swc.columns)
            swc = pd.concat([swc, disc], axis=0)

    if isinstance(coords, tm.Trimesh):
        coords = coords.vertices

    if not swc.empty:
        # Map x/y/z coordinates
        swc['x'] = coords[swc.node_id, 0]
        swc['y'] = coords[swc.node_id, 1]
        swc['z'] = coords[swc.node_id, 2]
    else:
        swc['x'] = swc['y'] = swc['z'] = None

    # Placeholder radius
    swc['radius'] = None

    if reindex:
        _, new_ids = reindex_swc(swc, inplace=True)
    else:
        swc = swc.sort_values('parent_id').reset_index(drop=True)

    if validate:
        # Check if any node has multiple parents
        if any(swc.node_id.duplicated()):
            raise ValueError('Nodes with multiple parents found.')

    if reindex:
        return swc, new_ids

    return swc


def reindex_swc(swc, inplace=False):
    """Reindex SWC such that parents always have a lower ID than their childs."""
    if not inplace:
        swc = swc.copy()

    # Sort such that the parent is always before the child
    swc.sort_values('parent_id', ascending=True, inplace=True)

    # Reset index
    swc.reset_index(drop=True, inplace=True)

    # Generate mapping
    new_ids = dict(zip(swc.node_id.values, swc.index.values))

    # -1 (root's parent) stays -1
    new_ids[-1] = -1

    swc['node_id'] = swc.node_id.map(new_ids)
    # Lambda prevents potential issue with missing parents
    swc['parent_id'] = swc.parent_id.map(lambda x: new_ids.get(x, -1))

    return swc, new_ids


def edges_to_graph(edges, nodes=None, vertices=None, fix_edges=True,
                   fix_tree=True, drop_disconnected=False, weight=True,
                   radii=None):
    """Create networkx Graph from edge list.

    Parameters
    ----------
    edges :         (N, 2) array
    nodes :         (M, ) array, optional
                    Node IDs. Should be provided in case of isolated nodes not
                    part of the edge list.
    vertices :      (M, 3) array, optional
                    X/Y/Z locations of nodes.
    fix_edges :     bool
                    If True (recommended!) will drop self-loops and remove
                    recurrent edges.
    fix_tree :      bool | "length" | "radius" | "degree"
                    If not False (recommended!) will fix the tree by removing
                    cycles. This is done using a minimum-spanning-tree or a
                    breadth-first search (see below). To improve this we can use
                    weights to increase the probability that cuts are made at
                    the right edges (i.e. preserving the "correct" topology of
                    the skeleton):

                      - "length" prioritizes cutting at long edges (requires
                        node positions as `vertex` to be provided)
                      - "radius" prioritizes cutting at edges with small radius
                        (requires `radii` to be provided)
                      - "degree" (default for `True`) prioritizes cutting at
                        edges between with low degree (i.e. non branch points)
                      - `True` will simply use a breadth-first search to
                        produce a directed graph without cycles.

    drop_disconnected : bool
                    Drops disconnected nodes from graph. Not recommended since
                    it breaks the vertex -> node mapping.
    weight :        bool
                    Whether to add edge lengths as weight to the final graph.
                    Requires `vertices` to be provided.
    radii :         (M, ) array, optional
                    Radii for each node. Only relevant if `fix_tree='radius'`.

    Returns
    -------
    G
                    networkx.DiGraph if `fix_tree` or networkx.Graph if not.

    """
    if fix_edges:
        # Drop self-loops
        edges = edges[edges[:, 0] != edges[:, 1]]

        # Make sure we don't have a->b and b<-a edges
        edges = unique(np.sort(edges, axis=1), axis=0)

    # Extract nodes from edges if not explicitly provided
    if isinstance(nodes, type(None)):
        nodes = unique(edges.flatten())

    # Start with undirected graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Fix tree with MST if specified
    if isinstance(fix_tree, str):
        if fix_tree == 'radius':
            # Ty cutting at edges with small radii
            if isinstance(radii, type(None)):
                raise ValueError('Must provided `radii` with `fix_tree="radius"`')
            weights = 1 / np.vstack((radii[edges[:, 0]],
                                     radii[edges[:, 1]])).mean(axis=0)
        elif fix_tree == 'length':
            # Ty cutting at long edges
            if isinstance(vertices, type(None)):
                raise ValueError('Must provided `vertices` with `fix_tree="length"`')
            vec = vertices[np.array(G.edges)[:, 0]] - vertices[np.array(G.edges)[:, 1]]
            weights = np.sqrt(np.sum(vec ** 2, axis=1))
        elif fix_tree == 'degree':
            # Else try cutting at edges with lowest degree
            weights = 1 / np.array([max(G.degree[e[0]], G.degree[e[1]]) for e in edges])
        else:
            raise ValueError(f'Unknown mode for `fix_tree`: "{fix_tree}"')
        # Note we are inverting the weights so that we cut either at a low
        # radius or a low degree
        weights[weights <= 0] = weights[weights > 0].min() / 2
        nx.set_edge_attributes(G, dict(zip([tuple(e) for e in edges], weights)), name='weight')

        # Get the minimum spanning tree
        G = nx.minimum_spanning_tree(G, weight='weight')

    # Even if we already ran the MST, we still need to orient the tree
    # This by itself also "fixes"  the tree (i.e. breaks cycles) but it doesn't
    # give us much control over it
    if fix_tree:
        trees = []
        for cc in nx.connected_components(G):
            # Get subgraph of this component
            SG = nx.subgraph(G, cc)
            # Create an oriented tree
            trees.append(nx.bfs_tree(SG, source=list(SG.nodes)[0]))

        # Create the union of all trees
        if len(trees) > 1:
            # For some reason this is much faster than nx.compose
            G = nx.DiGraph()
            for t in trees:
                G.add_edges_from(list(t.edges))
                G.add_nodes_from(list(t.nodes))
        else:
            G = trees[0]

        # Reverse to child -> parent
        G = G.reverse()

    if drop_disconnected:
        # Array of degrees [[node_id, degree], [....], ...]
        deg = np.array(G.degree)
        G.remove_nodes_from(deg[deg[:, 1] == 0][:, 0])

    if weight and isinstance(vertices, np.ndarray):
        final_edges = np.array(G.edges)
        vec = vertices[final_edges[:, 0]] - vertices[final_edges[:, 1]]
        weights = np.sqrt(np.sum(vec ** 2, axis=1))
        G.remove_edges_from(list(G.edges))
        G.add_weighted_edges_from([(e[0], e[1], w) for e, w in zip(final_edges, weights)])

    return G
