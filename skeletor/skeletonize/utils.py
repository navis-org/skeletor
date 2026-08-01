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
import scipy.sparse.csgraph

from .. import _fastcore
from ..utilities import get_edges_unique


def group_by_label(labels):
    """Group item indices by their label.

    The workhorse behind "give me the vertices of each connected component"
    (and of each cluster, and of each wavefront). Sorting by label puts every
    group in one contiguous run, which is both cheaper than a per-group pass
    over the whole array and the form the vectorized callers want.

    Parameters
    ----------
    labels :    (N, ) array
                One label per item. Need not be contiguous or start at zero.

    Returns
    -------
    order :     (N, ) array
                Item indices sorted by label. The sort is **stable**, so items
                stay in ascending index order within a label - that is what
                reproduces igraph's component ordering, and several call sites
                rely on the two backends agreeing on it.
    starts :    (n_groups, ) array
                Where each group's run begins in ``order``, groups in ascending
                label order.
    sizes :     (n_groups, ) array
                Length of each run. ``labels[order[starts]]`` gives the label
                each run belongs to.

    See Also
    --------
    :func:`split_by_label`
                The same thing as a list of per-group index arrays.

    """
    labels = np.asarray(labels)
    order = np.argsort(labels, kind='stable')
    if not len(order):
        empty = np.zeros(0, dtype=np.int64)
        return order, empty, empty
    srt = labels[order]
    starts = np.flatnonzero(np.r_[True, srt[1:] != srt[:-1]])
    return order, starts, np.diff(np.r_[starts, len(order)])


def split_by_label(labels):
    """Group item indices by their label, as a list of arrays.

    The convenience form of :func:`group_by_label` for callers that loop over
    groups rather than working on the runs directly.

    Parameters
    ----------
    labels :    (N, ) array
                One label per item.

    Returns
    -------
    list of arrays
                Item indices, one array per label, in ascending label order and
                ascending index order within each group.

    """
    order, starts, _ = group_by_label(labels)
    return np.split(order, starts[1:])


def prep_origins(origins):
    """Coerce the `origins` argument the wavefront skeletonizers share.

    Returns
    -------
    (N, ) int array
                Empty when ``origins`` is None.

    """
    if origins is None:
        return np.array([], dtype=np.int64)
    if isinstance(origins, (int, np.integer)):
        origins = [origins]
    elif not isinstance(origins, (set, list, tuple, np.ndarray)):
        raise TypeError('`origins` must be vertex ID (int) or list thereof, got '
                        f'"{type(origins)}"')
    return np.asarray(list(origins)).astype(np.int64)


def adjacency(edges, n_nodes, weights=None):
    """COO adjacency matrix for an undirected edge list."""
    if weights is None:
        weights = np.ones(len(edges))
    return scipy.sparse.coo_matrix((weights, (edges[:, 0], edges[:, 1])),
                                   shape=(n_nodes, n_nodes))


def connected_components(edges, n_nodes):
    """Connected components, each labelled by its lowest node ID.

    The label is part of the contract: callers key per-component data by it (see
    :func:`group_edges_by_component`), and it is what lets the fastcore and scipy
    backends agree on which component is which.
    """
    if _fastcore.has('connected_components_graph'):
        return np.asarray(_fastcore.fastcore.connected_components_graph(edges, n_nodes),
                          dtype=np.int64)

    _, labels = scipy.sparse.csgraph.connected_components(adjacency(edges, n_nodes),
                                                          directed=False)
    # scipy numbers components in discovery order. The first occurrence of a
    # label is the lowest node carrying it, so this relabels to the convention.
    return np.unique(labels, return_index=True)[1][labels]


def group_edges_by_component(edges, n_nodes):
    """Label components and group the edge list by them.

    Lets a per-component loop take its edges as a slice. Selecting them per
    component instead - the obvious reading - is one pass over the *whole* edge
    list per component, i.e. ``O(components * E)``. On a mesh that shatters into
    thousands of specks that dwarfs the actual work: it measured ~29s of a 33s
    run on a 447k-vertex, 12463-component mesh.

    Returns
    -------
    comp :      (n_nodes, ) array
                Component label of each node, being its component's lowest ID.
    ccs :       list of arrays
                Node indices per component, ascending.
    eorder :    (E, ) array
                Edge indices sorted by component. The grouping is stable, so each
                component's edges keep their original relative order.
    espan :     dict
                ``label -> (start, stop)`` into ``eorder``. A component with no
                edges at all is absent.

    """
    comp = connected_components(edges, n_nodes)

    # An edge never spans two components, so either endpoint labels it
    ecomp = comp[edges[:, 0]]
    eorder, estarts, esizes = group_by_label(ecomp)
    espan = dict(zip(ecomp[eorder[estarts]].tolist(),
                     zip(estarts.tolist(), (estarts + esizes).tolist())))

    return comp, split_by_label(comp), eorder, espan


def mst_over_mesh(mesh, verts, limit='auto'):
    """Generate minimum spanning tree by subsetting mesh to given vertices.

    Will (re-)connect vertices based on geodesic distance in original mesh
    using a minimum spanning tree.

    Parameters
    ----------
    mesh :      trimesh.Trimesh
                Mesh to subset.
    verts :     iterable
                Vertex indices to keep for the tree.
    limit :     float | np.inf | "auto"
                Use this to limit the distance for shortest path search. Can
                greatly speed up this function at the risk of producing
                disconnected components. By default (auto), we are using 3x the
                diagonal of the bounding box around ``verts`` - i.e. between 3x
                and 5.2x the largest distance between any two of them.

    Returns
    -------
    edges :     (N, 2) array
                The spanning tree as an undirected edge list, in vertex indices.
                Isolated vertices - ones the search reached from nothing else -
                contribute no row. Use :func:`orient_forest` to give it a
                direction.

    """
    # Make sure vertices to keep are unique
    keep = unique(verts)

    if limit == 'auto':
        # 3x the bounding box diagonal, which bounds the largest distance between
        # any two kept vertices from above (and by at most sqrt(3)x). This used to
        # take that largest distance exactly, via `pdist` - but exactly costs
        # O(k**2) in time *and* memory, 218 ms and 1.6 GB at k=20000 against
        # 0.6 ms and nothing here, and bought nothing for it: the search below
        # runs at the same speed under either limit.
        #
        # Being an upper bound the limit is never tighter than it was, only
        # looser - the safe direction, since it can only join more vertices,
        # never fewer. See `test_mst_auto_limit_is_an_upper_bound`.
        #
        # A single vertex has nothing to connect to, so any limit will do.
        if len(keep) > 1:
            v = mesh.vertices[keep]
            limit = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0))) * 3
        else:
            limit = np.inf

    if _fastcore.has('geodesic_mst_mesh'):
        return _mst_over_mesh_fastcore(mesh, keep, limit)

    return _mst_over_mesh_scipy(mesh, keep, limit)


def _mst_over_mesh_fastcore(mesh, keep, limit):
    """``mst_over_mesh`` backend using navis-fastcore.

    ``geodesic_mst_mesh`` spans ``keep`` under the geodesic metric without ever
    forming the ``k x k`` distance matrix the scipy backend needs - which is
    ``k**2`` distances to use ``k - 1`` of them, filled by ``k`` separate
    searches. It sweeps once, partitioning every mesh vertex by which of ``keep``
    is nearest, so the cost is flat in ``k`` rather than quadratic.

    ``limit`` prunes that sweep rather than merely discarding results, so unlike
    the scipy backend it buys time as well as bounding the output.
    """
    # Rows come back as positions *in* `keep`, so map them out to vertex indices
    mst_edges, _ = _fastcore.fastcore.geodesic_mst_mesh(
        mesh.faces, keep, vertices=np.asarray(mesh.vertices),
        limit=None if not np.isfinite(limit) else float(limit))

    return keep[mst_edges].reshape(-1, 2)


def _mst_over_mesh_scipy(mesh, keep, limit):
    """``mst_over_mesh`` backend without ``geodesic_mst_mesh``.

    Materialises the ``k x k`` geodesic distance matrix and hands it to scipy's
    matrix MST.
    """
    verts = mesh.vertices

    # Get geodesic distances between vertices
    if _fastcore.has('geodesic_matrix_mesh'):
        # Note that we only ever allocate the `keep` columns here - scipy has no
        # notion of targets and would materialise a (len(keep), len(verts))
        # matrix for us to subset afterwards.
        dist_matrix = _fastcore.fastcore.geodesic_matrix_mesh(
            mesh.faces, vertices=verts, sources=keep, targets=keep,
            limit=None if not np.isfinite(limit) else float(limit))
        # Unreachable comes back as -1 where scipy uses infinity. The MST below
        # reads anything non-positive as "no edge" but would take -1 at face
        # value as a *negative* weight.
        dist_matrix = np.where(dist_matrix < 0, np.inf,
                               dist_matrix).astype(np.float64)
    else:
        edges, edge_lengths = get_edges_unique(mesh, lengths=True)

        # Produce adjacency matrix from edges and edge lengths
        adj = adjacency(edges, verts.shape[0], edge_lengths)

        dist_matrix = scipy.sparse.csgraph.dijkstra(csgraph=adj, directed=False,
                                                    indices=keep, limit=limit)

        # Subset along second axis
        dist_matrix = dist_matrix[:, keep]

    # Get minimum spanning tree
    mst = scipy.sparse.csgraph.minimum_spanning_tree(dist_matrix, overwrite=True)

    # Turn into COO matrix
    coo = mst.tocoo()

    return np.column_stack((keep[coo.row], keep[coo.col])).astype(np.int64)


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

    # See if we need to manually add rows for root node(s)
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


def orient_forest(edges, n_nodes):
    """Orient an undirected edge list into a rooted forest.

    The half of "I have an edge list and I want a tree" that a minimum spanning
    tree does not answer: the MST picks *which* edges survive, this picks which
    way they point. Cycles are fine - each component contributes a spanning tree
    of itself - so this doubles as the cycle-breaker.

    Parameters
    ----------
    edges :     (N, 2) array
                Edge list over nodes ``0..n_nodes-1``. Direction is ignored.
                Self-loops and duplicate edges are tolerated.
    n_nodes :   int
                Total number of nodes. Nodes named by no edge are isolated roots.

    Returns
    -------
    parents :   (n_nodes, ) array
                Parent of each node, ``-1`` for roots. Each component is rooted
                at its lowest node ID.
    order :     (n_nodes, ) array
                Every node in the order it settled. A node always settles after
                its parent, so relabelling by this puts parents before their
                children - which is what SWC requires.

    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)

    if _fastcore.has('parents_from_edges'):
        parents, order = _fastcore.fastcore.parents_from_edges(edges, n_nodes)
        return np.asarray(parents, dtype=np.int64), np.asarray(order, dtype=np.int64)

    return _orient_forest_scipy(edges, n_nodes)


def _orient_forest_scipy(edges, n_nodes):
    """``orient_forest`` fallback. Same contract, an order of magnitude slower.

    Where fastcore sweeps the components one after another into a single
    ``n_nodes``-long column, this pays for a full ``connected_components`` pass
    plus a multi-source Dijkstra to get the same answer.
    """
    if not n_nodes:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty

    adj = adjacency(edges, n_nodes)
    _, labels = scipy.sparse.csgraph.connected_components(adj, directed=False)

    # First occurrence of each label = lowest node ID in that component, which
    # is the root `parents_from_edges` picks too
    roots = np.unique(labels, return_index=True)[1]

    # With unit weights this is effectively a BFS from all roots at once. On a
    # forest the resulting orientation is unique; where the input has cycles this
    # is a shortest-path tree, which breaks them just as validly (though not
    # necessarily in the same place fastcore would).
    dist, pred, _ = scipy.sparse.csgraph.dijkstra(adj, directed=False,
                                                  indices=roots,
                                                  return_predecessors=True,
                                                  min_only=True)

    # `pred` uses -9999 for roots and isolated nodes; we promise -1
    parents = np.where(pred < 0, -1, pred).astype(np.int64)

    # Sorting by BFS depth puts parents before their children
    return parents, np.argsort(dist, kind='stable')


def forest_to_swc(edges, coords, radii, n_nodes):
    """Generate SWC table directly from an edge list over ``0..n_nodes-1``.

    This is a fast alternative to the ``edges_to_graph`` -> ``make_swc`` ->
    ``reindex_swc`` pipeline for the case where the nodes are already numbered
    contiguously: each tree is rooted at its lowest node ID and oriented
    child -> parent, and new node IDs are assigned in traversal order which
    guarantees that parents have lower IDs than their children.

    See :func:`edges_to_swc` for the same thing over an arbitrary node space.

    Parameters
    ----------
    edges :     (N, 2) array
                Edge list over nodes ``0..n_nodes-1``. Isolated nodes are
                allowed and become roots; cycles are broken (see
                :func:`orient_forest`).
    coords :    (n_nodes, 3) array
                X/Y/Z locations of nodes.
    radii :     (n_nodes, ) array
                Radius for each node.
    n_nodes :   int
                Total number of nodes.

    Returns
    -------
    swc :       pandas.DataFrame
    new_ids :   (n_nodes, ) array
                Maps original node IDs (positional) to re-indexed node IDs.

    """
    parents, order = orient_forest(edges, n_nodes)

    new_ids = np.empty(n_nodes, dtype=np.int64)
    new_ids[order] = np.arange(n_nodes)

    parent_new = _reindex_parents(parents, order, new_ids)

    swc = pd.DataFrame({'node_id': np.arange(n_nodes),
                        'parent_id': parent_new,
                        'x': coords[order, 0],
                        'y': coords[order, 1],
                        'z': coords[order, 2],
                        'radius': radii[order]})

    return swc, new_ids


def _reindex_parents(parents, order, new_ids):
    """Express ``parents`` in the relabelled ID space, row-ordered by ``order``.

    Kept separate because the masking is the easy thing to get wrong: ``parents``
    carries ``-1`` for roots and ``new_ids[-1]`` silently reads the *last* node
    rather than raising.
    """
    parent_ord = parents[order]
    parent_new = np.full(len(order), -1, dtype=np.int64)
    has_parent = parent_ord >= 0
    parent_new[has_parent] = new_ids[parent_ord[has_parent]]
    return parent_new


def edges_to_swc(edges, coords, nodes=None):
    """Generate SWC table from an edge list over an arbitrary node space.

    Does in one pass what ``edges_to_graph(fix_tree=True)`` followed by
    ``make_swc(reindex=True)`` does via networkx: break cycles, orient every
    component child -> parent, and renumber so parents come before their
    children. Self-loops and duplicate edges are tolerated.

    :func:`forest_to_swc` is the same thing for nodes already numbered
    ``0..n_nodes-1``; this one accepts any node IDs (e.g. a subset of mesh
    vertices) and hands back the ID mapping as a dict, like :func:`make_swc`.

    Parameters
    ----------
    edges :     (N, 2) array
                Edge list. Node IDs index ``coords``.
    coords :    trimesh.Trimesh | (M, 3) array
                X/Y/Z locations, indexed by the *original* node IDs.
    nodes :     iterable, optional
                Additional node IDs to keep. Use this for isolated nodes that no
                edge names - without it the SWC holds exactly the nodes ``edges``
                mentions.
    Returns
    -------
    swc :       pandas.DataFrame
    new_ids :   dict
                Maps original node IDs to re-indexed node IDs. ``-1`` maps to
                itself so the root sentinel survives a lookup.

    """
    if isinstance(coords, tm.Trimesh):
        coords = coords.vertices

    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)

    # The node space: everything an edge names, plus whatever the caller wants
    # kept on top. `unique` sorts, which is what makes the searchsorted below
    # a valid original -> local index map.
    ids = edges.ravel()
    if nodes is not None:
        ids = np.concatenate((ids, np.asarray(nodes, dtype=np.int64).ravel()))
    ids = unique(ids)

    # Node IDs index `coords`, so a negative one would quietly pick a vertex off
    # the end rather than fail. Worth refusing outright: the natural mistake here
    # is feeding in a child -> parent list with its -1 roots still attached.
    if len(ids) and ids[0] < 0:
        raise ValueError('Node IDs must be non-negative, got '
                         f'{ids[ids < 0].tolist()[:5]}. If these are the `-1` '
                         'roots of a child -> parent edge list, drop those rows '
                         'and pass the root node IDs as `nodes` instead.')

    parents, order = orient_forest(np.searchsorted(ids, edges), len(ids))

    new_ids_arr = np.empty(len(ids), dtype=np.int64)
    new_ids_arr[order] = np.arange(len(ids))

    swc = pd.DataFrame({'node_id': np.arange(len(ids), dtype=np.int64),
                        'parent_id': _reindex_parents(parents, order,
                                                      new_ids_arr)})

    # `ids[order]` are the original IDs in their new row order
    original = ids[order]
    if len(ids):
        swc['x'] = coords[original, 0]
        swc['y'] = coords[original, 1]
        swc['z'] = coords[original, 2]
    else:
        swc['x'] = swc['y'] = swc['z'] = None
    swc['radius'] = None       # a placeholder, as `make_swc` leaves it

    new_ids = dict(zip(ids.tolist(), new_ids_arr.tolist()))
    new_ids[-1] = -1  # as `reindex_swc` does - the root's parent stays -1

    return swc, new_ids


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
