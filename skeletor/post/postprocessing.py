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

import ncollpyde
import numbers
import warnings

import networkx as nx
import numpy as np
import scipy.spatial

from ..utilities import make_trimesh


def clean_up(s, mesh=None, validate=False, inplace=False, **kwargs):
    """Clean up the skeleton.

    This function bundles a bunch of procedures to clean up the skeleton:

      1. Remove twigs that are running parallel to their parent branch
      2. Move nodes outside the mesh back inside (or at least snap to surface)

    Note that this is not a magic bullet and some of this will not work (well)
    if the original mesh was degenerate (e.g. internal faces or not watertight)
    to begin with.

    Parameters
    ----------
    s :         skeletor.Skeleton
                Skeleton to clean up.
    mesh :      trimesh.Trimesh, optional
                Original mesh (e.g. before contraction). If not provided will
                use the mesh associated with ``s``.
    validate :  bool
                If True, will try to fix potential issues with the mesh
                (e.g. infinite values, duplicate vertices, degenerate faces)
                before cleaning up. Note that this might change your mesh
                inplace!
    inplace :   bool
                If False will make and return a copy of the skeleton. If True,
                will modify the `s` inplace.

    **kwargs
                Keyword arguments are passed to the bundled function.

                For `skeletor.postprocessing.drop_parallel_twigs`::

                 theta :     float (default 0.01)
                             For each twig we generate the dotproduct between the tangent
                             vectors of it and its parents. If these line up perfectly the
                             dotproduct will equal 1. ``theta`` determines how much that
                             value can differ from 1 for us to still prune the twig: higher
                             theta = more pruning.

    Returns
    -------
    s_clean :   skeletor.Skeleton
                Hopefully improved skeleton.

    """
    if isinstance(mesh, type(None)):
        mesh = s.mesh

    mesh = make_trimesh(mesh, validate=validate)

    if not inplace:
        s = s.copy()

    # Drop parallel twigs
    _ = drop_parallel_twigs(s, theta=kwargs.get('theta', 0.01), inplace=True)

    # Recenter vertices
    _ = recenter_vertices(s, mesh, inplace=True)

    return s


def remove_hairs(s, mesh=None, inplace=False):
    """Remove "hairs" that sometimes occurr along the backbone.

    Works by finding terminal twigs that consist of only a single node. We will
    then remove those that are within line of sight of their parent.

    Note that this is currently not used for clean up as it does not work very
    well: removes as many correct hairs as genuine small branches.

    Parameters
    ----------
    s :         skeletor.Skeleton
                Skeleton to clean up.
    mesh :      trimesh.Trimesh, optional
                Original mesh (e.g. before contraction). If not provided will
                use the mesh associated with ``s``.
    inplace :   bool
                If False will make and return a copy of the skeleton. If True,
                will modify the `s` inplace.

    Returns
    -------
    SWC :       pandas.DataFrame
                SWC with line-of-sight twigs removed.

    """
    if isinstance(mesh, type(None)):
        mesh = s.mesh

    # Make a copy of the skeleton
    if not inplace:
        s = s.copy()

    # Find branch points
    pcount = s.swc[s.swc.parent_id >= 0].groupby('parent_id').size()
    bp = pcount[pcount > 1].index

    # Find terminal twigs
    twigs = s.swc[~s.swc.node_id.isin(s.swc.parent_id)]
    twigs = twigs[twigs.parent_id.isin(bp)]

    if twigs.empty:
        return s

    # Initialize ncollpyde Volume
    coll = ncollpyde.Volume(mesh.vertices, mesh.faces, validate=False)

    # Remove twigs that aren't inside the volume
    twigs = twigs[coll.contains(twigs[['x', 'y', 'z']].values)]

    # Generate rays between all pairs and their parents
    sources = twigs[['x', 'y', 'z']].values
    targets = s.swc.set_index('node_id').loc[twigs.parent_id,
                                             ['x', 'y', 'z']].values

    # Get intersections: `ix` points to index of line segment; `loc` is the
    #  x/y/z coordinate of the intersection and `is_backface` is True if
    # intersection happened at the inside of a mesh
    ix, loc, is_backface = coll.intersections(sources, targets)

    # Find pairs of twigs with no intersection - i.e. with line of sight
    los = ~np.isin(np.arange(sources.shape[0]), ix)

    # To remove: have line of sight
    to_remove = twigs[los]

    s.swc = s.swc[~s.swc.node_id.isin(to_remove.node_id)].copy()

    # Update the mesh map
    mesh_map = getattr(s, 'mesh_map', None)
    if not isinstance(mesh_map, type(None)):
        for t in to_remove.itertuples():
            mesh_map[mesh_map == t.node_id] = t.parent_id

    # Reindex nodes
    s.reindex(inplace=True)

    return s


def recenter_vertices(s, mesh=None, inplace=False):
    """Move nodes that ended up outside the mesh back inside.

    Nodes can end up outside the original mesh e.g. if the mesh contraction
    didn't do a good job (most likely because of internal/degenerate faces that
    messed up the normals). This function rectifies this by snapping those nodes
    nodes back to the closest vertex and then tries to move them into the
    mesh's center. That second step is not guaranteed to work but at least you
    won't have any more nodes outside the mesh.

    Please note that if connected (!) nodes end up on the same position (i.e
    because they snapped to the same vertex), we will collapse them.

    Parameters
    ----------
    s :         skeletor.Skeleton
    mesh :      trimesh.Trimesh
                Original mesh.
    inplace :   bool
                If False will make and return a copy of the skeleton. If True,
                will modify the `s` inplace.

    Returns
    -------
    SWC :       pandas.DataFrame
                SWC with line-of-sight twigs removed.

    """
    if isinstance(mesh, type(None)):
        mesh = s.mesh

    # Copy skeleton
    if not inplace:
        s = s.copy()

    # Find nodes that are outside the mesh
    coll = ncollpyde.Volume(mesh.vertices, mesh.faces, validate=False)
    outside = ~coll.contains(s.vertices)

    # Skip if all inside
    if not any(outside):
        return s

    # For each outside find the closest vertex
    tree = scipy.spatial.cKDTree(mesh.vertices)

    # Find nodes that are right on top of original vertices
    dist, ix = tree.query(s.vertices[outside])

    # We don't want to just snap them back to the closest vertex but try to find
    # the center. For this we will:
    # 1. Move each vertex inside the mesh by just a bit
    # 2. Cast a ray along the vertices' normals and find the opposite sides of the mesh
    # 3. Calculate the distance

    # Get the closest vertex...
    closest_vertex = mesh.vertices[ix]
    # .. and offset the vertex positions by just a bit so they "should" be
    # inside the mesh. In reality that doesn't always happen if the mesh is not
    # watertight
    vnormals = mesh.vertex_normals[ix]
    sources = closest_vertex - vnormals

    # Prepare rays to cast
    targets = sources - vnormals * 1e4

    # Cast rays
    ix, loc, is_backface = coll.intersections(sources, targets)

    # If no collisions
    if len(loc) == 0:
        return s

    # Get half-vector
    halfvec = np.zeros(sources.shape)
    halfvec[ix] = (loc - closest_vertex[ix]) / 2

    # Offset vertices
    final_pos = closest_vertex + halfvec

    # Keep only those that are properly inside the mesh and fall back to the
    # closest vertex if that's not the case
    now_inside = coll.contains(final_pos)
    final_pos[~now_inside] = closest_vertex[~now_inside]

    # Replace coordinates
    s.swc.loc[outside, 'x'] = final_pos[:, 0]
    s.swc.loc[outside, 'y'] = final_pos[:, 1]
    s.swc.loc[outside, 'z'] = final_pos[:, 2]

    # At this point we may have nodes that snapped to the same vertex and
    # therefore end up at the same position. We will collapse those nodes
    # - but only if they are actually connected!
    # First find duplicate locations
    u, i, c = np.unique(s.vertices, return_counts=True, return_inverse=True, axis=0)

    # If any coordinates have counter higher 1
    if c.max() > 1:
        rewire = {}
        # Find out which unique coordinates are duplicated
        dupl = np.where(c > 1)[0]

        # Go over each duplicated coordinate
        for ix in dupl:
            # Find the nodes on this duplicate coordinate
            node_ix = np.where(i == ix)[0]

            # Get their edges
            edges = s.edges[np.all(np.isin(s.edges, node_ix), axis=1)]

            # We will work on the graph to collapse nodes sequentially A->B->C
            G = nx.DiGraph()
            G.add_edges_from(edges)
            for cc in nx.connected_components(G.to_undirected()):
                # Root is the node without any outdegree in this subgraph
                root = [n for n in cc if G.out_degree[n] == 0][0]
                # We don't want to collapse into `root` because it's not actually
                # among the nodes with the same coordinates but rather the "last"
                # nodes parent
                clps_into = next(G.predecessors(root))
                # Keep track of how we need to rewire
                rewire.update({c: clps_into for c in cc if c not in {root, clps_into}})

        # Only mess with the skeleton if there were nodes to be merged
        if rewire:
            # Rewire
            s.swc['parent_id'] = s.swc.parent_id.map(lambda x: rewire.get(x, x))

            # Drop nodes that were collapsed
            s.swc = s.swc.loc[~s.swc.node_id.isin(rewire)]

            # Update mesh map
            if not isinstance(s.mesh_map, type(None)):
                s.mesh_map = [rewire.get(x, x) for x in s.mesh_map]

            # Reindex to make vertex IDs continous again
            s.reindex(inplace=True)

            # This prevents future SettingsWithCopy Warnings:
            if not inplace:
                s.swc = s.swc.copy()

    return s


def drop_line_of_sight_twigs(s, mesh=None, max_dist='auto', inplace=False):
    """Collapse twigs that are in line of sight to each other.

    Note that this only removes 1 layer of twigs (i.e. only the actual leaf
    nodes). Nothing is stopping you from running this function recursively
    though.

    Also note that this function needs a rework because it does not take
    connected components into account and hence collapses things that were
    not meant to be connected.

    Parameters
    ----------
    s :         skeletor.Skeleton
                Skeleton to clean up.
    mesh :      trimesh.Trimesh, optional
                Original mesh (e.g. before contraction). If not provided will
                use the mesh associated with ``s``.
    max_dist :  "auto" | int | float
                Maximum Eucledian distance allowed between leaf nodes for them
                to be considered for collapsing. If "auto", will use the length
                of the longest edge in skeleton as limit.
    inplace :   bool
                If False will make and return a copy of the skeleton. If True,
                will modify the `s` inplace.

    Returns
    -------
    SWC :       pandas.DataFrame
                SWC with line-of-sight twigs removed.

    """
    # Make a copy of the SWC
    if not inplace:
        s = s.copy()

    # Add distance to parents
    s.swc['parent_dist'] = 0
    not_root = s.swc.parent_id >= 0
    co1 = s.swc.loc[not_root, ['x', 'y', 'z']].values
    co2 = s.swc.set_index('node_id').loc[s.swc.loc[not_root, 'parent_id'],
                                         ['x', 'y', 'z']].values
    s.swc.loc[not_root, 'parent_dist'] = np.sqrt(np.sum((co1 - co2)**2, axis=1))

    # If max dist is 'auto', we will use the longest child->parent edge in the
    # skeleton as limit
    if max_dist == 'auto':
        max_dist = s.swc.parent_dist.max()

    # Initialize ncollpyde Volume
    coll = ncollpyde.Volume(mesh.vertices, mesh.faces, validate=False)

    # Find twigs
    twigs = s.swc[~s.swc.node_id.isin(s.swc.parent_id)]

    # Remove twigs that aren't inside the volume
    twigs = twigs[coll.contains(twigs[['x', 'y', 'z']].values)]

    # Generate rays between all pairs of twigs
    twigs_co = twigs[['x', 'y', 'z']].values
    sources = np.repeat(twigs_co, twigs.shape[0], axis=0)
    targets = np.tile(twigs_co, (twigs.shape[0], 1))

    # Keep track of indices
    pairs = np.stack((np.repeat(twigs.node_id, twigs.shape[0]),
                      np.tile(twigs.node_id, twigs.shape[0]))).T

    # If max distance, drop pairs that are too far appart
    if max_dist:
        d = scipy.spatial.distance.pdist(twigs_co)
        d = scipy.spatial.distance.squareform(d)
        is_close = d.flatten() <= max_dist
        pairs = pairs[is_close]
        sources = sources[is_close]
        targets = targets[is_close]

    # Drop self rays
    not_self = pairs[:, 0] != pairs[:, 1]
    sources, targets = sources[not_self], targets[not_self]
    pairs = pairs[not_self]

    # Get intersections: `ix` points to index of line segment; `loc` is the
    #  x/y/z coordinate of the intersection and `is_backface` is True if
    # intersection happened at the inside of a mesh
    ix, loc, is_backface = coll.intersections(sources, targets)

    # Find pairs of twigs with no intersection - i.e. with line of sight
    los = ~np.isin(np.arange(pairs.shape[0]), ix)

    # To collapse: have line of sight
    to_collapse = pairs[los]

    # Group into cluster we need to collapse
    G = nx.Graph()
    G.add_edges_from(to_collapse)
    clusters = nx.connected_components(G)

    # When collapsing the clusters, we need to decide which should be the
    # winning twig. For this we will use the twig lengths. In theory we ought to
    # be more fancy and ask for the distance to the root but that's more
    # expensive and it's unclear if it'll work any better.
    seg_lengths = twigs.set_index('node_id').parent_dist.to_dict()
    to_remove = []
    seen = set()
    for nodes in clusters:
        # We have to be careful not to generate chains here, e.g. A sees B,
        # B sees C, C sees D, etc. To prevent this, we will break up these
        # clusters into cliques and collapse them by order of size of the
        # cliques
        for cl in nx.find_cliques(nx.subgraph(G, nodes)):
            # Turn into set
            cl = set(cl)
            # Drop any node that has already been visited
            cl = cl - seen
            # Skip if less than 2 nodes left in clique
            if len(cl) < 2:
                continue
            # Add nodes to list of visited nodes
            seen = seen | cl
            # Sort by segment lenghts to find the loosing nodes
            loosers = sorted(cl, key=lambda x: seg_lengths[x])[:-1]
            to_remove += loosers

    # Drop the tips we flagged for removal and the new column we added
    s.swc = s.swc[~s.swc.node_id.isin(to_remove)].drop('parent_dist', axis=1)

    # Clean up node/vertex order
    s.reindex(inplace=True)

    return s


def drop_parallel_twigs(s, theta=0.01, inplace=False):
    """Remove 1-node twigs that run parallel to their parent branch.

    This happens e.g. for vertex clustering skeletonization.

    Parameters
    ----------
    s :         skeletor.Skeleton
                Skeleton to clean up.
    theta :     float
                For each twig we generate the dotproduct between the tangent
                vectors of it and its parents. If these line up perfectly the
                dotproduct will equal 1. ``theta`` determines how much that
                value can DIFFER from 1 for us to still prune the twig: higher
                theta = more pruning.
    inplace :   bool
                If False will make and return a copy of the skeleton. If True,
                will modify the `s` inplace.

    Returns
    -------
    SWC :       pandas.DataFrame
                SWC with parallel twigs removed.

    """
    assert isinstance(theta, numbers.Number), "theta must be a number"
    assert 0 <= theta <= 1, "theta must be between 0 and 1"

    # Work on a copy of the SWC table
    if not inplace:
        s = s.copy()

    # Find roots
    roots = s.swc[s.swc.parent_id < 0].node_id

    # Find branch points - we ignore roots that are also branch points because
    # that would cause headaches with tangent vectors further down
    cnt = s.swc.groupby('parent_id').node_id.count()
    bp = s.swc[s.swc.node_id.isin((cnt >= 2).index) & ~s.swc.node_id.isin(roots)]
    # Find 1-node twigs
    twigs = s.swc[~s.swc.node_id.isin(s.swc.parent_id) & s.swc.parent_id.isin(bp.node_id)]

    # Produce parent -> child tangent vectors for each node
    # Note that root nodes will have a NaN parent tangent vector
    coords = s.swc.set_index('node_id')[['x', 'y', 'z']]
    tangents = (s.swc[['x', 'y', 'z']].values - coords.reindex(s.swc.parent_id).values)
    tangents /= np.sqrt(np.sum(tangents**2, axis=1)).reshape(tangents.shape[0], 1)
    s.swc['tangent_x'] = tangents[:, 0]
    s.swc['tangent_y'] = tangents[:, 1]
    s.swc['tangent_z'] = tangents[:, 2]

    # For each node calculate a child vector
    child_tangent = s.swc[s.swc.parent_id >= 0].groupby('parent_id')
    child_tangent = child_tangent[['tangent_x', 'tangent_y', 'tangent_z']].sum()

    # Combine into a final vector and normalize again
    comb_tangent = s.swc[['tangent_x', 'tangent_y', 'tangent_y']].fillna(0).values \
        + child_tangent.reindex(s.swc.node_id).fillna(0).values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        comb_tangent /= np.sqrt(np.sum(comb_tangent**2, axis=1)).reshape(comb_tangent.shape[0], 1)

    # Replace tangent vectors in SWC dataframe
    s.swc['tangent_x'] = comb_tangent[:, 0]
    s.swc['tangent_y'] = comb_tangent[:, 1]
    s.swc['tangent_z'] = comb_tangent[:, 2]

    # Now get the dotproducts of the twigs' and their parent's tangent vectors
    twig_tangents = s.swc.set_index('node_id').loc[twigs.node_id,
                                                   ['tangent_x',
                                                    'tangent_y',
                                                    'tangent_z']].values
    parent_tangents = s.swc.set_index('node_id').loc[twigs.parent_id,
                                                     ['tangent_x',
                                                      'tangent_y',
                                                      'tangent_z']].values

    # Drop the tangent columns we made
    s.swc.drop(['tangent_x', 'tangent_y', 'tangent_z'], axis=1, inplace=True)

    # Generate dotproducts
    dot = np.einsum('ij,ij->i', twig_tangents, parent_tangents)

    # Basically we want to drop any twig for which the dotproduct is close to 1
    dot_diff = 1 - np.fabs(dot)
    # Remove twigs where the dotproduct is within `theta` to 1
    to_remove = twigs.loc[dot_diff <= theta]

    if not to_remove.empty:
        s.swc = s.swc[~s.swc.node_id.isin(to_remove.node_id)].copy()

        # Update the mesh map
        mesh_map = getattr(s, 'mesh_map', None)
        if not isinstance(mesh_map, type(None)):
            for t in to_remove.itertuples():
                mesh_map[mesh_map == t.node_id] = t.parent_id

        # Reindex nodes
        s.reindex(inplace=True)

    return s
