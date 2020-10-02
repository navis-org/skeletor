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
    import ncollpyde
except ImportError:
    ncollpyde = None
except BaseException:
    raise

import numbers

import networkx as nx
import numpy as np
import pandas as pd
import scipy.spatial

from .utilities import make_trimesh


def clean(swc, mesh, validate=True, copy=True, **kwargs):
    """Clean up the skeleton.

    This function bundles a bunch of procedures to clean up the skeleton:

      1. Remove twigs that are running parallel to their parent branch
      2. Collapse twigs that have line of sight to each other
      3. Move nodes outside the mesh back inside (or at least snap to surface)

    Note that this is not a magic bullet and some of this will not work (well)
    if the original mesh was degenerate (e.g. internal faces or not watertight)
    to begin with.

    Parameters
    ----------
    swc :       pandas.DataFrame
    mesh :      trimesh.Trimesh
                Original mesh.
    validate :  bool
                If True, will try to fix potential issues with the mesh
                (e.g. infinite values, duplicate vertices, degenerate faces)
                before cleaning up.
    copy :      bool
                If True will make and return a copy of the SWC table.

    **kwargs
                Keyword arguments are passed to the bundled function:

    For `skeletor.postprocessing.drop_line_of_sight_twigs`:

    max_dist :  "auto" (default) | int | float
                Maximum Eucledian distance allowed between leaf nodes for them
                to be considered for collapsing. If "auto", will use the length
                of the longest edge in skeleton as limit.

    For `skeletor.postprocessing.drop_parallel_twigs`:

    theta :     float (default 0.01)
                For each twig we generate the dotproduct between the tangent
                vectors of it and its parents. If these line up perfectly the
                dotproduct will equal 1. ``theta`` determines how much that
                value can differ from 1 for us to still prune the twig: higher
                theta = more pruning.

    Returns
    -------
    SWC :       pandas.DataFrame
                Hopefully improved SWC.

    """
    assert isinstance(swc, pd.DataFrame)

    mesh = make_trimesh(mesh, validate=validate)

    if copy:
        swc = swc.copy()

    # Drop parallel twigs
    swc = drop_parallel_twigs(swc, theta=kwargs.get('theta', 0.01), copy=False)

    # Recenter vertices
    swc = recenter_vertices(swc, mesh, copy=False)

    # Collapse twigs that in line of sight to one another
    swc = drop_line_of_sight_twigs(swc, mesh, copy=False,
                                   max_dist=kwargs.get('max_dist', 'auto'))

    return swc


def recenter_vertices(swc, mesh, copy=True):
    """Move nodes that ended up outside the mesh back inside.

    Nodes can end up outside the original mesh e.g. if the mesh contraction
    didn't do a good job (most likely because of internal/degenerate faces that
    messed up the normals). This function rectifies this by snapping those nodes
    nodes back to the closest vertex and then tries to move them into the
    mesh's center. That second step is not guaranteed to work but at least you
    won't have any more nodes outside the mesh.

    Parameters
    ----------
    swc :       pandas.DataFrame
    mesh :      trimesh.Trimesh
                Original mesh.
    copy :      bool
                If True will make and return a copy of the SWC table.

    Returns
    -------
    SWC :       pandas.DataFrame
                SWC with line-of-sight twigs removed.

    """
    if not ncollpyde:
        raise ImportError('skeletor.recenter_vertices() requires '
                          'the ncollpyde package: pip3 install ncollpyde')

    # Copy SWC
    if copy:
        swc = swc.copy()

    # Find nodes that are outside the mesh
    coll = ncollpyde.Volume(mesh.vertices, mesh.faces, validate=False)
    outside = ~coll.contains(swc[['x', 'y', 'z']].values)

    # For each outside find the closest vertex
    tree = scipy.spatial.cKDTree(mesh.vertices)

    # Find nodes that are right on top of original vertices
    dist, ix = tree.query(swc.loc[outside, ['x', 'y', 'z']].values)

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
    swc.loc[outside, 'x'] = final_pos[:, 0]
    swc.loc[outside, 'y'] = final_pos[:, 1]
    swc.loc[outside, 'z'] = final_pos[:, 2]

    return swc


def drop_line_of_sight_twigs(swc, mesh, max_dist='auto', copy=True):
    """Collapse twigs that are in line of sight to each other.

    Note that this only removes 1 layer of twigs (i.e. only the actual leaf
    nodes). Nothing is stopping you from running this function recursively
    though.

    Parameters
    ----------
    swc :       pandas.DataFrame
    mesh :      trimesh.Trimesh
                Original mesh.
    max_dist :  "auto" | int | float
                Maximum Eucledian distance allowed between leaf nodes for them
                to be considered for collapsing. If "auto", will use the length
                of the longest edge in skeleton as limit.
    copy :      bool
                If True will make and return a copy of the SWC table.

    Returns
    -------
    SWC :       pandas.DataFrame
                SWC with line-of-sight twigs removed.

    """
    if not ncollpyde:
        raise ImportError('skeletor.drop_line_of_sight_twigs() requires '
                          'the ncollpyde package: pip3 install ncollpyde')

    # Make a copy of the SWC
    if copy:
        swc = swc.copy()

    # Add distance to parents
    swc['parent_dist'] = 0
    not_root = swc.parent_id >= 0
    co1 = swc.loc[not_root, ['x', 'y', 'z']].values
    co2 = swc.set_index('node_id').loc[swc.loc[not_root, 'parent_id'],
                                       ['x', 'y', 'z']].values
    swc.loc[not_root, 'parent_dist'] = np.sqrt(np.sum((co1 - co2)**2, axis=1))

    # If max dist is 'auto', we will use the longest child->parent edge in the
    # skeleton as limit
    if max_dist == 'auto':
        max_dist = swc.parent_dist.max()

    # Initialize ncollpyde Volume
    coll = ncollpyde.Volume(mesh.vertices, mesh.faces, validate=False)

    # Find twigs
    twigs = swc[~swc.node_id.isin(swc.parent_id)]

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
    swc = swc[~swc.node_id.isin(to_remove)].drop('parent_dist', axis=1)

    return swc


def drop_parallel_twigs(swc, theta=0.01, copy=True):
    """Remove 1-node twigs that run parallel to their parent branch.

    This happens e.g. for vertex clustering skeletonization.

    Parameters
    ----------
    swc :       pandas.DataFrame
    theta :     float
                For each twig we generate the dotproduct between the tangent
                vectors of it and its parents. If these line up perfectly the
                dotproduct will equal 1. ``theta`` determines how much that
                value can DIFFER from 1 for us to still prune the twig: higher
                theta = more pruning.
    copy :      bool
                If True will make and return a copy of the SWC table.

    Returns
    -------
    SWC :       pandas.DataFrame
                SWC with parallel twigs removed.

    """
    assert isinstance(theta, numbers.Number), "theta must be a number"
    assert 0 <= theta <= 1, "theta must be between 0 and 1"

    # Work on a copy of the SWC table
    if copy:
        swc = swc.copy()

    # Find roots
    roots = swc[swc.parent_id < 0].node_id

    # Find branch points - we ignore roots that are also branch points because
    # that would cause headaches with tangent vectors further down
    cnt = swc.groupby('parent_id').node_id.count()
    bp = swc[swc.node_id.isin((cnt >= 2).index) & ~swc.node_id.isin(roots)]
    # Find 1-node twigs
    twigs = swc[~swc.node_id.isin(swc.parent_id) & swc.parent_id.isin(bp.node_id)]

    # Produce parent -> child tangent vectors for each node
    # Note that root nodes will have a NaN parent tangent vector
    coords = swc.set_index('node_id')[['x', 'y', 'z']]
    tangents = (swc[['x', 'y', 'z']].values - coords.reindex(swc.parent_id).values)
    tangents /= np.sqrt(np.sum(tangents**2, axis=1)).reshape(tangents.shape[0], 1)
    swc['tangent_x'] = tangents[:, 0]
    swc['tangent_y'] = tangents[:, 1]
    swc['tangent_z'] = tangents[:, 2]

    # For each node calculate a child vector
    child_tangent = swc[swc.parent_id >= 0].groupby('parent_id')
    child_tangent = child_tangent[['tangent_x', 'tangent_y', 'tangent_z']].sum()

    # Combine into a final vector and normalize again
    comb_tangent = swc[['tangent_x', 'tangent_y', 'tangent_y']].fillna(0).values \
        + child_tangent.reindex(swc.node_id).fillna(0).values
    comb_tangent /= np.sqrt(np.sum(comb_tangent**2, axis=1)).reshape(comb_tangent.shape[0], 1)
    # Replace tangent vectors in SWC dataframe
    swc['tangent_x'] = comb_tangent[:, 0]
    swc['tangent_y'] = comb_tangent[:, 1]
    swc['tangent_z'] = comb_tangent[:, 2]

    # Now get the dotproducts of the twigs' and their parent's tangent vectors
    twig_tangents = swc.set_index('node_id').loc[twigs.node_id, ['tangent_x',
                                                                 'tangent_y',
                                                                 'tangent_z']].values
    parent_tangents = swc.set_index('node_id').loc[twigs.parent_id, ['tangent_x',
                                                                     'tangent_y',
                                                                     'tangent_z']].values

    # Generate dotproducts
    dot = np.einsum('ij,ij->i', twig_tangents, parent_tangents)

    # Basically we want to drop any twig for which the dotproduct is close to 1
    dot_diff = 1 - np.fabs(dot)
    # Remove twigs where the dotproduct is within `theta` to 1
    to_remove = twigs.loc[dot_diff <= theta]
    swc = swc[~swc.node_id.isin(to_remove.node_id)]

    # Drop the tangent columns we made
    return swc.drop(['tangent_x', 'tangent_y', 'tangent_z'], axis=1)
