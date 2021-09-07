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

import itertools
import ncollpyde

import igraph as ig
import numpy as np
import scipy.sparse
import scipy.spatial
import trimesh as tm

from .base import Skeleton
from .utils import edges_to_graph, make_swc, reindex_swc

from ..utilities import make_trimesh

# Interesting references:
# - https://www.sciencedirect.com/science/article/pii/S037843710600464X
# - cool summary: https://hal.archives-ouvertes.fr/hal-01300281/file/3D_Skeletons_STAR.pdf

__all__ = ['by_tangent_ball']


def visualize_normals(mesh, le='auto', show=True):
    """Visualize vertex normals.

    Parameters
    ----------
    mesh :      trimesh.Trimesh
    le :        float
                Scale factor for length of normal vectors.
    show :      bool
                Whether to show the scene right away or just to return.

    Returns
    -------
    trimesh.Scene

    """
    mesh = make_trimesh(mesh, validate=False)

    # Make mesh semi-transparent
    mesh.visual.face_colors = [100, 100, 100, 100]

    # Generate the scene
    path = _make_normals(mesh, le=le)
    scene = tm.Scene([mesh.copy(), path])

    # I encountered some issues if object space is big and the easiest
    # way to work around this is to apply a transform such that the
    # coordinates have -5 to +5 bounds
    fac = 5 / mesh.bounds[1].max()
    scene.apply_transform(np.diag([fac, fac, fac, 1]))

    if show:
        return scene.show()
    else:
        return scene


def _make_normals(mesh, le='auto'):
    """Make path representing normals of the mesh."""
    if le == 'auto':
        le = mesh.edges_unique_length.mean()

    # Generate vertices
    vertices = np.append(mesh.vertices,
                         mesh.vertices + mesh.vertex_normals * le,
                         axis=0)

    # Generate edges
    edges = np.vstack((np.arange(mesh.vertices.shape[0]),
                       np.arange(mesh.vertices.shape[0]) + mesh.vertices.shape[0])).T

    # Put both together into individual normal lines and combine into path
    lines = [tm.path.entities.Line(e) for e in edges]
    path = tm.path.Path3D(entities=lines,
                          vertices=vertices,
                          process=False)

    return path


def _show_tangent_spheres(mesh, centers, radii, normals=False):
    """Visualize tangent spheres (debuggin)."""
    mesh = make_trimesh(mesh, validate=False)

    fac = 5 / mesh.bounds[1].max()

    mesh = mesh.copy()
    mesh.vertices *= fac

    spheres = []
    for c, r in zip(centers, radii):
        s = tm.primitives.Sphere(center=c * fac, radius=r * fac, subdivisions=0)
        spheres.append(s)

    # Make mesh semi-transparent
    mesh.visual.face_colors = [100, 100, 100, 100]

    # Generate the scene
    if not normals:
        scene = tm.Scene([mesh] + spheres)
    else:
        scene = tm.Scene([mesh, _make_normals(mesh)] + spheres)

    return scene.show()


def by_tangent_ball(mesh):
    """Skeletonize a mesh by finding the maximal tangent ball.

    This algorithm casts a ray from every mesh vertex along its inverse normals
    (requires `ncollpyde`). It then creates a sphere that is tangent to the
    vertex and to where the ray hit the inside of a face on the opposite side.
    Next it drops spheres that overlap with another, larger sphere. Modified
    from [1].

    The method works best on smooth meshes and is rather sensitive to errors in
    the mesh such as incorrect normals (see `skeletor.pre.fix_mesh`), internal
    faces, noisy surface (try smoothing or downsampling) or holes in the mesh.

    Parameters
    ----------
    mesh :              mesh obj
                        The mesh to be skeletonize. Can an object that has
                        ``.vertices`` and ``.faces`` properties  (e.g. a
                        trimesh.Trimesh) or a tuple ``(vertices, faces)`` or a
                        dictionary ``{'vertices': vertices, 'faces': faces}``.

    Returns
    -------
    skeletor.Skeleton
                        Holds results of the skeletonization and enables quick
                        visualization.

    Examples
    --------
    >>> import skeletor as sk
    >>> mesh = sk.example_mesh()
    >>> fixed = sk.pre.fix_mesh(mesh, fix_normals=True, remove_disconnected=10)
    >>> skel = sk.skeletonize.by_tangent_ball(fixed)

    References
    ----------
    [1] Ma, J., Bae, S.W. & Choi, S. 3D medial axis point approximation using
        nearest neighbors and the normal field. Vis Comput 28, 7–19 (2012).
        https://doi.org/10.1007/s00371-011-0594-7

    """
    mesh = make_trimesh(mesh, validate=False)

    # Generate the KD tree
    tree = scipy.spatial.cKDTree(mesh.vertices)

    dist = tree.query(mesh.vertices, k=2)[0][:, 1]

    centers = np.zeros(mesh.vertices.shape)
    radii = np.zeros(mesh.vertices.shape[0])

    coll = ncollpyde.Volume(mesh.vertices, mesh.faces, validate=False)
    sources = mesh.vertices - mesh.vertex_normals * 0.01
    targets = mesh.vertices - mesh.vertex_normals * (dist.max() * 10)
    ix, loc, is_backface = coll.intersections(sources, targets)

    # Now we need to invalidate centers
    intersects = np.zeros(mesh.vertices.shape[0]).astype(bool)
    intersects[ix[is_backface]] = True
    centers[ix] = mesh.vertices[ix] + (loc - mesh.vertices[ix]) / 2
    radii[ix] = np.linalg.norm(loc - mesh.vertices[ix], axis=1) / 2

    # Now we need to post processing
    inv = np.zeros(mesh.vertices.shape[0]).astype(bool)

    # Invalidate vertices that didn't intersect
    inv[~intersects] = True

    # Now invalidate any ball that is outside the mesh
    inv[~coll.contains(centers)] = True

    # Find tangent balls that are fully contained in another tangent ball
    # (those are not maximal inscribed)
    original_ind = np.arange(mesh.vertices.shape[0])
    while True:
        tree2 = scipy.spatial.cKDTree(centers[~inv])
        # For any not-yet-invalidated center find the closest other center
        dist, ix = tree2.query(centers[~inv], k=2)

        # Drop self-hits
        ix, dist = ix[:, 1], dist[:, 1]

        # In radius
        in_radius = dist < radii[~inv]

        # Stop if no more overlapping pairs
        if not in_radius.any():
            break

        # Collect radii to determine which of the overlapping ball survives
        pair_rad = np.vstack((radii[~inv][in_radius],
                              radii[~inv][ix[in_radius]])).T
        pair_ix = np.vstack((original_ind[~inv][in_radius],
                             original_ind[~inv][ix[in_radius]])).T

        # Invalidate the loosers
        looses = np.argmax(pair_rad, axis=1)
        looser_ix = np.unique(pair_ix[np.arange(pair_ix.shape[0]), looses])
        inv[looser_ix] = True

    # Now we need to collapse nodes into the remaining centers
    G = ig.Graph(n=mesh.vertices.shape[0],
                 edges=mesh.edges_unique,
                 directed=False)

    # Make sure that every connected component has at least one valid target
    for cc in G.clusters():
        if not np.isin(cc, original_ind[~inv]).any():
            inv[cc[0]] = False
            centers[cc[0]] = mesh.vertices[cc[0]]

    # For each invalidated vertex, find the closest vertex that is still valid
    # This works on unweighted edges but should be good enough - way faster
    # than a proper path search for sure
    pairs = find_closest(G, sources=original_ind[inv],
                         targets=original_ind[~inv])

    # Generate a mesh vertex to skeleton node map
    mesh_map = original_ind.copy()
    mesh_map[pairs[:, 0]] = pairs[:, 1]

    # Renumber the vertices from 0 -> N_vertices
    uni, ind, mesh_map = np.unique(mesh_map, return_inverse=True, return_index=True)

    # Make sure centers and radii match the new order
    centers = centers[uni]
    radii = radii[uni]

    # Contract vertices to nodes according to the mesh
    G.contract_vertices(mesh_map, combine_attrs=None)

    # This only drops duplicate and self-loop edges
    G = G.simplify()

    # Generate weights between remaining centers
    el = np.array(G.get_edgelist())
    weights = np.linalg.norm(centers[el[:, 0]] - centers[el[:, 1]], axis=1)

    # Generate hierarchical tree
    tree = G.spanning_tree(weights=weights)

    # Create a directed acyclic and hierarchical graph
    G_nx = edges_to_graph(edges=np.array(tree.get_edgelist()),
                          nodes=np.arange(0, len(G.vs)),
                          fix_tree=True,
                          drop_disconnected=False)

    # Generate the SWC table
    swc = make_swc(G_nx, coords=centers, reindex=False)
    swc['radius'] = radii[swc.node_id.values]
    _, new_ids = reindex_swc(swc, inplace=True)

    # Update vertex to node ID map
    mesh_map = np.array([new_ids[n] for n in mesh_map])

    return Skeleton(swc=swc, mesh=mesh, mesh_map=mesh_map,
                    method='tangent_ball')


def find_closest(G, sources, targets):
    """For each source find the closest node among targets.

    Parameters
    ----------
    G :                 igraph.Graph
                        Edge weights are ignored (i.e. we use unweighted edges).
    sources/targets :   iterable
                        Sources and targets to connect as vertex indices.

    Returns
    -------
    pairs :             (N, 2) numpy array
                        `[(source, target)]` pair. Note that source that are
                        disconnected from all targets will be omitted.

    """
    not_found = np.ones(len(sources)).astype(bool)

    dist = 1
    pairs = []
    while not_found.any():
        # Get neighbors
        neighbors = G.neighborhood(vertices=sources[not_found],
                                   order=dist, mindist=dist-1)

        # Turn into an array
        neigh_adj = np.array(list(itertools.zip_longest(*neighbors,
                                                        fillvalue=-1))).T

        # If no more neighbors stop
        # This happens if there are connected components containing no `targets`
        if not neigh_adj.shape[0]:
            break

        # Which of the neighbors is a target
        is_target = np.isin(neigh_adj, targets)
        # Which source has a target hit
        has_hit = np.max(is_target, axis=1)
        # All things being equal, pick the first hit among the targets
        hits = neigh_adj[np.arange(neigh_adj.shape[0])[has_hit],
                         np.argmax(is_target[has_hit], axis=1)]

        # Track source -> target pairs (IDs)
        pairs.append(np.vstack((sources[not_found][has_hit], hits)))

        # Tick off sources for which we found hits
        not_found[np.arange(sources.shape[0])[not_found][has_hit]] = False

        # Increase distance
        dist += 1

    # Combine pairs
    pairs = np.concatenate(pairs, axis=1).T

    return pairs
