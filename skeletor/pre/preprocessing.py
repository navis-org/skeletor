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

import os

import networkx as nx
import numpy as np
import scipy as sp
import trimesh as tm

from ..utilities import make_trimesh

try:
    import fastremap
except ImportError:
    fastremap = None
except BaseException:
    raise

__all__ = ['fix_mesh', 'simplify', 'remesh']


def fix_mesh(mesh, remote_infinite=True, merge_duplicate_verts=True,
             remove_degenerate_faces=True, remove_unreferenced_verts=True,
             drop_winglets=True, fix_normals=False, remove_disconnected=False,
             inplace=False):
    """Try to fix some common problems with mesh.

     1. Remove infinite values
     2. Merge duplicate vertices
     3. Remove duplicate and degenerate faces
     4. Remove unreference vertices
     5. Drop winglets (faces that have only one adjacent face)
     5. Fix normals (Optional)
     6. Remove disconnected fragments (Optional)

    Parameters
    ----------
    mesh :                  mesh-like object
                            Mesh to fix. Must have `.vertices` and `.faces`
                            properties.
    remove_disconnected :   False | int
                            If a number is given, will iterate over the mesh's
                            connected components and remove those consisting of
                            less than the given number of vertices. For example,
                            ``remove_disconnected=5`` will drop parts of the
                            mesh that consist of five or less connected
                            vertices.
    inplace :               bool
                            If True, will perform fixes on the input mesh.
                            If False, will make a copy first. This is silently
                            ignored if `mesh` is not already a trimesh.

    Returns
    -------
    fixed mesh :        trimesh.Trimesh

    """
    if not isinstance(mesh, tm.Trimesh):
        mesh = make_trimesh(mesh, validate=False)
    elif not inplace:
        mesh = mesh.copy()

    if remove_disconnected:
        to_drop = []
        for c in nx.connected_components(mesh.vertex_adjacency_graph):
            if len(c) <= remove_disconnected:
                to_drop += list(c)

        # Remove dropped vertices
        remove = np.isin(np.arange(mesh.vertices.shape[0]), to_drop)
        mesh.update_vertices(~remove)

    if remote_infinite:
        mesh.remove_infinite_values()

    if merge_duplicate_verts:
        mesh.merge_vertices()

    if remove_degenerate_faces:
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()

    if remove_unreferenced_verts:
        mesh.remove_unreferenced_vertices()

    if drop_winglets:
        mesh = remove_winglets(mesh)

    # This should be done after all clean-up operations
    if fix_normals:
        mesh.fix_normals()

    return mesh


def merge_vertices(mesh, dist='auto', inplace=False):
    """Merge vertices closer than a given distance.

    Parameters
    ----------
    mesh :      trimesh.Trimesh
                Mesh to merge vertices on.
    dist :      "auto" | number
                Distance at which to merge vertices. If "auto" will use
                ``mesh.edges_unique_length.mean() / 100``.
    inplace :   bool
                If True will modify the original mesh.

    Returns
    -------
    trimesh.Trimesh

    """
    assert isinstance(mesh, tm.Trimesh)

    if not inplace:
        mesh = mesh.copy()

    # Generate KDTree
    tree = sp.spatial.cKDTree(mesh.vertices)

    if dist == 'auto':
        dist = mesh.edges_unique_length.mean() / 100

    # Query tree
    pairs = tree.query_pairs(dist)

    # Facilitate remapping by removing extra steps: A->B->C to A->C, B->C
    G = nx.Graph()
    G.add_edges_from(pairs)
    mapping = {n: list(c)[0] for c in nx.connected_components(G) for n in list(c)[1:]}

    with mesh._cache:
        # Update faces
        if fastremap:
            mesh.faces = fastremap.remap(mesh.faces, mapping,
                                         preserve_missing_labels=True,
                                         in_place=True)
        else:
            for k, v in mapping.items():
                mesh.faces[mesh.faces == k] = v

    # Remove dropped vertices
    remove = np.isin(np.arange(mesh.vertices.shape[0]), list(mapping.keys()))
    mesh.update_vertices(~remove)

    # Remove degenerate and duplicate faces
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()

    # Fix normals
    mesh.fix_normals()

    return mesh


def remove_winglets(mesh):
    """Remove faces that have only one neighbor."""
    assert isinstance(mesh, tm.Trimesh)

    # Remove vertices that connected to only 2 or less other vertices
    vrem = np.arange(mesh.vertices.shape[0])[mesh.vertex_degree <= 2]

    if vrem.shape[0] == 0:
        return mesh

    # Get faces associated with these vertices
    frem = mesh.vertex_faces[vrem]

    # Faces that can be kept
    fkeep = np.arange(mesh.faces.shape[0])
    fkeep = fkeep[~np.isin(fkeep, frem)]

    return mesh.submesh([fkeep])[0]


def simplify(mesh, ratio):
    """Simplify mesh using Blender 3D.

    Uses Blender's "decimate" modifier in "collapse" mode.

    Parameters
    ----------
    mesh :  trimesh.Trimesh
            Mesh to simplify.
    ratio : float
            Factor to which to reduce faces. For example, a ratio of 0.5 will
            reduce the number of faces to 50%.

    Returns
    -------
    trimesh.Trimesh
            Simplified mesh.

    """
    if not tm.interfaces.blender.exists:
        raise ImportError('No Blender available (executable not found).')
    _blender_executable = tm.interfaces.blender._blender_executable

    assert ratio < 1 and ratio > 0, 'ratio must be between 0 and 1'

    # We need to import here to avoid circular imports
    from ..utilities import make_trimesh
    mesh = make_trimesh(mesh, validate=False)
    assert isinstance(mesh, tm.Trimesh)

    # Load the template
    temp_name = 'blender_decimate.py.template'
    if temp_name in _cache:
        template = _cache[temp_name]
    else:
        with open(os.path.join(_pwd, 'templates', temp_name), 'r') as f:
            template = f.read()
        _cache[temp_name] = template

    # Replace placeholder with actual ratio
    script = template.replace('$RATIO', str(ratio))

    # Let trimesh's MeshScript take care of exectution and clean-up
    with tm.interfaces.generic.MeshScript(meshes=[mesh],
                                          script=script,
                                          debug=False) as blend:
        result = blend.run(_blender_executable
                           + ' --background --python $SCRIPT')

    # Blender apparently returns actively incorrect face normals
    result.face_normals = None

    return result


def remesh(mesh, voxel_size=50, adaptivity=5):
    """Remesh mesh using Blender 3D.

    Uses Blender's "remesh" modifier in "voxel" mode.

    Parameters
    ----------
    mesh :  trimesh.Trimesh
            Mesh to remesh.
    voxel_size : float
            Size of individual voxels (edge length).
    adaptivity: float
            Reduces final face count where detail is not important.

    Returns
    -------
    trimesh.Trimesh
            Remeshed mesh.

    """
    if not tm.interfaces.blender.exists:
        raise ImportError('No Blender available (executable not found).')
    _blender_executable = tm.interfaces.blender._blender_executable

    assert voxel_size > 0, '`voxel_size` must be a positive number'
    assert adaptivity > 0, '`adaptivity` must be a positive number'

    # We need to import here to avoid circular imports
    from ..utilities import make_trimesh
    mesh = make_trimesh(mesh, validate=False)
    assert isinstance(mesh, tm.Trimesh)

    # Load the template
    temp_name = 'blender_remesh.py.template'
    if temp_name in _cache:
        template = _cache[temp_name]
    else:
        with open(os.path.join(_pwd, 'templates', temp_name), 'r') as f:
            template = f.read()
        _cache[temp_name] = template

    # Replace placeholder with actual ratio
    script = template.replace('$VOXEL_SIZE', str(voxel_size)
                              ).replace('$ADAPTIVITY', str(adaptivity))

    # Let trimesh's MeshScript take care of exectution and clean-up
    with tm.interfaces.generic.MeshScript(meshes=[mesh],
                                          script=script,
                                          debug=False) as blend:
        result = blend.run(_blender_executable
                           + ' --background --python $SCRIPT')

    # Blender apparently returns actively incorrect face normals
    result.face_normals = None

    return result


# find the current absolute path to this directory
_pwd = os.path.expanduser(os.path.abspath(os.path.dirname(__file__)))

# Use to cache templates
_cache = {}
