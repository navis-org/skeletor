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

import trimesh as tm


def make_trimesh(mesh, validate=True, **kwargs):
    """Construct ``trimesh.Trimesh`` from input data.

    Parameters
    ----------
    meshdata :      tuple | dict | mesh-like object
                    Tuple: (vertices, faces)
                    dict: {'vertices': [], 'faces': []}
                    mesh-like object: mesh.vertices, mesh.faces
    validate :      bool
                    If True, will try to fix potential issues with the mesh
                    (e.g. infinite values, duplicate vertices, degenerate faces).
    **kwargs
                    Keyword arguments are passed through to
                    `skeletor.pre.fix_mesh` if `validate=True`.

    Returns
    -------
    trimesh.Trimesh

    """
    from .pre import fix_mesh

    if isinstance(mesh, tm.Trimesh):
        pass
    elif isinstance(mesh, (tuple, list)):
        if len(mesh) == 2:
            mesh = tm.Trimesh(vertices=mesh[0],
                              faces=mesh[1],
                              process=validate)
    elif isinstance(mesh, dict):
        mesh = tm.Trimesh(vertices=mesh['vertices'],
                          faces=mesh['faces'],
                          process=validate)
    elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
        mesh = tm.Trimesh(vertices=mesh.vertices,
                          faces=mesh.faces,
                          process=validate)
    else:
        raise TypeError('Unable to construct a trimesh.Trimesh from object of '
                        f'type "{type(mesh)}"')

    if validate:
        mesh = fix_mesh(mesh, inplace=True, **kwargs)

    return mesh
