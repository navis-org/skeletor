#    This script is part of skeletor (http://www.github.com/navis-org/skeletor).
#    Copyright (C) 2018 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.

"""
The `skeletor.pre` module contains functions to pre-process meshes before
skeletonization.

#### Fixing faulty meshes

Some skeletonization methods are susceptible to faulty meshes (degenerate faces,
wrong normals, etc.). If your skeleton looks off, it might be worth a shot
trying to fix the mesh using `skeletor.pre.fix_mesh()`.

#### Mesh contraction

As a rule of thumb: the more your mesh looks like a skeleton, the easier it is
to extract one (duh). Mesh contraction using `skeletor.pre.contract()` [1] can
help you to get your mesh "in shape".

References
----------
[1] Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh
    contraction. ACM Transactions on Graphics (TOG). 2008 Aug 1;27(3):44.

"""

from .meshcontraction import contract
from .preprocessing import fix_mesh, simplify, remesh

__docformat__ = "numpy"
__all__ = ['fix_mesh', 'simplify', 'remesh', 'contract']
