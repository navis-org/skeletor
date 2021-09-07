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

r"""
The `skeletor.skeletonize` module contains functions to for skeletonization
of meshes.

There are several approaches to skeletonizing a mesh. Which one to pick depends
(among other things) on the shape of your mesh and the skeleton quality you want
to get out of it. In general, unless you mesh already looks like a tube I
recommend looking into mesh contraction [^1].

Please see the documentation of the individual functions for details but here
is a quick summary:

| function                                    | speed | robust | radii [^2] | mesh map [^3] | description                                        |
| ------------------------------------------- | :---: | :----: | :--------: | :-----------: | ---------------------------------------------------|
| `skeletor.skeletonize.by_wavefront()`       | +++   | ++     | yes        | yes           | works well for tubular meshes                      |
| `skeletor.skeletonize.by_vertex_clusters()` | ++    | +      | no         | yes           | best with contracted meshes [^1]                   |
| `skeletor.skeletonize.by_teasar()`          | +     | ++     | no         | yes           | works on mesh surface                              |
| `skeletor.skeletonize.by_tangent_ball()`    | ++    | 0      | yes        | yes           | works with mesh normals                            |
| `skeletor.skeletonize.by_edge_collapse()`   | -     | 0      | no         | no            | published with [1] - never got this to work well   |

[^1]: use `skeletor.pre.contract()`
[^2]: radii can also be added in postprocessing with `skeletor.post.radii()`
[^3]: a mapping from the meshes vertices to skeleton nodes

## References

`[1] Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh contraction. ACM Transactions on Graphics (TOG). 2008 Aug 1;27(3):44.`

"""

from .edge_collapse import *
from .vertex_cluster import *
from .wave import *
from .teasar import *
from .tangent_ball import *

__docformat__ = "numpy"
__all__ = ['by_teasar', 'by_wavefront', 'by_vertex_clusters',
           'by_edge_collapse', 'by_tangent_ball']
