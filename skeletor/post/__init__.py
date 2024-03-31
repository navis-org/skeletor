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
The `skeletor.post` module contains functions to post-process skeletons after
skeletonization.

### Fixing issues with skeletons

Depending on your mesh, pre-processing and the parameters you chose for
skeletonization, chances are that your skeleton will not come out perfectly.

`skeletor.post.clean_up` can help you solve some potential issues:

- skeleton nodes (vertices) that outside or right on the surface instead of
  centered inside the mesh
- superfluous "hairs" on otherwise straight bits

`skeletor.post.smooth` will smooth out the skeleton.

`skeletor.post.despike` can help you remove spikes in the skeleton where
single nodes are out of aligment.

`skeletor.post.remove_bristles` will remove bristles from the skeleton.

### Computing radius information

Only `skeletor.skeletonize.by_wavefront()` provides radii off the bat. For all
other methods, you might want to run `skeletor.post.radii` can help you
(re-)generate radius information for the skeletons.

"""

from .radiusextraction import radii
from .postprocessing import clean_up, smooth, despike, remove_bristles

__docformat__ = "numpy"
__all__ = ["radii", "clean_up", "smooth", "despike", "remove_bristles"]
