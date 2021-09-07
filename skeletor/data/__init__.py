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
Example data
------------
At this point skeletor ships with a single example mesh: a neuron reconstructed
from the brain of a fruit fly. It was segmented from an EM image data set and is
part of the Janelia hemibrain data set ([link](https://neuprint.janelia.org)) [1].

References
----------

[1] Louis K. Scheffer et al., eLife. 2020. doi: 10.7554/eLife.57443
A connectome and analysis of the adult Drosophila central brain
"""

import os

import trimesh as tm

# Load the example mesh (a neuron)
fp = os.path.dirname(__file__)
obj_path = os.path.join(fp, '722817260.obj')

__docformat__ = "numpy"


def example_mesh():
    """Load and return example mesh.

    The example mesh is a fruit fly neuron (an olfactory projection neuron of
    the DA1 glomerulus) segmented from an EM image data set. It is part of the
    Janelia hemibrain data set (see [here](https://neuprint.janelia.org)) [1].

    References
    ----------
    [1] Louis K. Scheffer et al., eLife. 2020. doi: 10.7554/eLife.57443
    A connectome and analysis of the adult Drosophila central brain

    Returns
    -------
    trimesh.Trimesh

    Examples
    --------
    >>> import skeletor as sk
    >>> # Load this example mesh
    >>> mesh = sk.example_mesh()

    """
    return tm.load_mesh(obj_path)
