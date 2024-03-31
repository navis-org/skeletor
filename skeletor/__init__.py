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
# What is skeletor?

Unlike its [namesake](https://en.wikipedia.org/wiki/Skeletor), this `skeletor`
does not (yet) seek to conquer Eternia but to turn meshes into skeletons.

Before we get started some terminology:

- a *mesh* is something that consists of *vertices* and *faces*
- a *skeleton* is a (hierarchical) tree-like structure consisting of *vertices*
  (also called *nodes*) and *edges* that connect them

Skeletons are useful for a range of reasons. For example:

1. Typically smaller (less vertices) than the mesh
2. Have an implicit sense of topology (e.g. "this node is distal to that node")

Extracting skeletons from meshes (or other types of data such as voxels) is
non-trivial and there are a great many research papers exploring various
different approaches (see [Google scholar](https://scholar.google.com/scholar
?hl=en&as_sdt=0%2C5&q=skeleton+extraction&btnG=)).

`skeletor` implements some algorithms that I found useful in my work with
neurons. In my experience there is unfortuntely no magic bullet when it
comes to skeletonization and chances are you will have to fiddle around a bit
to get decent results.

# Installation

From PyPI:
```bash
pip3 install skeletor
```

For the bleeding-edge version from Github:
```bash
pip3 install git+https://github.com/navis-org/skeletor@master
```

# Getting started

A skeletonization pipeline typically consists of:

1. Some pre-processing of the mesh (e.g. fixing some potential errors like
   degenerate faces, unreferenced vertices, etc.)
2. The skeletonization itself
3. Some post-processing of the skeleton (e.g. adding radius information, smoothing, etc.)

------

Here is a complete list of available functions:

| function                                    | description                                                 |
| ------------------------------------------- | ----------------------------------------------------------- |
| **example data**                            |                                                             |
| `skeletor.example_mesh()`                   | load an example mesh                                        |
| **pre-processing**                          |                                                             |
| `skeletor.pre.fix_mesh()`                   | fix some common errors found in meshes                      |
| `skeletor.pre.remesh()`                     | re-generate mesh (uses Blender 3D)                          |
| `skeletor.pre.simplify()`                   | reduce mesh complexity (uses Blender 3D)                    |
| `skeletor.pre.contract()`                   | contract mesh to facilitate skeletonization [1]             |
| **skeletonization**                         |                                                             |
| `skeletor.skeletonize.by_wavefront()`       | very fast, works well for tubular meshes (like neurons)     |
| `skeletor.skeletonize.by_vertex_clusters()` | very fast but needs mesh to be contracted (see above)       |
| `skeletor.skeletonize.by_edge_collapse()`   | presented in [1] but never got this to work well            |
| `skeletor.skeletonize.by_teasar()`          | very fast and robust, works on mesh surface                 |
| `skeletor.skeletonize.by_tangent_ball()`    | very fast, best on smooth meshes                            |
| **postprocessing**                          |                                                             |
| `skeletor.post.clean_up()`                  | fix some potential errors in the skeleton                   |
| `skeletor.post.radii()`                     | add radius information using various method                 |
| `skeletor.post.smooth()`                    | smooth the skeleton                                         |
| `skeletor.post.remove_bristles()`           | remove single-node bristles from the skeleton               |
| `skeletor.post.despike()`                   | smooth out jumps in the skeleton                            |

------

See docstrings of the respective functions for details.

A pipeline might look like this:

 1. `skeletor.pre.fix_mesh()` to fix the mesh
 2. `skeletor.pre.simplify()` to simplify the mesh
 3. `skeletor.pre.contract()` to contract the mesh [1]
 4. `skeletor.skeletonize.by_vertex_clusters()` to generate a skeleton
 5. `skeletor.post.clean_up()` to clean up some potential issues with the skeleton
 6. `skeletor.post.smooth()` to smooth the skeleton
 7. `skeletor.post.radii()` to extract radii either by k-nearest neighbours or ray-casting

In my experience there is no one-size-fits-all. You will have to play around to
find the right approach and parameters to get nice skeletons for your meshes.
If you need help just open an [issue](https://github.com/navis-org/skeletor/issues).

Also check out the Gotchas below!

# Examples

First load the example mesh (a fruit fly neuron):

```Python
>>> import skeletor as sk
>>> mesh = sk.example_mesh()
>>> mesh
<trimesh.Trimesh(vertices.shape=(6582, 3), faces.shape=(13772, 3))>
```

Next see if there is stuff to fix in the mesh (degenerate faces, duplicate
vertices, etc.):

```Python
>>> fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
>>> fixed
<trimesh.Trimesh(vertices.shape=(6213, 3), faces.shape=(12805, 3))>
```

Now for tubular meshes like this neuron, the "wave front" skeletonization method
performs really well: it works by casting waves across the mesh and collapsing
the resulting rings into a skeleton (kinda like when you throw a stone in a
pond and track the expanding ripples).

```Python
>>> skel = sk.skeletonize.by_wavefront(fixed, waves=1, step_size=1)
>>> skel
<Skeleton(vertices=(1258, 3), edges=(1194, 2), method=wavefront)>
```

All skeletonization methods return a `Skeleton` object. These are just
convenient objects to bundle the various outputs of the skeletonization.

```Python
>>> # x/y/z location of skeleton vertices (nodes)
>>> skel.vertices
array([[16744, 36720, 26407],
       ...,
       [22076, 23217, 24472]])
>>> # child -> parent edges
>>> skel.edges
array([[  64,   31],
       ...,
       [1257, 1252]])
>>> # Mapping for mesh to skeleton vertex indices
>>> skel.mesh_map
array([ 157,  158, 1062, ...,  525,  474,  547])
>>> # SWC table
>>> skel.swc.head()
   node_id  parent_id             x             y             z    radius
0        0         -1  16744.005859  36720.058594  26407.902344  0.000000
1        1         -1   5602.751953  22266.756510  15799.991211  7.542587
2        2         -1  16442.666667  14999.978516  10887.916016  5.333333
```

SWC is a commonly used format for saving skeletons. `Skeleton` objects
have a method for quickly saving a correctly formatted SWC file:

```Python
>>> skel.save_swc('~/Documents/my_skeleton.swc')
```

If you installed `pyglet` (see above) you can also use `trimesh`'s plotting
capabilities to inspect the results:

```Python
>>> skel.show(mesh=True)
```

<img src="https://github.com/navis-org/skeletor/raw/master/_static/example1.png" alt="skeletor_example" width="100%"/>

That looks pretty good already but let's run some pro-forma postprocessing.

```Python
>>> sk.post.clean_up(skel, inplace=True)
<Skeleton(vertices=(1071, 3), edges=(1070, 2))>
```

So that would be a full pipeline mesh to skeleton. Don't expect your own meshes
to produce such nice results off the bat though. Chances are you will need to
play around to find the right recipe. If you don't know where to start, I suggest
you try out mesh contraction + vertex clustering first:

```Python
>>> import skeletor as sk
>>> # Load the example mesh that ships with skeletor
>>> mesh = sk.example_mesh()
>>> # Alternatively use trimesh to load/construct your own mesh:
>>> # import trimesh as tm
>>> # mesh = tm.Trimesh(vertices, faces)
>>> # mesh = tm.load_mesh('some/mesh.obj')
>>> # Run some general clean-up (see docstring for details)
>>> fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
>>> # Contract mesh to 10% (0.1) of original volume
>>> cont = sk.pre.contract(fixed, epsilon=0.1)
>>> # Skeletonize
>>> skel = sk.skeletonize.by_vertex_clusters(cont, sampling_dist=100)
>>> # Replace contracted mesh with original for postprocessing and plotting
>>> skel.mesh = fixed
>>> # Add radii (vertex cluster method does not do that automatically)
>>> sk.post.radii(skel, method='knn')
>>> skel.show(mesh=True)
```

# Gotchas

- while this is a general purpose library, my personal focus is on neurons and
  this has certainly influenced things like default parameter values and certain
  post-processing steps
- meshes need to be triangular (we are using `trimesh`)
- use `sk.pre.simplify` if your mesh is very complex (half a million vertices is
  where things start getting sluggish)
- a good mesh contraction is often half the battle but it can be tricky to get
  to work
- if the mesh consists of multiple disconnected pieces the skeleton will
  likewise be fragmented (i.e. will have multiple roots)
- it's often a good idea to fix issues with the skeleton in postprocessing rather
  than trying to get the skeletonization to be perfect

# Benchmarks

<img src="https://github.com/navis-org/skeletor/raw/master/benchmarks/benchmark_2.png" alt="skeletor_benchmark" width="100%"/>

[Benchmarks](https://github.com/navis-org/skeletor/blob/master/benchmarks/skeletor_benchmark.ipynb)
were run on a 2018 MacBook Pro (2.2 GHz Core i7, 32Gb memory) with optional
`fastremap` dependency installed. Note some of these functions (e.g.
contraction and TEASAR/vertex cluster skeletonization) vary a lot in
speed based on parameterization.

# What about algorithm `X`?

`skeletor` contains some algorithms that I found easy enough to implement
and useful for my work with neurons. If you have some interesting paper/approach
that could make a nice addition to `skeletor`, please get in touch on Github.
Pull requests are always welcome!

# References

`[1] Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh contraction. ACM Transactions on Graphics (TOG). 2008 Aug 1;27(3):44.`

The abstract and the paper can be found [here](http://visgraph.cse.ust.hk/projects/skeleton/).
Also see [this](https://www.youtube.com/watch?v=-H7n59YQCRM&feature=youtu.be) YouTube video.

Some of the code in skeletor was modified from the
[Py_BL_MeshSkeletonization](https://github.com/aalavandhaann/Py_BL_MeshSkeletonization)
addon created by #0K Srinivasan Ramachandran and published under GPL3.

# Top-level functions and classes
At top-level we only expose `example_mesh()` and the `Skeleton` class (which
you probably won't ever need to touch manually). Everything else is neatly
tucked away into submodules (see side-bar or above table).

"""

__version__ = "1.3.0"
__version_vector__ = (1, 3, 0)

from . import skeletonize
from . import pre
from . import post

from .skeletonize.base import Skeleton
from .data import example_mesh

__docformat__ = "numpy"

__all__ = ['Skeleton', 'example_mesh', 'pre', 'post', 'skeletonize']
