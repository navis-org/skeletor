# Skeletor
Unlike its [namesake](https://en.wikipedia.org/wiki/Skeletor), this Python 3
library does not (yet) seek to conquer Eternia but to turn meshes into skeletons.

_Heads-up: in preparation for skeletor `1.0.0` we are currently reorganizing/refactoring many functions and modules._

## Install

```bash
pip3 install skeletor
```

For the dev version:
```bash
pip3 install git+git://github.com/schlegelp/skeletor@master
```

#### Dependencies
Automatically installed with `pip`:
- `networkx`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `trimesh`
- `tqdm`
- `python-igraph`

Optional because not strictly required for the core functions but highly recommended:
- [pyglet](https://pypi.org/project/pyglet/) is required by trimesh to preview meshes/skeletons in 3D: `pip3 install pyglet`
- [fastremap](https://github.com/seung-lab/fastremap) for sizeable speed-ups: `pip3 install fastremap`
- [ncollpyde](https://github.com/clbarnes/ncollpyde) for ray-casting (radii, clean-up): `pip3 install ncollpyde`

## Usage

Skeletonization typically consists of (optional) preprocessing of the mesh,
skeletonization and (optional) postprocessing of the skeleton.

| preprocessing             | description                                        |
| --------------------------| -------------------------------------------------- |
| `skeletor.pre.fix_mesh()` | fix some common errors found in meshes             |
| `skeletor.pre.remesh()`   | re-generate mesh (uses Blender 3D)                 |
| `skeletor.pre.simplify()` | reduce mesh complexity (uses Blender 3D)           |
| `skeletor.pre.contract()` | contract mesh to facilitate skeletonization [1]    |

| skeletonization                             | description                                                 |
| --------------------------------------------| ----------------------------------------------------------- |
| `skeletor.skeletonize.by_wavefront()`       | very fast, works well for tubular meshes (like neurons)     |
| `skeletor.skeletonize.by_vertex_clusters()` | very fast but needs mesh to be contracted (see above)       |
| `skeletor.skeletonize.by_edge_collapse()`   | presented in [1] but never got this to work well            |
| `skeletor.skeletonize.by_teasar()`          | very fast and robust, works on mesh surface                 |
| `skeletor.skeletonize.by_tangent_ball()`    | coming soon                                                 |

| postprocessing             | description                                        |
| ---------------------------| -------------------------------------------------- |
| `skeletor.post.clean_up()` | fix some potential errors in the skeleton          |
| `skeletor.post.radii()`    | add radius information using various method        |

See docstrings of the respective functions for details and caveats.

A full pipeline might look like this:

 1. `skeletor.pre.fix_mesh()` to fix the mesh
 2. `skeletor.pre.simplify()` to simplify the mesh
 3. `skeletor.pre.contract()` to contract the mesh [1]
 4. `skeletor.skeletonize.vertex_clusters()` to generate a skeleton
 5. `skeletor.post.clean_up()` to clean up some potential issues with the skeleton
 6. `skeletor.post.radii()` to extract radii either by k-nearest neighbours or ray-casting

In my experience there is no one-size-fits-all. You will have to play around to
find the right approach and parameters to get nice skeletons for your meshes.
If you need help just open an [issue](https://github.com/schlegelp/skeletor/issues).

Also check out the Gotchas below!

### Example

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

Now for tubular meshes like this neuron, the "wave" skeletonization method
performs really well: it works by casting waves across the mesh (kind of like
ripples in a pond) and collapsing the resulting rings into a skeleton.

```Python
>>> skel = sk.skeletonize.by_wavefront(fixed, waves=1, step_size=1)
>>> skel
<Skeleton(vertices=(1096, 3), edges=(1095, 2))>
```

All skeletonization methods return a `Skeleton` object. These are just
convenient objects to represent and inspect the results.

```Python
>>> # location of vertices (nodes)
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

If you installed `pyglet` (see above) you can also use `trimesh`'s plotting
capabilities to inspect the results:

```Python
>>> skel.show(mesh=True)
```

![skeletor_example](https://github.com/schlegelp/skeletor/raw/master/_static/example1.png)

That looks pretty good already but let's run some pro forma postprocessing.

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
>>> mesh = sk.example_mesh()
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

### Random Gotchas
- while this is a general purpose library, my personal focus is on neurons and
  this has certainly influenced things like default parameter values and certain
  post-processing steps
- meshes need to be triangular (we are using `trimesh`)
- use `sk.pre.simplify` if your mesh is very complex (>1e5 vertices)
- a good mesh contraction is often half the battle  
- if the mesh consists of multiple disconnected pieces the skeleton will
  likewise be fragmented (i.e. will have multiple roots)

## Benchmarks
![skeletor_examples](https://github.com/schlegelp/skeletor/raw/master/benchmarks/benchmark_2.png)

[Benchmarks](https://github.com/schlegelp/skeletor/blob/master/benchmarks/skeletor_benchmark.ipynb)
were run on a 2018 MacBook Pro (2.2 GHz Core i7, 32Gb memory) with optional
`fastremap` dependency installed. Note that the contraction speed heavily
depends on the shape of the mesh - thin meshes like the neurons used here
take fewer steps. Likewise skeletonization using vertex clustering is very
dependant on the `sampling_dist` parameter.

## References
`[1] Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh contraction. ACM Transactions on Graphics (TOG). 2008 Aug 1;27(3):44.`

The abstract and the paper can be found [here](http://visgraph.cse.ust.hk/projects/skeleton/).
Also see [this](https://www.youtube.com/watch?v=-H7n59YQCRM&feature=youtu.be) YouTube video.

Some of the code in skeletor was modified from the
[Py_BL_MeshSkeletonization](https://github.com/aalavandhaann/Py_BL_MeshSkeletonization)
addon created by #0K Srinivasan Ramachandran and published under GPL3.
