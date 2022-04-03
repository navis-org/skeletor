[![Generic badge](https://img.shields.io/badge/Github-pages-green)](https://navis-org.github.io/skeletor/) [![Tests](https://github.com/navis-org/skeletor/actions/workflows/test-package.yml/badge.svg)](https://github.com/navis-org/skeletor/actions/workflows/test-package.yml) [![DOI](https://zenodo.org/badge/153085435.svg)](https://zenodo.org/badge/latestdoi/153085435)

# Skeletor
Unlike its [namesake](https://en.wikipedia.org/wiki/Skeletor), this Python 3
library does not (yet) seek to conquer Eternia but to turn meshes into skeletons.

_Heads-up: skeletor `1.0.0` introduced some breaking changes and major reorganizations._
_Please see the [changelog](https://github.com/navis-org/skeletor/blob/master/NEWS.md)_
_for details._

## Install
```bash
pip3 install skeletor
```

For the dev version:
```bash
pip3 install git+https://github.com/navis-org/skeletor@master
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
- `ncollpyde`

Optional because not strictly required for the core functions but highly recommended:
- [pyglet](https://pypi.org/project/pyglet/) is required by trimesh to preview meshes/skeletons in 3D: `pip3 install pyglet`
- [fastremap](https://github.com/seung-lab/fastremap) for sizeable speed-ups with some methods: `pip3 install fastremap`

## Documentation
Please see the [documentation](https://navis-org.github.io/skeletor/) for details.

The changelog can be found [here](https://github.com/navis-org/skeletor/blob/master/NEWS.md).

## Quickstart
For the impatient a quick example:

```Python
>>> import skeletor as sk
>>> mesh = sk.example_mesh()
>>> fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
>>> skel = sk.skeletonize.by_wavefront(fixed, waves=1, step_size=1)
>>> skel
<Skeleton(vertices=(1258, 3), edges=(1194, 2), method=wavefront)>
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
>>> # Save SWC file
>>> skel.save_swc('skeleton.swc')
```

If you installed `pyglet` (see above) you can also use `trimesh`'s plotting
capabilities to inspect the results:

```Python
>>> skel.show(mesh=True)
```

![skeletor_example](https://github.com/navis-org/skeletor/raw/master/_static/example1.png)

## Benchmarks
![skeletor_examples](https://github.com/navis-org/skeletor/raw/master/benchmarks/benchmark_2.png)

[Benchmarks](https://github.com/navis-org/skeletor/blob/master/benchmarks/skeletor_benchmark.ipynb)
were run on a 2018 MacBook Pro (2.2 GHz Core i7, 32Gb memory) with optional
`fastremap` dependency installed. Note some of these functions (e.g.
contraction and TEASAR/vertex cluster skeletonization) can vary a lot in
speed based on parameterization.

## Contributing
Pull requests are always welcome!

## References & Acknowledgments
Mesh contraction and the edge collapse approach are based on this paper:
`[1] Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh contraction. ACM Transactions on Graphics (TOG). 2008 Aug 1;27(3):44.`
The abstract and the paper can be found [here](http://visgraph.cse.ust.hk/projects/skeleton/).
Also see [this](https://www.youtube.com/watch?v=-H7n59YQCRM&feature=youtu.be) YouTube video.

Some of the code in skeletor was modified from the
[Py_BL_MeshSkeletonization](https://github.com/aalavandhaann/Py_BL_MeshSkeletonization)
addon for Blender 3D created by #0K Srinivasan Ramachandran and published under GPL3.

The mesh TEASAR approach was adapted from the implementation in
[meshparty](https://github.com/sdorkenw/MeshParty) by Sven Dorkenwald, Casey
Schneider-Mizell and Forrest Collman.
