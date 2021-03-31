# Skeletor
Unlike its [namesake](https://en.wikipedia.org/wiki/Skeletor), this Python 3
library does not (yet) seek to conquer Eternia but to turn meshes into skeletons.

A skeletonization typically consists of (optional) preprocessing of the mesh,
skeletonization and (optional) postprocessing of the skeleton.

_Heads-up: in preparation for skeletor `1.0.0` we are currently reorganizing/refactoring many functions and modules._

| preprocessing             | description                                        |
| --------------------------| -------------------------------------------------- |
| `skeletor.pre.fix_mesh()` | fix some common errors found in meshes             |
| `skeletor.pre.remesh()`   | re-generate mesh (uses Blender 3D)                 |
| `skeletor.pre.simplify()` | reduce mesh complexity (uses Blender 3D)           |
| `skeletor.pre.contract()` | contract mesh to facilitate skeletonization [1]    |

| skeletonization         | description                                                                     |
| --------------------------------------------| ----------------------------------------------------------- |
| `skeletor.skeletonize.by_vertex_clusters()` | very fast but needs mesh to be contracted                   |
| `skeletor.skeletonize.by_wavefront()`       | very fast, works well for tubular meshes (like neurons)     |
| `skeletor.skeletonize.by_edge_collapse()`   | presented in [1] but never got this to work well            |
| `skeletor.skeletonize.by_teasar()`          | coming soon                                                 |
| `skeletor.skeletonize.by_tangent_ball()`    | coming soon                                                 |

| postprocessing            | description                                        |
| --------------------------| -------------------------------------------------- |
| `skeletor.post.cleanup()` | fix some potential errors in the skeleton          |
| `skeletor.post.radii()`   | add radius information using various method        |

See docstrings of the respective functions for details.

A full pipeline might look like this:

 1. `skeletor.pre.fix_mesh()` to fix the mesh
 2. `skeletor.pre.simplify()` to simplify the mesh
 3. `skeletor.pre.contract()` to contract the mesh [1]
 4. `skeletor.skeletonize.vertex_clusters()` to generate a skeleton
 5. `skeletor.post.clean()` to clean up some potential issues with the skeleton
 6. `skeletor.post.radii()` to extract radii either by k-nearest neighbours or ray-casting

In my experience there is no one size fits all. You will have to play around to find the right approach
and parameters to get nice skeletons for your meshes. If you need help just open an issue.

Also check out the Gotchas below!

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
- [fastremap](https://github.com/seung-lab/fastremap) for sizeable speed-ups: `pip3 install fastremap`
- [ncollpyde](https://github.com/clbarnes/ncollpyde) for ray-casting (radii, clean-up): `pip3 install ncollpyde`

## Usage

Fetch an example mesh:
```Python
# Requires cloudvolume: pip3 install cloud-volume
>>> from cloudvolume import CloudVolume
# Load a neuron from the Janelia Research Campus hemibrain connectome
>>> vol = CloudVolume('precomputed://gs://neuroglancer-janelia-flyem-hemibrain/segmentation_52a13', fill_missing=True)
# mesh is a trimesh.Trimesh
>>> mesh = vol.mesh.get(5812983825, lod=2)[5812983825]
>>> mesh
Mesh(vertices<79947>, faces<149224>, normals<0>, segid=None, encoding_type=<draco>)
```

Generate the skeleton
```Python
>>> import skeletor as sk
# Contract the mesh -> doesn't have to be perfect but you should aim for <10%
>>> cont = sk.pre.contract(mesh, iter_lim=4)
# Extract the skeleton from the contracted mesh
>>> swc = sk.skeletonize.by_vertex_clusters(cont, sampling_dist=50, output='swc')
# Clean up the skeleton
>>> swc = sk.post.cleanup(swc, mesh)
# Add/update radii
>>> swc['radius'] = sk.post.radii(swc, mesh, method='knn', n=5, aggregate='mean')
>>> swc.head()
   node_id  parent_id             x             y             z     radius
0        1        108  15171.575407  36698.832858  25797.208983  38.545553
1        2         75   5673.874254  21973.874094  15498.255429  79.262464
2        3        866  21668.461494  25084.044197  25855.263837  58.992209
3        4        212  16397.298583  35225.165481  24259.994014  20.213940
```

For visualisation check out [navis](https://navis.readthedocs.io/en/latest/index.html):

```Python
>>> import navis
>>> skeleton = navis.TreeNeuron(swc.copy(), units='8nm', soma=None)
>>> meshneuron = navis.MeshNeuron(mesh, units='8nm')
>>> navis.plot3d([skeleton, meshneuron], color=[(1, 0, 0), (1, 1, 1, .1)])
```

![skeletor_examples](https://user-images.githubusercontent.com/7161148/87663989-6eea7800-c75c-11ea-985d-058d22300b62.png)

## Benchmarks
![skeletor_examples](https://github.com/schlegelp/skeletor/raw/master/benchmarks/benchmark_2.png)

[Benchmarks](https://github.com/schlegelp/skeletor/blob/master/benchmarks/skeletor_benchmark.ipynb)
were run on a 2018 MacBook Pro (2.2 GHz Core i7, 32Gb memory) with optional
`fastremap` dependency installed. Note that the contraction speed heavily
depends on the shape of the mesh - thin meshes like the neurons used here
take fewer steps. Likewise skeletonization using vertex clustering is very
dependant on the `sampling_dist` parameter.

## Gotchas
- the mesh contraction is often the linchpin: insufficient/bad contraction will result
  in a sub-optimal skeleton
- to save time you should try to contract the mesh in as few steps as possible:
  try playing around with increasing the `SL` parameter - I've occasionally gone
  up as far a 1000 (from the default 10)
- if the contracted mesh looks funny (e.g. large spikes sticking out) try using
  the more robust "umbrella" Laplacian operator:
  `contract(mesh, operator='umbrella')`

### Additional Notes
- while this is a general purpose library, my personal focus is on neurons and
  this has certainly influenced things like default parameter values and certain
  post-processing steps
- meshes need to be triangular (we are using `trimesh`)
- if the mesh consists of multiple disconnected pieces the skeleton will
  likewise be fragmented (i.e. will have multiple roots)
- strictly speaking, the SWC [format](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html)
  requires continuous node IDs but the table generated by `skeletonize`
  currently uses the original vertex indices as node IDs (to facilitate e.g.
  mapping back and forth between mesh and SWC) and is therefore not continuous

## References
`[1] Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh contraction. ACM Transactions on Graphics (TOG). 2008 Aug 1;27(3):44.`

The abstract and the paper can be found [here](http://visgraph.cse.ust.hk/projects/skeleton/).
Also see [this](https://www.youtube.com/watch?v=-H7n59YQCRM&feature=youtu.be) YouTube video.

Some of the code in skeletor was modified from the
[Py_BL_MeshSkeletonization](https://github.com/aalavandhaann/Py_BL_MeshSkeletonization)
addon created by #0K Srinivasan Ramachandran and published under GPL3.
