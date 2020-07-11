# Skeletor
Unlike its [namesake](https://en.wikipedia.org/wiki/Skeletor), this Python 3
library does not (yet) seek to conquer Eternia but to turn meshes into skeletons.

Most notably it currently implements:

 1. **mesh contraction** via `skeletor.contract` [1]
 2. **skeleton extraction** by _edge collapse_ [1] (slow and very accurate) or by _vertex clustering_ (very fast, less accurate) via `skeletor.skeletonize`

Some of the code was modified from the
[Py_BL_MeshSkeletonization](https://github.com/aalavandhaann/Py_BL_MeshSkeletonization)
addon created by #0K Srinivasan Ramachandran and published under GPL3.

## Install

`pip3 install git+git://github.com/schlegelp/skeletor@master`

#### Dependencies
Automatically installed with `pip`:
- `networkx`
- `numpy`
- `pandas`
- `scipy`
- `trimesh`
- `tqdm`

Optional but highly recommended:
- [fastremap](https://github.com/seung-lab/fastremap) for sizeable speed-ups: `pip3 install fastremap`

### Usage

```Python
# Requires cloudvolume: pip3 install cloud-volume
>>> from cloudvolume import CloudVolume
# Load a neuron from the Janelia Research Campus hemibrain connectome
>>> vol = CloudVolume('precomputed://gs://neuroglancer-janelia-flyem-hemibrain/segmentation_52a13', fill_missing=True)
# m is a trimesh.Trimesh
>>> m = vol.mesh.get(5812983825, lod=2)[5812983825]

>>> import skeletor as sk
# Contract the mesh
>>> cont = sk.contract(m, iterations=3)
# Extract the skeleton from the contracted mesh
>>> swc = sk.skeletonize(cont,  method='vertex_clusters', sampling_dist=50, output='swc')
>>> swc.head()
   node_id  parent_id             x             y             z  radius
0        1         30  16368.385248  34698.573907  27412.404067       1
1        2         32  16381.961369  36869.856299  25817.206930       1
2        3        162  16382.860533  36371.151954  26461.314717       1
3        4        526  16334.821842  15626.407053  11060.381910       1
4        5        145  16379.626005  36098.558696  24579.275974       1
```

For visualisation check out [navis](https://navis.readthedocs.io/en/latest/index.html):

```Python
>>> import navis
>>> neuron = navis.TreeNeuron(swc)
>>> mesh = navis.MeshNeuron(m)
>>> navis.plot3d([neuron, mesh], color=[(1, 0, 0), (1, 1, 1, .1)])
```

Full mesh in white, skeleton in red.

![full-neuron](https://user-images.githubusercontent.com/7161148/84507953-89db4f80-acb9-11ea-8da0-b2e598a2bdb0.png "full") ![zoom1](https://user-images.githubusercontent.com/7161148/84507964-8c3da980-acb9-11ea-941a-c95a2328eabd.png "zoom1") ![zoom2](https://user-images.githubusercontent.com/7161148/84507966-8cd64000-acb9-11ea-98bd-87e140f6584e.png "zoom2")

## Notes
- meshes need to be triangular

# References
`[1] Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh contraction. ACM Transactions on Graphics (TOG). 2008 Aug 1;27(3):44.`

The abstract and the paper can be found [here](http://visgraph.cse.ust.hk/projects/skeleton/).
Also see [this](https://www.youtube.com/watch?v=-H7n59YQCRM&feature=youtu.be) YouTube video.
