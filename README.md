# Mesh Skeletonization
This is a Python 3 implementation of Skeleton Extraction by Mesh contraction published here:

`Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh contraction. ACM Transactions on Graphics (TOG). 2008 Aug 1;27(3):44.`

The abstract and the paper can be found [here](http://visgraph.cse.ust.hk/projects/skeleton/).
Also see [this](https://www.youtube.com/watch?v=-H7n59YQCRM&feature=youtu.be) YouTube video.

Code modified from the
[Py_BL_MeshSkeletonization](https://github.com/aalavandhaann/Py_BL_MeshSkeletonization)
addon created by #0K Srinivasan Ramachandran and published under GPL3.

## Important
**Currently only the mesh contraction has been implemented!** (see example below)

## Install

`pip3 install git+git://github.com/schlegelp/skeletonizer@master`

#### Dependencies
Automatically installed with `pip`:
- numpy
- scipy
- trimesh

### Usage

``` Python
# Requires cloudvolume: pip install cloud-volume
from cloudvolume import CloudVolume
# Load a neuron from the Janelia Research Campus hemibrain connectome
# (see references)
vol = CloudVolume('precomputed://gs://neuroglancer-janelia-flyem-hemibrain/segmentation_52a13', fill_missing=True)
m = vol.mesh.get(5812983825, lod=2)[5812983825]

import skeletonizer as sk
# Contract the mesh
cont = sk.contract_mesh(m, iterations=3)
```

Full mesh in white, contract meshes in red.

![full-neuron](https://user-images.githubusercontent.com/7161148/84507953-89db4f80-acb9-11ea-8da0-b2e598a2bdb0.png "full") ![zoom1](https://user-images.githubusercontent.com/7161148/84507964-8c3da980-acb9-11ea-941a-c95a2328eabd.png "zoom1") ![zoom2](https://user-images.githubusercontent.com/7161148/84507966-8cd64000-acb9-11ea-98bd-87e140f6584e.png "zoom2")

## Notes
- meshes need to be triangular


## TO-DO:
- mesh contraction algorithm (done)
- skeletonization of contracted mesh

# References
