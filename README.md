# Mesh Skeletonization
This is a Python 3 implementation of Skeleton Extraction by Mesh contraction published here:

`Au OK, Tai CL, Chu HK, Cohen-Or D, Lee TY. Skeleton extraction by mesh contraction. ACM Transactions on Graphics (TOG). 2008 Aug 1;27(3):44.`

The abstract and the paper can be found [here](http://visgraph.cse.ust.hk/projects/skeleton/).
Also see [this](https://www.youtube.com/watch?v=-H7n59YQCRM&feature=youtu.be) YouTube video.

Code modified from the
[Py_BL_MeshSkeletonization](https://github.com/aalavandhaann/Py_BL_MeshSkeletonization)
addon created by #0K Srinivasan Ramachandran and published under GPL3.

## Install

`pip3 install git+git://github.com/schlegelp/skeletonizer@master`

#### Dependencies
Automatically installed with `pip`:
- numpy
- scipy
- trimesh

### Usage

``` Python
import skeletonizer

```

## Notes
- meshes need to be triangular


## TO-DO:
- mesh contraction algorithm (done)
- skeletonization of contracted mesh