#    This script is part of skeletonizer (http://www.github.com/schlegelp/skeletonizer).
#    Copyright (C) 2018 Philipp Schlegel
#    Modified from https://github.com/aalavandhaann/Py_BL_MeshSkeletonization
#    by #0K Srinivasan Ramachandran.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.

import logging
import time

import trimesh as tm

from .meshcontraction import contract_mesh
from .skeleton import skeletonize, tree_from_mesh, edges_to_swc, add_radius

logger = logging.getLogger('skeletonizer')

if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


class Skeletonizer:
    """Skeletonization class.

    Parameters
    ----------
    mesh :              trimesh.Trimesh
                        Mesh to skeletonize.

    cnt_iterations :    int
                        Number of iterations for mesh contraction.
    cnt_SL :            float, optional
                        Factor by which the contraction matrix is multiplied
                        for each iteration.
    cnt_WC :            float, optional
                        Weight factor that affects the attraction constraint.

    sk_shape_weight :   float, optional
                        Weight for skeletonization shape costs which represent
                        impact of merging two nodes on the shape of the object.
    sk_sample_weight :  float, optional
                        Weight for skeletonization sampling costs which increase
                        if a merge would generate prohibitively long edges.

    radii :             "knn" | "ray" (TODO)
                        Whether and how to add radius information to nodes::

                            - "knn" uses k-nearest-neighbors to get radii: fast but potential for being wrong
                            - "ray" uses ray-casting to get radii: slow but "more right"

    progress :          bool
                        If False, will disable all progress bars.

    """

    def __init__(self,
                 mesh,
                 cnt_iterations=10, cnt_SL=10, cnt_WC=2,
                 sk_shape_weight=1, sk_sample_weight=0.1,
                 radii='knn',
                 progress=True):
        """Initialize class."""
        assert isinstance(mesh, tm.Trimesh)

        self.mesh = mesh

        # Parameters for mesh contraction
        self.cnt_iterations = cnt_iterations
        self.cnt_SL = cnt_SL
        self.cnt_WC = cnt_WC

        # Parameters for skeletonization
        self.sk_shape_weight = sk_shape_weight
        self.sk_sample_weight = sk_sample_weight

        self.radii = radii

        self.progress = progress

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """Return string representation."""
        contracted = hasattr(self, '_mesh_contraced')
        skeletonized = hasattr(self, 'swc')
        n_verts = len(self.mesh.vertices)
        n_faces = len(self.mesh.faces)

        return str(f'Skeletonizer ({n_verts} vertices; {n_faces} faces). '
                   f'Contracted: {contracted}; Skeletonized: {skeletonized}')

    @property
    def mesh_contracted(self):
        """Contracted mesh."""
        if not hasattr(self, '_mesh_contracted'):
            raise ValueError('Please run .contract() to generate the contracted mesh.')
        return self._mesh_contracted

    @mesh_contracted.setter
    def mesh_contracted(self, mesh):
        assert isinstance(mesh, tm.Trimesh)
        self._mesh_contracted = mesh

    def run(self):
        """Perform a full skeletonization run."""
        start = time.time()
        self.contract()
        self.skeletonize()
        logger.info(f'Run finished in {int(time.time() - start)}s. '
                    'Results stored as properties: `.swc`')

    def contract(self, **kwargs):
        """Contract mesh.

        Parameters
        ----------
        **kwargs
                 Keyword arguments superseed parametres set during
                 initialization and are passed to meshcontraction.contract_mesh().

        """
        params = dict(iterations=self.cnt_iterations,
                      SL=self.cnt_SL,
                      WC=self.cnt_WC,
                      progress=self.progress)
        params.update(kwargs)

        if params['iterations']:
            self.mesh_contracted = contract_mesh(self.mesh, **params)
        else:
            self.mesh_contracted = self.mesh.copy()

    def skeletonize(self, use_contracted=True, **kwargs):
        """Skeletonize mesh.

        Parameters
        ----------
        use_contracted :    bool
                            If False will use original mesh.
        **kwargs
                            Keyword arguments superseed parametres set during
                            initialization and are passed to skeleton.skeletonize().

        """
        mesh = self.mesh_contracted if use_contracted else self.mesh

        params = dict(shape_weight=self.sk_shape_weight,
                      sample_weight=self.sk_sample_weight,
                      progress=self.progress)
        params.update(kwargs)

        self.nodes = skeletonize(mesh, **params)
        self.edges = tree_from_mesh(self.nodes, mesh)
        self.swc = edges_to_swc(self.edges, mesh, reindex=True)

        # Radius always comes from the full mesh
        add_radius(self.swc, self.mesh, method=self.radii)
