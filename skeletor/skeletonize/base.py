#    This script is part of skeletor (http://www.github.com/navis-org/skeletor).
#    Copyright (C) 2018 Philipp Schlegel
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

import csv
import datetime

import numpy as np
import pandas as pd
import trimesh as tm
import networkx as nx

from textwrap import dedent

from .utils import reindex_swc

__docformat__ = "numpy"


class Skeleton:
    """Class representing a skeleton.

    Typically returned as results from a skeletonization.

    Attributes
    ----------
    swc :       pd.DataFrame, optional
                SWC table.
    vertices :  (N, 3) array
                Vertex (node) positions.
    edges :     (M, 2) array
                Indices of connected vertex pairs.
    radii :     (N, ) array, optional
                Radii for each vertex (node) in the skeleton.
    mesh :      trimesh, optional
                The original mesh.
    mesh_map :  array, optional
                Same length as ``mesh``. Maps mesh vertices to vertices (nodes)
                in the skeleton.
    skel_map :  array of arrays, optional
                Inverse of `mesh_map`: maps skeleton vertices (nodes) to mesh
                vertices.
    method :    str, optional
                Which method was used to generate the skeleton.

    """

    def __init__(self, swc, mesh=None, mesh_map=None, method=None):
        self.swc = swc
        self.mesh = mesh
        self.mesh_map = mesh_map
        self.method = method

    def __str__(self):
        """Summary."""
        return self.__repr__()

    def __repr__(self):
        """Return quick summary of the skeleton's geometry."""
        elements = []
        if hasattr(self, 'vertices'):
            elements.append(f'vertices={self.vertices.shape}')
        if hasattr(self, 'edges'):
            elements.append(f'edges={self.edges.shape}')
        if hasattr(self, 'method'):
            elements.append(f'method={self.method}')
        return f'<Skeleton({", ".join(elements)})>'

    @property
    def edges(self):
        """Return skeleton edges."""
        return self.swc.loc[self.swc.parent_id >= 0,
                            ['node_id', 'parent_id']].values

    @property
    def vertices(self):
        """Return skeleton vertices (nodes)."""
        return self.swc[['x', 'y', 'z']].values

    @property
    def radius(self):
        """Return radii."""
        if 'radius' not in self.swc.columns:
            raise ValueError('No radius info found. Run `skeletor.post.radii()`'
                             ' to get them.')
        return self.swc['radius'].values,

    @property
    def skeleton(self):
        """Skeleton as trimesh Path3D."""
        if not hasattr(self, '_skeleton'):
            lines = [tm.path.entities.Line(e) for e in self.edges]

            self._skeleton = tm.path.Path3D(entities=lines,
                                            vertices=self.vertices,
                                            process=False)
        return self._skeleton

    @property
    def skel_map(self):
        """Skeleton vertex (nodes) to mesh vertices. Based on `mesh_map`."""
        if isinstance(self.mesh_map, type(None)):
            return None
        return pd.DataFrame(self.mesh_map
                            ).reset_index(drop=False
                                          ).groupby(0)['index'].apply(np.array
                                                                      ).values

    @property
    def leafs(self):
        """Leaf nodes (includes root)."""
        swc = self.swc
        leafs = swc[~swc.node_id.isin(swc.parent_id.values) | (swc.parent_id < 0)]
        return leafs.copy()

    def reindex(self, inplace=False):
        """Clean up skeleton."""
        x = self
        if not inplace:
            x = x.copy()

        # Re-index to make node IDs continous again
        x.swc, new_ids = reindex_swc(x.swc)

        # Update mesh map
        if not isinstance(x.mesh_map, type(None)):
            x.mesh_map = np.array([new_ids.get(i, i) for i in x.mesh_map])

        if not inplace:
            return x

    def copy(self):
        """Return copy of the skeleton."""
        return Skeleton(swc=self.swc.copy() if not isinstance(self.swc, type(None)) else None,
                        mesh=self.mesh.copy() if not isinstance(self.mesh, type(None)) else None,
                        mesh_map=self.mesh_map.copy() if not isinstance(self.mesh_map, type(None)) else None)

    def get_graph(self):
        """Generate networkX representation of the skeletons.

        Distance between nodes will be used as edge weights.

        Returns
        -------
        networkx.DiGraph

        """
        not_root = self.swc.parent_id >= 0
        nodes = self.swc.loc[not_root]
        parents = self.swc.set_index('node_id').loc[self.swc.loc[not_root, 'parent_id'].values]

        dists = nodes[['x', 'y', 'z']].values - parents[['x', 'y', 'z']].values
        dists = np.sqrt((dists ** 2).sum(axis=1))

        G = nx.DiGraph()
        G.add_nodes_from(self.swc.node_id.values)
        G.add_weighted_edges_from(zip(nodes.node_id.values, nodes.parent_id.values, dists))

        return G

    def get_segments(self,
                     weight = 'weight',
                     return_lengths = False):
        """Generate a list of linear segments while maximizing segment lengths.

        Parameters
        ----------
        weight :    'weight' | None, optional
                    If ``"weight"`` use physical, geodesic length to determine
                    segment length. If ``None`` use number of nodes (faster).
        return_lengths : bool
                    If True, also return lengths of segments according to ``weight``.

        Returns
        -------
        segments :  list
                    Segments as list of lists containing node IDs. List is
                    sorted by segment lengths.
        lengths :   list
                    Length for each segment according to ``weight``. Only provided
                    if `return_lengths` is True.

        """
        assert weight in ('weight', None), f'Unable to use weight "{weight}"'

        # Get graph representation
        G = self.get_graph()

        # Get distances to root
        dists = {}
        for root in self.swc[self.swc.parent_id < 0].node_id.values:
            dists.update(nx.shortest_path_length(G, target=root, weight=weight))

        # Sort leaf nodes
        endNodeIDs = self.leafs[self.leafs.parent_id >= 0].node_id.values
        endNodeIDs = sorted(endNodeIDs, key=lambda x: dists.get(x, 0), reverse=True)

        seen: set = set()
        sequences = []
        for nodeID in endNodeIDs:
            sequence = [nodeID]
            parents = list(G.successors(nodeID))
            while True:
                if not parents:
                    break
                parentID = parents[0]
                sequence.append(parentID)
                if parentID in seen:
                    break
                seen.add(parentID)
                parents = list(G.successors(parentID))

            if len(sequence) > 1:
                sequences.append(sequence)

        # Sort sequences by length
        lengths = [dists[s[0]] - dists[s[-1]] for s in sequences]
        sequences = [x for _, x in sorted(zip(lengths, sequences), reverse=True)]

        if return_lengths:
            return sequences, sorted(lengths, reverse=True)
        else:
            return sequences

    def save_swc(self, filepath):
        """Save skeleton in SWC format.

        Parameters
        ----------
        filepath :      path-like
                        Filepath to save SWC to.

        """
        header = dedent(f"""\
        # SWC format file
        # based on specifications at http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
        # Created on {datetime.date.today()} using skeletor (https://github.com/navis-org/skeletor)
        # PointNo Label X Y Z Radius Parent
        # Labels:
        # 0 = undefined, 1 = soma, 5 = fork point, 6 = end point
        """)

        # Make copy of SWC table
        swc = self.swc.copy()

        # Set all labels to undefined
        swc['label'] = 0
        swc.loc[~swc.node_id.isin(swc.parent_id.values), 'label'] = 6
        n_childs = swc.groupby('parent_id').size()
        bp = n_childs[n_childs > 1].index.values
        swc.loc[swc.node_id.isin(bp), 'label'] = 5

        # Add radius if missing
        if 'radius' not in swc.columns:
            swc['radius'] = 0

        # Get things in order
        swc = swc[['node_id', 'label', 'x', 'y', 'z', 'radius', 'parent_id']]

        # Adjust column titles
        swc.columns = ['PointNo', 'Label', 'X', 'Y', 'Z', 'Radius', 'Parent']

        with open(filepath, 'w') as file:
            # Write header
            file.write(header)

            # Write data
            writer = csv.writer(file, delimiter=' ')
            writer.writerows(swc.astype(str).values)

    def scene(self, mesh=False, **kwargs):
        """Return a Scene object containing the skeleton.

        Returns
        -------
        scene :     trimesh.scene.scene.Scene
                    Contains the skeleton and optionally the mesh.

        """
        if mesh:
            if isinstance(self.mesh, type(None)):
                raise ValueError('Skeleton has no mesh.')

            self.mesh.visual.face_colors = [100, 100, 100, 100]

            # Note the copy(): without it the transform in show() changes
            # the original meshes
            sc = tm.Scene([self.mesh.copy(), self.skeleton.copy()], **kwargs)
        else:
            sc = tm.Scene(self.skeleton.copy(), **kwargs)

        return sc

    def show(self, mesh=False, **kwargs):
        """Render the skeleton in an opengl window. Requires pyglet.

        Parameters
        ----------
        mesh :      bool
                    If True, will render transparent mesh on top of the
                    skeleton.

        Returns
        --------
        scene :     trimesh.scene.Scene
                    Scene with skeleton in it.

        """
        scene = self.scene(mesh=mesh)

        # I encountered some issues if object space is big and the easiest
        # way to work around this is to apply a transform such that the
        # coordinates have -5 to +5 bounds
        fac = 5 / np.fabs(self.skeleton.bounds).max()
        scene.apply_transform(np.diag([fac, fac, fac, 1]))

        return scene.show(**kwargs)
