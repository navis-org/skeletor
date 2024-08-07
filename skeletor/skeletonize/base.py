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
        if hasattr(self, "vertices"):
            elements.append(f"vertices={self.vertices.shape}")
        if hasattr(self, "edges"):
            elements.append(f"edges={self.edges.shape}")
        if hasattr(self, "method"):
            elements.append(f"method={self.method}")
        return f'<Skeleton({", ".join(elements)})>'

    @property
    def edges(self):
        """Return skeleton edges."""
        return self.swc.loc[self.swc.parent_id >= 0, ["node_id", "parent_id"]].values

    @property
    def vertices(self):
        """Return skeleton vertices (nodes)."""
        return self.swc[["x", "y", "z"]].values

    @property
    def radius(self):
        """Return radii."""
        if "radius" not in self.swc.columns:
            raise ValueError(
                "No radius info found. Run `skeletor.post.radii()`" " to get them."
            )
        return self.swc["radius"].values

    @property
    def skeleton(self):
        """Skeleton as trimesh Path3D."""
        if not hasattr(self, "_skeleton"):
            lines = [tm.path.entities.Line(e) for e in self.edges]

            self._skeleton = tm.path.Path3D(
                entities=lines, vertices=self.vertices, process=False
            )
        return self._skeleton

    @property
    def skel_map(self):
        """Skeleton vertex (nodes) to mesh vertices. Based on `mesh_map`."""
        if isinstance(self.mesh_map, type(None)):
            return None
        return (
            pd.DataFrame(self.mesh_map)
            .reset_index(drop=False)
            .groupby(0)["index"]
            .apply(np.array)
            .values
        )

    @property
    def roots(self):
        """Root node(s)."""
        return self.swc.loc[self.swc.parent_id < 0].index.values

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

    def reroot(self, new_root):
        """Reroot the skeleton.

        Parameters
        ----------
        new_root :  int
                    Index of node to use as new root. If the skeleton
                    consists of multiple trees, only the tree containing the
                    new root will be updated.

        Returns
        -------
        Skeleton
                    Skeleton with new root.

        """
        assert new_root in self.swc.index.values, f"Node index {new_root} not in skeleton."

        # Make copy of self
        x = self.copy()

        # Check if the new root is already a root
        if new_root in x.roots:
            return x

        # Get graph representation
        G = x.get_graph()

        # Get the path from the new root to the current root (of the same tree)
        for r in x.roots:
            try:
                path = nx.shortest_path(G, source=new_root, target=r)
                break
            except nx.NetworkXNoPath:
                continue

        # Now we need to invert the path from the old root to the new root
        new_parents = x.swc.set_index('node_id').parent_id.to_dict()
        new_parents.update({c: p for p, c in zip(path[:-1], path[1:])})
        new_parents[new_root] = -1

        # Update the SWC table
        x.swc["parent_id"] = x.swc.node_id.map(new_parents)

        return x

    def copy(self):
        """Return copy of the skeleton."""
        return Skeleton(
            swc=self.swc.copy() if not isinstance(self.swc, type(None)) else None,
            mesh=self.mesh.copy() if not isinstance(self.mesh, type(None)) else None,
            mesh_map=self.mesh_map.copy()
            if not isinstance(self.mesh_map, type(None))
            else None,
            method=self.method,
        )

    def get_graph(self):
        """Generate networkX representation of the skeletons.

        Distance between nodes will be used as edge weights.

        Returns
        -------
        networkx.DiGraph

        """
        not_root = self.swc.parent_id >= 0
        nodes = self.swc.loc[not_root]
        parents = self.swc.set_index("node_id").loc[
            self.swc.loc[not_root, "parent_id"].values
        ]

        dists = nodes[["x", "y", "z"]].values - parents[["x", "y", "z"]].values
        dists = np.sqrt((dists**2).sum(axis=1))

        G = nx.DiGraph()
        G.add_nodes_from(self.swc.node_id.values)
        G.add_weighted_edges_from(
            zip(nodes.node_id.values, nodes.parent_id.values, dists)
        )

        return G

    def get_segments(self, weight="weight", return_lengths=False):
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
        assert weight in ("weight", None), f'Unable to use weight "{weight}"'

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
        swc["label"] = 0
        swc.loc[~swc.node_id.isin(swc.parent_id.values), "label"] = 6
        n_childs = swc.groupby("parent_id").size()
        bp = n_childs[n_childs > 1].index.values
        swc.loc[swc.node_id.isin(bp), "label"] = 5

        # Add radius if missing
        if "radius" not in swc.columns:
            swc["radius"] = 0

        # Get things in order
        swc = swc[["node_id", "label", "x", "y", "z", "radius", "parent_id"]]

        # Adjust column titles
        swc.columns = ["PointNo", "Label", "X", "Y", "Z", "Radius", "Parent"]

        with open(filepath, "w") as file:
            # Write header
            file.write(header)

            # Write data
            writer = csv.writer(file, delimiter=" ")
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
                raise ValueError("Skeleton has no mesh.")

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

    def mend_breaks(self, dist_mult=5, dist_min=0, dist_max=np.inf):
        """Mend breaks in the skeleton using the original mesh.

        This works by comparing the connectivity of the original mesh with that
        of the skeleton. If the shortest path between two adjacent vertices on the mesh
        is shorter than the distance between the nodes in the skeleton, a new edge
        is added to the skeleton.

        Parameters
        ----------
        dist_mult : float, optional
                    Factor by which the new edge should be shorter than the
                    current shortest path between two nodes to be added.
                    Lower values = fewer false negatives; higher values = fewer
                    false positive edges.
        dist_min :  float, optional
                    Minimum distance between nodes to consider adding an edge.
                    Use this to avoid adding very short edges.
        dist_max :  float, optional
                    Maximum distance between nodes to consider adding an edge.
                    Use this to avoid adding very long edges.

        Returns
        -------
        edges :     (N, 2) array
                    Edges connecting the skeleton nodes.
        vertices :  (N, 3) array
                    Positions of the skeleton nodes.

        """
        # We need `.mesh_map` and `.mesh` to exist
        if self.mesh_map is None:
            raise ValueError("Skeleton must have a `mesh_map` to mend breaks.")
        if self.mesh is None:
            raise ValueError("Skeleton must have a `mesh` to mend breaks.")

        # Make a copy of the mesh edges
        edges = self.mesh.edges.copy()
        # Map mesh vertices to skeleton vertices
        edges[:, 0] = self.mesh_map[self.mesh.edges[:, 0]]
        edges[:, 1] = self.mesh_map[self.mesh.edges[:, 1]]
        # Deduplicate
        edges = np.unique(edges, axis=0)
        # Remove self edges
        edges = edges[edges[:, 0] != edges[:, 1]]

        G = self.get_graph().to_undirected()

        # Remove edges that are already in the skeleton
        edges = np.array([e for e in edges if not G.has_edge(*e)])

        # Calculate distance between these new edge candidates
        dists = np.sqrt(
            ((self.vertices[edges[:, 0]] - self.vertices[edges[:, 1]]) ** 2).sum(axis=1)
        )

        # Sort by distance (lowest first)
        edges = edges[np.argsort(dists)]
        dists = dists[np.argsort(dists)]

        for e, d in zip(edges, dists):
            # Check if the new path would be shorter than the current shortest path
            if (d * dist_mult) < nx.shortest_path_length(G, e[0], e[1]):
                continue
            # Check if the distance is within bounds
            elif d < dist_min:
                continue
            elif d > dist_max:
                continue
            # Add edge
            G.add_edge(*e, weight=d)

        # The above may have introduced small triangles which we should try to remove
        # by removing the longest edge in a triangle. I have also spotted more
        # complex cases of four or more nodes forming false-positive loops but
        # these will be harder to detect and remove.

        # First collect neighbors for each node
        later_nbrs = {}
        for node, neighbors in G.adjacency():
            later_nbrs[node] = {
                n for n in neighbors if n not in later_nbrs and n != node
            }

        # Go over each node
        triangles = set()
        for node1, neighbors in later_nbrs.items():
            # Go over each neighbor
            for node2 in neighbors:
                # Check if there is one or more third nodes that are connected to both
                third_nodes = neighbors & later_nbrs[node2]
                for node3 in third_nodes:
                    # Add triangle (sort to deduplicate)
                    triangles.add(tuple(sorted([node1, node2, node3])))

        # Remove longest edge in each triangle
        for t in triangles:
            e1, e2, e3 = t[:2], t[1:], t[::2]
            # Make sure all edges still exist (we may have already removed edges
            # that were part of a previous triangle)
            if any(not G.has_edge(*e) for e in (e1, e2, e3)):
                continue
            # Remove the longest edge
            G.remove_edge(*max((e1, e2, e3), key=lambda e: G.edges[e]["weight"]))

        return np.array(G.edges), self.vertices.copy()
