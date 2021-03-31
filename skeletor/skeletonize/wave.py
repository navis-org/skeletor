#    This script is part of skeletor (http://www.github.com/schlegelp/skeletor).
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

import igraph as ig
import numpy as np

from tqdm.auto import tqdm

from ..utilities import make_trimesh
from .utils import make_swc, edges_to_graph

__all__ = ['by_wavefront']


def by_wavefront(mesh, waves=1, step_size=1, output='swc',
                 drop_disconnected=False, progress=True):
    """Skeletonize a mesh using wave fronts.

    In a nutshell this tries to find rings of vertices and collapse them to
    their center. This is done by propagating a wave across the mesh starting at
    a single seed vertex. As the wave travels across the mesh we keep track of
    which vertices are are encountered at each step. Groups of connected
    vertices that are "hit" by the wave at the same time are considered rings
    and subsequently collapsed. By its nature this works best with tubular meshes.

    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to be skeletonize. Can an object that has
                    ``.vertices`` and ``.faces`` properties  (e.g. a
                    trimesh.Trimesh) or a tuple ``(vertices, faces)`` or a
                    dictionary ``{'vertices': vertices, 'faces': faces}``.
    waves :         int
                    Number of waves to run across the mesh. Each wave is
                    initialized at a different vertex which produces slightly
                    different rings. The final skeleton is produced from a mean
                    across all waves. More waves produce higher resolution
                    skeletons but also introduce more noise.
    step_size :     int
                    Values greater 1 effectively lead to binning of rings. For
                    example a stepsize of 2 means that two adjacent vertex rings
                    will be collapsed to the same center. This can help reduce
                    noise in the skeleton (and as such counteracts a large
                    number of waves).
    output :        "swc" | "graph" | "both"
                    Determines the function's output. See ``Returns``.
    drop_disconnected : bool
                    If True, will drop disconnected nodes from the skeleton.
                    Note that this might result in empty skeletons.
    progress :      bool
                    If True, will show progress bar.

    Returns
    -------
    "swc" :         pandas.DataFrame
                    SWC representation of the skeleton.
    "graph" :       networkx.Graph
                    Graph representation of the skeleton.
    "both" :        tuple
                    Both of the above: ``(swc, graph)``.

    """
    assert output in ['swc', 'graph', 'both']
    mesh = make_trimesh(mesh, validate=False)

    # Generate Graph (must be undirected)
    G = ig.Graph(edges=mesh.edges_unique, directed=False)
    #G.es['weight'] = mesh.edges_unique_length

    # Prepare empty array to fill with centers
    centers = np.full((mesh.vertices.shape[0], 3, waves), fill_value=np.nan)

    # Go over each connected component
    with tqdm(desc='Skeletonizing', total=len(G.vs), disable=not progress) as pbar:
        for cc in G.clusters():
            # Make a subgraph for this connected component
            SG = G.subgraph(cc)
            cc = np.array(cc)

            # Select seeds according to the number of waves
            seeds = np.linspace(0, len(cc) - 1,
                                min(waves, len(cc))).astype(int)

            # Get the distance between the seeds and all other nodes
            dist = np.array(SG.shortest_paths(source=seeds, target=None, mode='all'))

            if step_size > 1:
                mx = dist.flatten()
                mx = mx[mx < float('inf')].max()
                dist = np.digitize(dist, bins=np.arange(0, mx, step_size))

            # Go over each wave
            for w in range(dist.shape[0]):
                this_wave = dist[w, :]
                # Collect groups
                mx = this_wave[this_wave < float('inf')].max()
                for i in range(0, int(mx) + 1):
                    this_dist = this_wave == i
                    ix = np.where(this_dist)[0]
                    SG2 = SG.subgraph(ix)
                    for cc2 in SG2.clusters():
                        this_verts = cc[ix[cc2]]
                        this_center = mesh.vertices[this_verts].mean(axis=0)
                        centers[this_verts, :, w] = this_center

            pbar.update(len(cc))

    centers_final = np.nanmean(centers, axis=2)

    (node_centers,
     vertex_to_node_map) = np.unique(centers_final,
                                     return_inverse=True, axis=0)

    # Contract vertices
    G.contract_vertices(vertex_to_node_map)

    # Remove self loops and duplicate edges
    G.simplify()

    # Generate weights
    el = np.array(G.get_edgelist())
    weights = np.linalg.norm(node_centers[el[:, 0]] - node_centers[el[:, 1]], axis=1)

    # Generate hierarchical tree
    tree = G.spanning_tree(weights=weights)

    # Create a directed acyclic and hierarchical graph
    G_nx = edges_to_graph(np.array(tree.get_edgelist()), fix_tree=True,
                          drop_disconnected=drop_disconnected)

    # Make the SWC table
    if output == 'graph':
        return G_nx

    swc = make_swc(G_nx, coords=node_centers)

    if output == 'swc':
        return swc

    return swc, G_nx
