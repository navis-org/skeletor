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

import igraph as ig
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from itertools import combinations

from ..utilities import make_trimesh
from .base import Skeleton

try:
    import fastremap
except ModuleNotFoundError:
    fastremap = None


__all__ = ["by_wavefront_exact"]


def by_wavefront_exact(mesh, step_size, origins=None, radius_agg="mean", progress=True):
    """Skeletonize a mesh using wave fronts with an exact step size.

    This is a version of the `by_wavefront` function in which the wave front
    moves _exactly_ the given distance along the mesh (see `step_size` parameter) instead
    of hopping from vertex to vertex. This can give better results on meshes with a
    low vertex density but is computationally more expensive: the difference between
    the two methods depends heavily on the mesh and the step size, but as a rule of thumb,
    the exact version is about 3-4x slower than the non-exact version.

    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to be skeletonize. Can an object that has
                    ``.vertices`` and ``.faces`` properties  (e.g. a
                    trimesh.Trimesh) or a tuple ``(vertices, faces)`` or a
                    dictionary ``{'vertices': vertices, 'faces': faces}``.
    step_size :     float | int (>0)
                    The distance each the wave fronts move along the mesh
                    in each step. To get a feel for what a good value might be,
                    you can check the average edge length of the mesh
                    (see example).
    origins :       int | list of ints, optional
                    Vertex ID(s) where the wave(s) are initialized. If there
                    is no origin for a given connected component, will fall
                    back to semi-random origin.
    radius_agg :    "mean" | "median" | "max" | "min" | "percentile75" | "percentile25"
                    Function used to aggregate radii over sample (i.e. the
                    vertices forming a ring that we collapse to its center).
    progress :      bool
                    If True, will show progress bar.

    See Also
    --------
    `skeletor.skeletonize.by_wavefront()`
                    This is the original `by_wavefront` function in which the wave front
                    jumps from vertex to vertex. This is much faster but can give bad
                    results on low-resolution meshes.

    Returns
    -------
    skeletor.Skeleton
                    Holds results of the skeletonization and enables quick
                    visualization.

    Example
    -------
    >>> import skeletor as sk
    >>> mesh = sk.example_mesh()
    >>> # Check the average edge length
    >>> mesh.edges_unique_length.mean()
    119.325
    >>> # A good step size is usually a good bit smaller than the average edge length
    >>> skeleton = sk.skeletonize.by_wavefront_exact(mesh, step_size=50)

    """
    agg_map = {
        "mean": np.mean,
        "max": np.max,
        "min": np.min,
        "median": np.median,
        "percentile75": lambda x: np.percentile(x, 75),
        "percentile25": lambda x: np.percentile(x, 25),
    }
    assert radius_agg in agg_map, f'Unknown `radius_agg`: "{radius_agg}"'
    rad_agg_func = agg_map[radius_agg]

    mesh = make_trimesh(mesh, validate=False)

    centers_final, radii_final, parents = _cast_waves(
        mesh,
        step_size=step_size,
        origins=origins,
        rad_agg_func=rad_agg_func,
        progress=progress,
    )

    # Initialize the SWC dataframe
    swc = pd.DataFrame()
    swc["node_id"] = np.arange(0, len(centers_final))
    swc["parent_id"] = parents
    swc["x"] = centers_final[:, 0]
    swc["y"] = centers_final[:, 1]
    swc["z"] = centers_final[:, 2]
    swc["radius"] = radii_final

    return Skeleton(swc=swc, mesh=mesh, method="wavefront_exact")


def _cast_waves(mesh, step_size, origins=None, rad_agg_func=np.mean, progress=True):
    """Cast waves across mesh."""
    if not isinstance(origins, type(None)):
        if isinstance(origins, int):
            origins = [origins]
        elif not isinstance(origins, (set, list)):
            raise TypeError(
                "`origins` must be vertex ID (int) or list "
                f'thereof, got "{type(origins)}"'
            )
        origins = np.asarray(origins).astype(int)
    else:
        origins = np.array([])

    # Step size must be positive
    if step_size <= 0:
        raise ValueError("`step_size` must be > 0")

    # Generate Graph (must be undirected)
    G = ig.Graph(edges=mesh.edges_unique, directed=False)
    G.es["weight"] = mesh.edges_unique_length

    # For each edge, track which faces in the mesh it belong to
    edge2face = {}
    for i, f in enumerate(mesh.faces_unique_edges):
        edge2face.update({e: [i] + edge2face.get(e, []) for e in f})
    G.es['faces'] = [edge2face[i] for i in range(len(G.es))]

    # Prepare empty array to fill with centers
    centers = []
    radii = []
    parents = []

    # Go over each connected component
    with tqdm(
        desc="Skeletonizing", total=len(G.vs), disable=not progress, leave=False
    ) as pbar:
        for k, cc in enumerate(G.connected_components()):
            # Make a subgraph for this connected component
            SG = G.subgraph(cc)
            cc = np.array(cc)

            # Select seeds according to the number of waves
            pot_seeds = np.arange(len(cc))
            np.random.seed(1985)  # make seeds predictable
            # See if we can use any of the provided origins
            if len(origins):
                # Get those origins in this cc
                in_cc = np.isin(origins, cc)
                if any(in_cc):
                    # Map origins into cc
                    cc_map = dict(zip(cc, np.arange(0, len(cc))))
                    seed = np.array([cc_map[o] for o in origins[in_cc]])[0]
                else:
                    seed = np.random.choice(pot_seeds)
            else:
                # Use the vertex with the "peakiest" vertex degree as seed
                seed = np.argmin(SG.degree())

            # Track the seed
            SG.vs["is_seed"] = False
            SG.vs[seed]["is_seed"] = True

            # Get the distance between the seed and all other nodes
            dist = np.array(
                SG.distances(source=seed, target=None, mode="all", weights="weight")
            )[0]

            # Pre-emptively check if any of the distances is close to a multiple of step_size
            dist_is_step = np.isclose(dist % step_size, 0)

            # What's the max distance we can reach?
            mx = dist[dist < float("inf")].max()

            # If the max distance is less than the step size, we can just add the seed
            if mx < step_size:
                centers.append(mesh.vertices[cc[seed]])
                radii.append(0)
                parents.append(-1)
                continue

            # A dictionary to keep track of which vertices are at which `step_size`` distance
            nodes_at_stepsize = {d: set() for d in np.arange(0, mx, step_size)}
            nodes_at_stepsize[0].add(seed)

            # A dictionary to track, for each new vertex, which face it belongs to
            new_verts_faces = {}

            # Iterate over all edge and see if we need to insert a new vertex along the edge
            edges_to_delete = []  # edges we need to remove
            edges_to_add = []  # edges we need to add
            new_vertices = []  # vertices we need to add
            for i, edge in enumerate(SG.es):
                v1, v2 = edge.source, edge.target
                # Figure out which vertex is proximal and which one is distal wrt the seed
                if dist[v1] < dist[v2]:
                    prox, distal = v1, v2
                else:
                    prox, distal = v2, v1

                # If the proximal vertex is at a distance that is a multiple of the step size
                # we can start inserting new vertices at the next multiple of the step size + 1
                # In reality this really onla happens for the seed vertex
                if dist_is_step[prox]:
                    nodes_at_stepsize[dist[prox]].add(prox)
                    start_inserting = (np.ceil(dist[prox] / step_size) + 1) * step_size
                # If not, we need to start inserting at the next multiple of the step size
                else:
                    start_inserting = np.ceil(dist[prox] / step_size) * step_size

                # If the distal vertex is further away than the start_inserting, we need to insert
                if dist[distal] > start_inserting:
                    edges_to_delete.append((v1, v2))

                    start = prox
                    this_to_add = np.arange(
                        start_inserting,
                        dist[distal],
                        step_size,
                    )
                    # Note to self: this block is currently the most expensive step in the whole function.
                    #
                    if len(this_to_add):
                        # Calculate the unit vector along the edge
                        prox_pos = mesh.vertices[cc[prox]]
                        distal_pos = mesh.vertices[cc[distal]]
                        vect = distal_pos - prox_pos
                        vect_norm = vect / np.linalg.norm(vect)

                        # Add the new vertices along that vector
                        for d in this_to_add:
                            d_along_edge = d - dist[prox]
                            new_v_ix = len(cc) + len(new_vertices)  # new vertex ID
                            new_v_pos = prox_pos + vect_norm * d_along_edge
                            edges_to_add.append((start, new_v_ix, d_along_edge, edge['faces']))
                            nodes_at_stepsize[d].add(new_v_ix)
                            new_vertices.append(new_v_pos)

                            new_verts_faces[new_v_ix] = edge['faces']

                            start = new_v_ix  # update start for next iteration

                        # Add the last edge (from the last inserted vertex to the distal vertex)
                        edges_to_add.append((new_v_ix, distal, dist[distal] - d, edge['faces']))

            # Turn the new vertices into an array
            new_vertices = np.array(new_vertices)

            # Add the new vertices to the graph
            SG.add_vertices(len(new_vertices))

            # Set is_seed to False for all new vertices
            # (this avoids having "None" value for this attribute which will cause problems during the vertex contraction step)
            SG.vs[len(cc) :]["is_seed"] = False

            # Delete the old edges
            SG.delete_edges(edges_to_delete)

            # Add new edges
            SG.add_edges(
                [(e[0], e[1]) for e in edges_to_add],
                attributes={"weight": [e[2] for e in edges_to_add], "faces": [e[3] for e in edges_to_add]},
            )

            # To collect groups of new vertices that make up a wavefront, we need to connect new vertices
            # that are on the same face and have the same step distance from the seed.
            edges_to_add = []
            for d_, nodes_ in nodes_at_stepsize.items():
                if d_ == 0:
                    continue # skip the seed

                # Collect the faces we need to connect across
                faces_to_connect = {}
                for n in nodes_:
                    for face in new_verts_faces[n]:
                        if face not in faces_to_connect:
                            faces_to_connect[face] = []
                        faces_to_connect[face].append(n)

                # Collect edges
                for nodes_on_face in faces_to_connect.values():
                    if len(nodes_on_face) <= 1:
                        continue
                    for i, j in combinations(nodes_on_face, 2):
                        edges_to_add.append((i, j))

            # Add edges (note we're setting non-sense weights here because we don't really need it anyome)
            SG.add_edges(edges_to_add, attributes={"weight": [1.0] * len(edges_to_add)})

            # Remove duplicate edges
            SG = SG.simplify()

            # Array with all vertex coordinates (old + new)
            all_vertices = np.vstack([mesh.vertices[cc], new_vertices])

            # Now we need a graph containing only step-vertices
            all_step_nodes = np.concatenate(
                [list(s) for s in nodes_at_stepsize.values()]
            )
            SG.vs["is_step_node"] = np.isin(
                np.arange(0, len(SG.vs)), all_step_nodes
            )

            # First generate a graph where there are no edges between nodes of a different distance from the seed
            SG_no_across = SG.copy()

            # Map nodes to their step distance
            nodes2step = {n: d for d, nodes in nodes_at_stepsize.items() for n in nodes}

            # Assign non-step nodes to the closest step distance by rounding
            dist_round = np.round(dist / step_size) * step_size
            nodes2step.update(dict(zip(np.arange(len(dist_round)), dist_round)))

            # Delete all edges that connect nodes of different distances
            SG_no_across.delete_edges(
                [
                    e.index
                    for e in SG_no_across.es
                    if nodes2step[e.source] != nodes2step[e.target]
                ]
            )

            if False:
                # Check for weakly connected components. Something like this:
                # 0 - 1   5 - 7
                # |   |   |   |
                # 2 - 3 - 4 - 6
                # where we would want to remove the connections between 3 and 4
                # In practice this can lead to lots of side branches though
                # which is why we are not doing this for now.
                SG_no_across.delete_edges(SG_no_across.bridges())

            # Because there should only be within-layer edges, we can iterate over all
            # connected components in `SG_no_across` and collapse them into a single node
            # at the center of the connected component
            new_mapping = np.arange(
                0, len(SG.vs)
            )  # note: we're still mapping to SG, not SG_no_across!
            this_centers = {}
            this_radii = {}
            edges_to_delete = []
            for cc2 in SG_no_across.connected_components():
                # Subset cc2 to only step nodes (i.e. remove the "rounded" original nodes)
                cc2_step = [n for n in cc2 if SG_no_across.vs[n]["is_step_node"]]

                if not len(cc2_step):
                    # Connected components without a step node happen when there are normal
                    # nodes (typically just one or two) furher out than the last step node.
                    # We will manually disconnect these nodes.
                    for n in cc2:
                        edges_to_delete.extend([e.index for e in SG.es.select(_source=n)])
                    continue

                # Get the center of all step nodes in this connected component
                this_center = all_vertices[cc2_step].mean(axis=0)

                # Calculate the radius of this connected component
                this_radius = rad_agg_func(
                    np.sqrt(((all_vertices[cc2_step] - this_center) ** 2).sum(axis=1))
                )

                # If this is more than a single node, we need to collapse those nodes
                if len(cc2) > 1:
                    # If this is a group of nodes, we need to collapse them into a single node
                    # We will do this by mapping all nodes in `cc2` to the first node in `cc2`
                    new_mapping[cc2[1:]] = cc2[0]

                this_centers[cc2[0]] = this_center
                this_radii[cc2[0]] = this_radius

            # Remove the spurious edges (need to do that first before simplifying)
            SG.delete_edges(np.unique(edges_to_delete))

            # Contract vertices
            # Note that we're trying to mak sure to preserve the is_seed status:
            # if any of the vertices in the connected component is a seed, the resulting vertex will be a seed.
            # We're using a function because the built-in functions ("max", "first", etc.) run into issues
            # with e.g. None values or empty lists.
            def combine_is_seed(values):
                if any(values):
                    return True
                else:
                    return False

            SG.contract_vertices(new_mapping, combine_attrs={"is_seed": combine_is_seed})
            SG = SG.simplify()

            # Add properties before we delete vertices
            SG.vs["center"] = [this_centers.get(v.index, None) for v in SG.vs]
            SG.vs["radius"] = [this_radii.get(v.index, None) for v in SG.vs]

            # For debugging:
            SG.vs["has_center"] = [v.index in this_centers for v in SG.vs]
            SG.vs["step_dist"] = [nodes2step.get(v.index, None) for v in SG.vs]
            SG.vs["position"] = [v for v in all_vertices]

            # Drop disconnected nodes (nodes will be disconnected because they got contracted)
            SG.delete_vertices([v.index for v in SG.vs if len(v.neighbors()) == 0])

            # Last but not least: remove edges that do not connect nodes across step_dist distances
            edge_dists = np.array([SG.vs[e]['step_dist'] for e in SG.get_edgelist()])
            edge_dists_diff = np.abs(edge_dists[:, 0] - edge_dists[:, 1])
            edges_to_delete = np.where(edge_dists_diff != step_size)[0]

            # Make sure we don't break the connected components
            edges_to_delete = edges_to_delete[~np.isin(edges_to_delete, SG.bridges())]
            SG.delete_edges(edges_to_delete)

            # Generate a directed tree from the root
            # We need to use breadth-first because depth-first messes up branch points
            # We want this:
            # 0 - 1 - 2 - 3
            #     |
            #     4
            # But depth-first might get us this:
            # 0 - 1  2 - 3
            #     | /
            #     4
            # Also note that we're looking up the seed instead of using the original `seed`.
            # That's because the index may have changed due to the vertex contractions.
            _, _, this_parents = SG.bfs(SG.vs.find(is_seed=True))

            this_parents = np.array(this_parents)
            this_parents[this_parents >= 0] += len(centers)  # offset parents
            parents.extend(this_parents)

            radii.extend(SG.vs["radius"])
            centers.extend(SG.vs["center"])

            pbar.update(len(cc))

    return np.vstack(centers), np.array(radii), np.array(parents)


# Constants for the 2D projection
nA = np.array([0, 1, 0])
nB = np.array([1, 0, 0])


def project_to_2d(points, normal, normalize=True):
    # Ensure the normal is a unit vector
    if normalize:
        normal = normal / np.linalg.norm(normal)

    # Find two orthogonal vectors on the plane
    if np.abs(normal[0]) > 0.9:
        n1 = nA
    else:
        n1 = nB

    n1 = n1 - np.dot(n1, normal) * normal
    n1 = n1 / np.linalg.norm(n1)

    n2 = np.cross(normal, n1)

    # Project points onto the plane and return
    return np.dot(points, np.array([n1, n2]).T)


def calc_projection(normals, normalize=True):
    # Ensure the normals are unit vectors
    if normalize:
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # Find two orthogonal vectors on the plane for each normal
    n1 = np.repeat(nA[np.newaxis, :], len(normals), axis=0)
    n1[np.abs(normals[:, 0]) > 0.9] = nB

    # n1 = np.where(np.abs(normals[:, 0]) > 0.9, nA, nB)
    n1 = n1 - np.dot(n1, normals.T).T * normals
    n1 = n1 / np.linalg.norm(n1, axis=1)[:, np.newaxis]

    n2 = np.cross(normals, n1)

    return np.stack([n1, n2], axis=1)