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

from scipy.spatial import Delaunay
from tqdm.auto import tqdm, trange
from itertools import combinations

from ..utilities import make_trimesh
from .base import Skeleton

try:
    import fastremap
except ModuleNotFoundError:
    fastremap = None


def by_wavefront_exact(mesh, step_size, origins=None, radius_agg="mean", progress=True):
    """Skeletonize a mesh using wave fronts.

    This is the _exact_ version of the `by_wavefront` function meaning that each wave front
    moves _exactly_ the given distance (see `step_size` parameter) along the mesh, instead
    of hoping from vertex to vertex. This is computationally more expensive but also more
    accurate.

    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to be skeletonize. Can an object that has
                    ``.vertices`` and ``.faces`` properties  (e.g. a
                    trimesh.Trimesh) or a tuple ``(vertices, faces)`` or a
                    dictionary ``{'vertices': vertices, 'faces': faces}``.
    step_size :     float | int (>0)
                    The distance each the wave fronts move along the mesh
                    in each step.
    origins :       int | list of ints, optional
                    Vertex ID(s) where the wave(s) are initialized. If there
                    is no origin for a given connected component, will fall
                    back to semi-random origin.
    radius_agg :    "mean" | "median" | "max" | "min" | "percentile75" | "percentile25"
                    Function used to aggregate radii over sample (i.e. the
                    vertices forming a ring that we collapse to its center).
    progress :      bool
                    If True, will show progress bar.

    Returns
    -------
    skeletor.Skeleton
                    Holds results of the skeletonization and enables quick
                    visualization.

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

    centers_final, radii_final, parents = _cast_waves2(
        mesh,
        step_size=step_size,
        origins=origins,
        rad_agg_func=rad_agg_func,
        progress=progress,
    )

    # Map radii for individual vertices to the collapsed nodes
    # Using pandas is the fastest way here
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

    # Prepare empty array to fill with centers
    centers = []
    radii = []
    data = []

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
            # See if we can use any origins
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
                seed = np.random.choice(pot_seeds)

            # Get the distance between the seed and all other nodes
            dist = np.array(
                SG.distances(source=seed, target=None, mode="all", weights="weight")
            )[0]

            # What's the max distance we can reach?
            mx = dist[dist < float("inf")].max()

            # To keep track of which vertices we have already processed and can be
            # safely ignored
            is_inside_vert = np.zeros(len(dist)).astype(bool)

            # Collect groups
            for i, d in enumerate(np.arange(0, mx, step_size)):
                # Inner verts are all vertices that are within the current distance
                # (minus those that we have already processed)
                inside_verts = np.where((dist <= d) & ~is_inside_vert)[0]

                # To get the outer edge, we need to (1) find the neighbors of the inner edge
                neighbors = [
                    np.array(nn) for nn in SG.neighborhood(vertices=inside_verts)
                ]
                # (2) only keep those whose distance is above the current distance (i.e. those going outwards)
                outer_verts = [nn[dist[nn] > d] for nn in neighbors]

                # Inside vertices that are not part of the inner ring(s) have no neighbors outside
                is_inside_edge = np.array([len(ov) > 0 for ov in outer_verts])
                inner_verts = inside_verts[is_inside_edge]
                outer_verts = [
                    ov for ov, is_iv in zip(outer_verts, is_inside_edge) if is_iv
                ]

                # To avoid doing the same work twice, we will mark inner vertices
                is_inside_vert[inside_verts[~is_inside_edge]] = True

                # For each edge between inner-vertices and their outer neighbors calculate the
                # exact position for the exact step_size distance
                in_out_edges = np.array(
                    [
                        [iv, ov]
                        for iv, ovs in zip(inner_verts, outer_verts)
                        for ov in ovs
                    ]
                )
                in_pos = mesh.vertices[cc[in_out_edges[:, 0]]]
                out_pos = mesh.vertices[cc[in_out_edges[:, 1]]]

                # The distance between the inner and outer vertices
                in_out_dist = in_out_dist = np.sqrt(
                    ((in_pos - out_pos) ** 2).sum(axis=1)
                )

                # For each in_out_edge the remaining distance
                d_left = d - dist[in_out_edges[:, 0]]

                # Move along the edges until we hit the target distance
                new_verts = (
                    in_pos + (out_pos - in_pos) * (d_left / in_out_dist)[:, None]
                )

                # To group vertices into rings we need to first determine which vertices
                # are part of the same ring. We do this by checking which outer vertices
                # are in connected components
                outer_verts_unique = np.unique(np.concatenate(outer_verts))
                for cc2 in SG.subgraph(outer_verts_unique).connected_components():
                    # Translate this connected components back to the vertex IDs in SG
                    outer_verts_cc2 = outer_verts_unique[cc2]

                    # Get the new vertices that are part of this connected component
                    this_new_verts = new_verts[
                        np.isin(in_out_edges[:, 1], outer_verts_cc2)
                    ]

                    # Get the center of this connected component
                    this_center = this_new_verts.mean(axis=0)

                    # Calculate the radius of this connected component
                    this_radius = rad_agg_func(
                        np.sqrt((this_new_verts - this_center) ** 2).sum(axis=1)
                    )

                    centers.append(this_center)  # Store the center
                    radii.append(this_radius)  # Store the radius
                    data.append(
                        (k, i, cc[outer_verts_cc2[0]])
                    )  # Track the connected component, the step and a vertex from the outer ring for this center

                # pbar.update(len(cc))
                # Update progress bar based on the number of vertices we have invalidated
                pbar.update((~is_inside_edge).sum())

    centers = np.vstack(centers)
    radii = np.array(radii)
    data = np.array(data)

    # Next we have to reconstruct the parent-child relationships
    # For each center, we know _a_ vertex that is part of the (outer) ring
    # With that information we can ask, for each center, which center in the
    # previous step is closest to it (as per distance along the mesh)
    parents = np.full(len(centers), fill_value=-1, dtype=int)

    # This maps the step and the vertex ID to the index in the centers array
    step_id_map = {(step, id): i for i, (step, id) in enumerate(data[:, 1:])}

    # tree = G.spanning_tree()

    for i in trange(
        1, data[:, 1].max(), desc="Connecting", disable=not progress, leave=False
    ):
        is_this_step = data[:, 1] == i  # All centers in this step
        is_prev_step = data[:, 1] == i - 1  # All centers in the previous step

        # If there is only one center in the previous step, we can just go on and connect these
        if is_prev_step.sum() == 1:
            parents[is_this_step] = np.where(is_prev_step)[0][0]
            continue

        this_step_track_verts = np.unique(
            data[is_this_step, 2]
        )  # All track-vertices in this step
        prev_step_track_verts = np.unique(
            data[is_prev_step, 2]
        )  # All track-vertices in the previous step

        # Get distances between current and previous track vertices
        # Note to self: this step takes up about 60% of the time at the moment
        # There should be a way to speed this up.
        d = np.array(
            G.distances(
                source=this_step_track_verts,
                target=prev_step_track_verts,
                mode="all",
                weights="weight",
            )
        )

        # Get the closest previous track-vertex for each current track-vertex
        closest_prev_track_verts = prev_step_track_verts[d.argmin(axis=1)]

        for this, prev in zip(this_step_track_verts, closest_prev_track_verts):
            parents[is_this_step & (data[:, 2] == this)] = step_id_map[(i - 1, prev)]

    return centers, radii, parents


def _cast_waves2(mesh, step_size, origins=None, rad_agg_func=np.mean, progress=True):
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
                seed = mesh.vertex_degree.argmin()

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

            # Iterate over all edge and see if we need to insert a new vertex along the edge
            edges_to_delete = []  # edges we need to remove
            edges_to_add = []  # edges we need to add
            new_vertices = []  # vertices we need to add
            for i, (v1, v2) in enumerate(SG.get_edgelist()):
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
                            edges_to_add.append((start, new_v_ix, d_along_edge))
                            nodes_at_stepsize[d].add(new_v_ix)
                            new_vertices.append(new_v_pos)
                            start = new_v_ix  # update start for next iteration

                        # Add the last edge (from the last inserted vertex to the distal vertex)
                        edges_to_add.append((new_v_ix, distal, dist[distal] - d))

            # Turn the new vertices into an array
            new_vertices = np.array(new_vertices)

            # Add the new vertices to the graph
            SG.add_vertices(len(new_vertices))

            # Delete the old edges
            SG.delete_edges(edges_to_delete)

            # Add new edges
            SG.add_edges(
                [(e[0], e[1]) for e in edges_to_add],
                attributes={"weight": [e[2] for e in edges_to_add]},
            )

            # To extract collect groups of new vertices, we need to first re-triangulate the faces using scipy's Delaunay
            all_vertices = np.vstack([mesh.vertices[cc], new_vertices])

            # Uncomment to debug the new vertices
            # return {d: all_vertices[list(nodes)] for d, nodes in nodes_at_stepsize.items()}

            # Get all faces and their normals belonging to this connected component
            selection = np.isin(mesh.faces[:, 0], cc)
            cc_faces = mesh.faces[selection]
            cc_face_normals = mesh.face_normals[selection]

            # Translate vertex to node indices
            vert2cc = dict(zip(cc, np.arange(0, len(cc))))

            if fastremap:
                cc_faces_nodes = fastremap.remap(cc_faces, vert2cc)
            else:
                cc_faces_nodes = np.array([[vert2cc[v] for v in f] for f in cc_faces])

            # Note to self: this seems to take up the majority of the time
            # Check if we can speed this up!
            edges_to_add = []
            for f, n in zip(cc_faces_nodes, cc_face_normals):
                # Get the path around this face
                paths = [
                    SG.get_shortest_path(f[0], f[1]),
                    SG.get_shortest_path(f[1], f[2]),
                    SG.get_shortest_path(f[2], f[0]),
                ]

                # Flatten
                path = np.unique([p for path in paths for p in path])

                # If path contains only 3 vertices, we don't need to triangluate
                if len(path) == 3:
                    continue
                elif len(path) < 3:
                    raise ValueError("Something went wrong")

                # Get the coordinates for these points
                points = all_vertices[path]

                # The points are all on a face, so they are coplanar
                # For the Delaunay to be a triangulation (i.e. a 2d convex hull)
                # we need to project the points onto a plane.
                points_2d = project_to_2d(points, n, normalize=False)

                # Triangulate the points
                # The "QJ" option is used to ensure all points are represented
                # in the simplices
                tri = Delaunay(points_2d, qhull_options="QJ")

                # Add the new edges
                for s in path[tri.simplices]:
                    edges_to_add.append((s[0], s[1]))
                    edges_to_add.append((s[1], s[2]))
                    edges_to_add.append((s[2], s[0]))

            # Add new edges
            SG.add_edges(edges_to_add)

            # Because this will have introduces duplicate edges, we will do one round of simplification
            SG = SG.simplify()

            # At this point we want to excise all original, non-step vertices and directly connect the
            # step vertices wherever we remove a non-step vertex
            all_step_nodes = np.concatenate(
                [list(s) for s in nodes_at_stepsize.values()]
            )
            SG.vs["is_step_node"] = is_step_node = np.isin(
                np.arange(0, len(SG.vs)), all_step_nodes
            )

            # while True:
            #     edges_to_add = []
            #     edges_to_delete = []
            #     for v in SG.vs.select(is_step_node=False):
            #         # Directly connect all neighbors of this vertex
            #         for v1, v2 in combinations(v.neighbors(), 2):
            #             edges_to_add.append((v1.index, v2.index))

            #         # Delete all edges connected to this vertex
            #         # (_source works both ways because the graph is undirected)
            #         edges_to_delete.extend(SG.es.select(_source=v.index))

            #     if len(edges_to_add) == 0:
            #         break

            #     SG.add_edges(edges_to_add)
            #     SG.delete_edges(edges_to_delete)
            #     SG.simplify()

            # Other idea:
            # Assign non-step nodes to the closest step distance (round?) and

            # # This runs in a loop until we can't simplify anymore
            # # If this turns out to be expensive, we can just find a shortest path from each non-step node
            # # to a step node and collapse along that path
            # while True:
            #     was_simplified = False
            #     new_mapping = np.arange(0, len(SG.vs))
            #     for e in SG.es:
            #         if (
            #             not SG.vs[e.source]["is_step_node"]
            #             and SG.vs[e.target]["is_step_node"]
            #         ):
            #             was_simplified = True
            #             new_mapping[e.source] = e.target
            #         elif (
            #             SG.vs[e.source]["is_step_node"]
            #             and not SG.vs[e.target]["is_step_node"]
            #         ):
            #             was_simplified = True
            #             new_mapping[e.target] = e.source

            #     # If we didn't simplify anything in this round, we can stop
            #     if not was_simplified:
            #         break

            #     # Contract vertices. This will not remove any vertices but instead will remap
            #     # only vertex IDs in edges.
            #     SG.contract_vertices(new_mapping)

            #     # Contraction will have dropped the `is_step_node` attribute, so we need to re-add it
            #     SG.vs["is_step_node"] = is_step_node

            #     # One round of simplification to get rid of duplicate edges and self-loops
            #     SG = SG.simplify()

            # Now the graph has only have edges between nodes that are at `step_size` distance
            # Next, we need to collapse these nodes into their center.
            # First generate a graph where there are no edges between nodes of a different distance from the seed
            SG_no_across = SG.copy()

            # Delete all edges that connect nodes of different distances
            nodes2step = {n: d for d, nodes in nodes_at_stepsize.items() for n in nodes}

            # Assign non-step nodes to the closest step distance (round?)
            dist_round = np.round(dist / step_size) * step_size
            nodes2step.update(dict(zip(np.arange(len(dist_round)), dist_round)))

            SG_no_across.delete_edges(
                [
                    e.index
                    for e in SG_no_across.es
                    if nodes2step[e.source] != nodes2step[e.target]
                ]
            )

            # Check for weakly connected components. Something like this:
            # 0 - 1   5 - 7
            # |   |   |   |
            # 2 - 3 - 4 - 6
            # where we would want to remove the connections between 3 and 4
            # In practice this can lead to lots of side branches though
            if False:
                SG_no_across.delete_edges(SG_no_across.bridges())

            # return SG_no_across

            # Because there should only be within-layer edges, we can iterate over all
            # connected components in SG_no_across and collapse them into a single node
            # at the center of the connected component
            new_mapping = np.arange(
                0, len(SG.vs)
            )  # note: we're still mapping to SG, not SG_no_across!
            this_centers = {}
            this_radii = {}
            edges_to_delete = []
            for cc2 in SG_no_across.connected_components():
                # Subset cc2 to only step nodes
                cc2_step = [n for n in cc2 if SG_no_across.vs[n]["is_step_node"]]

                if not len(cc2_step):
                    # Connected components without a step node happen when there is are normal
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
            SG.contract_vertices(new_mapping)
            SG = SG.simplify()

            # Add properties before we delete vertices
            SG.vs["center"] = [this_centers.get(v.index, None) for v in SG.vs]
            SG.vs["radius"] = [this_radii.get(v.index, None) for v in SG.vs]
            SG.vs[seed]["seed"] = True  # Mark the seed

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
            # Instead of this:
            # 0 - 1 - 2 - 3
            #     |
            #     4
            # We might get this:
            # 0 - 1  2 - 3
            #     | /
            #     4
            _, _, this_parents = SG.bfs(SG.vs.select(seed=True)[0].index)

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


def project_to_2d_proj(points, proj):
    # Find two orthogonal vectors on the plane
    if np.abs(normal[0]) > 0.9:
        n1 = nA
    else:
        n1 = nB

    n1 = n1 - np.dot(n1, normal) * normal
    n1 = n1 / np.linalg.norm(n1)

    n2 = np.cross(normal, n1)

    # Project points onto the plane and return
    return np.dot(points, proj)