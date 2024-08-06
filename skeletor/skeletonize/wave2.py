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

from tqdm.auto import tqdm, trange

from ..utilities import make_trimesh
from .base import Skeleton


def by_wavefront_exact(mesh, step_size, origins=None, radius_agg="mean", progress=True):
    """Skeletonize a mesh using wave fronts.

    This is the _exact_ version of the `by_wavefront` function meaning
    that each wave front moves exactly the given distance (see `step_size`)
    along the mesh, instead of hoping from vertex to vertex. This is
    computationally more expensive but also more accurate.

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

    centers_final, radii_final, parents = _cast_waves(
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
    if step_size < 0:
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

    tree = G.spanning_tree()

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
            tree.distances(
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
