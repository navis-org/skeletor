# News

## 1.7.0 (2026-07-21)
- fix: `by_vertex_clusters` produced a scrambled `mesh_map` - vertices were
  mapped to essentially arbitrary skeleton nodes (mean distance to the assigned
  node was ~100x the distance to the nearest one). The map was indexed by
  cluster-discovery order rather than by vertex ID
- fix: `by_vertex_clusters` clusters now hold the vertices within
  `sampling_dist` geodesic distance of their seed, as documented. Previously
  the distance was accumulated along whatever path the traversal happened to
  take, which overshoots, so clusters were too large and skeletons too coarse.
  Expect noticeably more nodes for a given `sampling_dist`
- fix: `by_teasar` raised on meshes containing isolated vertices
- `by_teasar` now raises a descriptive `ValueError` when `min_length` filters
  out every path (or the mesh has no edges) instead of an opaque `IndexError`
- `navis-fastcore` is now used to accelerate most methods if it is installed
  (`pip install navis-fastcore`); without it skeletor falls back to equivalent
  implementations
  - affected: `by_wavefront` (~2x faster; no graph object is built at all),
  `by_teasar` (~2x), `by_edge_collapse` (~2x), `by_tangent_ball` (~2x),
  `by_vertex_clusters` (~4-5x), `by_wavefront_exact`, and the shared
  mesh -> edge list conversion used by every method (~3.8x)
  - results are the same either way except where the underlying problem has more
  than one correct answer: `by_wavefront` with `waves > 1` produces many exactly
  tied edge weights, and the two spanning tree implementations break those ties
  differently (the trees have identical total weight). `by_teasar` with a small
  `inv_dist` is similar - it zeroes the weights along each extracted path, and
  equidistant routes are then resolved differently (2 of 5773 edges on the
  example mesh, with identical node positions)

## 1.6.1 (2026-07-16)
- fixed `by_edge_collapse` method
- general improvements to `by_wavefront` and `by_tangent_ball` methods

## 1.6.0 (2026-06-27)
- new skeletonization method: `skeletor.skeletonize.by_mean_curvature`
- `sk.skeletonize.by_wavefront_exact` now returns a `mesh_map` property in the `Skeleton` object
- fixes + improvements to mesh contraction (`sk.pre.contract`)
- fixes + improvements to `sk.skeletonize.by_edge_collapse`

## 1.5.0 (2026-02-25)
- new skeletonization method: `skeletor.skeletonize.by_wavefront_exact`
- new post-processing methods: `skeletor.post.recenter_vertices` and `skeletor.post.fix_outside_edges` (by @zyx287)
- a number of small fixes

## 1.4.0 (2025-12-02)
- new method: `Skeleton.mend_breaks`
- fix `Skeleton.radius`
- follow changes in `igraph` and `trimesh`
- a large number of small fixes

## 1.3.0 (2023-03-31)

New post-processing methods:
- `skeletor.post.smooth` to smooth the skeleton
- `skeletor.post.despike` remove spikes from skeleton
- `skeletor.post.remove_bristles` to remove single-node bristles from the skeleton

## 1.2.0 (2022-04-03)

This release mainly improves `skeletor.skeletonize.by_wavefront` but it also
adds a new method to get a graph representation of skeletons:
`Skeleton.get_graph`.

## 1.1.0 (2021-07-26)

This is a small release that adds a single method to the `Skeleton` class to
quickly save results as
[SWC](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html)
file:

```Python
>>> skel.to_swc('skeleton.swc')
```

## 1.0.0 (2021-04-09)

This release represents a major rework of `skeletor`. Notably, functions that
were previously exposed at top level (e.g. `skeletor.contract`) have been
moved to submodules sorting them by their purpose:

| < 1.0.0                     | >= 1.0.0                  |
| --------------------------- | ------------------------- |
| `skeletor.fix_mesh`         | `skeletor.pre.fix_mesh`   |
| `skeletor.simplify`         | `skeletor.pre.simplify`   |
| `skeletor.remesh`           | `skeletor.pre.remesh`     |
| `skeletor.contract`         | `skeletor.pre.contract`   |
| `skeletor.cleanup`          | `skeletor.post.clean_up`  |
| `skeletor.radii`            | `skeletor.post.radii`     |

Similarly `skeletor.skeletonize(method='some method')` has been broken out into
separate functions:

| < 1.0.0                                                | >= 1.0.0                                             |
| ------------------------------------------------------ | ---------------------------------------------------- |
| `skeletor.skeletonize(mesh, method='vertex_clusters')` | `skeletor.skeletonize.by_vertex_clusters(mesh)`      |
| `skeletor.skeletonize(mesh, method='edge_collapse')`   | `skeletor.skeletonize.by_edge_collapse(mesh)`        |


### New Features
- new skeletonization methods `skeletor.skeletonize.by_tangent_ball`,
  `skeletor.skeletonize.by_wavefront` and `skeletor.skeletonize.by_teasar`
- all skeletonization functions return a `Skeleton` object that bundles the
  results (SWC, nodes/vertices, edges, etc.) and allows for quick visualization
- SWC tables are now strictly conforming to the format (continuous node IDs,
  parents always listed before their childs, etc)
- `Skeleton` results contain a mesh to skeleton mapping as `.mesh_map` property
- added an example mesh: to load it use `skeletor.example_mesh()`
- `skeletor` now has proper tests and a simple [documentation](https://navis-org.github.io/skeletor/)

### Removals
- removed `drop_disconnected` parameter from all skeletonization functions -
  pleases either use `fix_mesh` to remove small disconnected pieces from the
  mesh or drop those in postprocessing
- `output` parameter has been removed from all skeletonization functions as the
  output is now always a `Skeleton`


### Bugfixes
- plenty
