# News

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
- `skeletor` now has proper tests and a simple [documentation](https://schlegelp.github.io/skeletor/)

### Removals
- removed `drop_disconnected` parameter from all skeletonization functions -
  pleases either use `fix_mesh` to remove small disconnected pieces from the
  mesh or drop those in postprocessing
- `output` parameter has been removed from all skeletonization functions as the
  output is now always a `Skeleton`


### Bugfixes
- plenty
