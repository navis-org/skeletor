import pytest

import skeletor as sk
import trimesh as tm
import networkx as nx
import numpy as np
import scipy.spatial
import scipy.sparse
import scipy.sparse.csgraph


class TestPreprocessing:
    def test_example_mesh(self):
        """Test loading the example mesh."""
        assert isinstance(sk.example_mesh(), tm.Trimesh)

    def test_fix_mesh(self):
        fixed = sk.pre.fix_mesh(sk.example_mesh(),
                                remove_disconnected=True,  # this is off by default
                                inplace=False)
        assert isinstance(fixed, tm.Trimesh)

    @pytest.mark.parametrize('operator', ['cotangent', 'umbrella', 'robust'])
    def test_contraction(self, operator):
        if operator == 'robust':
            pytest.importorskip('robust_laplacian')

        m = sk.example_mesh()
        # Use an aggressive epsilon - this is the setting that exposes
        # numerical breakdown of the solve (vertices smearing out of bounds)
        cont = sk.pre.contract(m, epsilon=1e-6, operator=operator,
                               progress=False)

        assert isinstance(cont, tm.Trimesh)
        # Vertices must stay finite (no NaN/inf leaking from the solve)
        assert np.isfinite(cont.vertices).all()
        # The contraction flow is contractive: vertices must not smear outside
        # the input bounding box (allowing a small numerical margin)
        diag = np.linalg.norm(m.bounds[1] - m.bounds[0])
        assert (cont.vertices.min(axis=0) >= m.bounds[0] - 0.01 * diag).all()
        assert (cont.vertices.max(axis=0) <= m.bounds[1] + 0.01 * diag).all()
        # Contraction must actually make meaningful progress
        assert cont.epsilon < 0.5


class TestSkeletonization:
    def test_wave_skeletonization(self):
        one_wave = sk.skeletonize.by_wavefront(sk.example_mesh(), waves=1)
        two_wave = sk.skeletonize.by_wavefront(sk.example_mesh(), waves=2)
        stepsize = sk.skeletonize.by_wavefront(sk.example_mesh(), waves=1,
                                               step_size=2)

        for s in [one_wave, two_wave, stepsize]:
            assert len(s.mesh_map) == len(s.mesh.vertices)
            assert all(np.isin(s.mesh_map, s.swc.node_id.values))

    @pytest.mark.parametrize('waves,step_size', [(1, 1), (1, 3), (2, 1), (3, 4)])
    @pytest.mark.parametrize('fragmented', [False, True])
    def test_wave_backends_agree(self, waves, step_size, fragmented):
        """The fastcore and igraph `_cast_waves` backends must not drift apart.

        They are structurally very different - fastcore vectorizes over the
        whole mesh, igraph loops per component - and several invariants keeping
        them in step (component ordering, the per-component binning bounds) are
        otherwise only asserted in comments. Levels are integer, so unlike the
        spanning tree there is no tie-breaking slack to excuse a difference.
        """
        pytest.importorskip('navis_fastcore')

        from skeletor.skeletonize import wave
        from skeletor.utilities import make_trimesh, get_edges_unique

        m = sk.example_mesh()
        if fragmented:
            # Lots of tiny components is the case that separates the two: it is
            # what made the per-component backend quadratic in the first place.
            m = tm.util.concatenate([m] + [tm.creation.box().apply_translation(
                [i * 5, 1e4, 0]) for i in range(20)])
        m = make_trimesh(m, validate=False)
        edges = get_edges_unique(m)

        kwargs = dict(origins=np.array([]), step_size=step_size)
        fc_c, fc_r = wave._cast_waves_fastcore(m, edges, waves,
                                               radius_agg='mean',
                                               rad_agg_func=np.mean,
                                               progress=False, **kwargs)
        ig_c, ig_r = wave._cast_waves_igraph(m, edges, waves,
                                             rad_agg_func=np.mean,
                                             progress=False, **kwargs)

        assert np.allclose(fc_c, ig_c, atol=1e-9, equal_nan=True)
        assert np.allclose(fc_r, ig_r, atol=1e-9, equal_nan=True)

    def test_wave_exact(self):
        s = sk.skeletonize.by_wavefront_exact(sk.example_mesh(), step_size=50)

        assert len(s.mesh_map) == len(s.mesh.vertices)
        assert all(np.isin(s.mesh_map, s.swc.node_id.values))

    @pytest.mark.parametrize('step_frac,radius_agg,fragmented',
                             [(1.0, 'mean', False), (0.4, 'mean', False),
                              (0.4, 'percentile75', False), (1.0, 'max', False),
                              (1.0, 'mean', True)])
    def test_wave_exact_backends_agree(self, monkeypatch, step_frac, radius_agg,
                                       fragmented):
        """`by_wavefront_exact` must give the same skeleton with and without fastcore.

        Unlike the other backend pairs in this file, there is only one
        implementation here: `_cast_waves` carries its graph as arrays and asks a
        handful of primitives - components, distances, level sets, contraction,
        bridges, orientation - each of which has a fastcore and a scipy/igraph
        form. So this pins the *primitives* to each other rather than two rival
        implementations, and it is exact rather than approximate: whether an
        optional dependency happens to be installed must not be visible in
        anyone's skeleton.

        The one thing that needs care is the breadth-first orientation, where the
        depths are forced but the parent within a layer is not - see
        `wave2._bfs_tree`.
        """
        pytest.importorskip('navis_fastcore')

        from skeletor import _fastcore
        from skeletor.skeletonize import wave2
        from skeletor.utilities import make_trimesh

        m = sk.example_mesh()
        if fragmented:
            m = tm.util.concatenate([m] + [
                tm.creation.box(extents=[80] * 3).apply_translation([i * 200, 1e4, 0])
                for i in range(20)])
        m = make_trimesh(m, validate=False)

        agg = {'mean': np.mean, 'max': np.max,
               'percentile75': lambda x: np.percentile(x, 75)}[radius_agg]
        kwargs = dict(radius_agg=radius_agg, rad_agg_func=agg, progress=False)
        step = float(np.mean(m.edges_unique_length)) * step_frac

        fast = wave2._cast_waves(m, step, **kwargs)

        monkeypatch.setattr(_fastcore, 'fastcore', None)
        assert not _fastcore.has('parents_from_edges'), 'fastcore not disabled'
        slow = wave2._cast_waves(m, step, **kwargs)

        # Topology and the vertex mapping exactly; coordinates and radii to
        # within float reassociation
        assert np.array_equal(fast[2], slow[2]), 'parents differ'
        assert np.array_equal(fast[3], slow[3]), 'mesh_map differs'
        assert np.allclose(fast[0], slow[0], rtol=1e-12, atol=0)
        assert np.allclose(np.asarray(fast[1], dtype=float),
                           np.asarray(slow[1], dtype=float), rtol=1e-10)

    def test_wave_exact_no_fastcore(self, monkeypatch):
        """`by_wavefront_exact` must work at all without fastcore."""
        from skeletor import _fastcore

        monkeypatch.setattr(_fastcore, 'fastcore', None)
        s = sk.skeletonize.by_wavefront_exact(sk.example_mesh(), step_size=50,
                                              progress=False)

        assert len(s.mesh_map) == len(s.mesh.vertices)
        assert all(np.isin(s.mesh_map, s.swc.node_id.values))

    def test_components_labelled_by_lowest(self, monkeypatch):
        """Both `connected_components` backends label a component by its lowest ID.

        `group_edges_by_component` keys its per-component edge slices by that
        label, so a backend numbering components any other way would hand
        `by_wavefront_exact` and `by_teasar` the wrong edges.
        """
        pytest.importorskip('navis_fastcore')

        from skeletor import _fastcore
        from skeletor.skeletonize.utils import connected_components
        from skeletor.utilities import make_trimesh, get_edges_unique

        m = make_trimesh(tm.util.concatenate(
            [sk.example_mesh()] + [tm.creation.box().apply_translation([i * 5, 1e4, 0])
                                   for i in range(15)]), validate=False)
        edges = np.asarray(get_edges_unique(m), dtype=np.int64)
        n = len(m.vertices)

        fast = connected_components(edges, n)
        monkeypatch.setattr(_fastcore, 'fastcore', None)
        slow = connected_components(edges, n)

        assert np.array_equal(fast, slow)
        # The label of a component is the lowest node in it
        for members in np.split(np.argsort(fast, kind='stable'),
                                np.flatnonzero(np.diff(np.sort(fast))) + 1):
            assert fast[members[0]] == members.min()

    def test_vertex_cluster(self):
        s = sk.skeletonize.by_vertex_clusters(sk.example_mesh(),
                                              sampling_dist=100)

        assert len(s.mesh_map) == len(s.mesh.vertices)
        assert all(np.isin(s.mesh_map, s.swc.node_id.values))

    @pytest.mark.parametrize('method,kwargs',
                             [('by_vertex_clusters', {'sampling_dist': 200}),
                              ('by_wavefront', {'waves': 1}),
                              ('by_teasar', {'inv_dist': 200})])
    def test_mesh_map_is_local(self, method, kwargs):
        """Each vertex must map to a *nearby* node.

        `np.isin(mesh_map, node_id)` - the check the other tests use - passes
        for any permutation, so it happily accepted a mesh_map that was indexed
        by cluster-discovery order rather than by vertex ID.
        """
        m = sk.example_mesh()
        s = getattr(sk.skeletonize, method)(m, progress=False, **kwargs)

        mapped = s.swc.set_index('node_id')[['x', 'y', 'z']].loc[s.mesh_map].values
        d_mapped = np.linalg.norm(np.asarray(m.vertices) - mapped, axis=1)

        tree = scipy.spatial.cKDTree(s.swc[['x', 'y', 'z']].values)
        d_nearest = tree.query(np.asarray(m.vertices))[0]

        # Nodes are collapsed clusters of vertices, so the mapped node need not
        # be the closest one - but it must be in the same neighbourhood, not
        # across the mesh
        assert d_mapped.mean() < 3 * d_nearest.mean() + 1e-9

    def test_edge_collapse(self):
        s = sk.skeletonize.by_edge_collapse(sk.example_mesh())

        # Skeleton must be valid: finite vertices and well-formed edges that
        # stay within the input bounding box (no spurious long "zipper" jumps)
        assert isinstance(s.vertices, np.ndarray)
        assert np.isfinite(s.vertices).all()
        assert s.edges.shape[1] == 2
        m = sk.example_mesh()
        assert (s.vertices >= m.bounds[0]).all() and (s.vertices <= m.bounds[1]).all()

    def test_teasar(self):
        s = sk.skeletonize.by_teasar(sk.example_mesh(), 500)

        assert len(s.mesh_map) == len(s.mesh.vertices)
        assert all(np.isin(s.mesh_map, s.swc.node_id.values))

    def test_teasar_nothing_left(self):
        """A `min_length` nothing can satisfy must say so, not IndexError."""
        with pytest.raises(ValueError, match='min_length'):
            sk.skeletonize.by_teasar(sk.example_mesh(), 500, min_length=1e9)

    @pytest.mark.parametrize('inv_dist,min_length', [(500, None), (200, None),
                                                     (500, 300)])
    def test_teasar_backends_agree(self, inv_dist, min_length):
        """The fastcore and igraph `_invalidate` backends must not drift apart.

        The fastcore one drives two prepared `GeodesicGraph`s - one it
        re-weights as it extracts paths, one pristine for the invalidation
        radius - where the igraph one rebuilds a subgraph and hands scipy a
        snapshot adjacency. Very different machinery for an algorithm whose
        every step depends on the last, so the skeletons are pinned to each
        other here rather than only to their own shape.

        Only the *edges* are pinned. `mesh_map` is allowed to differ: a vertex
        equidistant from two nodes of the path invalidating it goes to whichever
        the underlying search settled first, which the two decide differently.
        The vertices that do differ are checked to be plausible ties rather than
        one backend picking a distant node - not proof of a geodesic tie, which
        would mean re-running the invalidation search here, but enough to catch
        a map that has gone wrong rather than merely gone the other way.
        """
        pytest.importorskip('navis_fastcore')

        from skeletor.skeletonize import teasar
        from skeletor.utilities import make_trimesh, get_edges_unique

        def as_set(e):
            return set(map(tuple, np.sort(e, axis=1)))

        m = make_trimesh(sk.example_mesh(), validate=False)
        edges, lengths = get_edges_unique(m, lengths=True)
        args = (m, edges, lengths, inv_dist, min_length, 0, False)

        fc_edges, fc_map = teasar._invalidate_fastcore(*args)
        ig_edges, ig_map = teasar._invalidate_igraph(*args)

        assert as_set(fc_edges) == as_set(ig_edges)

        differ = np.flatnonzero(fc_map != ig_map)
        v = np.asarray(m.vertices)
        assert np.allclose(np.linalg.norm(v[differ] - v[fc_map[differ]], axis=1),
                           np.linalg.norm(v[differ] - v[ig_map[differ]], axis=1))
        assert len(differ) < 0.01 * len(fc_map)

    def test_mean_curvature(self):
        s = sk.skeletonize.by_mean_curvature(sk.example_mesh(), progress=False)

        assert len(s.mesh_map) == len(s.mesh.vertices)
        assert all(np.isin(s.mesh_map, s.swc.node_id.values))
        # Skeleton must be valid: finite vertices that stay within the input
        # bounding box (no spurious long jumps)
        assert np.isfinite(s.vertices).all()
        m = sk.example_mesh()
        assert (s.vertices >= m.bounds[0]).all() and (s.vertices <= m.bounds[1]).all()

    def test_tangent(self):
        s = sk.skeletonize.by_tangent_ball(sk.example_mesh())

        assert len(s.mesh_map) == len(s.mesh.vertices)
        assert all(np.isin(s.mesh_map, s.swc.node_id.values))

    def test_graph(self):
        s = sk.skeletonize.by_wavefront(sk.example_mesh(), waves=1)

        assert isinstance(s.get_graph(), nx.DiGraph)


class TestUtils:
    """`group_by_label` underpins component/cluster/wavefront grouping.

    Its stability is what keeps the fastcore and igraph backends producing the
    same component order, so it is worth pinning down directly rather than only
    through the skeletonization results.
    """

    @pytest.mark.parametrize('labels', [
        [],                                  # empty
        [0],                                 # single item
        [7, 7, 7],                           # one group, label != 0
        [2, 0, 1, 0, 2, 1],                  # contiguous, interleaved
        [50, 3, 50, 3, 999],                 # sparse labels
        [-1, 5, -1, 5, -1],                  # negative labels (the -1 sentinel)
        [0, 0, 0, 1],                        # unequal group sizes
    ])
    def test_group_by_label(self, labels):
        from skeletor.skeletonize.utils import group_by_label, split_by_label

        labels = np.asarray(labels, dtype=np.int64)
        order, starts, sizes = group_by_label(labels)
        groups = split_by_label(labels)

        # Every item appears exactly once, and the runs tile `order`
        assert sorted(order.tolist()) == list(range(len(labels)))
        assert sizes.sum() == len(labels)
        assert np.array_equal(starts, np.cumsum(np.r_[0, sizes])[:-1].astype(starts.dtype))

        # Groups come out in ascending label order...
        keys = labels[order[starts]] if len(starts) else np.zeros(0, dtype=labels.dtype)
        assert np.array_equal(keys, np.unique(labels))

        # ...and are stable: ascending item index within each group
        for start, size in zip(starts, sizes):
            run = order[start:start + size]
            assert np.array_equal(run, np.sort(run))
            assert len(np.unique(labels[run])) == 1

        # `split_by_label` agrees with the run form, and with a naive reference
        if len(labels):
            assert [g.tolist() for g in groups] == [
                np.flatnonzero(labels == k).tolist() for k in np.unique(labels)]
            assert [g.tolist() for g in groups] == [
                order[s:s + n].tolist() for s, n in zip(starts, sizes)]

    @pytest.mark.parametrize('case', ['mesh', 'fragmented', 'cyclic',
                                      'self_and_duplicate', 'no_edges', 'empty'])
    def test_orient_forest_backends_agree(self, case):
        """The fastcore and scipy `orient_forest` backends must not drift apart.

        Everything that turns an edge list into a skeleton goes through here, so
        a difference would show up as a differently-shaped tree three call sites
        away. Both must root each component at its lowest node ID - that is the
        contract `forest_to_swc` and `edges_to_swc` rely on - and both must hand
        back a topological order, since SWC requires parents to be renumbered
        below their children.
        """
        from skeletor.skeletonize.utils import orient_forest, _orient_forest_scipy
        from skeletor.utilities import make_trimesh, get_edges_unique

        if case in ('mesh', 'fragmented'):
            m = sk.example_mesh()
            if case == 'fragmented':
                m = tm.util.concatenate([m] + [tm.creation.box().apply_translation(
                    [i * 5, 1e4, 0]) for i in range(20)])
            m = make_trimesh(m, validate=False)
            edges, n = get_edges_unique(m), len(m.vertices)
        elif case == 'cyclic':
            # Dense random edges: many cycles, so the two have to break them
            # somewhere - the roots and the partition must still agree
            edges, n = np.random.default_rng(7).integers(0, 500, (2000, 2)), 500
        elif case == 'self_and_duplicate':
            edges, n = np.array([[0, 0], [0, 1], [1, 0], [0, 1], [2, 2]]), 4
        elif case == 'no_edges':
            edges, n = np.zeros((0, 2), dtype=np.int64), 5
        else:
            edges, n = np.zeros((0, 2), dtype=np.int64), 0

        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        pf, of = orient_forest(edges, n)
        ps, os_ = _orient_forest_scipy(edges, n)

        assert np.array_equal(np.flatnonzero(pf < 0), np.flatnonzero(ps < 0))

        if not n:
            return

        adj = scipy.sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                      shape=(n, n))
        labels = scipy.sparse.csgraph.connected_components(adj, directed=False)[1]
        # First occurrence of a label is that component's lowest node ID
        lowest = np.sort(np.unique(labels, return_index=True)[1])
        undirected = set(map(tuple, np.sort(edges[edges[:, 0] != edges[:, 1]], axis=1)))

        for parents, order in ((pf, of), (ps, os_)):
            assert np.array_equal(np.sort(np.flatnonzero(parents < 0)), lowest)

            has_parent = parents >= 0
            child = np.flatnonzero(has_parent)
            # Parent links stay inside their component and are real input edges
            assert np.array_equal(labels[parents[has_parent]], labels[child])
            assert set(map(tuple, np.sort(np.column_stack(
                (child, parents[has_parent])), axis=1))) <= undirected

            # `order` is a permutation, and relabelling by it puts every parent
            # below its child
            assert np.array_equal(np.sort(order), np.arange(n))
            new_ids = np.empty(n, dtype=np.int64)
            new_ids[order] = np.arange(n)
            assert (new_ids[parents[has_parent]] < new_ids[child]).all()

    def test_edges_to_swc_matches_networkx(self):
        """`edges_to_swc` must reproduce `edges_to_graph` -> `make_swc`.

        It replaced that pipeline at three call sites. Node *IDs* are allowed to
        differ - the two renumber differently - so this compares the skeleton in
        the original ID space, plus the isolated nodes `nodes=` is there to keep.
        """
        from skeletor.skeletonize.utils import edges_to_swc, edges_to_graph, make_swc

        coords = np.random.default_rng(3).random((900, 3)) * 100
        rng = np.random.default_rng(7)
        cyclic = np.unique(np.column_stack((rng.integers(0, 800, 1200),
                                            rng.integers(0, 800, 1200))), axis=0)

        for edges, nodes in [(cyclic, None),
                             (np.array([[1, 2], [2, 3], [7, 8]]), [1, 2, 3, 7, 8, 42])]:
            new, new_ids = edges_to_swc(edges, coords, nodes=nodes)
            G = edges_to_graph(np.asarray(edges), nodes=nodes, fix_tree=True,
                               weight=False, drop_disconnected=False)
            old, old_ids = make_swc(G, coords, reindex=True, validate=False)

            def original(swc, ids):
                inv = {v: k for k, v in ids.items() if k != -1}
                e = swc[swc.parent_id >= 0][['node_id', 'parent_id']].values
                edges_ = {tuple(sorted((inv[a], inv[b]))) for a, b in e}
                return edges_, {inv[n] for n in swc.node_id.values}

            assert original(new, new_ids) == original(old, old_ids)

            # SWC invariants that `make_swc(validate=True)` would have checked
            assert not new.node_id.duplicated().any()
            not_root = new[new.parent_id >= 0]
            assert (not_root.parent_id.values < not_root.node_id.values).all()

            # Coordinates must follow the node through the renumbering
            inv = {v: k for k, v in new_ids.items() if k != -1}
            orig = np.array([inv[n] for n in new.node_id.values])
            assert np.allclose(new[['x', 'y', 'z']].values, coords[orig])

    def test_mst_over_mesh_backends_agree(self):
        """The `geodesic_mst_mesh` backend must span exactly what scipy's did.

        The fastcore one never forms the k x k geodesic matrix - it sweeps once
        and spans the distance network - so this pins it to the matrix route it
        replaced rather than only to its own shape.
        """
        pytest.importorskip('navis_fastcore')

        from skeletor.skeletonize.utils import (_mst_over_mesh_fastcore,
                                                _mst_over_mesh_scipy)
        from skeletor.utilities import make_trimesh

        m = make_trimesh(sk.example_mesh(), validate=False)
        rng = np.random.default_rng(0)

        for k in (250, 1500):
            keep = np.unique(rng.choice(len(m.vertices), k, replace=False))
            limit = np.max(scipy.spatial.distance.pdist(m.vertices[keep])) * 3

            fc = _mst_over_mesh_fastcore(m, keep, limit)
            sp = _mst_over_mesh_scipy(m, keep, limit)

            def tree(e):
                return set(map(tuple, np.sort(e, axis=1)))

            assert tree(fc) == tree(sp)
            # A spanning forest over the vertices it reached: no more edges than
            # nodes, and no vertex outside `keep`
            assert len(fc) == len(tree(fc))
            assert np.isin(fc, keep).all()
            assert len(fc) < len(np.unique(fc))

    def test_mst_auto_limit_is_an_upper_bound(self):
        """`limit='auto'` must never be tighter than the distance it bounds.

        It uses the bounding box diagonal rather than the exact largest pairwise
        distance, because exact is O(k**2) in time and memory for a number the
        search is insensitive to. That is only safe while the bound stays *above*
        the true value - a bound that came in under it would disconnect vertices
        the old limit joined. The k=2 case is the one that actually binds in
        practice, and there the two must agree exactly.
        """
        rng = np.random.default_rng(0)

        for k in (2, 3, 10, 500):
            for _ in range(20):
                v = rng.random((k, 3)) * rng.choice([1, 1e4])

                exact = np.max(scipy.spatial.distance.pdist(v))
                bound = np.linalg.norm(v.max(axis=0) - v.min(axis=0))

                assert bound >= exact - 1e-9
                # ...and not loosely so: sqrt(3) is the worst case in 3D
                assert bound <= exact * np.sqrt(3) + 1e-9
                if k == 2:
                    assert np.isclose(bound, exact)

    def test_fastcore_has(self):
        """`has` must not claim a feature the installed fastcore lacks.

        The keyword form is the one worth pinning: a function that exists but
        does not take the argument a call site needs would raise a TypeError at
        the point `has` was meant to prevent it.
        """
        pytest.importorskip('navis_fastcore')

        from skeletor import _fastcore

        assert _fastcore.has('connected_components_graph')
        assert not _fastcore.has('no_such_function_here')
        assert _fastcore.has('GeodesicGraph.distances')
        assert not _fastcore.has('GeodesicGraph.no_such_method')

        # Keyword form
        assert _fastcore.has('geodesic_matrix_graph(sources)')
        assert not _fastcore.has('geodesic_matrix_graph(no_such_kwarg)')
        # A name that resolves to something with no signature must not blow up
        assert not _fastcore.has('__doc__(anything)')

        # Several at once: all must hold
        assert _fastcore.has('bridges', 'contract_vertices')
        assert not _fastcore.has('bridges', 'no_such_function_here')

    def test_edges_unique_dtype_matches_across_backends(self):
        """Both `get_edges_unique` backends must return the same dtype.

        fastcore returns node ids as uint32 where the numpy backend (and trimesh)
        use int64. Unsigned arithmetic wraps instead of going negative, so left
        alone this would make any signed arithmetic downstream depend on whether
        fastcore happens to be installed.
        """
        pytest.importorskip('navis_fastcore')

        from skeletor.utilities import (make_trimesh, get_edges_unique,
                                        _unique_edges_numpy)

        m = make_trimesh(sk.example_mesh(), validate=False)
        faces = np.asarray(m.faces)

        eu = get_edges_unique(m)
        expected = _unique_edges_numpy(m, faces, False, False)[0]

        assert eu.dtype == expected.dtype == np.int64
        assert np.array_equal(eu, expected)
        # The failure this actually guards against
        assert (eu[:, 0] - eu[:, 1]).min() < 0

    def test_group_by_label_matches_igraph_components(self):
        """The ordering invariant the two skeletonization backends share."""
        pytest.importorskip('navis_fastcore')
        import igraph as ig
        from skeletor import _fastcore
        from skeletor.skeletonize.utils import split_by_label
        from skeletor.utilities import make_trimesh, get_edges_unique

        m = make_trimesh(tm.util.concatenate(
            [sk.example_mesh()] + [tm.creation.box().apply_translation([i * 5, 1e4, 0])
                                   for i in range(15)]), validate=False)
        edges = get_edges_unique(m)
        n_verts = m.vertices.shape[0]

        comp = _fastcore.fastcore.connected_components_graph(edges, n_verts)
        ours = split_by_label(comp)
        theirs = ig.Graph(n=n_verts, edges=edges, directed=False).connected_components()

        assert len(ours) == len(theirs)
        for a, b in zip(ours, theirs):
            assert a.tolist() == sorted(b)


class TestPostprocessing:
    def test_cleanup(self):
        mesh = sk.example_mesh()
        s = sk.skeletonize.by_wavefront(mesh, waves=1)
        clean = sk.post.clean_up(s, inplace=False)

    def test_radius(self):
        mesh = sk.example_mesh()
        s = sk.skeletonize.by_wavefront(mesh, waves=1)

        rad_knn = sk.post.radii(s, method='knn')
        rad_ray = sk.post.radii(s, method='ray')

    def test_fix_outside_edges(self):
        mesh = sk.example_mesh()
        s = sk.skeletonize.by_wavefront(mesh, waves=1)
        fixed = sk.post.fix_outside_edges(s, inplace=False, eps='auto')


class TestExamples:
    def test_readme_example(self):
        mesh = sk.example_mesh()
        fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
        skel = sk.skeletonize.by_wavefront(fixed, waves=1, step_size=1)

        assert isinstance(skel.vertices, np.ndarray)
        assert isinstance(skel.edges, np.ndarray)
        assert isinstance(skel.mesh_map, np.ndarray)
        assert skel.vertices.shape[1] == 3
        assert skel.edges.shape[1] == 2
        assert skel.mesh_map.shape[0] == len(skel.mesh.vertices)
