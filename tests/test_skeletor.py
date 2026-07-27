import pytest

import skeletor as sk
import trimesh as tm
import networkx as nx
import numpy as np
import scipy.spatial


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
