import pytest

import skeletor as sk
import trimesh as tm
import networkx as nx
import numpy as np


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

    def test_wave_exact(self):
        s = sk.skeletonize.by_wavefront_exact(sk.example_mesh(), step_size=50)

        assert len(s.mesh_map) == len(s.mesh.vertices)
        assert all(np.isin(s.mesh_map, s.swc.node_id.values))

    def test_vertex_cluster(self):
        s = sk.skeletonize.by_vertex_clusters(sk.example_mesh(),
                                              sampling_dist=100)

        assert len(s.mesh_map) == len(s.mesh.vertices)
        assert all(np.isin(s.mesh_map, s.swc.node_id.values))

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
