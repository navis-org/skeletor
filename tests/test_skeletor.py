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

    def test_contraction(self):
        cont = sk.pre.contract(sk.example_mesh(), epsilon=0.1)
        assert isinstance(cont, tm.Trimesh)


class TestSkeletonization:
    def test_wave_skeletonization(self):
        one_wave = sk.skeletonize.by_wavefront(sk.example_mesh(), waves=1)
        two_wave = sk.skeletonize.by_wavefront(sk.example_mesh(), waves=2)
        stepsize = sk.skeletonize.by_wavefront(sk.example_mesh(), waves=1,
                                               step_size=2)

        for s in [one_wave, two_wave, stepsize]:
            assert len(s.mesh_map) == len(s.mesh.vertices)
            assert all(np.isin(s.mesh_map, s.swc.node_id.values))

    def test_vertex_cluster(self):
        s = sk.skeletonize.by_vertex_clusters(sk.example_mesh(),
                                              sampling_dist=100)

        assert len(s.mesh_map) == len(s.mesh.vertices)
        assert all(np.isin(s.mesh_map, s.swc.node_id.values))

    def test_edge_collapse(self):
        s = sk.skeletonize.by_edge_collapse(sk.example_mesh())

    def test_teasar(self):
        s = sk.skeletonize.by_teasar(sk.example_mesh(), 500)

        assert len(s.mesh_map) == len(s.mesh.vertices)
        assert all(np.isin(s.mesh_map, s.swc.node_id.values))

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
