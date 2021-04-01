import skeletor as sk
import trimesh as tm


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

    def test_vertex_cluster(self):
        s = sk.skeletonize.by_vertex_clusters(sk.example_mesh(),
                                              sampling_dist=100)

    def test_edge_collapse(self):
        s = sk.skeletonize.by_edge_collapse(sk.example_mesh())


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
