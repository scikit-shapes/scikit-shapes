import pytest
import skshapes as sks



@pytest.mark.skipif(True, reason="This test is not ready yet")
def test_sphere_trimesh():
    import matplotlib.pyplot as plt
    import pyvista as pv

    import torch

    import trimesh
    from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure, sphere_ball_intersection

    def compute_mean_gauss_curvatures(mesh, scale, method = 'skshapes') :

        if method == 'skshapes':
            shape = sks.PolyData(points=mesh.points,
                                triangles=mesh.triangles,
                                )
            kmax, kmin = shape.point_principal_curvatures(scale=scale)
            Hs = 0.5 * (kmax + kmin)
            Ks = kmax * kmin

        elif method == 'trimesh':
            Hs = discrete_mean_curvature_measure(mesh, mesh.vertices, scale)/sphere_ball_intersection(1, scale)
            Ks = discrete_gaussian_curvature_measure(mesh, mesh.vertices, scale)/sphere_ball_intersection(1, scale)

        return Hs, Ks


    def plot_histogram_curvatures(curvs_1, curvs_2) :
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        # Display statistics on curvs_1
        for curv in [curvs_1, curvs_2]:
            print('Mean:', curv.mean())
            print('Std:', curv.std())
            print('Min:', curv.min())
            print('Max:', curv.max())

        axs[0].hist(curvs_1, bins=100)
        axs[0].set_title('Mean Curvature')
        axs[1].hist(curvs_2, bins=100)
        axs[1].set_title('Gaussian Curvature')
        plt.show()


    def build_sphere(radius, n_points):
        points = torch.randn(n_points, 3)
        points = radius * torch.nn.functional.normalize(points, p=2, dim=1)
        shape = sks.PolyData(points=points)
        return shape

    def build_sphere_2(radius, mode = 'skshapes') :
        sphere = trimesh.creation.icosphere(radius = radius)
        if mode == 'trimesh':
            return sphere
        elif mode == 'skshapes':
            return sks.PolyData(points=sphere.vertices, triangles = sphere.faces)
        

    RADIUS = 1
    #sphere = build_sphere(radius = 1, n_points = 500)
    sphere = build_sphere_2(radius = RADIUS)
    mean_curvs, gauss_curvs = compute_mean_gauss_curvatures(sphere,
                                                            scale = 0.1,
                                                            method = 'skshapes')
    plot_histogram_curvatures(mean_curvs, gauss_curvs)

    print(sphere.triangle_normals)
    print(sphere.point_normals(scale=None))


    pl = pv.Plotter(shape=(1, 3))
    pl.subplot(0, 0)
    #sphere.to_pyvista().plot_normals(mag=0.1, faces=True, show_edges=True)

    pv_sphere = sphere.to_pyvista()
    pl.add_mesh(pv_sphere)

    pv_sphere["vectors"] = sphere.point_normals(scale=None) * 0.3
    pv_sphere.set_active_vectors("vectors")

    # plot just the arrows
    pl.add_mesh(pv_sphere.arrows)


    pl.subplot(0, 1)
    pl.add_mesh(sphere.points.numpy(), scalars=mean_curvs)
    pl.link_views()
    pl.show()

    # trimesh
    sphere = build_sphere_2(radius = RADIUS, mode='trimesh')
    mean_curvs_t, gauss_curvs_t = compute_mean_gauss_curvatures(sphere,
                                                            scale = 5,
                                                            method = 'trimesh')
    plot_histogram_curvatures(mean_curvs_t, gauss_curvs_t)