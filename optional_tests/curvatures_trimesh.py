# ruff: noqa

import numpy as np

import skshapes as sks


def test_sphere_trimesh():
    import matplotlib.pyplot as plt
    import pyvista as pv
    import trimesh
    from trimesh.curvature import (
        discrete_gaussian_curvature_measure,
        discrete_mean_curvature_measure,
        sphere_ball_intersection,
    )

    def compute_mean_gauss_curvatures(mesh, scale, method="skshapes"):

        if method == "skshapes":
            shape = sks.PolyData(
                points=mesh.points,
                triangles=mesh.triangles,
            )
            Hs, Ks = shape.point_mean_gauss_curvatures(scale=scale)

        elif method == "trimesh":
            Hs = discrete_mean_curvature_measure(
                mesh, mesh.vertices, scale
            ) / sphere_ball_intersection(1, scale)
            Ks = discrete_gaussian_curvature_measure(
                mesh, mesh.vertices, scale
            ) / sphere_ball_intersection(1, scale)

        return Hs, Ks

    def plot_histogram_curvatures(curvs_1, curvs_2):
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        # Display statistics on curvs_1
        for curv in [curvs_1, curvs_2]:
            print(
                "Mean:",
                curv.mean(),
                "Std:",
                curv.std(),
                "Min:",
                curv.min(),
                "Max:",
                curv.max(),
            )

        axs[0].hist(curvs_1, bins=100)
        axs[0].set_title("Mean Curvature")
        axs[1].hist(curvs_2, bins=100)
        axs[1].set_title("Gaussian Curvature")
        plt.show()

    RADIUS = 2

    if False:
        shape_tri = trimesh.creation.icosphere(radius=RADIUS, subdivisions=4)
    elif True:
        shape_tri = trimesh.creation.torus(
            major_radius=1, major_sections=100, minor_radius=0.45
        )
    elif False:
        shape_tri = trimesh.creation.capsule(radius=1, height=0.2)
    else:
        shape_pv = pv.ParametricRandomHills(
            u_res=30, v_res=30, w_res=30
        ).triangulate()
        diam = shape_pv.points.max() - shape_pv.points.min()

        shape_tri = trimesh.Trimesh(
            vertices=shape_pv.points / diam,
            faces=shape_pv.faces.reshape(-1, 4)[:, 1:],
        )
        print(shape_tri)
        print(shape_tri.vertices.max(), shape_tri.vertices.min())

    shape_sks = sks.PolyData(
        points=shape_tri.vertices, triangles=shape_tri.faces
    )
    print(shape_sks.n_points)

    for SCALE in [0.1, 0.2, 0.3, 0.4, 0.5, 0.1]:
        for mean_only in [True, False]:
            mean_curvs, gauss_curvs = shape_sks.point_mean_gauss_curvatures(
                scale=SCALE, mean_only=mean_only
            )
            kmax, kmin = shape_sks.point_principal_curvatures(
                scale=SCALE, mean_only=mean_only
            )
            print(
                SCALE,
                mean_only,
                "mean:",
                mean_curvs.mean().item(),
                "gauss:",
                gauss_curvs.mean().item(),
                "kmax:",
                kmax.mean(),
                "kmin:",
                kmin.mean(),
            )
        print("")

    # print(shape_sks.triangle_normals)
    # print(shape_sks.point_normals(scale=None))

    # sphere.to_pyvista().plot_normals(mag=0.1, faces=True, show_edges=True)

    # trimesh
    mean_curvs_t, gauss_curvs_t = compute_mean_gauss_curvatures(
        shape_tri, scale=SCALE, method="trimesh"
    )

    # Compute kmax_t and kmin_t from mean_curvs_t and gauss_curvs_t
    kmax_t = mean_curvs_t + np.sqrt(
        np.maximum(0, mean_curvs_t**2 - gauss_curvs_t)
    )
    kmin_t = mean_curvs_t - np.sqrt(
        np.maximum(0, mean_curvs_t**2 - gauss_curvs_t)
    )

    # Use seaborn to plot the histograms
    import seaborn as sns

    data = {
        "mean_curvs": mean_curvs,
        "gauss_curvs": gauss_curvs,
        "kmax": kmax,
        "kmin": kmin,
        "mean_curvs_t": mean_curvs_t,
        "gauss_curvs_t": gauss_curvs_t,
        "kmax_t": kmax_t,
        "kmin_t": kmin_t,
    }
    print(mean_curvs.mean(), mean_curvs_t.mean())

    # sns.jointplot(data, x="mean_curvs", y="gauss_curvs", kind="kde")
    sns.jointplot(data, x="kmax", y="kmin", kind="kde")
    sns.jointplot(data, x="kmax_t", y="kmin_t", kind="kde")
    plt.show()

    # Normalize the curvatures by max absolute value
    # mean_curvs = mean_curvs / torch.abs(mean_curvs).max()
    # gauss_curvs = gauss_curvs / torch.abs(gauss_curvs).max()

    # kmax = kmax / torch.abs(kmax).max()
    # kmin = kmin / torch.abs(kmin).max()

    # mean_curvs_t = mean_curvs_t / np.abs(mean_curvs_t).max()
    # gauss_curvs_t = gauss_curvs_t / np.abs(gauss_curvs_t).max()

    shape_pv = shape_sks.to_pyvista()

    pl = pv.Plotter(shape=(2, 3))

    pl.subplot(0, 0)
    shape_pv["smooth_normals"] = shape_sks.point_normals(scale=SCALE) * 0.3
    shape_pv.set_active_vectors("smooth_normals")
    pl.add_mesh(shape_pv.arrows)

    pl.subplot(1, 0)
    pl.add_mesh(shape_pv, color="white", show_edges=True)
    shape_pv["vectors"] = shape_sks.point_normals(scale=None) * 0.3
    shape_pv.set_active_vectors("vectors")
    pl.add_mesh(shape_pv.arrows)

    pl.subplot(0, 1)
    pl.add_mesh(shape_sks.points.numpy(), scalars=kmin)

    pl.subplot(0, 2)
    pl.add_mesh(shape_sks.points.numpy(), scalars=kmax)

    pl.subplot(1, 1)
    pl.add_mesh(shape_sks.points.numpy(), scalars=kmin_t)
    pl.subplot(1, 2)
    pl.add_mesh(shape_sks.points.numpy(), scalars=kmax_t)

    pl.link_views()
    pl.show()


test_sphere_trimesh()
