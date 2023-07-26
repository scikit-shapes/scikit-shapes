import skshapes as sks
import pyvista
import torch
import pytest


def test_decimation_basic():
    ## Assert that the if we fit_transform the decimator from sks on a mesh, and then transform a copy of the mesh
    ## the result is the same

    sphere = sks.PolyData(pyvista.Sphere())
    sphere_copy = sks.PolyData(pyvista.Sphere())

    decimation = sks.QuadricDecimation(n_points=15, implementation="sks")

    # fit the decimator on the mesh
    decimated_sphere = decimation.fit_transform(sphere)
    # transform a copy of the mesh
    decimated_sphere_copy = decimation.transform(sphere_copy)

    # Assert that the result is the same
    assert torch.allclose(decimated_sphere.points, decimated_sphere_copy.points)
    assert torch.allclose(decimated_sphere.triangles, decimated_sphere_copy.triangles)

    ## Assert that the decimator with implementation="vtk" gives the same result as pyvista

    decimation = sks.QuadricDecimation(target_reduction=0.8, implementation="vtk")
    decimated_sphere_sks = decimation.fit_transform(sphere)
    decimated_sphere_vtk = sks.PolyData(
        sphere.to_pyvista().decimate(target_reduction=0.8)
    )

    assert torch.allclose(decimated_sphere_sks.points, decimated_sphere_vtk.points)
    assert torch.allclose(
        decimated_sphere_sks.triangles, decimated_sphere_vtk.triangles
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Cuda is required for this test"
)
def test_decimation_gpu():
    # Assert that the decimation works with PolyData on the gpu
    sphere = sks.PolyData(pyvista.Sphere(), device="cuda")

    dec = sks.QuadricDecimation(n_points=15, implementation="sks")
    dec.fit(sphere)
    newsphere = dec.transform(sphere.to("cpu"))
    assert newsphere.points.device.type == "cpu"

    newsphere = dec.transform(sphere.to("cuda"))
    assert newsphere.points.device.type == "cuda"
