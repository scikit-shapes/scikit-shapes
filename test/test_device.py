import pyvista.examples
import skshapes as sks
import torch


def test():

    bunny = sks.PolyData.from_pyvista(pyvista.examples.download_bunny_coarse())
    airplane = sks.PolyData.from_pyvista(pyvista.examples.load_airplane())

    bunny = bunny.to("cpu")
    airplane = airplane.to("cuda")

    for model in [sks.RigidMotion(), sks.ElasticMetric()]:

        r = sks.Registration(
            model=model,
            loss=sks.NearestNeighborsLoss(),
            optimizer=sks.SGD(),
            n_iter=2,
            device="cuda",
        )

        r.fit_transform(source=bunny, target=airplane)

        cuda_device = torch.Tensor().cuda().device
        cpu_device = torch.Tensor().cpu().device

        # Test that the device of the parameter is correct
        assert r.parameter.device == cuda_device

        # Check that the device of the shapes is not changed
        assert airplane.points.device == cuda_device
        assert airplane.triangles.device == cuda_device
        assert bunny.points.device == cpu_device
        assert bunny.triangles.device == cpu_device

        # Check that Model.morph gives priority to the device of the shape
        newshape = model.morph(shape=bunny, parameter=r.parameter).morphed_shape
        assert newshape.points.device == cpu_device
        assert newshape.triangles.device == cpu_device

    # Check that the loss cannot compare shapes on different devices
    bunny_cuda = bunny.to("cuda")
    bunny_cpu = bunny.to("cpu")

    for loss in [
        sks.NearestNeighborsLoss(),
        sks.OptimalTransportLoss(),
        sks.L2Loss(),
    ]:

        try:
            sks.NearestNeighborsLoss()(bunny_cuda, bunny_cpu)
        except:
            pass
        else:
            raise AssertionError(
                "Loss should not be able to compare shapes on different devices"
            )
