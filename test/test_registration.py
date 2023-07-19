import skshapes as sks


def test_registration():
    # Load two meshes
    datafolder = "data/SCAPE_low_resolution"
    print(datafolder + "/" + "mesh001.ply")
    source = sks.PolyData(datafolder + "/" + "mesh001.ply")
    target = sks.PolyData(datafolder + "/" + "mesh041.ply")

    # Few type checks
    assert isinstance(source, sks.ShapeType)
    assert isinstance(target, sks.PolyDataType)
    assert source.landmarks is None

    losses = [
        sks.loss.NearestNeighborsLoss(),
        sks.loss.OptimalTransportLoss(),
        sks.loss.L2Loss(),
        0.8 * sks.loss.L2Loss() + 2 * sks.loss.OptimalTransportLoss(),
    ]
    models = [sks.morphing.ElasticMetric(), sks.morphing.RigidMotion()]
    for loss in losses:
        for model in models:
            r = sks.tasks.Registration(
                model=model,
                loss=loss,
                optimizer=sks.LBFGS(),
                n_iter=1,
                gpu=False,
                regularization=1,
            )
            print(loss, model)
            r.fit_transform(source=source, target=target)
