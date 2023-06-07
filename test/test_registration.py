def test_registration():

    from skshapes.tasks import Registration
    from skshapes.data import read
    import torch
    import skshapes

    from skshapes.loss import NearestNeighborsLoss, OptimalTransportLoss, LandmarkLoss
    from skshapes.morphing import ElasticMetric, RigidMotion
    from skshapes.optimization import LBFGS

    # Load two meshes
    datafolder = "data/SCAPE_low_resolution"
    source = read(datafolder + "/" + "mesh001.ply")
    target = read(datafolder + "/" + "mesh041.ply")

    # Few type checks
    assert isinstance(source, skshapes.Shape)
    assert isinstance(target, skshapes.PolyData)

    assert source.landmarks is None
    # Add landmarks
    source.landmarks = torch.arange(source.points.shape[0], dtype=skshapes.int)
    target.landmarks = torch.arange(target.points.shape[0], dtype=skshapes.int)
    assert source.landmarks is not None

    for loss in [NearestNeighborsLoss(), OptimalTransportLoss(), LandmarkLoss()]:
        for model in [ElasticMetric(), RigidMotion()]:

            r = Registration(
                model=model,
                loss=loss,
                optimizer=LBFGS(),
                n_iter=1,
                device="cpu",
                regularization=1,
            )
            print(loss, model)
            r.fit_transform(source=source, target=target)
