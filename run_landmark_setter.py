from skshapes.applications import LandmarkSetter
from skshapes.data import read

meshes = [read("data/SCAPE_low_resolution/mesh00{}.ply".format(i)) for i in range(1, 5)]

import torch

meshes[0].landmarks = torch.Tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])

app = LandmarkSetter(meshes=meshes)
app.start()


for mesh in meshes:
    print(mesh.landmarks)
