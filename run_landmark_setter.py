from skshapes.applications import LandmarkSetter
from skshapes.data import read

meshes = [read("data/SCAPE_low_resolution/mesh00{}.ply".format(i)) for i in range(1, 5)]

app = LandmarkSetter(meshes=meshes)
app.start()


import torch

for i, mesh in enumerate(meshes):
    print(mesh.landmarks_3d)
    torch.save(mesh.landmarks, "data/SCAPE_low_resolution/landmarks00{}.pt".format(i))
