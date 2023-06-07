from skshapes.applications import LandmarkSetter
from skshapes.data import read

meshes = [read("data/SCAPE_low_resolution/mesh00{}.ply".format(i)) for i in range(1, 5)]

app = LandmarkSetter(meshes=meshes)
app.start()

print(app.landmarks)
