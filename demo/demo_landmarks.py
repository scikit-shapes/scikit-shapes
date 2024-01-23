from typing import Optional

shape1 = MultiShape(
    {
        "heart": {
            "shape": "surface.vtk",
            "landmarks": SlidingLandmarks(SparseMatrix, normals=normals),
            "controls": DeformationModule(
                SparseMatrix,
            ),
        },
        "lungs": "lungs.vtk",
    }
)

shape2 = Shape(
    shape="heart.vtk",
    landmarks=SlidingLandmarks(SparseMatrix, normals=normals),
    controls=DeformationModule(
        SparseMatrix,
    ),
)

shape1["heart"].landmarks
shape2.landmarks


Loss = KernelLoss()["heart"] + LandmarksLoss(p=2)


shape_3 = MultiResolution(
    shapes=["fine.vtk", "medium.vtk", "coarse.vtk"],
    landmarks=[landmarks_fine, landmarks_medium, landmarks_coarse],
    controls=[...],
    scales=[0.01, 0.1, 0.5],
    downsampling=[fine_to_medium, medium_to_coarse],
    upsampling="auto",
)

shape_4 = Shape(
    "fine.vtk", landmarks=landmarks, controls=controls
).multiresolution(scales=[0.01, 0.1, 0.5])

# fine_to_medium = a linear operator of shape (n_medium, n_fine)
# it should support "@" and ".shape"
# Users can specify "scales" or "n_points"


class KernelLoss:
    def __init__(self, kernel, subshape: Optional[str] = None):
        self.kernel = kernel
        self.subshape = subshape

    def __getitem__(self, subshape: str):
        return KernelLoss(self.kernel, subshape=subshape)

    def __call__(self, shape1, shape2):
        if self.subshape is None:
            return self.kernel(shape1, shape2)
        else:
            return self.kernel(shape1[self.subshape], shape2[self.subshape])
