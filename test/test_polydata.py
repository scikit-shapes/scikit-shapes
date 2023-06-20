import sys

sys.path.append(sys.path[0][:-4])
from skshapes.data import PolyData
import torch


def test_polydata_creation():

    # Shape with points and triangles
    points = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.float32)
    triangles = torch.tensor([[0], [1], [2]], dtype=torch.int64)
    triangle = PolyData(points=points, triangles=triangles)

    # edges are computed on the fly when the getter is called, and _edges remains None
    assert triangle.edges is not None
    assert triangle._triangles is not None
    assert triangle._edges is None
    assert triangle.n_triangles == 1
    assert triangle.n_edges == 3  # Should be 3 or not ?
    assert triangle.n_points == 3

    assert triangle.edge_centers is not None
    assert triangle.edge_lengths is not None
    assert triangle.triangle_areas is not None

    # /!\ Assinging edges manually will delete the triangles
    triangle.edges = triangle.edges
    assert triangle.edges is not None
    assert triangle._triangles is None
    assert triangle._edges is not None
    assert triangle.n_triangles == 0
    assert triangle.n_edges == 3
    assert triangle.n_points == 3

    assert triangle.edge_centers is not None
    assert triangle.edge_lengths is not None

    try:
        triangle.triangle_areas
    except ValueError:
        pass
    else:
        raise AssertionError("Assigning edges should delete triangles")
