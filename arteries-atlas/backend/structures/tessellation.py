import numpy as np

from ..utils.numpy_utils import angle_to_vect, orthogonal


class BoundingBox:
    def __init__(self, anchors=None, cone=None, orientation=None):
        self.anchors = anchors
        self.cone = cone
        self.orientation = orientation

        self._base_vector = None
        self._border_vector = None
        self._orientation_vector = None

    def update(self, left_anchor=None, right_anchor=None, left_cone=None, right_cone=None, orientation=None):
        if left_anchor is not None:
            self.anchors[0] = left_anchor.copy()
        if right_anchor is not None:
            self.anchors[1] = right_anchor.copy()
        if left_cone is not None:
            self.cone[0] = left_cone.copy()
            self._border_vector = None
        if right_cone is not None:
            self.cone[1] = right_cone.copy()
            self._border_vector = None
        if orientation is not None:
            self.orientation = orientation
            self._orientation_vector = None
            self._base_vector = None

    def orientation_vector(self):
        if self._orientation_vector is None:
            self._orientation_vector = angle_to_vect(self.orientation).flatten()

        return self._orientation_vector

    def base_vector(self, orient='left'):
        if self._base_vector is None:
            self._base_vector = orthogonal(self.orientation_vector())
        return self._base_vector if orient == 'left' else - self._base_vector

    def border_vectors(self):
        if self._border_vector is None:
            self._border_vector = np.stack([angle_to_vect(self.cone[0]), angle_to_vect(self.cone[1])])
        return self._border_vector

class Tessellation:
    def __init__(self, size, root, root_anchors, root_cone, root_orientation):
        self.size = size
        self.boxes = [BoundingBox() for _ in range(size)]
        self.boxes[root] = BoundingBox(anchors=root_anchors, cone=root_cone, orientation=root_orientation)

        self.parents = np.zeros(size)
        self.children = [[] for _ in range(size)]

    def cone(self, node):
        return self.boxes[node].cone

    def anchors(self, node):
        return self.boxes[node].anchors

    def update(self, nodes, left_anchors=None, right_anchors=None, left_cones=None, right_cones=None, orientations=None):
        for i, node in enumerate(nodes):
            left_anchor = left_anchors[i] if left_anchors is not None else None
            right_anchor = right_anchors[i] if right_anchors is not None else None
            left_cone = left_cones[i] if left_cones is not None else None
            right_cone = right_cones[i] if right_cones is not None else None
            orientation = orientations[i] if orientations is not None else None
            self.boxes[node].update(left_anchor, right_anchor, left_cone, right_cone, orientation)

    def partition(self, parent, children, proportions):
        bounds = np.cumsum(proportions)
        cones = self.boxes[parent].cone

        cone_limits = cones[0] + (cones[1] - cones[0]) * np.concatenate([[0], bounds / bounds[-1]])
        for i, child in enumerate(children):
            self.boxes[child].cone = [cone_limits[i], cone_limits[i + 1]]
            self.boxes[child].anchors = self.anchors(parent).copy()
            self.boxes[child].orientation = (self.boxes[child].cone[0] + self.boxes[child].cone[1]) / 2

        self.parents[children] = parent
        self.children[parent] = children

    def children_boxes(self, parent: int):
        return [self.boxes[i] for i in self.children[parent]]
