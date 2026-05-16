import numpy as np

from ..utils.numpy_utils import integrate_angular_path


class Trajectory:
    def __init__(self, lengths, angles, initial_node=0):
        self.lengths = lengths.copy()
        self.angles = angles.copy()
        self.initial_node = initial_node

        self.emb = None
        self.generate_embeddings()

        self.shape = self.angles.shape[0]

    def update_angles(self, angles):
        self.angles = angles
        self.generate_embeddings()

    def generate_embeddings(self):
        embedding_path = integrate_angular_path(self.lengths, self.angles, initial_position=self.initial_node)

        if self.emb is None:
            self.emb = np.append([self.initial_node], embedding_path, axis=0)
        else:
            self.emb[1:] = embedding_path


class BranchedTrajectories:
    def __init__(self, trajectories_lengths, trajectories_angles, initial_node=0):
        self.size = len(trajectories_angles)
        self.trajectories = [Trajectory(trajectories_lengths[i], trajectories_angles[i], initial_node) for i in
                             range(self.size)]

    def generate_embeddings(self):  # TODO supprimer ?
        for traj in self.trajectories:
            traj.generate_embeddings()

    def get_embeddings(self):  # TODO supprimer ?
        return [traj.emb for traj in self.trajectories]

    def order_consistency(self):
        init_angles = [traj.angles[0] for traj in self.trajectories]
        return np.all(np.argsort(np.squeeze(init_angles)) == np.arange(len(init_angles)))
