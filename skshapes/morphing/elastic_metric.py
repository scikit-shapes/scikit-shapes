import torch


class IntrinsicMetric:
    def __init__(self, n_steps) -> None:
        self.n_steps = n_steps

    def morph(self, parameter, return_path=False):
        N, D = self.source_points.shape

        if return_path:
            # Compute the cumulative sum of the velocity sequence
            cumvelocities = torch.cat(
                (
                    torch.zeros(size=(1, N, D), device=parameter.device),
                    torch.cumsum(parameter, dim=0),
                )
            )
            # Compute the path by adding the cumulative sum of the velocity sequence to the initial shape
            return (
                self.source_points.repeat(self.n_steps + 1, 1).reshape(
                    self.n_steps + 1, N, D
                )
                + cumvelocities
            )

        else:
            return self.source_points + torch.sum(parameter, dim=0)

    def regularization(self, parameter):
        shape_sequence = self.morph(parameter, return_path=True)
        reg = 0
        for i in range(self.n_steps):
            reg += self.metric(shape_sequence[i], parameter[i])
        return reg / (2 * self.n_steps)

    @property
    def parameter_template(self):
        return torch.zeros(
            self.n_steps, *self.source_points.shape, device=self.source_points.device
        )


class ElasticMetric(IntrinsicMetric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fit(self, *, source):
        assert hasattr(
            source, "edges"
        ), "The shape must have edges to use the as-isometric-as-possible metric"

        self.edges_0 = source.edges[0]
        self.edges_1 = source.edges[1]

        self.source_points = source.points
        return self

    def metric(self, points, parameter):
        a1 = (
            (parameter[self.edges_0] - parameter[self.edges_1])
            * (points[self.edges_0] - points[self.edges_1])
        ).sum(dim=1)
        a2 = (
            (parameter[self.edges_0] - parameter[self.edges_1])
            * (points[self.edges_0] - points[self.edges_1])
        ).sum(dim=1)

        return torch.sum(a1 * a2)
