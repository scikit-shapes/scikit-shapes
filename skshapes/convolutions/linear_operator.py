class LinearOperator:
    """A simple wrapper for scaled linear operators."""

    def __init__(self, matrix, input_scaling=None, output_scaling=None):
        M, N = matrix.shape
        assert matrix.shape == (M, N)
        assert input_scaling is None or input_scaling.shape == (N,)
        assert output_scaling is None or output_scaling.shape == (M,)

        self.matrix = matrix
        self.input_scaling = input_scaling
        self.output_scaling = output_scaling

    def __matmul__(self, other):
        assert other.shape[0] == self.matrix.shape[1]
        i_s = self.input_scaling if self.input_scaling is not None else 1
        o_s = self.output_scaling if self.output_scaling is not None else 1

        if len(other.shape) in (1, 2):
            if len(other.shape) == 2:
                if self.input_scaling is not None:
                    i_s = i_s.view(-1, 1)
                if self.output_scaling is not None:
                    o_s = o_s.view(-1, 1)

            return o_s * (self.matrix @ (i_s * other))

        else:
            ndims = len(other.shape) - 1
            if self.input_scaling is not None:
                i_s = self.input_scaling.view(-1, *(ndims * (1,)))
            if self.output_scaling is not None:
                o_s = self.output_scaling.view(-1, *(ndims * (1,)))

            return (self @ other.view(other.shape[0], -1)).reshape(
                self.shape[0], *other.shape[1:]
            )

    @property
    def T(self):
        return LinearOperator(
            self.matrix.T,
            input_scaling=self.output_scaling,
            output_scaling=self.input_scaling,
        )

    @property
    def shape(self):
        return self.matrix.shape
