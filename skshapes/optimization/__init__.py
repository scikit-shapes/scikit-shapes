import torch.optim
from ..types import Optimizer


class BaseOptimizer(Optimizer):
    def __init__(self, name, **kwargs):
        """Initialize the optimizer with the given hyperparameters"""
        self.kwargs = kwargs
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, params):
        """Return the optimizer object"""
        return torch.optim.__dict__[self.name](params, **self.kwargs)


class LBFGS(BaseOptimizer):
    def __init__(self, line_search_fn="strong_wolfe", **kwargs):
        super().__init__("LBFGS", line_search_fn=line_search_fn, **kwargs)


class Adam(BaseOptimizer):
    def __init__(self, **kwargs):
        super().__init__("Adam", **kwargs)


class Adagrad(BaseOptimizer):
    def __init__(self, **kwargs):
        super().__init__("Adagrad", **kwargs)


class SGD(BaseOptimizer):
    def __init__(self, lr=0.01, **kwargs):
        super().__init__("SGD", lr=lr, **kwargs)
