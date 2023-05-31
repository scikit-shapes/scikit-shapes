import torch.optim

from .._typing import *


class BaseOptimizer(OptimizerType):
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


# def Optimizer_fromtorch(name, default_kwargs=dict()):

#     # Check that the optimizer exists in torch.optim
#     if name not in dir(torch.optim):
#         raise ValueError(f"Optimizer {name} not found in torch.optim")

#     def __init__(self, **kwargs):

#         print(default_kwargs)
#         self.kwargs = {**default_kwargs, **kwargs}

#         for key, value in self.kwargs.items():
#             setattr(self, key, value)

#     def __call__(self, params):
#         return torch.optim.__dict__[name](params, **self.kwargs)

#     #TODO define setters and getters for the attributes ?

#     base = (OptimizerType,)

#     attrs = {
#         "__init__": __init__,
#         "__call__": __call__,
#     }

#     return type(name, base, attrs)


# # Get the list of optimizers in torch.optim
# # There are the modules in torch.optim that start with a capital letter
# optimizers = [name for name in dir(torch.optim) if name[0].isupper()]

# # Create a class for each optimizer
# for optimizer in optimizers:
#     if optimizer == "LBFGS":
#         default_kwargs = {'line_search_fn': 'strong_wolfe'}
#     if optimizer == "SGD":
#         default_kwargs = {'lr': 0.1}
#     else:
#         default_kwargs = dict()

#     globals()[optimizer] = Optimizer_fromtorch(name=optimizer, default_kwargs=default_kwargs)
