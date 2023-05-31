import torch

from .._typing import *


class Registration:
    def __init__(
        self,
        *,
        model,
        loss,
        optimizer,
        regularization=1,
        n_iter=10,
        verbose=0,
        device="auto",
        **kwargs,
    ) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.verbose = verbose
        self.n_iter = n_iter
        self.regularization = regularization

        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

    def fit(
        self,
        *,
        source,
        target,
    ) -> None:
        # Make copies of the source and target and move them to the device
        source = source.copy().to(self.device)
        target = target.copy().to(self.device)

        # Load the tensors and fit the models/loss
        self.model.fit(source=source)
        self.loss.fit(source=source, target=target)

        def loss_fn(parameter):

            if self.regularization == 0:
                return self.loss(self.model.morph(parameter=parameter))

            return self.loss(
                self.model.morph(parameter=parameter)
            ) + self.regularization * self.model.regularization(parameter=parameter)

        # Initialize the parameter tensor using the template provided by the model
        parameter = self.model.parameter_template
        parameter.requires_grad = True

        optimizer = self.optimizer([parameter])

        # Initialize the optimizer
        # if self.optimizer == "LBFGS":
        #     optimizer = torch.optim.LBFGS(
        #         params=[parameter], line_search_fn="strong_wolfe"
        #     )
        # else:
        #     # TODO : add other optimizers
        #     raise NotImplementedError

        # Define the closure
        def closure():
            optimizer.zero_grad()
            loss_value = loss_fn(parameter)
            loss_value.backward()
            return loss_value

        # Run the optimization
        for i in range(self.n_iter):
            loss_value = optimizer.step(closure)
            if self.verbose > 0:
                print(f"Loss value at iteration {i} : {loss_value}")

        self.parameter = parameter.detach()
        self.distance = self.model.regularization(
            parameter=parameter
        ).detach()  # Is it the right way to compute the distance ?

    def transform(self, *, source) -> torch.Tensor:

        morphed_points = self.model.morph(parameter=self.parameter)
        morphed_shape = source.copy()
        morphed_shape.points = morphed_points
        return morphed_shape

    def fit_transform(
        self,
        *,
        source,
        target,
    ) -> torch.Tensor:
        self.fit(source=source, target=target)
        return self.transform(source=source)


class Registration2:
    def __init__(
        self,
        *,
        model,
        loss,
        optimizer="LBFGS",
        regularization=1,
        n_iter=10,
        verbose=0,
        device="auto",
        **kwargs,
    ) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.verbose = verbose
        self.n_iter = n_iter
        self.regularization = regularization

        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

    def fit(
        self,
        *,
        source,
        target,
    ) -> None:
        # Make copies of the source and target and move them to the device
        if source.device != self.device:
            source = source.to(self.device)
        if target.device != self.device:
            target = target.to(self.device)

        # Load the tensors and fit the models/loss
        # self.model.fit(source=source)
        self.loss.fit(source=source, target=target)

        def loss_fn(parameter):

            if self.regularization == 0:
                return self.loss(
                    self.model.morph(shape=source, parameter=parameter).points
                )

            return self.loss(
                self.model.morph(shape=source, parameter=parameter).points
            ) + self.regularization * self.model.regularization(
                shape=source, parameter=parameter
            )

        # Initialize the parameter tensor using the template provided by the model
        N, D = source.points.shape
        parameter = torch.zeros((self.model.n_steps, N, D), device=self.device)
        # parameter = self.model.parameter_template
        parameter.requires_grad = True

        # Initialize the optimizer
        optimizer = self.optimizer([parameter])
        # if self.optimizer == "LBFGS":
        #     optimizer = torch.optim.LBFGS(
        #         params=[parameter], line_search_fn="strong_wolfe"
        #     )
        # else:
        #     # TODO : add other optimizers
        #     raise NotImplementedError

        # Define the closure
        def closure():
            optimizer.zero_grad()
            loss_value = loss_fn(parameter)
            loss_value.backward()
            return loss_value

        # Run the optimization
        for i in range(self.n_iter):
            loss_value = optimizer.step(closure)
            if self.verbose > 0:
                print(f"Loss value at iteration {i} : {loss_value}")

        self.parameter = parameter.detach()
        self.distance = self.model.regularization(
            shape=source, parameter=parameter
        ).detach()  # Is it the right way to compute the distance ?

    def transform(self, *, source) -> torch.Tensor:

        return self.model.morph(shape=source, parameter=self.parameter)

    def fit_transform(
        self,
        *,
        source,
        target,
    ) -> torch.Tensor:
        if source.device != self.device:
            source = source.to(self.device)
        if target.device != self.device:
            target = target.to(self.device)
        self.fit(source=source, target=target)
        return self.transform(source=source)
