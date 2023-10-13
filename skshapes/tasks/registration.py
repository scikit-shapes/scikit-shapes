"""Registration between two shapes."""

from ..optimization import Optimizer
from ..types import typecheck, Union, shape_type
from ..loss import Loss
from ..morphing import Model
import torch


class Registration:
    @typecheck
    def __init__(
        self,
        *,
        model: Model,
        loss: Loss,
        optimizer: Optimizer,
        regularization: Union[int, float] = 1,
        n_iter: int = 10,
        verbose: int = 0,
        gpu: bool = True,
        device: Union[str, torch.device] = "auto",
        **kwargs,
    ) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.verbose = verbose
        self.n_iter = n_iter
        self.regularization = regularization

        if gpu:
            self.optim_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        else:
            self.optim_device = torch.device("cpu")

        self.device = device

    @typecheck
    def fit(
        self,
        *,
        source: shape_type,
        target: shape_type,
    ) -> None:
        # Check that the shapes are on the same device
        assert (
            source.device == target.device
        ), "Source and target shapes must be on the same device, found source on device {} and target on device {}".format(
            source.device, target.device
        )
        self.output_device = (
            source.device
        )  # Save the device on which the output will be

        # Make copies of the source and target and move them to the optimization device
        if source.device != self.optim_device:
            source = source.to(self.optim_device)
        if target.device != self.optim_device:
            target = target.to(self.optim_device)

        print("source.device", source.device)
        print("target.device", target.device)

        # Define the loss function
        def loss_fn(parameter):
            return_regularization = False if self.regularization == 0 else True
            morphing = self.model.morph(
                shape=source,
                parameter=parameter,
                return_path=False,
                return_regularization=return_regularization,
            )

            return (
                self.loss(source=morphing.morphed_shape, target=target)
                + self.regularization * morphing.regularization
            )

        # Initialize the parameter tensor using the template provided by the model, and set it to be optimized
        parameter_shape = self.model.parameter_shape(shape=source)
        parameter = torch.zeros(parameter_shape, device=self.optim_device)
        parameter.requires_grad = True

        # Initialize the optimizer
        optimizer = self.optimizer([parameter])

        # Define the closure for the optimizer
        def closure():
            optimizer.zero_grad()
            loss_value = loss_fn(parameter)
            loss_value.backward()
            return loss_value

        print("ok so far")

        # Run the optimization
        for i in range(self.n_iter):
            loss_value = optimizer.step(closure)
            if self.verbose > 0:
                print(f"Loss value at iteration {i} : {loss_value}")

        # Store the device type of the parameter (useful for testing purposes)
        self.internal_parameter_device_type = parameter.device.type

        # Move the parameter to the output device
        self.parameter_ = parameter.clone().detach().to(self.output_device)

        print("few chances to get here")

        morphing = self.model.morph(
            shape=source,
            parameter=parameter,
            return_path=True,
            return_regularization=True,
        )

        print("few chances to get here")

        self.regularization_ = morphing.regularization.to(self.output_device)
        self.transformed_shape_ = morphing.morphed_shape
        self.path_ = morphing.path

        if self.transformed_shape_.device != self.output_device:
            self.transformed_shape_ = self.transformed_shape_.to(self.output_device)
            self.path_ = [s.to(self.output_device) for s in self.path_]

    @typecheck
    def transform(self, *, source: shape_type) -> shape_type:
        transformed_shape = self.model.morph(
            shape=source, parameter=self.parameter_
        ).morphed_shape

        if transformed_shape.device != self.output_device:
            return transformed_shape.to(self.output_device)
        else:  # If transformed_shape is already on the output device, avoid a copy
            return transformed_shape

    @typecheck
    def fit_transform(
        self,
        *,
        source: shape_type,
        target: shape_type,
    ) -> shape_type:
        # if source.device != self.device:
        #     source = source.to(self.device)
        # if target.device != self.device:
        #     target = target.to(self.device)
        self.fit(source=source, target=target)
        return self.transform(source=source)
