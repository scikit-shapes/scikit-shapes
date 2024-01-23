"""Registration between two shapes."""
from __future__ import annotations

from typing import get_args

import torch

from ..errors import DeviceError, NotFittedError, ShapeError
from ..input_validation import convert_inputs, typecheck
from ..loss import Loss
from ..morphing import Model
from ..optimization import LBFGS, Optimizer
from ..types import (
    FloatTensor,
    shape_type,
)


class Registration:
    """Registration class.

    This class implements the registration between two shapes. It must be
    initialized with a model, a loss and an optimizer. The registration is
    performed by calling the fit method with the source and target shapes as
    arguments. The transform method can then be used to transform a new shape
    using the learned registration's parameter.
    """

    @typecheck
    def __init__(
        self,
        *,
        model: Model,
        loss: Loss,
        optimizer: Optimizer | None = None,
        regularization: int | float = 1,
        n_iter: int = 10,
        verbose: int = 0,
        gpu: bool = True,
    ) -> None:
        """Initialize the registration object.

        Parameters
        ----------
        model
            a model object (from skshapes.morphing)
        loss
            a loss object (from skshapes.loss)
        optimizer
            an optimizer object (from skshapes.optimization)
        regularization
            the regularization parameter for the criterion :
            loss + regularization * reg.
        n_iter
            number of iteration for optimization process.
        verbose
            positive to print the losses after each optimization loop iteration
        gpu:
            do intensive numerical computations on a nvidia gpu with a cuda
            backend if available.
        """
        if optimizer is None:
            optimizer = LBFGS()

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.verbose = verbose
        self.n_iter = n_iter
        self.regularization = regularization

        if gpu:
            if torch.cuda.is_available():
                self.optim_device = "cuda"
            else:
                self.optim_device = "cpu"

        else:
            self.optim_device = "cpu"

    @convert_inputs
    @typecheck
    def fit(
        self,
        *,
        source: shape_type,
        target: shape_type,
        initial_parameter: FloatTensor | None = None,
    ) -> Registration:
        """Fit the registration between the source and target shapes.

        After calling this method, the registration's parameter can be accessed
        with the parameter_ attribute, the transformed shape with the
        transformed_shape_ attribute and the list of successives shapes during
        the registration process with the path_ attribute.

        Parameters
        ----------
        source : shape_object
            a shape object (from skshapes.shapes)
        target
            a shape object (from skshapes.shapes)
        initial_parameter
            an initial parameter tensor for the optimization process.
            Defaults to None.

        Raises
        ------
        DeviceError
            if the source and target shapes are not on the same device.

        Returns
        -------
        Registration
            self

        """
        # Check that the shapes are on the same device
        if source.device != target.device:
            raise DeviceError(
                "Source and target shapes must be on the same device, found"
                + "source on device {} and target on device {}".format(
                    source.device, target.device
                )
            )

        self.output_device = (
            source.device
        )  # Save the device on which the output will be

        # Make copies of the source and target and move them to the
        # optimization device
        if source.device != self.optim_device:
            source = source.to(self.optim_device)
        if target.device != self.optim_device:
            target = target.to(self.optim_device)

        # Define the loss function
        def loss_fn(parameter):
            return_regularization = self.regularization != 0
            morphing = self.model.morph(
                shape=source,
                parameter=parameter,
                return_path=False,
                return_regularization=return_regularization,
            )

            loss = self.loss(source=morphing.morphed_shape, target=target)
            total_loss = loss + self.regularization * morphing.regularization

            self.current_loss = loss.clone().detach()
            self.current_path_length = morphing.regularization.clone().detach()

            return total_loss

        # Initialize the parameter tensor
        parameter_shape = self.model.parameter_shape(shape=source)
        if initial_parameter is not None:
            # if a parameter is provided, check that it has the right shape
            # and move it to the optimization device
            if initial_parameter.shape != parameter_shape:
                msg = (
                    f"Initial parameter has shape {initial_parameter.shape}"
                    + f" but the model expects shape {parameter_shape}"
                )
                raise ShapeError(msg)

            parameter = initial_parameter.clone().detach()
            parameter = parameter.to(self.optim_device)
        else:
            # if no parameter is provided, initialize it with zeros
            parameter = self.model.inital_parameter(shape=source)

        parameter.requires_grad = True

        # Initialize the optimizer
        optimizer = self.optimizer([parameter])

        # Define the closure for the optimizer
        def closure():
            optimizer.zero_grad()
            loss_value = loss_fn(parameter)
            loss_value.backward()
            return loss_value

        loss_value = loss_fn(parameter)
        if self.verbose > 0:
            if self.regularization != 0:
                pass
            else:
                pass

        fidelity_history = []
        path_length_history = []

        fidelity_history.append(self.current_loss)
        path_length_history.append(self.current_path_length)
        # Run the optimization
        for _i in range(self.n_iter):
            loss_value = optimizer.step(closure)
            fidelity_history.append(self.current_loss)
            path_length_history.append(self.current_path_length)
            if self.verbose > 0:
                loss_value = loss_fn(parameter)
                if self.regularization != 0:
                    pass
                else:
                    pass

        # Store the device type of the parameter (useful for testing purposes)
        self.internal_parameter_device_type = parameter.device.type

        # Move the parameter to the output device
        self.parameter_ = parameter.clone().detach().to(self.output_device)

        morphing = self.model.morph(
            shape=source,
            parameter=parameter,
            return_path=True,
            return_regularization=True,
        )

        # Automatically add attributes with final values of the morphing
        for attr in dir(morphing):
            if not attr.startswith("_") and attr not in ["count", "index"]:
                attribute_name = attr + "_"
                attribute_value = getattr(morphing, attr)

                if isinstance(attribute_value, torch.Tensor):
                    attribute_value = attribute_value.detach().to(
                        self.output_device
                    )

                if isinstance(attribute_value, get_args(shape_type)):
                    attribute_value = attribute_value.to(self.output_device)
                if isinstance(attribute_value, list) and all(
                    isinstance(s, get_args(shape_type))
                    for s in attribute_value
                ):
                    attribute_value = [
                        s.to(self.output_device) for s in attribute_value
                    ]

                setattr(self, attribute_name, attribute_value)

        if self.n_iter == 0:
            loss_value = loss_fn(parameter=parameter)

        self.loss_ = loss_value.detach().to(self.output_device)

        self.fidelity_history_ = torch.stack(fidelity_history).to(
            self.output_device
        )
        self.path_length_history_ = torch.stack(path_length_history).to(
            self.output_device
        )
        return self

    @typecheck
    def transform(self, *, source: shape_type) -> shape_type:
        """Apply the registration to a new shape.

        Parameters
        ----------
        source
            the shape to transform.

        Returns
        -------
        shape_type
            the transformed shape.
        """
        if not hasattr(self, "morphed_shape_"):
            msg = "The registration must be fitted before calling transform"
            raise NotFittedError(msg)
        return self.model.morph(
            shape=source,
            parameter=self.parameter_,
            return_path=True,
            return_regularization=True,
        ).morphed_shape

    @convert_inputs
    @typecheck
    def fit_transform(
        self,
        *,
        source: shape_type,
        target: shape_type,
        initial_parameter: FloatTensor | None = None,
    ) -> shape_type:
        """Fit the registration and apply it to the source shape."""
        self.fit(
            source=source, target=target, initial_parameter=initial_parameter
        )
        return self.transform(source=source)
