"""Registration between two shapes."""

from __future__ import annotations

from typing import get_args

import torch

from ..errors import DeviceError, NotFittedError, ShapeError
from ..input_validation import convert_inputs, typecheck
from ..loss.baseloss import BaseLoss
from ..morphing.basemodel import BaseModel
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

    It must be initialized with a model, a loss and an optimizer. The
    registration is performed by calling the fit method with the source and
    target shapes as arguments. The transform method can then be used to
    transform the source shape using the learned registration's parameter.

    The optimization criterion is the sum of the fidelity and the regularization
    term, weighted by the regularization_weight parameter:

    $$ \\text{loss}(\\theta) = \\text{fid}(\\text{Morph}(\\theta, \\text{source}), \\text{target}) + \\text{regularization_weight} \\times \\text{reg}(\\theta)$$

    The fidelity term $\\text{fid}$ is given by the loss object, the
    regularization term $\\text{reg}$ and the morphing $\\text{Morph}$ are
    given by the model.

    Parameters
    ----------
    model
        a model object (from skshapes.morphing)
    loss
        a loss object (from skshapes.loss)
    optimizer
        an optimizer object (from skshapes.optimization)
    regularization_weight
        the regularization_weight parameter for the criterion :
        fidelity + regularization_weight * regularization.
    n_iter
        number of iteration for optimization loop.
    verbose
        positive to print the losses after each optimization loop iteration
    gpu:
        do intensive numerical computations on a nvidia gpu with a cuda
        backend if available.
    debug:
        if True, information will be stored during the optimization process

    Examples
    --------

    >>> model = sks.RigidMotion()
    >>> loss = sks.OptimalTransportLoss()
    >>> optimizer = sks.SGD(lr=0.1)
    >>> registration = sks.Registration(
    ...     model=model, loss=loss, optimizer=optimizer
    ... )
    >>> registration.fit(source=source, target=target)
    >>> transformed_source = registration.transform(source=source)
    >>> # Access the parameter
    >>> parameter = registration.parameter_
    >>> # Access the loss
    >>> loss = registration.loss_
    >>> # Access the fidelity term
    >>> fidelity = registration.fidelity_
    >>> # Access the regularization term
    >>> regularization = registration.regularization_

    More examples can be found in the
    [gallery](../../../generated/gallery/#registration).

    """

    @typecheck
    def __init__(
        self,
        *,
        model: BaseModel,
        loss: BaseLoss,
        optimizer: Optimizer | None = None,
        regularization_weight: int | float = 1,
        n_iter: int = 10,
        verbose: int = 0,
        gpu: bool = True,
        debug=False,
    ) -> None:
        if optimizer is None:
            optimizer = LBFGS()

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.verbose = verbose
        self.n_iter = n_iter
        self.regularization_weight = regularization_weight
        self.debug = debug

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
        with the ``parameter_`` attribute, the transformed shape with the
        ``transformed_shape_`` attribute and the list of successives shapes during
        the registration process with the ``path_`` attribute.

        Parameters
        ----------
        source : shape_object
            a shape object (from skshapes.shapes)
        target
            a shape object (from skshapes.shapes)
        initial_parameter
            an initial parameter tensor for the optimization process. If None,
            the parameter is initialized with zeros. Defaults to None.

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
                + f"source on device {source.device} and target on device {target.device}"
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
            return_regularization = self.regularization_weight != 0
            morphing = self.model.morph(
                shape=source,
                parameter=parameter,
                return_path=False,
                return_regularization=return_regularization,
            )

            loss = self.loss(source=morphing.morphed_shape, target=target)
            total_loss = (
                loss + self.regularization_weight * morphing.regularization
            )

            self.current_loss = loss.clone().detach()
            self.current_regularization = (
                morphing.regularization.clone().detach()
            )

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
            if self.debug:
                parameters_list[-1].append(parameter.clone().detach())
                gradients_list[-1].append(parameter.grad.clone().detach())
            return loss_value

        loss_value = loss_fn(parameter)
        if self.verbose > 0:
            loss_value = loss_fn(parameter)
            print(f"Initial loss : {loss_fn(parameter):.2e}")  # noqa: T201
            print(f"  = {self.current_loss:.2e}", end="")  # noqa: T201
            if self.regularization_weight != 0:
                print(  # noqa: T201
                    f" + {self.regularization_weight} * "
                    f"{self.current_regularization:.2e}",
                    end="",
                )
            else:
                print(  # noqa: T201
                    " + 0",
                    end="",
                )
            print(  # noqa: T201
                " (fidelity + regularization_weight * regularization)"
            )

        fidelity_history = []
        regularization_history = []

        fidelity_history.append(self.current_loss)
        regularization_history.append(self.current_regularization)

        if self.debug:
            parameters_list = [[]]
            gradients_list = [[]]

        # Run the optimization
        for _i in range(self.n_iter):
            loss_value = optimizer.step(closure)
            if self.debug and _i < self.n_iter - 1:
                parameters_list.append([])
                gradients_list.append([])
            fidelity_history.append(self.current_loss)
            regularization_history.append(self.current_regularization)
            if self.verbose > 0:
                loss_value = loss_fn(parameter)
                print(  # noqa: T201
                    f"Loss after {_i + 1} iteration(s) : {loss_value:.2e}"
                )
                print(f"  = {self.current_loss:.2e}", end="")  # noqa: T201
                if self.regularization_weight != 0:
                    print(  # noqa: T201
                        f" + {self.regularization_weight} * "
                        f"{self.current_regularization:.2e}",
                        end="",
                    )
                else:
                    print(  # noqa: T201
                        " + 0",
                        end="",
                    )
                print(  # noqa: T201
                    " (fidelity + regularization_weight * regularization)"
                )

        # Store the device type of the parameter (useful for testing purposes)
        self.internal_parameter_device_type = parameter.device.type

        # Move the parameter to the output device
        self.parameter_ = parameter.clone().detach().to(self.output_device)

        if self.debug:
            self.parameters_list_ = parameters_list
            self.gradients_list_ = gradients_list

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
        self.fidelity_ = self.fidelity_history_[-1]

        self.regularization_history_ = torch.stack(regularization_history).to(
            self.output_device
        )
        self.regularization_ = self.regularization_history_[-1]

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
