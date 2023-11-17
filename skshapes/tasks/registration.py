"""Registration between two shapes."""

from ..optimization import Optimizer, LBFGS
from ..types import (
    typecheck,
    convert_inputs,
    shape_type,
    float_dtype,
    FloatTensor,
)
from typing import Union, Optional
from ..loss import Loss
from ..morphing import Model
import torch


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
        optimizer: Optional[Optimizer] = None,
        regularization: Union[int, float] = 1,
        n_iter: int = 10,
        verbose: int = 0,
        gpu: bool = True,
        **kwargs,
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
                print(
                    "Warning : gpu is set to True but no cuda device is "
                    + "available. The optimization will be performed on cpu."
                )
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
        initial_parameter: Optional[FloatTensor] = None,
    ) -> None:
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
        ValueError
            if the source and target shapes are not on the same device.
        """
        # Check that the shapes are on the same device
        if source.device != target.device:
            raise ValueError(
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
            return_regularization = False if self.regularization == 0 else True
            morphing = self.model.morph(
                shape=source,
                parameter=parameter,
                return_path=False,
                return_regularization=return_regularization,
            )

            loss = self.loss(source=morphing.morphed_shape, target=target)
            total_loss = loss + self.regularization * morphing.regularization

            if self.verbose > 0:
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
                raise ValueError(
                    f"Initial parameter has shape {initial_parameter.shape}"
                    + f" but the model expects shape {parameter_shape}"
                )

            parameter = initial_parameter.clone().detach()
            parameter = parameter.to(self.optim_device)
        else:
            # if no parameter is provided, initialize it with zeros
            parameter = torch.zeros(
                parameter_shape, device=self.optim_device, dtype=float_dtype
            )

        parameter.requires_grad = True

        # Initialize the optimizer
        optimizer = self.optimizer([parameter])

        # Define the closure for the optimizer
        def closure():
            optimizer.zero_grad()
            loss_value = loss_fn(parameter)
            loss_value.backward()
            return loss_value

        if self.verbose > 0:
            loss_value = loss_fn(parameter)
            print(f"Initial loss : {loss_fn(parameter):.2e}")
            print(f"    fidelity = {self.current_loss:.2e}")
            print(f"    regularization = {self.current_regularization:.2e}")

        # Run the optimization
        for i in range(self.n_iter):
            loss_value = optimizer.step(closure)
            if self.verbose > 0:
                loss_value = loss_fn(parameter)
                print(f"Loss after {i + 1} iteration(s) : {loss_value:.2e}")
                print(f"    fidelity = {self.current_loss:.2e}")
                print(
                    f"    regularization = {self.current_regularization:.2e}"
                )

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

        self.loss_ = loss_value.detach().to(self.output_device)
        self.regularization_ = morphing.regularization.to(self.output_device)
        self.transformed_shape_ = morphing.morphed_shape
        self.path_ = morphing.path

        # If needed : move the transformed shape and the path to the output
        # device
        if self.transformed_shape_.device != self.output_device:
            self.transformed_shape_ = self.transformed_shape_.to(
                self.output_device
            )
            self.path_ = [s.to(self.output_device) for s in self.path_]

        self.transformed_shape_.points
        for s in self.path_:
            s.points

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
        if not hasattr(self, "parameter_"):
            raise ValueError(
                "The registration must be fitted before calling transform"
            )

        transformed_shape = self.model.morph(
            shape=source, parameter=self.parameter_
        ).morphed_shape

        if transformed_shape.device != self.output_device:
            return transformed_shape.to(self.output_device)
        else:  # If transformed_shape is on the output device don't copy
            return transformed_shape

    @convert_inputs
    @typecheck
    def fit_transform(
        self,
        *,
        source: shape_type,
        target: shape_type,
        initial_parameter: Optional[FloatTensor] = None,
    ) -> shape_type:
        """Fit the registration and apply it to the source shape."""
        self.fit(
            source=source, target=target, initial_parameter=initial_parameter
        )
        return self.transform(source=source)
