"""Custom errors for the skshapes package."""

from beartype.roar import BeartypeCallHintParamViolation
from jaxtyping import TypeCheckError

InputTypeError = (TypeCheckError, BeartypeCallHintParamViolation)


class InputStructureError(Exception):
    """Raised when the input structure is not valid."""


class ShapeError(Exception):
    """Raised when an input has an invalid shape."""


class DeviceError(Exception):
    """Raised when devices mismatch."""


class NotFittedError(Exception):
    """Raised when the model is not fitted."""
