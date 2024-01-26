"""Not-None rules.

This module define decorators that allows to do some checks for functions that
can be called by different arguments combinations. For example, a function can
be called with one of the arguments `a`, `b` or `c` but not with more than one
of them. This can be done with the `one_and_only_one` decorator:
```python
@one_and_only_one(["a", "b", "c"])
def foo(a=None, b=None, c=None):
    pass


foo(a=1)  # OK
foo(b=1)  # OK
foo(c=1)  # OK
foo(a=1, b=1)  # InputStructureError
```

Decorators already implemented:

- `one_and_only_one` : one and only one of the parameters must be not None
- `no_more_than_one` : no more than one of the parameters must be not None
"""
from functools import wraps

from ..errors import InputStructureError


def generator_notnone_rule(rule):
    """Not-None rules decorator generator.

    A not-None rule decorator decorates functions that have a number of
    keywords arguments that are None by default and that must be provided
    within certain rules (for example, one and only one of these arguments must
    be specified)

    Parameters
    ----------
    rule
        A function with the number of not none keywords arguments from a list
        that raises an error if this number does not satisfies certain
        condition

    Returns
    -------
    Callable
        A decorator that can be parametrized with a list of parameters and a
        boolean arguments to determine if typecheck must be applied
    """

    def ruler(parameters):
        def decorator(func):
            """Actual decorator."""

            @wraps(func)
            def wrapper(*args, **kwargs):
                """Actual wrapper.

                Returns
                -------
                callable
                    the decorated function

                Raises
                ------
                InputStructureError
                    if more than of the parameters list is not None
                """
                # Check that only one of the parameters is not None
                not_none = 0

                for key, value in kwargs.items():
                    if key in parameters and value is not None:
                        not_none += 1

                rule(not_none, parameters)

                return func(*args, **kwargs)

            # Copy annotations (if not, beartype does not work)
            wrapper.__annotations__ = func.__annotations__
            return wrapper

        return decorator

    return ruler


def rule_one_and_only_one(not_none: int, parameters: list[str]) -> None:
    """Rule that checks that one and only one of the parameters is not None."""
    if not_none != 1:
        raise InputStructureError(
            f"One and only one of the parameters {parameters} must be"
            + " not None and they must be passed as keyword arguments"
        )


def rule_no_more_than_one(not_none: int, parameters: list[str]) -> None:
    """Rule that checks that no more than one of the parameters is not None."""
    if not_none > 1:
        raise InputStructureError(
            f"No more than one if the parameters {parameters} must be"
            + " not None and they must be passed as keyword arguments"
        )


def one_and_only_one(parameters):
    """Checker for only one not None parameter.

    Parameters
    ----------
    parameters : list[str]
        the list of parameters to check

    Returns
    -------
    callable
        the decorated function

    Raises
    ------
    InputStructureError
        if more than one parameter is not None

    Examples
    --------
    >>> @one_and_only_one(["a", "b"])
    >>> def func(a=None, b=None):
    >>>     pass
    >>> func(a=1)
    >>> func(b=1)
    >>> func(a=1, b=1)
    InputStructureError: Only one of the parameters a, b must be not None

    """
    return generator_notnone_rule(rule_one_and_only_one)(parameters)


def no_more_than_one(parameters):
    """Checker for less than one not None parameter.

    Parameters
    ----------
    parameters : list[str]
        the list of parameters to check

    Returns
    -------
    callable
        the decorated function

    Raises
    ------
    InputStructureError
        if more than one parameter is not None

    Examples
    --------
    >>> @no_more_than_one(["a", "b"])
    >>> def func(a=None, b=None):
    >>>     pass
    >>> func()
    >>> func(a=1)
    >>> func(b=1)
    >>> func(a=1, b=1)
    InputStructureError: No more than one of the parameters a, b must be not None  # noqa E501

    """
    return generator_notnone_rule(rule_no_more_than_one)(parameters)
