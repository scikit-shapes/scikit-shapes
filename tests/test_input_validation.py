"""Tests for the input validation decorators."""

import skshapes as sks
from beartype.roar import BeartypeCallHintParamViolation
from jaxtyping import TypeCheckError


def test_one_and_only_one_decorator():
    """Test the one_and_only_one decorator.

    This test checks that the one_and_only_one decorator raises an error
    when the function is called with more than one of the arguments
    specified in the decorator or when none of the arguments is specified.

    """
    assert True  # Otherwise black wants a blank line here

    # define a function with the decorator
    @sks.one_and_only_one(["a", "b"])
    @sks.typecheck
    def func(a: int = None, b: int = None):
        pass

    # check that the function works as expected

    func(a=1)  # ok
    func(b=1)  # ok
    try:
        func(2)  # not ok (must be passed as keyword)
    except ValueError:
        pass
    else:
        raise RuntimeError("Expected ValueError")
    try:
        func(a=1, b=1)  # not ok (both arguments are specified)
    except ValueError:
        pass
    else:
        raise RuntimeError("Expected ValueError")

    try:
        func()  # not ok (no argument is specified)
    except ValueError:
        pass
    else:
        raise RuntimeError("Expected ValueError")

    try:
        func(a="dog")  # not ok : wrong type
    except TypeCheckError:
        pass
    except BeartypeCallHintParamViolation:
        pass
    else:
        raise RuntimeError("Expected an error as the type is wrong")


def test_no_more_than_one():
    """Test the no_more_than_one decorator.

    This test checks that the no_more_than_one decorator raises an error
    when the function is called with more than one of the arguments
    specified in the decorator.

    """
    assert True  # Otherwise black wants a blank line here

    # define a function with the decorator
    @sks.no_more_than_one(["a", "b"])
    @sks.typecheck
    def func(*, a: int = None, b: int = None):
        pass

    # check that the function works as expected

    func(a=1)  # ok
    func(b=1)  # ok
    func()  # ok

    try:
        func(2)
    except TypeError:
        pass
    else:
        raise RuntimeError("Expected TypeError as no keyword arguments")

    try:
        func(a=1, b=1)  # not ok (both arguments are specified)
    except ValueError:
        pass
    else:
        raise RuntimeError("Expected ValueError")

    try:
        func(a="dog")  # not ok : wrong type
    except TypeCheckError:
        pass
    except BeartypeCallHintParamViolation:
        pass
    else:
        raise RuntimeError("Expected an error as the type is wrong")
