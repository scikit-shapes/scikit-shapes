import skshapes as sks


def test_one_and_only_one_decorator():
    """Test the one_and_only_one decorator.

    This test checks that the one_and_only_one decorator raises an error
    when the function is called with more than one of the arguments
    specified in the decorator or when none of the arguments is specified.
    """

    # define a function with the decorator
    @sks.one_and_only_one(["a", "b"])
    def func(a=None, b=None):
        pass

    # check that the function works as expected

    func(a=1)  # ok
    func(b=1)  # ok
    func(2)  # ok
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
