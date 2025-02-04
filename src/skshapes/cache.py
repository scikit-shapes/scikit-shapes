import functools
import weakref
from collections.abc import Callable
from functools import cached_property, lru_cache, partial, update_wrapper
from typing import TypeVar


def cached_method(*lru_args, **lru_kwargs):
    """Least-recently-used cache decorator for instance methods."""

    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator


T = TypeVar("T")


def instance_lru_cache(
    method: Callable[..., T] | None = None,
    *,
    maxsize: int | None = 128,
    typed: bool = False,
) -> Callable[..., T] | Callable[[Callable[..., T]], Callable[..., T]]:
    """Least-recently-used cache decorator for instance methods.

    The cache follows the lifetime of an object (it is stored on the object,
    not on the class) and can be used on unhashable objects. Wrapper around
    functools.lru_cache.

    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.

    If *typed* is True, arguments of different types will be cached separately.
    For example, f(3.0) and f(3) will be treated as distinct calls with
    distinct results.

    Arguments to the cached method (other than 'self') must be hashable.

    View the cache statistics named tuple (hits, misses, maxsize, currsize)
    with f.cache_info().  Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.

    """

    def decorator(wrapped: Callable[..., T]) -> Callable[..., T]:
        def wrapper(self: object) -> Callable[..., T]:
            return lru_cache(maxsize=maxsize, typed=typed)(
                update_wrapper(partial(wrapped, self), wrapped)
            )

        return cached_property(wrapper)  # type: ignore[code]

    return decorator if method is None else decorator(method)


def cache_clear(self):
    """Reload all cached properties."""
    cls = self.__class__
    # Below, the third argument to getattr is there to return a default value
    # instead of throwing an AttributeError.
    attrs = [
        a
        for a in dir(self)
        if isinstance(getattr(cls, a, cls), cached_property)
    ]
    for a in attrs:
        self.__dict__.pop(a, None)
        # delattr(self, a)

    if hasattr(self, "_cached_methods"):
        for a in self._cached_methods:
            cached_method = getattr(self, a)
            # N.B.: in the __init__ of PolyData, cached methods may not have been
            #       set to an actual cached method yet, with a cache_clear method.
            if hasattr(cached_method, "cache_clear"):
                cached_method.cache_clear()

    if hasattr(self, "_cached_properties"):
        for a in self._cached_properties:
            if hasattr(self, "_cached_" + a):
                delattr(self, "_cached_" + a)


def immutable_cached_property(*, function, cache):
    """This decorator is roughly equivalent to @cached_property, but better suited here.

    Notably, it does not allow the cached property to be set to a new value,
    and allows the docstrings to be discovered by pytest.
    """

    def cached_func(self):
        if not cache:
            return function(self)
        else:
            if not hasattr(self, "_cached" + function.__name__):
                setattr(self, "_cached" + function.__name__, function(self))
            return getattr(self, "_cached" + function.__name__)

    return property(cached_func)


def add_cached_methods_to_sphinx(cls):
    """Ensures that e.g. ``PolyData.point_normals`` is documented in Sphinx.

    Cached methods are instance methods that are memoized with ``functools.lru_cache``.
    This small decorator ensures that although ``PolyData.point_normals`` is a cached
    front-end for the private method ``PolyData._point_normals`` that is instantiated
    in the `__init__` method, the Sphinx documentation will look as though
    it was a regular method.
    """
    for method_name in cls._cached_methods + cls._cached_properties:
        # As far as Sphinx is concerned,
        # self.method_name = self._method_name
        # Then, at the end of the __init__, we overwrite self.method_name
        # with a memoized version of self._method_name.
        setattr(cls, method_name, getattr(cls, "_" + method_name))
    return cls


def cache_methods_and_properties(*, cls, instance, cache_size):
    for method_name in instance._cached_methods:
        setattr(
            instance,
            method_name,
            functools.lru_cache(maxsize=cache_size)(
                getattr(instance, "_" + method_name)
            ),
        )

    # Cached properties are not cached if cache_size is 0
    for method_name in instance._cached_properties:
        setattr(
            cls,
            method_name,
            immutable_cached_property(
                function=getattr(cls, "_" + method_name),
                cache=cache_size != 0,
            ),
        )
