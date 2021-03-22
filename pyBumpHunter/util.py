import functools
import warnings
from collections import defaultdict
from typing import Optional

warned_funcs = set()
warned_args = defaultdict(dict)


def deprecated_arg(oldarg: str, newarg: str, comment: Optional[str] = None):
    """Decorator to deprecate *oldarg* in favour of *newarg*

    Args:
        oldarg: Deprecated argument
        newarg: Argument to use instead.
        comment: An additional comment

    """
    if comment is None:
        comment = ""

    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            if oldarg not in warned_args[func]:
                warnings.warn(
                    f"The argument {oldarg} in function {func} is deprecated and will be removed"
                    f" in a future release. Use {newarg} instead. {comment}",
                    category=FutureWarning,
                    stacklevel=2,
                )

                warned_args[func] = oldarg

            return func(*args, **kwargs)

        return wrapped_func

    return decorator


def deprecated(instruction):
    """Decorator to deprecate a function with *instruction* how to update."""

    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            if func not in warned_funcs:
                warnings.warn(
                    f"The function {func} is deprecated and will be removed in a future release."
                    f" {instruction}",
                    category=FutureWarning,
                    stacklevel=2,
                )
                warned_funcs.add(func)
            return func(*args, **kwargs)

        return wrapped_func

    return decorator
