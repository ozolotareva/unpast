"""TODO: check and remove this file if not used anymore"""

from .logs import get_logger

logger = get_logger(__name__)

TRY_USE_NUMBA = True


def jit_if_available(func):
    """Decorator that applies numba JIT compilation if available, otherwise returns unmodified function.

    Args:
        func: function to potentially JIT compile

    Returns:
        function: JIT-compiled function if numba available, otherwise original function
    """

    # default "do nothing" decorator with the numba-like interface
    def decorated(*args, **kwargs):
        return func(*args, **kwargs)

    return decorated


if TRY_USE_NUMBA:
    try:
        from numba import jit  # as jit_if_available

        jit_if_available = jit()
    except:
        logger.info(
            "Numba is not available. Install numba for a bit faster calculations"
        )
