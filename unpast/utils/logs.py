import logging
from logging import getLogger as get_logger
import time
from functools import wraps
import sys

LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def setup_logging(log_file=None, log_level=logging.INFO, log_file_level=logging.DEBUG):
    """Set up logging configuration.
    log_level messages will be logged to the console (stdout)
    log_file_level messages will be logged to the specified log file if provided.

    Args:
        log_file (str): Path to the log file. If None, no file logging is done.
        log_level: Logging level for console output.
        log_file_level: Logging level for file output.
    """
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(
        logging.DEBUG
    )  # Set to lowest level to allow all handlers to filter

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    # console_formatter = logging.Formatter(
    #     "%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    # )
    # file_formatter = logging.Formatter(
    #     "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    # )

    # tmp compatible formatters untill enh:print->log everywhere
    console_formatter = logging.Formatter("%(message)s", datefmt="%H:%M:%S")
    file_formatter = logging.Formatter("%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    return logger


def log_function_duration(name=None):
    """
    Decorator to log the duration of a function call.

    Usage:
        @log_function_duration(name="MyFunction")
        def my_function():
            # Function implementation
            pass

        @log_function_duration()
        def my_function():
            # Function implementation
            pass

    Args:
        name (str, optional): Name to use in the log message.
        If not provided, the function's name will be used.


    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger for the module where the decorated function is defined
            func_logger = get_logger(func.__module__)
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time

            if name:
                step_name = name
            else:
                step_name = func.__name__

            func_logger.debug(f"{step_name} completed in: {duration:7.2f} seconds")
            return result

        return wrapper

    return decorator


@log_function_duration(name="Function Test")
def _test_func():
    """A simple test function to demonstrate logging."""
    time.sleep(0.5)  # Simulate some processing time
    return "Test function completed."


# initialize logging with default settings
setup_logging()
logger = get_logger(__name__)
