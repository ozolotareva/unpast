import logging
import sys
import time
from functools import wraps
from typing import Union

LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

GLOBAL_PREFIX = ""  # for log_duration_decorator
INDENT = "  "


class PrefixLogger:
    """Wrapper around logging.Logger with prefix functionality"""

    def __init__(self, logger):
        self.logger = logger

    def info(self, msg, *args, **kwargs):
        """Log an info message with the global prefix"""
        self.logger.info(f"{GLOBAL_PREFIX} {msg}", *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """Log a debug message with the global prefix"""
        self.logger.debug(f"{GLOBAL_PREFIX} {msg}", *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log a warning message with the global prefix"""
        self.logger.warning(f"{GLOBAL_PREFIX} {msg}", *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log an error message with the global prefix"""
        self.logger.error(f"{GLOBAL_PREFIX} {msg}", *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Log a critical message with the global prefix"""
        self.logger.critical(f"{GLOBAL_PREFIX} {msg}", *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        """Log an exception message with the global prefix"""
        self.logger.exception(f"{GLOBAL_PREFIX} {msg}", *args, **kwargs)


def get_logger(name: Union[str, None] = None) -> PrefixLogger:
    """Get a logger with the specified name.

    Args:
        name (str, optional): Name of the logger.
    """
    return PrefixLogger(logging.getLogger(name))


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
    formatter = logging.Formatter(
        "%(asctime)s %(relativeCreated)8dms %(levelname)7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%SUTC",
    )
    formatter.converter = time.gmtime  # Use UTC time for log timestamps

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    return logger


def log_function_duration(name=None, indent=INDENT):
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
        indent (str, optional): Indentation to apply to the log message.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger for the module where the decorated function is defined
            func_logger = get_logger(func.__module__)
            if name:
                step_name = name
            else:
                step_name = func.__name__

            func_logger.debug(f"{step_name} started...")

            global GLOBAL_PREFIX
            GLOBAL_PREFIX = indent + GLOBAL_PREFIX

            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            GLOBAL_PREFIX = GLOBAL_PREFIX[len(indent) :]
            func_logger.debug(f"{step_name} completed in: {duration:.2f} seconds")
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
