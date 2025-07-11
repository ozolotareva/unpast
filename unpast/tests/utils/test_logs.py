import os
import time

from unpast.utils import logs


def test_log_file_creation_and_write(tmp_path):
    log_path = tmp_path / "test.log"
    logs.setup_logging(log_file=str(log_path))
    logger = logs.get_logger(__name__)
    logger.debug("Test log message")
    assert os.path.exists(log_path)
    with open(log_path) as f:
        content = f.read()
    assert "Test log message" in content


def test_log_file_append(tmp_path):
    log_path = tmp_path / "test_append.log"
    # First write using file append
    with open(log_path, "w") as f:
        f.write("First message\n")

    logs.setup_logging(log_file=str(log_path))
    logger = logs.get_logger(__name__)
    logger.debug("Second message")

    with open(log_path) as f:
        lines = f.read()

    assert "First message" in lines
    assert "Second message" in lines


def test_log_file_custom_level(tmp_path):
    log_path = tmp_path / "test_level.log"
    logs.setup_logging(log_file=str(log_path), log_file_level=40)  # 40 = ERROR
    logger = logs.get_logger(__name__)
    logger.error("Error message")
    with open(log_path) as f:
        content = f.read()
    assert "Error message" in content
    logger.debug("Info message")
    with open(log_path) as f:
        content = f.read()
    assert "Info message" not in content


def test_log_function_duration(tmp_path):
    log_path = tmp_path / "test_duration.log"
    logs.setup_logging(log_file=str(log_path))

    @logs.log_function_duration(name="timed_func")
    def timed_func():
        time.sleep(0.1)
        return 42

    result = timed_func()
    assert result == 42
    with open(log_path) as f:
        content = f.read()
    assert "timed_func completed in:" in content
