import pytest
from unittest.mock import patch, MagicMock
from trainer import DataDirectoryEventHandler
import logging
from logging.handlers import MemoryHandler


# def test_on_created_directory():
#     mock_event = MagicMock()
#     mock_event.is_directory = True
#     mock_event.src_path = "/path/to/directory"

#     obj = DataDirectoryEventHandler()

#     memory_handler = MemoryHandler(capacity=100)
#     logger = logging.getLogger()
#     logger.addHandler(memory_handler)

#     obj.on_created(mock_event)

#     # Check the first log record
#     log_record = memory_handler.buffer[0]
#     assert log_record.getMessage() == "New directory created: /path/to/directory"


@patch("builtins.print")
def test_on_created_file(mock_print):
    # Arrange
    mock_event = MagicMock()
    mock_event.is_directory = False
    mock_event.src_path = "/path/to/file"

    obj = DataDirectoryEventHandler()
    obj.on_created(mock_event)

    # Assert
    mock_print.assert_not_called()
