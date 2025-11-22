"""
Comprehensive Logging System for KV-1

Saves all output to both console AND timestamped log files.
Every run creates a new log file in ./logs/ directory.

Usage:
    from core.logger import setup_logging, log

    setup_logging()
    log("This goes to both console and file!")
"""

import sys
import os
from datetime import datetime
from pathlib import Path


class DualLogger:
    """Logger that writes to both console and file simultaneously."""

    def __init__(self, log_file_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'a', encoding='utf-8')

    def write(self, message):
        """Write to both terminal and file."""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure it's written immediately

    def flush(self):
        """Flush both streams."""
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        """Close log file."""
        self.log_file.close()


class DualErrorLogger:
    """Logger for stderr that writes to both console and file."""

    def __init__(self, log_file_path: str):
        self.terminal = sys.stderr
        self.log_file = open(log_file_path, 'a', encoding='utf-8')

    def write(self, message):
        """Write to both terminal and file."""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        """Flush both streams."""
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        """Close log file."""
        self.log_file.close()


# Global loggers
_stdout_logger = None
_stderr_logger = None
_current_log_file = None


def setup_logging(
    log_dir: str = "./logs",
    session_name: str = None,
    enabled: bool = True
) -> str:
    """
    Set up dual logging to both console and file.

    Args:
        log_dir: Directory to store log files
        session_name: Optional custom name for this session
        enabled: Whether to enable file logging (default: True)

    Returns:
        Path to the created log file

    Example:
        setup_logging()  # Creates logs/session_2025-11-22_14-30-45.log
        setup_logging(session_name="curriculum_phase1")  # logs/curriculum_phase1_2025-11-22_14-30-45.log
    """
    global _stdout_logger, _stderr_logger, _current_log_file

    if not enabled:
        print("[i] File logging disabled")
        return None

    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)

    # Generate log file name with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if session_name:
        log_filename = f"{session_name}_{timestamp}.log"
    else:
        log_filename = f"session_{timestamp}.log"

    log_file_path = os.path.join(log_dir, log_filename)
    _current_log_file = log_file_path

    # Create dual loggers
    _stdout_logger = DualLogger(log_file_path)
    _stderr_logger = DualErrorLogger(log_file_path)

    # Redirect stdout and stderr
    sys.stdout = _stdout_logger
    sys.stderr = _stderr_logger

    # Log header
    print("="*80)
    print(f"KV-1 Learning Session Started")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Log file: {log_file_path}")
    print("="*80)
    print()

    return log_file_path


def get_current_log_file() -> str:
    """Get path to current log file."""
    return _current_log_file


def close_logging():
    """Close log files and restore normal stdout/stderr."""
    global _stdout_logger, _stderr_logger

    if _stdout_logger:
        print()
        print("="*80)
        print(f"Session Ended: {datetime.now().isoformat()}")
        print(f"Log saved to: {_current_log_file}")
        print("="*80)

        # Restore original streams
        sys.stdout = _stdout_logger.terminal
        sys.stderr = _stderr_logger.terminal

        # Close files
        _stdout_logger.close()
        _stderr_logger.close()

        _stdout_logger = None
        _stderr_logger = None


def log(message: str, prefix: str = "[i]"):
    """
    Convenience function to log a message.

    Args:
        message: Message to log
        prefix: Prefix for the message (default: [i])
    """
    print(f"{prefix} {message}")


# Ensure logging is closed on exit
import atexit
atexit.register(close_logging)
