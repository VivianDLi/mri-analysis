"""Basic utility functions for the module."""

from datetime import datetime


def get_time_identifier():
    """Returns a unique string identifier based on the current time."""
    return datetime.now().strftime("%Y%m%d%H%M%S")
