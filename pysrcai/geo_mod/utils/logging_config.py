"""Logging configuration for the geopolitical simulation."""
import logging
import sys


def setup_logging(level=logging.INFO):
    """
    Configures logging for the geopolitical simulation module.

    Logging levels:
    - DEBUG: Detailed information, typically of interest only when diagnosing problems.
    - INFO: Confirmation that things are working as expected.
    - WARNING: An indication that something unexpected happened, or indicative of some problem in the near future.
    - ERROR: Due to a more serious problem, the software has not been able to perform some function.
    - CRITICAL: A very serious error, indicating that the program itself may be unable to continue running.
    """
    logger = logging.getLogger('geo_mod')
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
