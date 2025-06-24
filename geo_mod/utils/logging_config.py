"""Logging configuration for the geopolitical simulation."""
import logging
import sys

def setup_logging(level=logging.INFO):
    """Configures logging for the geopolitical simulation module."""
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
