"""Logging configuration for vector_rag."""

import logging
import sys

def configure_logging(level=logging.DEBUG):
    """Configure logging for the application."""
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicate logs
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s: %(message)s (%(filename)s:%(lineno)s)',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(console_handler)
    
    # Set specific loggers
    logging.getLogger('vector_rag').setLevel(level)
    logging.getLogger('vector_rag.db').setLevel(level)
    logging.getLogger('vector_rag.db.db_file_handler').setLevel(level)
    
    # Quiet down some noisy libraries
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    
    return root_logger
