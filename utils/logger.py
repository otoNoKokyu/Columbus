import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# Define colors for the console formatter
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to standard logging output."""
    COLORS = {
        logging.DEBUG: "\033[94m",    # Blue
        logging.INFO: "\033[92m",     # Green
        logging.WARNING: "\033[93m",  # Yellow
        logging.ERROR: "\033[91m",    # Red
        logging.CRITICAL: "\033[1;91m" # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        format_str = f"%(asctime)s [{color}%(levelname)s{self.RESET}] %(name)s: %(message)s"
        formatter = logging.Formatter(format_str, datefmt="%H:%M:%S")
        return formatter.format(record)

def setup_logger():
    # Base logger name for the whole project
    logger = logging.getLogger("Columbus")
    
    # Avoid adding handlers multiple times if imported multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # 1. Console Handler
    ch = logging.StreamHandler(sys.stdout)
    # Default to INFO for console to avoid noise, can be overridden by env var
    console_level = os.environ.get("COLUMBUS_LOG_LEVEL", "INFO").upper()
    ch.setLevel(getattr(logging, console_level, logging.INFO))
    ch.setFormatter(ColoredFormatter())
    logger.addHandler(ch)

    # 2. File Handler (Optional, logs everything DEBUG and above)
    # Try to make logs directory relative to the project root
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = RotatingFileHandler(os.path.join(log_dir, "columbus.log"), maxBytes=10*1024*1024, backupCount=3)
        fh.setLevel(logging.DEBUG)
        file_format = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        fh.setFormatter(file_format)
        logger.addHandler(fh)
    except Exception as e:
        # Fallback if file logging fails (e.g., permissions)
        pass

    # Silence noisy 3rd-party loggers globally
    noisy_loggers = ["httpx", "httpcore", "urllib3", "sentence_transformers", "pypdf"]
    for nl in noisy_loggers:
        logging.getLogger(nl).setLevel(logging.WARNING)

    return logger

# Singleton logger instance
logger = setup_logger()
