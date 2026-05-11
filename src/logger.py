# Centralized logging setup of entire pipeline here
# Write every log message to both terminal and file  with consistent format and log level control 
# This module is used by all other modules to ge the logger so guard against duplicate handlers on repeat call  


import logging 
import os
import sys 
import logging.handlers
from src.config_loader import LoggingConfig 

# First block is LOG level map
_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

# setup function which is public entry point 
def setup_logging(cfg: LoggingConfig) -> logging.Logger:
    """
    Configure the root logger with terminal + file handlers based on the provided configuration.
    """

    # First step is to configure root logger and resolve the log level
    level_str = cfg.level.upper() 
    level = _LEVEL_MAP.get(level_str,logging.INFO) # .get(value,default) is convert the level string to python logging module log level, if the level string is not found in the level string, then it will return with INFO default setting rather tahn crash

    # create log folder 
    log_directory =  os.path.dirname(cfg.Log_file)
    if log_directory:
        os.makedirs(log_directory, exist_ok = True)

    # Configure logger 
    root_logger = logging.getLogger() # this is the parent of all module loggers, control everything 
    root_logger.setLevel(level) # set level on root 

    # Ensure no duplicate handlers
    if root_logger.handlers:
        root_logger.handlers.clear()

    # setup format for terminal and file handlers
    format = logging.Formatter(
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )
    # Terminal handler function as send log output to terminal in real time
    console_handler = logging.StreamHandler(sys.stdout) # 
    console_handler.setLevel(level)
    console_handler.setFormatter(format)

    # File handler function as write log output to .log on disk
    file_handler = logging.handlers.RotatingFileHandler(  #File handler with rotation to cap file size and automatically roll over if just file handler alone the disk fill up and log file become slow to open 
        filename = cfg.Log_file, 
        maxBytes = 5 * 1024 * 1024, # 5MB max file size
        backupCount = 3, # keep 3 old files after rotation
        encoding = "utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(format)

    # Add handlers to root Logger and now every log message travel to both destination simultaneously
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Silence noisy third party libraries 
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)
    
    # Log logger itself as startup confirmation
    startup_log = logging.getLogger(__name__)
    startup_log.info("Logger initialised")
    startup_log.info(f"  Level    : {level_str}")
    startup_log.info(f"  Log file : {cfg.Log_file}")
    startup_log.info(f"  Terminal : colour enabled")
    startup_log.info(f"  Rotation : 5MB × 3 backups")
    startup_log.info("=" * 60)

    return root_logger

def get_logger(name:str) -> logging.Logger:
    """
    fetches or creates a logger with this specific name from Python's global logger registry
    """
    return logging.getLogger(name)

def log_step(logger: logging.Logger, step_num: int, total: int, description: str) -> None:
    """
    Log a clean step banner for each pipeline stage
    """
    banner = f" STEP {step_num}/{total}: {description}"
    logger.info(banner)

