from src.config_loader import load_config
from src.exceptional import ConfigError
from src.logger import setup_logging, get_logger, log_step
import sys 

def main():
    try: 
        cfg = load_config("config.yaml")
        print("Config loaded succesfully")
    except ConfigError as e:
        print(f"error loading config: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"unexpected error: {e}")
        raise e
    
    try:
        log = setup_logging(cfg.logging)
        log.info("Logger test started")

        # Test all severity levels
        log.debug("DEBUG — detailed info (only in log file)")
        log.info("INFO — normal pipeline progress")
        log.warning("WARNING — something unexpected")
        log.error("ERROR — something broke")

        # Test helper functions
        log_step(log, 1, 3, "Loading data")
        log_step(log, 2, 3, "Training model")
        log_step(log, 3, 3, "Saving outputs")
        

        log.info(" Logger test complete")

    except Exception as e:
        print(f" Logger error: {e}")
        raise

if __name__ == "__main__":
    main()
