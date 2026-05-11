from src.config_loader import load_config
from src.exceptional import ConfigError
import sys 

def main():
    try: 
        config = load_config("config.yaml")
        print("Config loaded succesfully")
    except ConfigError as e:
        print(f"error loading config: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"unexpected error: {e}")
        raise e


if __name__ == "__main__":
    main()