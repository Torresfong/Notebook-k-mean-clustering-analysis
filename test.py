from src.config_loader import load_config
from src.exceptional import ConfigError,DataValidationError,FeatureEngineeringError
from src.logger import setup_logging, get_logger, log_step
import sys
from src.data_loader import data_ingestion 
from src.feature_engineering import build_rfm

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

    # data loading 
    log_step(log,1,5,"loading data")
    try: 
        df = data_ingestion(cfg.data)
    except DataValidationError as e:
        log.error(f"Data loading failed:{e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"unexpected error during data loading: {e}")
        raise

    # sanity check/ data validation before proceding
    log.info("Running post-load sanity check")
    log.info(f"  Shape            : {df.shape}")
    log.info(f"  Columns          : {list(df.columns)}")
    log.info(f"  Dtypes           :\n{df.dtypes.to_string()}")
    log.info(f"  Null counts      :\n{df.isnull().sum().to_string()}")
    log.info(f"  Unique customers : {df[cfg.data.customer_id_column].nunique()}")
    log.info(f"  Date range       : {df[cfg.data.date_column].min().date()} → " f"{df[cfg.data.date_column].max().date()}")

    # feature engineering 
    log_step(log,2,5,"Feature engineering")
    try:
         rfm_clean, rfm_outlier, outlier_labels = build_rfm(
            df          = df,
            feature_cfg = cfg.features,
            data_cfg    = cfg.data,
            outlier_cfg = cfg.outliers,
            )
    except FeatureEngineeringError as e:
        log.error(f"feature error:{e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"unexpected error during data loading: {e}")
        raise   

if __name__ == "__main__":
    main()
