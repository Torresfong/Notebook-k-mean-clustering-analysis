import pandas as pd 
import logging 

from src.config_loader import DataConfig
from src.exceptional import DataValidationError

# create logger name = "src.data_loader"
log = logging.getLogger(__name__)

def data_ingestion(cfg: DataConfig) -> pd.DataFrame:
    """
    This function first reads the data from the given file path and returns the dataframe object.
    Second validate schema
    Third clean data and return a cleaned dataframe as artifact 
    """
    # First data loader 
    log.info("Data ingestion start")
    log.info(f"Source: {cfg.filepath}")
    log.info(f"Encoding : {cfg.encoding}")

    try: 
        df = pd.read_csv(cfg.filepath, encoding=cfg.encoding)
    
    except FileNotFoundError:
        raise DataValidationError(f"Data file not found: {cfg.filepath}")
        
    except UnicodeDecodeError:
         raise DataValidationError(
             f"Encoding error reading: {cfg.filepath}"
             f"Current encoding in config.yaml: {cfg.encoding}"
             )
    
    except Exception as e:
        raise DataValidationError(f"Failed to read csv: {cfg.filepath}")
    

    # Second data validation 
    log.info("Data validation")
    missing_columns = [col for col in cfg.required_columns 
                       if col not in df.columns]
    
    if missing_columns: 
        raise DataValidationError(f"Missing required columns: {missing_columns}")
    
    # thrid parse invoice date
    log.info(f"Parsing {cfg.date_column} with format {cfg.date_format}")
    
    df[cfg.date_column] = pd.to_datetime(
        df[cfg.date_column],
        format = cfg.date_format,
        errors = "coerce",
        dayfirst = True
    )
    # Calculate the Nan value in InvoceDate
    nat_count = df[cfg.date_column].isna().sum() #nat = Not a Time / Nan
    if nat_count > 0:
        log.warning(
            f"{nat_count} rows have unparseable {cfg.date_column} values " )

    # Drop row with NAT dates 
    before = len(df)  # capture row count before dropping 
    df = df.dropna(subset=[cfg.date_column]) # drop row with InvoiceDate is Nat 
    log.info(
        f"After date parsing: {len(df)} rows remain "
        f"({before - len(df)} dropped)"
    )

    # Fourth remove invalud invoice prefixes 
    if cfg.invalid_invoice_prefixes:
        log.info(f"Filtering invoice prefixes: {cfg.invalid_invoice_prefixes}")

        # build regex pattern
        prefix_chars = "".join(cfg.invalid_invoice_prefixes)
        pattern = f"^[{prefix_chars}]"

        before = len(df)
        df = df[
            ~df[cfg.invoice_column]
            .astype(str)
            .str.match(pattern, na=False)
        ] 
        log.info(f"Removed {before - len(df)} rows with invalid invoice prefixes ")      

    else:
         log.info("No invalid invoice prefixes configured — skipping")

    # Fifth remove invalid stockcode 
    if cfg.invalid_stockcodes:
        log.info(f" Filtering invalid stockcode; {cfg.invalid_stockcodes}")

        before = len(df)
        df = df[
            ~df[cfg.stock_column]
            .astype(str)
            .isin(cfg.invalid_stockcodes)   
        ]
        log.info(f"Removed {before - len(df)} rows with invalid stockcodes")
    else:
        log.info("No invalid stockcodes configured — skipping")

    # sixth remove null customer ID 
    log.info(f"removing null {cfg.customer_id_column}")

    before = len(df)
    df = df.dropna(subset = [cfg.customer_id_column])
    log.info(f" Removed {before - len(df)} rows with null {cfg.customer_id_column}")

    # Seventh filter non positive quantity
    log.info(f" Remove non positive {cfg.quantity_column} ")

    before =len(df)
    df = df[df[cfg.quantity_column] > 0]
    log.info(f"Removed {before - len(df)} rows with Quantity <= 0 ")

    # Eighth filter non positive price 
    log.info(f" Filter non positive {cfg.price_column}")

    before = len(df)
    df = df[df[cfg.price_column] > 0]
    log.info(f"Removed {before - len(df)} rows with Price <= 0 ")

    # Nineth filter duplicates 
    log.info("Remove duplicate rows")

    before = len(df)
    df = df.drop_duplicates()
    log.info(f"Removed {before - len(df)} exact duplicate rows")

    # tenth standardise customer ID format 
    log.info(f" standardise {cfg.customer_id_column} format")

    df[cfg.customer_id_column] = (
    df[cfg.customer_id_column]
    .astype(float)   # first handles "12345.0" from Excel exports
    .astype(int)     # then removes decimal → 12345
    .astype(str)     # final clean string → "12345"
    )
    log.info(f"Customer ID standardised to clean string format")

    # Compute sale column that needed for RFM features 
    log.info("Computing Sales = Quantity × Price...")

    df[cfg.sales_column] = df[cfg.quantity_column] * df[cfg.price_column]
    log.info(f"Sales column created — ")

    before = len(df)
    df = df[df[cfg.sales_column]>0]
    if before - len(df) > 0:
        log.warning("Negative Quantity rows survived ")
    else:
        log.info("Sales guard passed — all Sales values are positive")

    # Post clean validation 
    if df.empty:
        raise DataValidationError("DataFrame is empty after all cleaning steps")
    
    # check wheter enough unique customer for kmeans 
    unique_customers = df[cfg.customer_id_column].nunique()
    if unique_customers < 2:
        raise DataValidationError(f"Only {unique_customers} unique customer(s) found after cleaning.")
    
    # summary 
    log.info("Data loading complete")

    return df