# Use this file to load and read the config yaml > validate every value > map to typed dataclasses > return AppConfig object 
# Every module import AppConfig from here.   

import yaml 
import os 
from dataclasses import dataclass
from typing import Dict, List, Tuple
from src.exceptional import ConfigError

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create dataclasses object. One data class per yaml section. mapping yaml to dataclass in builder function.
# AppConfig is the master config that holds all sub-config as typed attrbutes.
@dataclass
class DataConfig:
    """Control raw csv loading and cleaning filters"""
    filepath:                   str
    encoding:                   str
    date_column:                str  
    date_format:                str          
    customer_id_column:         str           
    invoice_column:             str           
    quantity_column:            str           
    price_column:               str           
    required_columns:           List[str]
    invalid_invoice_prefixes:   List[str]     # invoice prefixes to exclude (C, A)
    invalid_stockcodes:         List[str]     # non-product stockcodes to exclude

@dataclass
class FeatureConfig:
    rfm_columns:            List[str]
    log_transform:          bool
    recency_offset_days:    int
    filter_positive_sales:  bool

@dataclass
class OutlierConfig:
    """Control IQR outlier detection and two stage segmentation filters"""
    method:                   str         # "IQR" — only supported method
    iqr_multiplier:           float       # 1.5 = mild | 3.0 = extreme only
    remove_outlier_features:  List[str]   # features whose outliers get separated
    keep_outlier_features:    List[str]   # features whose outliers stay in data
    manual_labels:            Dict[str, int]  # monetary_only, frequency_only, both

@dataclass
class ModelConfig:
    n_clusters  : int
    random_state: int
    n_init      : int
    max_iter    : int
    algorithm   : str
    k_search_min: int
    k_search_max: int

@dataclass
class ClusterLabelConfig:
    labels: Dict[int,str]

@dataclass
class LoggingConfig:
    Log_file: str
    level   : str

@dataclass
class MLflowConfig:
    experiment_name:    str    # groups all runs under one experiment
    tracking_uri:       str    # "mlruns" local | "http://..." remote
    run_name:           str    # descriptive name — update when changing k
    log_artifacts:      bool       

@dataclass 
class AppConfig:
    data:           DataConfig
    features:       FeatureConfig
    outliers:       OutlierConfig
    model:          ModelConfig
    cluster_labels: ClusterLabelConfig
    logging:        LoggingConfig
    mlflow:         MLflowConfig


def _resolve_path(relative_path:str) -> str:
    """
    Convert relative path from config.yaml into absolute path based on project root.
    Anchor to ROOT_DIR to ensure it works regradless of where the script is run from.
    """
    if os.path.isabs(relative_path):
        return relative_path # already absolute, so leave it as is
    return os.path.join(ROOT_DIR,relative_path)


def _validate_config(cfg: AppConfig):
    """
    Validate config values with fail fast checks, prevent waste of computing on invalid config when run pipeline.
    Called by load_config() before returning AppConfig object.
    Ensure all value in config.yaml are valid and raise ConfigError if any value is missing or invalid 
    """
    # Data file existence check 
    if not os.path.exists(cfg.data.filepath):
        raise ConfigError(f"Data file not found at {cfg.data.filepath}")
    
    # Data column check 
    if not cfg.data.required_columns:
        raise ConfigError("Required columns is missing")
    
    # Data date formate check 
    if not cfg.data.date_format.startswith("%"):
        raise ConfigError(f"date_format {cfg.data.date_format} is invalid")
    
    # Features columns check
    if not cfg.features.rfm_columns:
        raise ConfigError("rfm_columns is missing")
    
    if len(cfg.features.rfm_columns) < 2:
        raise ConfigError(" rfm_columns should have at least 2 features for k-mean")
    
    # Features recency offset check
    if cfg.features.recency_offset_days < 0:
        raise ConfigError("recency_offset_days should be non-negative")
    
    # check outlier method 
    outlier_method = {"IQR"}
    if cfg.outliers.method not in outlier_method:
        raise ConfigError(f"outlier method {cfg.outliers.method} is not supported")
    
    if cfg.outliers.iqr_multiplier <= 0 :
        raise ConfigError("iqr_multiplier should be positive")
    
    # Ensure all outlier features are valid features 
    all_outlier_features = set(cfg.outliers.remove_outlier_features + cfg.outliers.keep_outlier_features)
    
    unknown_outlier_features = [x for x in all_outlier_features if x not in cfg.features.rfm_columns]

    if unknown_outlier_features:
        raise ConfigError(f"outlier features {unknown_outlier_features} not found in rfm_columns")

    # K-mean k number checking
    if cfg.model.n_clusters < 2:
        raise ConfigError("n_cluster should be at least 2 for k-mean")
    
    if cfg.model.max_iter < 10:
        raise ConfigError("max_iter should be at least 10 for k-mean to converge, standard is 1000, recommended min is 300")
    
    valid_algo = {"lloyd", "elkan"}
    if cfg.model.algorithm not in valid_algo:
        raise ConfigError(f"algorithm {cfg.model.algorithm} is not supported for k-mean")
    
    if cfg.model.k_search_min < 2:
        raise ConfigError("k_search_min should be at least 2")
    
    if cfg.model.k_search_max < cfg.model.k_search_min:
        raise ConfigError(" k search max should be greater than k search min")
    
    if not (cfg.model.k_search_min <= cfg.model.n_clusters <= cfg.model.k_search_max):
        raise ConfigError("n_clusters should be between k_search_min and k_search_max")
    
    # cluster label check 
    if not cfg.cluster_labels.labels:
        raise ConfigError("cluster labels is missing")
    
    # Ensure cluster label keys match as -1,-2,-3 cluster are from manual labelling, 0,1,2 are from k-mean output 
    expected_kmean_id = set(range(cfg.model.n_clusters))
    manual_kmean_id = set(cfg.cluster_labels.labels.keys()) # manually added cluster label for outlier 
    missing_id = expected_kmean_id - manual_kmean_id 
    if missing_id:
        raise ConfigError(f"cluster labels is missing label for cluster id {missing_id}")
    
    # logging config check 
    valid_levels = {"DEBUG","INFO","WARNING","ERROR","CRITICAL"}
    if cfg.logging.level not in valid_levels:
        raise ConfigError(f"logging level {cfg.logging.level} is not valid")
    
    # matplotlib raises clear errors at render time if values are invalid


def _build_config(raw: dict) -> AppConfig:
    """
    Builder are map all raw yaml dict to typed dataclass object
    seperate from load config for better readability 
    """ 
    return AppConfig(
        data = DataConfig(
        filepath                 = _resolve_path(raw["data"]["filepath"]),
        encoding                 = raw["data"]["encoding"],
        date_column              = raw["data"]["date_column"],
        date_format              = raw["data"]["date_format"],
        customer_id_column       = raw["data"]["customer_id_column"],
        invoice_column           = raw["data"]["invoice_column"],
        quantity_column          = raw["data"]["quantity_column"],
        price_column             = raw["data"]["price_column"],
        required_columns         = raw["data"]["required_columns"],
        invalid_invoice_prefixes = raw["data"]["invalid_invoice_prefixes"],
        invalid_stockcodes       = raw["data"]["invalid_stockcodes"],
    ),

    features = FeatureConfig(
        rfm_columns           = raw["features"]["rfm_columns"],
        log_transform         = raw["features"]["log_transform"],
        recency_offset_days   = raw["features"]["recency_offset_days"],
        filter_positive_sales = raw["features"]["filter_positive_sales"],
    ),

    outliers = OutlierConfig(
        method                  = raw["outliers"]["method"],
        iqr_multiplier          = float(raw["outliers"]["iqr_multiplier"]),
        remove_outlier_features = raw["outliers"]["remove_outlier_features"],
        keep_outlier_features   = raw["outliers"]["keep_outlier_features"],
        manual_labels           = raw["outliers"]["manual_labels"],
    ),

    model = ModelConfig(
        n_clusters   = raw["model"]["n_clusters"],
        random_state = raw["model"]["random_state"],
        n_init       = raw["model"]["n_init"],
        max_iter     = raw["model"]["max_iter"],
        algorithm    = raw["model"]["algorithm"],
        k_search_min = raw["model"]["k_search_min"],
        k_search_max = raw["model"]["k_search_max"],
    ),

    cluster_labels = ClusterLabelConfig(
        labels = {int(k): v for k, v in raw["cluster_labels"].items()}
        # int(k) — yaml dict keys are strings, convert to int for cluster ID mapping
        # "0":"New Customer" → 0:"New Customer"
    ),

    logging = LoggingConfig(
        Log_file = _resolve_path(raw["logging"]["log_file"]),
        level    = raw["logging"]["level"].upper(),
    ),

    mlflow = MLflowConfig(
        experiment_name = raw["mlflow"]["experiment_name"],
        tracking_uri    = raw["mlflow"]["tracking_uri"],
        run_name        = raw["mlflow"]["run_name"],
        log_artifacts   = raw["mlflow"]["log_artifacts"],
    ),
)

def load_config(config_path: str) -> AppConfig:
    """
    Read yaml.config, build typed AppConfig object, validate value 
    """
    # resolve config path
    if not os.path.isabs(config_path):
        config_path = os.path.join(ROOT_DIR, config_path)
    
    if not os.path.exists(config_path):
        raise ConfigError(f"Config file not found at {config_path}")
    
    # read and parse yaml 
    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f) # safe load only parses data only, not code, prevents yaml code injection attack
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse config.yaml: {e}")
    
    # check raw config is not empty
    if not raw_config:
        raise ConfigError("Config.yaml is empty")
    
    # check all required sections are present in yaml
    required_sections = {"data", "features", "outliers", "model", "cluster_labels", "logging", "mlflow"}
    
    missing_sections = required_sections - set(raw_config.keys())
    if missing_sections:
        raise ConfigError(f"config.yaml is missing required sections: {missing_sections}")
    
    # build typed config object 
    try:
        cfg = _build_config(raw_config)
    except KeyError as e:
        raise ConfigError(f"failed to build config object from yaml: {e}")
    except(TypeError, ValueError) as e:
        raise ConfigError(f"invalid value type in config.yaml: {e}")
    
    # validate config values 
    _validate_config(cfg)

    return cfg
    
