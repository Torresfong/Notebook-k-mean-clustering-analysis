import logging 
import numpy as np
import pandas as pd
from typing import Dict,List,Tuple

from src.config_loader import DataConfig,FeatureConfig,OutlierConfig
from src.exceptional import FeatureEngineeringError

log = logging.getLogger(__name__)

def _compute_iqr_bounds(
        df: pd.DataFrame,
        features: List[str],
        multiplier: float,
) -> Dict[str,tuple[float,float]]:
    bounds = {}
    for feature in features:
        Q1  = df[feature].quantile(0.25)
        Q3  = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        bounds[feature] = (lower, upper)
        log.info(
            f"  {feature:<12} Q1={Q1:>8.3f} | Q3={Q3:>8.3f} | "
            f"IQR={IQR:>8.3f} | upper fence={upper:>10.3f}"
        )
    return bounds    

def _build_outlier_mask(
        rfm: pd.DataFrame,
        bounds: Dict[str,Tuple[float,float]],
)-> pd.Series:
    outlier_mask = pd.Series(False, index=rfm.index)
    for feature, (_, upper) in bounds.items():
        feature_flag  = rfm[feature] > upper
        outlier_mask  = outlier_mask | feature_flag
        log.info(
            f"  {feature}: {feature_flag.sum()} customers above upper fence"
        )
    return outlier_mask

def _assign_outlier_labels(
        rfm_outlier: pd.DataFrame,
        outlier_cfg: OutlierConfig,
        bounds:Dict[str, Tuple[float, float]], 
) -> pd.Series:
    # Pull label integers from config — -1, -2, -3 never appear below
    label_monetary_only  = outlier_cfg.manual_labels["monetary_only"]
    label_frequency_only = outlier_cfg.manual_labels["frequency_only"]
    label_both           = outlier_cfg.manual_labels["monetary_and_frequency"]

    # Re-derive per-feature breach flags on the outlier subset
    feat0 = outlier_cfg.remove_outlier_features[0]   # "Monetary"
    feat1 = outlier_cfg.remove_outlier_features[1]   # "Frequency"

    _, upper0 = bounds[feat0]
    _, upper1 = bounds[feat1]

    flag_monetary  = rfm_outlier[feat0] > upper0
    flag_frequency = rfm_outlier[feat1] > upper1

    # Start with NaN so any unassigned rows surface as visible gaps
    labels = pd.Series(np.nan, index=rfm_outlier.index)
    labels[flag_monetary  & ~flag_frequency] = label_monetary_only
    labels[~flag_monetary &  flag_frequency] = label_frequency_only
    labels[flag_monetary  &  flag_frequency] = label_both

    # Safety net — fires only if flag logic has a gap
    unassigned = labels.isna().sum()
    if unassigned > 0:
        log.warning(
            f"{unassigned} outlier row(s) could not be assigned a label.\n"
            f"Defaulting to monetary_only ({label_monetary_only}).\n"
            f"Inspect IQR bounds for unexpected values."
        )
        labels = labels.fillna(label_monetary_only)

    log.info("  Outlier label distribution:")
    for val, count in labels.value_counts().sort_index().items():
         log.info(f" Cluster_ID {int(float(str(val))):>2} : {count} customers")

    return labels.astype(int)

def build_rfm(
        df: pd.DataFrame,
        feature_cfg: FeatureConfig,
        data_cfg: DataConfig,
        outlier_cfg: OutlierConfig,
) -> Tuple[pd.DataFrame,pd.DataFrame,pd.Series]:
    
    log.info("Featutes engieering started")

    # sales guard - check sales wheter have any values < 0 
    if feature_cfg.filter_positive_sales:
        before = len(df)
        df = df[df[data_cfg.sales_column]>0]
        Negative_sale_col = before -len(df)

        if Negative_sale_col > 0:
            log.warning(f"sales remove {Negative_sale_col} rows with {data_cfg.sales_column} <=0")

        else:
            log.info(" All values are positive")
    else:
        log.info("Skip - all sales rows are clean ")

    # step 2 compute snapshot data= latest transaction date + offset
    log.info("Computing snapshot date")
    latest_date = df[data_cfg.date_column].max()
    snapshot_date = latest_date + pd.Timedelta(days=feature_cfg.recency_offset_days)

    # step 3 aggregate RFM per customer 
    log.info("Aggregating RFM features per customer")

    rfm = (
        df.groupby(data_cfg.customer_id_column)
        .agg(
            Recency = (
                data_cfg.date_column,
                lambda x: (snapshot_date - x.max()).days
            ),
            Frequency=(
                data_cfg.invoice_column,
                "nunique"
            ),
            Monetary=(
                data_cfg.sales_column,
                "sum"
            ),
        )
        .reset_index()
    )

    log.info(f"RFM table shape: {rfm.shape[0]} customers x {rfm.shape[1]} columns")

    # step 4 Post aggregation validation
    log.info("aggregation validation")
    if rfm.empty:
        raise FeatureEngineeringError(
            "RFM table is empty after aggregation")
    
    missing_value = rfm[feature_cfg.rfm_columns].isnull().sum()
    if missing_value.any():
        raise FeatureEngineeringError("missing vlaue found in RFM columns")
    
    negative_recency = (rfm["Recency"] < 0 ).sum()
    if negative_recency > 0 : 
        raise FeatureEngineeringError(
            f"{negative_recency} customer have negative Recency\n"
            "Root cause is wrong date_format"
        )
    
    log.info("Post-aggregation validation passed")

    # step 5 log transformation
    # Show raw RFM distribution before any transformation
    log.info("Pre-transform RFM")
    log.info(
            f"  Recency   — "
        f"min: {rfm['Recency'].min():.1f} days | "
        f"max: {rfm['Recency'].max():.1f} days | "
        f"mean: {rfm['Recency'].mean():.1f} days"
    )
    log.info(
        f"  Frequency — "
        f"min: {rfm['Frequency'].min():.0f} orders | "
        f"max: {rfm['Frequency'].max():.0f} orders | "
        f"mean: {rfm['Frequency'].mean():.1f} orders"
    )
    log.info(
        f"  Monetary  — "
        f"min: {rfm['Monetary'].min():.2f} | "
        f"max: {rfm['Monetary'].max():.2f} | "
        f"mean: {rfm['Monetary'].mean():.2f}"
    )

    # Log1 transformation = compress right-skewed distribution
    if feature_cfg.log_transform:
        log.info(f"Applying log1p transformation to: {feature_cfg.rfm_columns}")

        for col in feature_cfg.rfm_columns:
            rfm[col] = np.log1p(rfm[col])

        log.info("Log1p transfromation applied")
        log.info(
            f"  Recency   — "
            f"min: {rfm['Recency'].min():.3f} | "
            f"max: {rfm['Recency'].max():.3f} | "
            f"mean: {rfm['Recency'].mean():.3f}"
        )
        log.info(
            f"  Frequency — "
            f"min: {rfm['Frequency'].min():.3f} | "
            f"max: {rfm['Frequency'].max():.3f} | "
            f"mean: {rfm['Frequency'].mean():.3f}"
        )
        log.info(
            f"  Monetary  — "
            f"min: {rfm['Monetary'].min():.3f} | "
            f"max: {rfm['Monetary'].max():.3f} | "
            f"mean: {rfm['Monetary'].mean():.3f}"
        )

    else:
        log.info("Log transform disabled in config.yaml (log_transform: false) ")

    # step 7 post transformation validation 
    log.info("Post transformation validation")
    
    # check for Null value after log transform 
    nan_value = rfm[feature_cfg.rfm_columns].isnull().sum()
    if nan_value.any():
        raise FeatureEngineeringError( f"NaN values found after log transform" )
    
    # check infinite values 
    inf_count = np.isinf(rfm[feature_cfg.rfm_columns].values).sum()
    if inf_count > 0:
        raise FeatureEngineeringError(f"{inf_count} infinite values found after log transformation")
    
    log.info("log transform validation done ")

    log.info(f"Computing IQR bounds on: {outlier_cfg.remove_outlier_features}")
    log.info(
        f"  method     : {outlier_cfg.method} "
        f"(multiplier = {outlier_cfg.iqr_multiplier})"
    )
    log.info(
        f"  Skipping   : {outlier_cfg.keep_outlier_features} "
        f"(Recency outliers stay in clean set)"
    )
    bounds = _compute_iqr_bounds(
        rfm,
        outlier_cfg.remove_outlier_features,
        outlier_cfg.iqr_multiplier,
    )
    log.info("Detecting outliers...")
    outlier_mask = _build_outlier_mask(rfm, bounds)
    log.info(
        f"  Total flagged : {outlier_mask.sum()} / {len(rfm)} customers"
    )


    # Step 8 Split into clean and outlier sets 
    log.info("Splitting RFM table into clean and outlier sets...")

    rfm_clean   = rfm[~outlier_mask].copy()
    rfm_outlier = rfm[ outlier_mask].copy()

    log.info(f"  Clean set   : {len(rfm_clean)} customers → KMeans")
    log.info(f"  Outlier set : {len(rfm_outlier)} customers → manual labels")

    if rfm_clean.empty:
        raise FeatureEngineeringError(
            "No clean customers remain after outlier separation.\n"
            "All customers were flagged as outliers.\n"
            "Increase iqr_multiplier in config.yaml to reduce sensitivity."
        )

    if rfm_outlier.empty:
        log.warning(
            "No outliers detected — entire RFM table goes to KMeans.\n"
            "Outlier labels (-1, -2, -3) will not appear in output.\n"
            "Consider reducing iqr_multiplier in config.yaml."
        )


    #  Step 9 Assign manual labels to outlier customers 
    # -1 / -2 / -3 come from manual_labels in config — never hardcoded
    log.info("Assigning manual labels to outlier customers...")

    if not rfm_outlier.empty:
        outlier_labels = _assign_outlier_labels(
            rfm_outlier, outlier_cfg, bounds
        )
    else:
        # Return empty Series with int dtype — model.py checks .empty
        outlier_labels = pd.Series(dtype=int)
        log.info("  No outliers to label — returning empty Series")

    log.info("Feature engineering complete")
    log.info(f"  Output shape  : {rfm.shape[0]} customers × {rfm.shape[1]} columns")
    log.info(f"  RFM columns   : {feature_cfg.rfm_columns}")
    log.info(f"  Log transform : {feature_cfg.log_transform}")
    log.info(f"  Snapshot date : {snapshot_date.date()}")
    log.info("  Sample (first 3 rows):")
    for _, row in rfm.head(3).iterrows():
        log.info(
            f"    {data_cfg.customer_id_column}: {row[data_cfg.customer_id_column]} | "
            f"Recency: {row['Recency']:.3f} | "
            f"Frequency: {row['Frequency']:.3f} | "
            f"Monetary: {row['Monetary']:.3f}"
        )


    return rfm_clean,rfm_outlier,outlier_labels