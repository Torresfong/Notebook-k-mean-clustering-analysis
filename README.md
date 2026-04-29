# Customer Segmentation — RFM + K-Means Clustering

An end-to-end unsupervised machine learning pipeline that transforms 5.25 million raw e-commerce transactions into six actionable customer segments using the RFM framework and K-Means clustering.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Notebooks](#notebooks)
- [Artifacts](#artifacts)
- [Methodology](#methodology)
  - [Stage 1 — EDA & Data Cleaning](#stage-1--eda--data-cleaning)
  - [Stage 2 — Feature Engineering (RFM)](#stage-2--feature-engineering-rfm)
  - [Stage 3 — K-Means Clustering](#stage-3--k-means-clustering)
- [Segment Definitions](#segment-definitions)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)

---

## Project Overview

Retail businesses accumulate millions of transactions but often lack visibility into **who their customers really are**. This project solves that by:

1. Cleaning raw transactional data at scale
2. Engineering three customer-level behavioural features using the **RFM (Recency, Frequency, Monetary)** framework
3. Applying a **two-stage segmentation strategy** — K-Means on the majority population and rule-based labelling on extreme outliers
4. Producing six named customer segments with concrete business action recommendations

---

## Dataset

| Property | Detail |
|---|---|
| Source | [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| Raw shape | 5,254,618 rows × 8 columns |
| Cleaned shape | 350,916 rows × 8 columns |
| Data loss | 33.21% filtered during cleaning |
| Granularity | One row per line item per invoice |

**Columns:** `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`

---

## Pipeline Architecture

```
Raw Excel (5.25M rows)
        │
        ▼
┌─────────────────────┐
│   EDA & Cleaning    │  → Drop cancellations, nulls, invalid StockCodes, zero-price rows
│   (EDA.ipynb)       │
└────────┬────────────┘
         │  cleaned_online_retail.csv (350,916 rows)
         ▼
┌─────────────────────┐
│ Feature Engineering │  → Aggregate to customer level → RFM features → IQR outlier split → StandardScaler
│ (Feature_eng...ipynb)│
└────────┬────────────┘
         │
         ├── non_outlier_df.csv     (3,785 customers)
         ├── Monetary_outlier_df.csv  (412 customers)
         ├── Frequency_outlier_df.csv (269 customers)
         └── scaled_data_df.csv     (3,785 customers, scaled)
         │
         ▼
┌─────────────────────┐
│ K-Means Clustering  │  → Elbow + Silhouette → k=4 → Violin plot interpretation → Outlier labelling
│ (kmean_clust...ipynb)│
└────────┬────────────┘
         │
         ▼
  6 Named Customer Segments
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `EDA.ipynb` | Data exploration, anomaly detection via regex, data cleaning |
| `Feature_engieering.ipynb` | RFM aggregation, IQR outlier isolation, StandardScaler |
| `kmean_clustering_model.ipynb` | Elbow/silhouette k-selection, K-Means training, cluster interpretation, outlier labelling |

---

## Artifacts

| File | Description | Rows |
|---|---|---|
| `cleaned_online_retail.csv` | Cleaned transaction-level data | 350,916 |
| `cleaned_df.csv` | Intermediate cleaned dataset | 350,916 |
| `non_outlier_df.csv` | Customer-level RFM table (outliers removed) | 3,785 |
| `Monetary_outlier_df.csv` | Customers with extreme `Total_sales` | 412 |
| `Frequency_outlier_df.csv` | Customers with extreme `Frequency` | 269 |
| `scaled_data_df.csv` | StandardScaler output — input to K-Means | 3,785 |

---

## Methodology

### Stage 1 — EDA & Data Cleaning

**Anomaly detection using regex:**

| Issue | Detection Method | Action |
|---|---|---|
| Cancelled invoices (prefix `C`) | `^\\d{6}$` regex | Removed |
| Adjustment invoices (prefix `A`) | String match | Removed |
| Non-product StockCodes (`POST`, `DOT`, `M`, `D`, `S`, `C2`, `DCGS`) | Unique string match | Removed |
| Missing `Customer ID` | `isnull()` | Removed |
| Zero-price rows | `Price <= 0` | Removed |
| Duplicate rows | `duplicated()` | Removed |

**Key insight:** `PADS` (padding) was deliberately kept as it represents a real product. All StockCode decisions are documented in EDA as a decision log — a good habit for production pipelines.

---

### Stage 2 — Feature Engineering (RFM)

**Aggregation from transaction → customer level:**

```python
agg_df = cleaned_df.groupby("Customer ID").agg(
    Total_sales = ("Sales", "sum"),       # Monetary — total revenue per customer
    Frequency   = ("Invoice", "nunique"), # Frequency — number of unique orders
    LastInvoiceDate = ("InvoiceDate", "max")
)
agg_df["Recency"] = (reference_date - agg_df["LastInvoiceDate"]).dt.days
```

**Outlier isolation (IQR method):**

Monetary and Frequency outliers are separated **before** clustering. This is the **two-stage segmentation** pattern — standard K-Means on the majority, rule-based labelling on extremes. Recency outliers were kept in the main cluster because they represent genuinely inactive customers which is valid signal for segmentation.

```
Upper fence = Q3 + 1.5 × IQR
```

**Feature scaling:**

`StandardScaler` is applied so that Monetary (large values) does not dominate the Euclidean distance used by K-Means, giving Recency and Frequency equal weight.

---

### Stage 3 — K-Means Clustering

**Optimal k selection — Elbow + Silhouette:**

| k | Silhouette Score | Decision |
|---|---|---|
| 3 | Highest | Statistically best |
| **4** | **0.41 (2nd)** | **Chosen — more actionable segments** |

K=4 was chosen over K=3 because it produces four meaningfully distinct business segments while maintaining a strong silhouette score. This is a valid product decision — pure metric optimisation does not always maximise business value.

**Outlier labelling (rule-based):**

| Overlap | Cluster ID | Label |
|---|---|---|
| Monetary only | `-1` | VIP |
| Both Monetary & Frequency | `-3` | Champion |
| Frequency only | `-2` | Absorbed into `-3` |

---

## Segment Definitions

| Cluster | Label | Recency | Frequency | Monetary | Recommended Action |
|---|---|---|---|---|---|
| 0 | **New Customer** | Low | Low | Low | Build relationship, onboarding incentives |
| 1 | **Key Customer** | Low | High | High | Loyalty rewards, early access |
| 2 | **Dormant** | High | Very Low | Low | Re-engagement campaigns |
| 3 | **Occasional** | High | Moderate | Moderate | Personalised offers, targeted promotions |
| -1 | **VIP** | Varies | Low–Moderate | Extreme | Personalised service, white-glove experience |
| -3 | **Champion** | Varies | Extreme | Extreme | Exclusive VIP programme, prevent churn at all cost |

> **Note:** Low Recency = purchased recently. High Recency = has not purchased in a long time.

---

## Tech Stack

```
Python 3.x
├── pandas          — data manipulation, aggregation
├── numpy           — numerical operations
├── matplotlib      — visualisations (histogram, boxplot, 3D scatter)
├── seaborn         — violin plots, cluster distribution
├── scikit-learn
│   ├── KMeans      — clustering
│   ├── silhouette_score — k selection metric
│   └── StandardScaler   — feature scaling
└── scipy           — supporting statistics
```

---

## Project Structure

```
.
├── notebook/
│   ├── EDA.ipynb
│   ├── Feature_engieering.ipynb
│   └── kmean_clustering_model.ipynb
├── artifact/
│   ├── cleaned_online_retail.csv
│   ├── cleaned_df.csv
│   ├── non_outlier_df.csv
│   ├── Monetary_outlier_df.csv
│   ├── Frequency_outlier_df.csv
│   └── scaled_data_df.csv
└── README.md
```

---

## How to Run

**1. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl
```

**2. Download the dataset**

Download `online_retail_II.xlsx` from [UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and place it in your working directory.

**3. Run notebooks in order**

```
EDA.ipynb  →  Feature_engieering.ipynb  →  kmean_clustering_model.ipynb
```

Each notebook reads from the artifact CSVs produced by the previous step. Update the file paths at the top of each notebook to match your local directory.

---

> Built as a portfolio project demonstrating a production-style unsupervised ML pipeline covering data cleaning, feature engineering, outlier strategy, model selection, and business interpretation.
