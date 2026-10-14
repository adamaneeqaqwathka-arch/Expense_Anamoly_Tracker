# Expense Anomaly Auditor

An unsupervised machine learning tool that learns what normal spending looks like and flags unusual transactions for audit review, using **K-Means clustering**.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-K--Means-orange)
![Streamlit](https://img.shields.io/badge/app-Streamlit-red)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Getting the Data](#getting-the-data)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Tech Stack](#tech-stack)

---

## Overview

Reviewing expense records by hand is slow and inconsistent, and fixed rules such as "flag anything above a set amount" miss new patterns while raising many false alarms. Without labelled fraud examples, a supervised classifier is not an option either.

**Expense Anomaly Auditor** takes an unsupervised approach. It groups transactions by spending behaviour, then flags the ones that sit far from their group or in unusually small groups. Flagged transactions are ranked by an anomaly score so a reviewer can start with the riskiest.

> **Note:** A flagged transaction is *unusual*, not necessarily fraudulent. The tool prioritises human attention and does not replace human judgement.

---

## Features

- Data cleaning and exploratory analysis of transaction data
- Feature engineering: category-relative z-scores, merchant frequency, time-based features, rolling statistics
- K-Means clustering with *k* chosen using the elbow method and silhouette score
- Anomaly scoring from distance to the cluster centre and small-cluster membership
- Isolation Forest baseline for comparison
- 2D PCA cluster visualisation with anomalies highlighted
- Exportable, ranked audit table
- Streamlit app for file upload, sensitivity control, and report download

---

## How It Works

```
Transaction CSV
      │
      ▼
Cleaning ──► EDA ──► Feature engineering ──► Scaling and encoding
                                                    │
                                                    ▼
                                         K-Means (k from elbow + silhouette)
                                                    │
                                   ┌────────────────┴────────────────┐
                                   ▼                                 ▼
                       Distance to centroid                Small-cluster check
                                   └────────────────┬────────────────┘
                                                    ▼
                                         Combined anomaly score
                                                    │
                                                    ▼
                                  Ranked audit table, charts, Streamlit app
```

1. **Clean** the data: fix types, handle missing values, remove duplicates.
2. **Engineer features** that describe behaviour, for example how far an amount is from the usual amount for its category.
3. **Scale** all numeric features, because K-Means is distance-based and sensitive to scale.
4. **Cluster** with K-Means and profile each cluster.
5. **Score** each transaction by its distance from its centroid, adjusted for cluster size.
6. **Rank and review** the highest-scoring transactions.

---

## Project Structure

```
expense-anomaly-auditor/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # Dataset goes here (not tracked by Git)
├── notebooks/             # Exploration and analysis notebooks
├── src/                   # Reusable code: cleaning, features, model, scoring
├── outputs/               # Generated tables and plots (not tracked)
├── models/                # Saved scaler and model (not tracked)
└── app.py                 # Streamlit application
```

> Adjust this tree to match your final layout.

---

## Getting Started

### Prerequisites

- Python 3.9 or newer
- A Kaggle account (to download the dataset)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/expense-anomaly-auditor.git
cd expense-anomaly-auditor

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Getting the Data

The dataset is not included in the repository. Download it from Kaggle.

**1. Create a Kaggle API token**

Go to Kaggle → **Settings → API → Create New Token**. Place the downloaded `kaggle.json` in:

- Windows: `C:\Users\<YourName>\.kaggle\`
- Mac / Linux: `~/.kaggle/` (then run `chmod 600 ~/.kaggle/kaggle.json`)

**2. Download the dataset**

```python
import kagglehub

path = kagglehub.dataset_download("<owner>/<dataset-name>")
print(path)
```

or with the command line:

```bash
kaggle datasets download -d <owner>/<dataset-name> --unzip -p data/
```

**Dataset used:** `<TODO: dataset name and Kaggle link>`

**Required columns:** a transaction amount, a date, and a category or merchant field.

> Never commit `kaggle.json`. It is already listed in `.gitignore`.

---

## Usage

### Run the analysis notebooks

```bash
jupyter notebook
```

Open the notebooks in `notebooks/` in order.

### Run the Streamlit app

```bash
streamlit run app.py
```

Then:

1. Upload a transactions CSV.
2. Adjust the sensitivity slider to control how many transactions are flagged.
3. Review the ranked table and charts.
4. Download the audit report as CSV.

---

## Results

> Fill this section in after running your analysis.

| Item | Value |
|---|---|
| Dataset size | `<TODO>` rows |
| Number of clusters (k) | `<TODO>` |
| Silhouette score | `<TODO>` |
| Transactions flagged | `<TODO>` (`<TODO>`%) |
| Overlap with Isolation Forest | `<TODO>` |
| Precision / recall (if labels exist) | `<TODO>` |

**Suggested figures to add:**

- Elbow and silhouette plots
- PCA cluster plot with anomalies highlighted
- Anomaly score distribution
- Example of the top flagged transactions

```markdown
![Cluster plot](outputs/pca_clusters.png)
```

---

## Limitations

- **Anomaly does not mean fraud.** Rare behaviour can be completely legitimate.
- **K-Means assumes roughly round, similarly sized clusters** and is sensitive to scaling, outliers, and the choice of *k*.
- **Detection depends on the features.** Only anomalies visible through the engineered features can be found.
- **Without labels, accuracy is hard to measure.** Validation relies on cluster quality, synthetic anomaly injection, a baseline comparison, and manual review.
- **Results depend on the dataset** and may not transfer to other organisations.

---

## Roadmap

- [x] Project planning and documentation
- [ ] Data download and exploration
- [ ] Cleaning and feature engineering
- [ ] K-Means clustering and cluster profiling
- [ ] Anomaly scoring and ranking
- [ ] Validation and Isolation Forest baseline
- [ ] Streamlit app and final packaging

**Possible extensions:** additional detectors (DBSCAN, Local Outlier Factor), duplicate-payment rules, per-department profiles, reviewer feedback to tune thresholds, and explanations of why each transaction was flagged.

---

## Tech Stack

Python · pandas · NumPy · SciPy · scikit-learn · matplotlib · seaborn · Plotly · Streamlit · joblib · kagglehub

---

## License

`<TODO: choose a license, for example MIT>`

## Author

`<TODO: your name and contact or GitHub profile>`