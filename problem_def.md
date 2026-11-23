# Expense Anomaly Auditor
### Problem Definition Document: First Project Review

| | |
|---|---|
| **Project title** | Expense Anomaly Auditor: Detecting Unusual Transactions Using K-Means Clustering |
| **Domain** | Finance / Audit analytics |
| **Approach** | Unsupervised machine learning (clustering and anomaly detection) |
| **Deliverable type** | Analysis notebooks, ranked audit report, and an interactive web app |

---

## 1. Problem Definition (the 4 Ws)

### WHO has the problem?

Finance and audit teams who review expense and transaction records, and the managers who depend on their findings. Individuals tracking their own spending face a smaller version of the same problem.

### WHAT is the problem?

Transaction records run into thousands or millions of rows per month. Reviewing them by hand is slow and inconsistent, and it tends to catch only obvious irregularities. Fixed rules (for example "flag anything above a set amount") miss new patterns and raise many false alarms, because a large amount is normal in one category and suspicious in another. Fraud labels are usually unavailable, so a supervised model cannot be trained.

**Problem statement:** *Auditors lack a scalable, label-free way to identify which transactions deserve attention. This project builds an unsupervised system that learns normal spending behaviour with K-Means and produces a ranked, explainable list of unusual transactions for human review.*

### WHERE does the problem occur?

In the expense-approval, accounts-payable, and audit workflows of organisations, and in card or payment transaction monitoring. Anywhere a large ledger must be checked by a small number of reviewers.

### WHY does it matter?

- Unnoticed errors, duplicate payments, inflated claims, and fraud cause direct financial loss.
- Auditor time is limited, and reviewing everything is not realistic.
- Rule-only systems become outdated as behaviour changes.
- A prioritised list lets reviewers spend time where it is most likely to matter.

---

## 2. Objective

**Main objective:** Design and build a K-Means based system that flags and ranks unusual transactions in a transaction dataset, without needing labelled fraud examples.

**Specific objectives**

1. Clean and explore a transaction dataset to understand normal spending patterns.
2. Engineer features that describe behaviour (for example how far an amount is from the usual amount for its category).
3. Cluster transactions with K-Means, choosing the number of clusters with justified evidence.
4. Convert cluster results into an anomaly score and a ranked list of flagged transactions.
5. Validate the detector against a baseline and against known or injected anomalies.
6. Present results in charts, an exportable audit table, and a simple web interface.

---

## 3. Expected Deliverables

| # | Deliverable | Description |
|---|---|---|
| D1 | Cleaned dataset and data dictionary | Documented columns, cleaning decisions, and assumptions |
| D2 | EDA notebook | Distributions, category and time patterns, written observations |
| D3 | Feature-engineering pipeline | Reusable code that turns raw rows into scaled model inputs |
| D4 | K-Means model and cluster profiles | Chosen *k* with elbow and silhouette evidence, plain-language description of each cluster |
| D5 | Ranked audit table | CSV of transactions with cluster, anomaly score, and flag, sorted by risk |
| D6 | Validation report | Comparison with an Isolation Forest baseline, injected-anomaly test, and label-based metrics if labels exist |
| D7 | Visualisations | Cluster plot (PCA) with anomalies highlighted, score distribution, flags over time |
| D8 | Streamlit app | Upload a CSV, adjust sensitivity, view flags, download the report |
| D9 | Documentation | README, requirements, and project documentation |

---

## 4. Success Metrics

Because the method is unsupervised, success is judged from several angles instead of a single accuracy number. The targets below are **proposed** and may be revised once the dataset is chosen and baseline results are known.

| # | Metric | How it is measured | Proposed target |
|---|---|---|---|
| M1 | **Cluster quality** | Silhouette score across candidate *k* values | At least 0.25 at the chosen *k*, with interpretable clusters |
| M2 | **Detection of injected anomalies** | Insert known unusual transactions (for example very large amounts for a category) and check how many appear in the top-flagged set | At least 80% recalled within the top 5% of ranked transactions |
| M3 | **Lift over random** (if the dataset has fraud labels) | Fraud rate among flagged transactions divided by the overall fraud rate | At least 3x |
| M4 | **Precision of top flags** | Manual review of the top 20 flagged transactions, judged as genuinely unusual or not | At least 60% judged genuinely unusual |
| M5 | **Agreement with baseline** | Overlap between K-Means flags and Isolation Forest flags | Reported, with disagreements examined |
| M6 | **Stability** | Share of top-flagged transactions that stay flagged across 5 random seeds | At least 80% |
| M7 | **Review workload reduction** | Flagged transactions as a share of all transactions | At most 5% of the ledger |
| M8 | **Usability** | A new CSV can be uploaded and reviewed in the app without code | Works end to end on a file not used during development |

**Overall success** means M1, M2, M4 and M8 are met, with the remaining metrics reported honestly even if they fall short.

---

## 5. Stakeholders and Target Users

### Key stakeholders

| Stakeholder | Interest in the project |
|---|---|
| Auditors and finance reviewers | Want a short, trustworthy list to investigate |
| Finance managers / CFO office | Want reduced loss and a defensible review process |
| Compliance and risk teams | Want transparency, an audit trail, and fair treatment of transactions |
| Data analysts / developers (project team) | Want reproducible, maintainable code |
| Course evaluators / mentors | Want a sound method, validation, and clear documentation |
| Employees or customers whose transactions are reviewed | Indirectly affected, so flags must not be treated as accusations |

### Target users

- **Primary:** internal auditors and expense reviewers with no coding background who need a ready-to-use interface.
- **Secondary:** data analysts who want to inspect the model and adapt it to their own data.

---

## 6. Scope

### Within scope

- Batch analysis of a historical transaction dataset provided as a CSV
- Data cleaning, EDA, and feature engineering
- K-Means clustering with elbow and silhouette analysis
- Anomaly scoring from centroid distance and small-cluster membership
- Isolation Forest as a comparison baseline
- Validation using injected anomalies, manual review, and labels where available
- Charts, a ranked audit table, and a Streamlit app
- Project documentation

### Outside scope

- Real-time or streaming detection
- Integration with live banking, ERP, or accounting systems
- Automatic blocking, rejection, or approval of transactions
- Legal determination that a transaction is fraudulent
- Supervised fraud classification as the main method
- User authentication, multi-user management, and production-grade deployment
- Use of real personal or confidential financial data

---

## 7. Data Sources

### 7.1 Selection criteria

A suitable dataset has an **amount**, a **date or time**, and a **category or merchant** field, with readable (not anonymised) columns, enough rows for clustering, and ideally some labels for evaluation.

### 7.2 Candidate datasets (Kaggle)

| Priority | Dataset | What it offers | Points to check |
|---|---|---|---|
| **Primary** | **Credit Card Transactions Dataset** (`priyamchoksi/credit-card-transactions-dataset`) | Over 1.85 million credit card transactions with transaction time, amount, merchant, category, and anonymised customer details | Very large, so work on a sample. It appears to be simulated. Confirm on download whether a fraud label column is present |
| **Alternative 1** | **Comprehensive Credit Card Transactions Dataset** (`rajatsurana979/comprehensive-credit-card-transactions-dataset`) | Transaction amount, merchant name, and category | Check size, date field, and whether labels exist |
| **Alternative 2** | **Synthetic Financial Datasets for Fraud Detection (PaySim)** (`ealaxi/paysim1`) | About 6.3 million simulated mobile-money transactions over a 30-day period, with a fraud label, derived from real mobile money logs | Has no merchant category (it has a transaction type instead), and fraud is only about 0.13% of rows. The balance columns should not be used for fraud detection, as the dataset notes |
| **Fallback** | Self-generated synthetic expense data | Python script that generates normal spending plus injected anomalies | Gives full control and known ground truth, but is artificial |

### 7.3 Other source types considered

Public repositories such as UCI, government open-data portals, and AIKosh (IndiaAI) can be searched for expense or procurement data. Real organisational expense data is rarely public, which is why simulated datasets are the practical choice. These sources were not evaluated in detail for this document.

### 7.4 Data handling and ethics

- Only public, anonymised or simulated data is used. No personal or confidential data is collected.
- The dataset licence and citation are recorded in the README.
- Because the data is simulated, results show how the method behaves, not how it would perform on a real organisation's ledger. This limitation is stated in the final report.
- Raw data is kept out of the Git repository and re-downloaded through the Kaggle API.

---

## 8. Key Assumptions and Risks

| Item | Detail | Mitigation |
|---|---|---|
| Assumption | Most transactions are normal and anomalies are a small minority | Check with EDA |
| Assumption | Behaviour can be described by the engineered features | Test several feature sets |
| Risk | The chosen dataset lacks labels | Use injected anomalies, manual review, and the baseline comparison |
| Risk | K-Means gives unstable or unclear clusters | Scale features, try several seeds and *k* values, profile each cluster |
| Risk | Skewed amounts dominate the clusters | Log-transform amounts and use category-relative z-scores |
| Risk | Too many false flags | Rank by score and offer a sensitivity control instead of a hard cutoff |
| Risk | Schedule slips | Prioritise the core pipeline and defer optional items (interactive plots, extra baselines) |

---

## 9. Timeline Summary

| Week | Focus |
|---|---|
| 1 | Setup, dataset selection, data dictionary |
| 2 | Cleaning and EDA |
| 3 | Feature engineering |
| 4 | K-Means and cluster profiling |
| 5 | Anomaly scoring and ranking |
| 6 | Validation, baseline, and visuals |
| 7 | Streamlit app, README, and presentation |