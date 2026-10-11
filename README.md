# Amazon Prime Video Content & Market Trends Analysis

An end-to-end data analytics and machine learning capstone project exploring Amazon Prime Video's catalogue, investigating content and rating patterns, and predicting titles likely to receive strong critical ratings.

> **Project focus:** Exploratory Data Analysis (EDA), statistical hypothesis testing, feature engineering, text processing, classification, model evaluation, and model explainability.

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Key Questions](#key-questions)
- [Workflow](#workflow)
- [Key Findings](#key-findings)
- [Machine Learning](#machine-learning)
- [Model Evaluation](#model-evaluation)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Using the Saved Model](#using-the-saved-model)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Overview

This project studies the Amazon Prime Video titles catalogue and associated cast/crew records to understand the catalogue's composition, content trends, and relationships with critical reception. It combines descriptive analytics and statistical testing with a binary classification task that estimates whether a title is likely to be a critically well-rated title.

For this project, a **“hit” is defined as a title with an IMDb score of 7.0 or higher**. This is a proxy for critical reception, not a direct measure of commercial success, viewing hours, subscriber retention, or revenue.

## Objectives

- Understand the balance between movies and TV shows in the catalogue.
- Explore genre, release-year, country, age-certification, runtime, and rating patterns.
- Combine title metadata with cast and crew records to derive additional features.
- Test selected hypotheses about content type, certification, production country, and ratings.
- Build and compare classification models for the defined `is_hit` target.
- Explain model behaviour using permutation feature importance.
- Save the selected model and preprocessing objects for potential reuse.

## Dataset

The repository contains two CSV files:

| File | Description | Rows in supplied file | Columns |
|---|---|---:|---:|
| `titles.csv` | Title-level metadata, including type, description, release year, runtime, genres, production countries, IMDb and TMDB fields | 9,871 | 15 |
| `credits.csv` | Cast and crew records associated with titles, including person, role, and character information | 124,235 | 5 |

### Important fields

**`titles.csv`** includes `id`, `title`, `type`, `description`, `release_year`, `age_certification`, `runtime`, `genres`, `production_countries`, `seasons`, `imdb_score`, `imdb_votes`, `tmdb_popularity`, and `tmdb_score`.

**`credits.csv`** includes `person_id`, `id`, `name`, `character`, and `role`. The `id` field links credit records to titles.

The notebook cleans the data before analysis; therefore, the row counts used in individual analyses may be lower than the raw CSV counts due to duplicate removal and missing values.

> **Dataset provenance:** The supplied project files do not establish a verifiable original download URL. Add the exact dataset source and its license here if you have that information.

## Key Questions

- How is the catalogue divided between movies and TV shows?
- Which genres and production countries appear most frequently?
- How do ratings vary by content type, certification, and country?
- What do release-year and runtime patterns reveal about the catalogue?
- Are selected differences in ratings statistically significant?
- Can pre-release metadata and description text help distinguish titles with IMDb scores of at least 7.0?

## Workflow

1. **Load and inspect data** — review shapes, data types, duplicates, and missing values.
2. **Clean and join data** — prepare title metadata and aggregate credits to derive cast/crew counts.
3. **Explore and visualize** — investigate catalogue composition, genre and country distributions, temporal patterns, ratings, and relationships among variables.
4. **Test hypotheses** — use statistical tests to evaluate selected differences and associations.
5. **Engineer features** — handle structural missingness, impute selected fields, cap runtime outliers, encode categorical variables, and prepare numerical and text features.
6. **Process descriptions** — clean synopsis text, transform it with TF-IDF, and reduce text dimensions with Truncated SVD.
7. **Train and compare models** — fit Logistic Regression, Random Forest, and Gradient Boosting classifiers.
8. **Tune and evaluate** — use stratified cross-validation and `RandomizedSearchCV`, with F1-score as the tuning objective; evaluate on a held-out test set.
9. **Explain and save** — use permutation importance to inspect influential features and save the selected model bundle in pickle and joblib formats.

## Key Findings

The notebook's reported conclusions include:

- The cleaned catalogue contains approximately **9,868 unique titles**; movies make up roughly **86%** of titles, with TV shows accounting for the remainder.
- The catalogue is led by US-produced titles, and drama and comedy are among the most common genres.
- In the notebook's statistical tests, average IMDb ratings differed by content type; age certification was associated with the defined hit label; and average ratings varied across production countries. These are findings for the available dataset and should not be interpreted as causal relationships.
- Documentary/history content was identified in the notebook as potentially performing well relative to its catalogue share; this is a hypothesis for content strategy, not proof of future commercial demand.

## Machine Learning

### Target definition

The notebook creates a binary target:

- `is_hit = 1` when `imdb_score >= 7.0`
- `is_hit = 0` otherwise

The modelling workflow intentionally excludes post-engagement indicators such as IMDb votes, TMDB popularity, and TMDB score from the predictive features, aiming to reduce target leakage and better represent information that could be available before audience response is known.

### Models compared

- **Logistic Regression** — linear baseline; tuned with `RandomizedSearchCV` over regularization strength `C`.
- **Random Forest** — tree-ensemble baseline; tuned over estimator count, depth, and minimum-sample parameters.
- **Gradient Boosting Classifier** — boosting-based model; tuned over estimator count, learning rate, and tree depth.

The search uses **3-fold stratified cross-validation** and F1-score. The notebook also addresses class imbalance and evaluates models using accuracy, precision, recall, F1-score, ROC-AUC, classification reports, and confusion matrices.

## Model Evaluation

The notebook selects the tuned **Gradient Boosting Classifier** as its final model. The reported approximate results are:

| Metric | Reported result | Interpretation |
|---|---:|---|
| ROC-AUC | ~0.83 | Good ranking ability relative to chance on the evaluated split |
| F1-score | ~0.60 | Balance between precision and recall for the hit class |
| Recall (hit class) | ~0.70 | Finds a larger share of actual high-rated titles than the compared alternatives, at the cost of false positives |

These values are rounded summaries from the notebook narrative, not a guarantee of performance on new data. Consult the notebook's executed metric outputs for exact scores. IMDb rating is used as a proxy for critical reception; it does not directly represent business success.

### Explainability

Permutation importance is used to estimate how much predictive performance changes when each feature is shuffled. The notebook reports primary genre indicators, cast size, runtime, content age, and some description-derived SVD components among the leading signals. Importance indicates predictive usefulness in this dataset, not causation.

## Repository Structure

```text
Amazon-Prime-Video-Analytics/
├── Amazon_Prime_TV_Shows_and_Movies_Capstone.ipynb
├── titles.csv
├── credits.csv
├── amazon_prime_hit_predictor_gb.pkl
├── amazon_prime_hit_predictor_gb.joblib
└── README.md
```

## Tech Stack

- **Python** — analysis and model development
- **Jupyter Notebook** — interactive workflow and reporting
- **Pandas, NumPy** — data manipulation
- **Matplotlib, Seaborn** — visualization
- **SciPy / statistical testing** — hypothesis tests
- **Scikit-learn** — preprocessing, feature extraction, model training, validation, metrics, and permutation importance
- **NLTK** — text preprocessing in the notebook
- **Pickle / Joblib** — model bundle serialization

The notebook is the authoritative record of imports and processing steps. Install any additional package that your local notebook reports as missing.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd Amazon-Prime-Video-Analytics
```

Replace the URL and folder name with your actual GitHub repository details.

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter ipykernel nltk joblib
```

If the notebook requires additional packages, install them in the same environment.

### 4. Run the notebook

```bash
jupyter notebook Amazon_Prime_TV_Shows_and_Movies_Capstone.ipynb
```

Run the notebook cells in order. Keep `titles.csv` and `credits.csv` in the same working directory expected by the notebook, or update the data paths accordingly.

## Using the Saved Model

The repository includes two serialized versions of the model bundle:

- `amazon_prime_hit_predictor_gb.pkl`
- `amazon_prime_hit_predictor_gb.joblib`

The bundle contains the fitted model and preprocessing objects (including scaler, TF-IDF vectorizer, SVD transformer, feature-column metadata, and categorical feature lists). It is designed to be loaded by Python code that recreates the same feature-preparation pipeline.

A saved estimator alone is not enough to predict reliably from arbitrary raw title details: incoming records must be transformed into the same feature schema and column order used during training. The notebook's final cells show loading and a sanity check on held-out, already-prepared test rows; they do not provide a complete web API or user-facing prediction app.

**Security note:** Only load pickle/joblib files from a source you trust. Deserializing untrusted files can execute malicious code.

## Limitations

- The catalogue only contains titles represented in the supplied dataset; it does not include rejected or never-acquired titles.
- IMDb score is an imperfect proxy for a “hit” and does not capture watch time, completion rate, renewals, revenue, or subscriber retention.
- Missing metadata and imputation can introduce noise.
- The reported results depend on the chosen data split, features, target threshold, and class-balancing method. External validation is needed before real-world use.
- Relationships in observational catalogue data are not causal evidence.
- The repository does not include a deployment interface or automated inference API.

## Future Improvements

- Add a verified dataset source, license, and data dictionary.
- Build a reproducible `requirements.txt` and a single reusable preprocessing/inference pipeline.
- Evaluate with repeated cross-validation and calibration; report exact metrics in a versioned results table.
- Test on a time-based split or external dataset to better assess generalization to future releases.
- Incorporate platform-level outcomes such as watch time, completion rate, retention, and licensing cost if legitimately available.
- Build a small interface or API for structured predictions, with clear explanations and uncertainty.
- Add charts or dashboard screenshots to make the exploratory findings easier to browse on GitHub.

## Author

**Nandini Pandey**  
GitHub profile - https://github.com/NandiniPandey2004, LinkedIn - https://www.linkedin.com/in/nandini-pandey-749325255/

---

*This project is intended for learning, exploratory analysis, and decision-support research. Model predictions should not be treated as guarantees of a title's future performance.*
