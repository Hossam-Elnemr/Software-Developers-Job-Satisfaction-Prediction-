# Software Developers' Job Satisfaction Prediction

**Machine learning classification of developer job satisfaction using the 2025 Stack Overflow Developer Survey.**

This project investigates whether developers' career profiles, work arrangements, experience, technology choices, AI-tool attitudes, compensation, and community participation can help predict their reported level of job satisfaction. It covers exploratory data analysis (EDA), data preparation, feature engineering and selection, model training, hyperparameter tuning, and comparative evaluation.

> **Project outcome:** XGBoost achieved the highest reported test accuracy (**53.12%**) among seven evaluated classifiers. Linear SVM achieved the highest **macro F1 score (49.42%)**. These are moderate results, and the predictions should not be treated as a definitive assessment of an individual's job satisfaction.

## Table of contents

- [Problem and dataset](#problem-and-dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Running predictions](#running-predictions)
- [Large files and GitHub](#large-files-and-github)
- [Limitations and future work](#limitations-and-future-work)
- [Data source and attribution](#data-source-and-attribution)

## Problem and dataset

**Task:** Multiclass classification of `JobSatLevel`:

| Encoded label | Job satisfaction level |
| --- | --- |
| `0` | Low |
| `1` | Average |
| `2` | High |

The project uses a prepared dataset based on the **2025 Stack Overflow Annual Developer Survey**. Its `JobSatLevel` target is already present in the project input file; the supplied notebooks do not document a separate procedure for reconstructing this label from the original public survey release.

| Dataset detail | Value |
| --- | ---: |
| Rows in the project's initial CSV | 44,196 |
| Input columns | 156 |
| Records with a non-null target | 23,961 |
| Training records (80%, stratified) | 19,168 |
| Test records (20%, stratified) | 4,793 |
| Features in the final processed matrix | 79 |
| Random seed | 42 |

**Class distribution (labeled records):** Low: **6,803**; Average: **11,296**; High: **5,862**. The majority-class baseline is approximately **47.14% accuracy**, which provides useful context for the model results.

The supplied `survey_results_schema.csv` explains survey variable names and question wording. The project's survey questionnaire is also provided as `2025_Developer_Survey_Tool.pdf`.

## Methodology

The analysis is organized into six thematic feature blocks:

1. **AI usage and work experience** — AI-tool preferences, AI attitudes, professional experience, and learning behavior.
2. **AI agents and developer profile** — developer role, organization size, individual-contributor/manager status, and AI-agent engagement.
3. **Compensation and technology attitudes** — salary conversion to USD and responses to technology-related survey questions.
4. **Work and databases** — remote-work arrangements, employment information, and database-related responses.
5. **Stack Overflow participation** — community involvement, participation frequency, and tag-related responses.
6. **Tools and platforms** — programming tools, development environments, and platform selections.

The preprocessing workflow includes:

- An **80/20 stratified train/test split** before fitting train-dependent transformations.
- Removal of input columns with **at least 70% missing values in the training set**.
- Missing-value handling, categorical and ordinal encoding, and multi-hot encoding of semicolon-separated multi-select answers.
- Engineered features, such as counts of selected AI-agent uses, grouped developer roles, and compensation converted to USD.
- Exploratory statistical association analysis and feature selection, including **mutual information**, **chi-square/Cramér's V**, **Kruskal–Wallis**, and **PCA** analysis where applicable.
- Alignment of train/test feature columns, followed by **StandardScaler** transformation.

Seven classifiers were evaluated:

- Logistic Regression
- Linear Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Decision Tree with cross-validation tuning
- Random Forest with cross-validation tuning
- XGBoost

The modeling notebook also includes hyperparameter experiments for Linear SVM's `C` and Random Forest's `n_estimators`.

## Results

Reported evaluation on the held-out test set (`Milestone 2 Report.pdf`):

| Model | Accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| **XGBoost** | **53.12%** | 45.41% | 45.21% |
| Random Forest (CV) | 52.60% | 43.71% | 42.22% |
| Logistic Regression | 52.12% | 44.54% | 44.12% |
| Decision Tree (CV) | 51.91% | 43.29% | 42.45% |
| **Linear SVM** | 50.64% | 49.45% | **49.42%** |
| Random Forest | 47.65% | **49.74%** | 47.82% |
| Decision Tree | 45.15% | 47.36% | 45.36% |

**Interpretation:** XGBoost leads on overall accuracy, but the ranking changes under class-sensitive metrics. Linear SVM produces the strongest macro F1, while the baseline Random Forest has the highest balanced accuracy in this comparison. XGBoost's accuracy is approximately **5.98 percentage points** above predicting the majority class every time; therefore, accuracy alone would overstate the strength of the result.

The reported experiments also measured training and prediction time. See **`Milestone 2 Report.pdf`** and **`Modelling.ipynb`** for the detailed comparisons, tuning runs, and plots.

## Project structure

The following layout shows the main inputs, notebooks, script, and generated artifacts used by the project. Some generated files may need to be recreated locally or stored separately because of GitHub size limits.

```text
Software-Developers-Job-Satisfaction-Prediction-/
├── README.md
├── Final notebook (EDA & preprocessing).ipynb  # EDA + feature engineering
├── Modelling.ipynb                             # Training + evaluation
├── script.py                                  # Batch prediction/evaluation CLI
├── survey_results_schema.csv                  # Survey field descriptions
├── 2025_Developer_Survey_Tool.pdf             # Survey questionnaire
├── Milestone 2 Report.pdf                     # Classification results
├── CSVs/
│   └── train_data.csv                         # Prepared source data (large)
├── final_train_data.csv                       # Generated processed training data
├── final_test_data.csv                        # Generated processed test data
└── preprocessing_objects/
    ├── raw_preprocessing.pkl                  # Generated preprocessing state
    └── models.pkl                             # Generated fitted models (large)
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Hossam-Elnemr/Software-Developers-Job-Satisfaction-Prediction-.git
cd Software-Developers-Job-Satisfaction-Prediction-
```

### 2. Set up Python

Python 3 with a virtual environment is recommended. Install the packages used in the notebooks:

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn xgboost notebook
```

> Package versions have not been pinned in the supplied project files. For fully reproducible execution, add and test a version-pinned `requirements.txt` or environment lockfile.

### 3. Place the prepared dataset in the expected location

The preprocessing notebook currently defines:

```python
DATA_PATH = "train_data.csv"
SCHEMA_PATH = "survey_results_schema.csv"
```

If your dataset is stored at **`CSVs/train_data.csv`**, change the notebook's `DATA_PATH` to that location (or copy the file to the repository root). The original survey download is **not** a drop-in replacement for the project's prepared CSV unless it has the required columns and `JobSatLevel` labels.

### 4. Run the notebooks in order

```bash
jupyter notebook
```

1. Open **`Final notebook (EDA & preprocessing).ipynb`** and run its cells in order. This produces `final_train_data.csv`, `final_test_data.csv`, and `preprocessing_objects/raw_preprocessing.pkl`.
2. Open **`Modelling.ipynb`** and run its cells in order. This evaluates the models and saves `preprocessing_objects/models.pkl`.

Run the notebooks from the repository root so their relative paths resolve correctly.

## Running predictions

The provided `script.py` loads saved preprocessing objects and trained models, transforms a **raw labeled test CSV**, predicts with each model, prints model accuracy and prediction timing, and writes a combined output CSV.

```bash
python script.py --test path/to/raw_labeled_test.csv --output predictions.csv
```

By default, it expects:

```text
preprocessing_objects/raw_preprocessing.pkl
preprocessing_objects/models.pkl
```

The output retains the input columns and appends one `<model_name>_prediction` column per saved classifier. Predictions use the numeric class encoding shown above.

**Current script limitation:** The script accesses `JobSatLevel` in the test file to compute accuracy; therefore, it is **not yet a standalone unlabeled-data inference interface**. An unlabeled CSV will fail without changing the script. The test file should also follow the expected raw survey feature schema.

> Only load `.pkl` artifacts from a source you trust. Python pickle files can execute arbitrary code when deserialized.

## Large files and GitHub

Two project files exceed GitHub's normal **100 MB per-file Git limit**:

| File | Approximate size |
| --- | ---: |
| `CSVs/train_data.csv` | 108.92 MB |
| `preprocessing_objects/models.pkl` | 229.10 MB |

Choose one distribution method:

- **Git LFS:** Keep the dataset and model available through the repository using [Git Large File Storage](https://git-lfs.com/). If these files were already committed normally, merely tracking them with LFS is insufficient; the existing commit history must also be migrated or rewritten.
- **External artifacts / local regeneration:** Exclude the large files from Git with `.gitignore`, provide a download location for the prepared dataset if redistribution is permitted, and regenerate the trained models by running the notebooks.

The project cannot be fully reproduced from source code alone unless the prepared input dataset (or a documented procedure to create it) is available.

## Limitations and future work

- **Moderate predictive power:** The best reported accuracy is 53.12%, compared with a 47.14% majority-class baseline.
- **Unequal class frequencies:** Compare macro F1, balanced accuracy, and per-class errors instead of relying only on accuracy.
- **Survey selection bias:** Survey respondents are not necessarily representative of all software developers.
- **Non-causal associations:** Predictive relationships among survey features do not establish what *causes* job satisfaction.
- **Reproducibility:** Pin dependency versions, document creation of `train_data.csv` and `JobSatLevel`, and package the preprocessing artifacts consistently.
- **Inference usability:** Extend `script.py` to accept unlabeled survey data and return readable class names.
- **Potential improvements:** Broader hyperparameter searches, stronger imbalance handling, feature interactions, model calibration, and external validation.

## Data source and attribution

- **Source survey:** [Stack Overflow Developer Survey 2025](https://survey.stackoverflow.co/2025/)
- **Survey archive and downloads:** [Stack Overflow Developer Survey](https://survey.stackoverflow.co/)
- **Question definitions:** `survey_results_schema.csv` and `2025_Developer_Survey_Tool.pdf` (provided with the project)
- **Project analysis:** `Final notebook (EDA & preprocessing).ipynb`, `Modelling.ipynb`, and `Milestone 2 Report.pdf`

Stack Overflow's public survey responses are provided under the **Open Database License (ODbL)**, as described in the accompanying survey questionnaire. Preserve appropriate source attribution and review the dataset terms before redistributing adapted data. **The survey's data license does not automatically license the original code in this repository**; add a separate software license if you want to authorize reuse of the code.

---

*Academic machine learning project — classification phase, Team CS_28.*
