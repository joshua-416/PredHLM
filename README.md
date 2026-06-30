# PredHLM

**PredHLM** is an interpretable machine-learning model for quantitative prediction of metabolic half-life (log<sub>10</sub> t<sub>1/2</sub>, minutes) in **human liver microsomes (HLM)**. The final model is a 5-fold ensemble of gradient-boosted decision trees (XGBoost) trained on RDKit 2D descriptors, and each prediction is accompanied by an applicability domain score so that out-of-domain compounds can be flagged.

---

## Developer

**Jidon Jang** (jdjang@krict.re.kr)

---

## Publication

> Jidon Jang, Nam-Chul Cho, Kwang-Seok Oh, **"PredHLM: an interpretable machine-learning model for quantitative half-life prediction in human liver microsomes"** (in preparation)

Please cite this paper if you use the code or model.

---

## Prerequisites

The model was developed and tested with **Python 3.9.18** and the following
packages:

| Package            | Version  |
|--------------------|----------|
| numpy              | 1.24.4   |
| pandas             | 1.4.2    |
| scipy              | 1.8.1    |
| scikit-learn       | 1.6.1    |
| xgboost            | 2.1.4    |
| lightgbm           | 4.6.0    |
| catboost           | 1.0.4    |
| optuna             | 4.8.0    |
| rdkit              | 2023.9.2 |
| mordredcommunity   | 2.0.7    |
| shap               | 0.49.1   |
| matplotlib         | 3.8.0    |
| joblib             | 1.5.1    |
| tqdm               | 4.66.1   |

Create the environment with conda:

```bash
conda env create -f environment.yaml
conda activate predhlm
```

---

## Repository structure

```
PredHLM_main/
├── train.py            # Train a K-fold ensemble regressor
├── predict.py          # Predict half-life + applicability domain (SDC) score
├── environment.yaml    # Conda environment specification
├── LICENSE             # MIT license
├── data/
│   ├── training.csv    # Training set  (columns: smiles, value)
│   └── test.csv        # Independent test set (columns: smiles, value)
├── model/              # Final model: 5-fold XGBoost ensemble (RDKit2D)
│   ├── fold_0/ … fold_4/   # model.pkl + scaler.pkl per fold
│   └── rdkit2d_names.csv    # descriptor columns used by the model
└── CMPNN/              # Secondary GNN baseline (CMPNN); shares data/ (see CMPNN/README.md)
```

`value` is the **log<sub>10</sub> half-life in minutes**.

The `CMPNN/` subfolder contains the graph neural network (CMPNN) baseline
reported as a secondary model in the paper. It is trained and evaluated on the
**same** `data/training.csv` and `data/test.csv` split used here; see
[CMPNN/README.md](CMPNN/README.md) for details.

---

## Usage

### 1. Prediction (using the provided model)

Predict half-life for a CSV containing a `smiles` column. The training set is
used as the reference for the applicability domain (SDC) score.

```bash
python predict.py \
    --input     data/test.csv \
    --model_dir model \
    --train     data/training.csv \
    --feature   rdkit2d \
    --output    predictions.csv
```

**Output** (`predictions.csv`):

| column               | description                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------------|
| `smiles`             | input SMILES                                                                                      |
| `value`              | true log<sub>10</sub> t<sub>1/2</sub> (only if present in the input)                              |
| `pred`               | predicted log<sub>10</sub> t<sub>1/2</sub> (ensemble mean)                                        |
| `pred_std`           | ensemble standard deviation in log<sub>10</sub> space (prediction uncertainty)                   |
| `pred_half_life_min` | predicted half-life in minutes (10<sup>`pred`</sup>)                                              |
| `sdc`                | applicability domain score (higher = better supported by structurally similar training compounds) |

The `sdc` column is the **raw** applicability domain score; no fixed
in/out-of-domain cutoff is imposed, so users can apply a threshold appropriate
for their use case. If the input contains a `value` column, regression metrics
(RMSE, MAE, R<sup>2</sup>) are printed.

### 2. Training (reproducing or retraining a model)

Train a K-fold ensemble from scratch. Hyperparameters are optimized with Optuna
(TPE sampler) using K-fold cross-validation; the independent test set is used only
for final evaluation.

```bash
python train.py \
    --train     data/training.csv \
    --test      data/test.csv \
    --feature   rdkit2d \
    --model     xgboost \
    --n_trials  100 \
    --n_folds   5 \
    --output_dir results_rdkit2d_xgboost
```

**Options**

| argument      | choices / default                                                       |
|---------------|-------------------------------------------------------------------------|
| `--feature`   | `rdkit2d` (default), `mordred`, `morgan`, `maccs`                       |
| `--model`     | `xgboost` (default), `catboost`, `lightgbm`, `random_forest`, `knn`, `svm`, `krr`, `ann` |
| `--n_trials`  | number of Optuna trials (default 100)                                  |
| `--n_folds`   | number of CV folds (default 5)                                         |
| `--output_dir`| output directory                                                       |

**Outputs**

```
output_dir/
├── model/
│   ├── fold_0/ … fold_{K-1}/   # model.pkl + scaler.pkl per fold
│   └── {feature}_names.csv      # descriptor columns (rdkit2d / mordred only)
└── results/
    ├── cv_results.csv           # per-fold mean ± std + ensemble metrics
    ├── test_results.csv         # per-molecule test predictions (+ per-fold)
    ├── oof_predictions.csv       # out-of-fold CV predictions on the training set
    └── optuna_trials.csv         # full Optuna trial log
```

The eight models (XGBoost, CatBoost, LightGBM, Random Forest, SVM, KRR, k-NN,
MLP) and four molecular representations (RDKit2D, Mordred, Morgan/ECFP4, MACCS)
benchmarked in the paper can all be reproduced by varying `--model` and
`--feature`. The best configuration reported in the paper is
**RDKit2D + XGBoost**, which is the model shipped in `model/`.

---

### 3. Interpretation (SHAP)

Compute SHAP values for the ensemble and produce the feature-impact figures and
the per-feature importance table used in the paper.

```bash
python shap_analysis.py \
    --input     data/test.csv \
    --model_dir model \
    --feature   rdkit2d \
    --output_dir shap_results
```

**Outputs** (`shap_results/`)

| file                  | description                                            |
|-----------------------|--------------------------------------------------------|
| `beeswarm.png`        | feature impact direction & magnitude (top-N)           |
| `bar_importance.png`  | mean \|SHAP\| ranking (top-N)                          |
| `shap_importance.csv` | **all** features: rank, mean \|SHAP\| (magnitude), mean SHAP (signed), and correlation between feature value and SHAP value (direction: positive / negative) |

SHAP values are computed in the model's scaled space (for correctness) and
averaged across the K fold models (SHAP additivity); figures display the raw,
interpretable feature values.

---

## Method summary

- **Task** — regression of log<sub>10</sub> half-life (minutes) in HLM.
- **Features** — RDKit 2D descriptors (default), Mordred, Morgan/ECFP4, or MACCS.
  Binary fingerprints are not standardized; continuous descriptors are
  standardized with a `StandardScaler` fitted only on the training fold.
- **Model selection** — Optuna (TPE) with 5-fold cross-validation; the
  hyperparameters minimizing the mean CV error are selected.
- **Final model** — an ensemble of 5 models (one per CV fold). The prediction is
  the mean of the 5 models and the standard deviation is reported as an
  uncertainty estimate.
- **Applicability domain** — Sum of Distance-weighted Contributions (SDC) using
  ECFP4 fingerprints; quantifies how well a query molecule is supported by
  structurally similar training compounds.

---

## License

This project is released under the [MIT License](LICENSE).
