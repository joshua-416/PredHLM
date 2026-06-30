# CMPNN for HLM Half-life Regression

**CMPNN** is the graph neural network baseline of the **PredHLM** study for
quantitative prediction of metabolic half-life (log<sub>10</sub> t<sub>1/2</sub>,
minutes) in **human liver microsomes (HLM)**. It is a Communicative Message
Passing Neural Network (CMPNN) that learns directly from the molecular graph,
optionally augmented with a self-attention readout and molecule-level descriptor
features, and ensembled across 5 cross-validation folds.

This folder is a **sub-module** of the main PredHLM repository. In the paper,
CMPNN is reported as a secondary (graph neural network) baseline; the
best-performing model overall is the descriptor-based model (RDKit2D + XGBoost)
in the parent repository. The code here trains and evaluates four CMPNN schemes:

| Scheme         | Self-attention readout | Global feature (RDKit 2D) |
| :------------- | :--------------------: | :-----------------------: |
| `baseline`     |           –            |             –             |
| `attn`         |           ✓            |             –             |
| `rdkit2d`      |           –            |             ✓             |
| `attn_rdkit2d` |           ✓            |             ✓             |

The code additionally supports `mordred`, `morgan`, and `maccs` global features.

---

## Developer

**Jidon Jang** (jdjang@krict.re.kr)

---

## Publication

> Jidon Jang, Nam-Chul Cho, Kwang-Seok Oh, **"PredHLM: an interpretable machine-learning model for quantitative half-life prediction in human liver microsomes"** (in preparation)

Please cite this paper if you use the code or model.

---

## Prerequisites

The model was developed and tested with **Python 3.10** and the following
packages (a CUDA-capable GPU is recommended; CPU works with `--gpu -1`):

| Package          | Version          |
|------------------|------------------|
| pytorch          | 2.5.1 (CUDA 12.1)|
| numpy            | 2.2.6            |
| pandas           | 2.3.3            |
| scikit-learn     | 1.7.2            |
| rdkit            | 2026.3.2         |
| mordredcommunity | 2.0.7            |
| networkx         | 3.4.2            |
| optuna           | 4.8.0            |
| tqdm             | 4.67.3           |

Create the environment with conda:

```bash
conda env create -f environment.yaml
conda activate cmpnn-hlm
```

> Install the PyTorch build matching your CUDA toolkit (see https://pytorch.org).

---

## Repository structure

```
PredHLM_main/
├── data/                   # Shared dataset (used by both the main model and CMPNN)
│   ├── training.csv        # Training set  (columns: smiles, value)
│   └── test.csv            # Independent test set (columns: smiles, value)
└── CMPNN/
    ├── train.py            # Train a 5-fold CMPNN ensemble (+ optional Optuna)
    ├── predict.py          # Predict with a trained ensemble
    ├── environment.yaml    # Conda environment specification
    ├── chemprop/           # CMPNN/DMPNN library (modified; see Acknowledgements)
    └── model/              # Released checkpoints: 4 schemes x 5 folds
        ├── ckpt_baseline/
        ├── ckpt_attn/
        ├── ckpt_rdkit2d/
        └── ckpt_attn_rdkit2d/
```

The dataset lives in the parent repository's `data/` folder and is **shared**
with the main descriptor-based model, so the CMPNN and main models are trained
and evaluated on identical compounds. Both CSVs have two columns: `smiles` and
`value` (the log<sub>10</sub> half-life in minutes). The commands below are run
from inside the `CMPNN/` directory and reference the data as `../data/`. Each
`model/ckpt_*` directory contains
the five fold checkpoints (`fold_{0..4}/best_model.pt`), the tuned
hyperparameters (`optuna/best_params.json`), the descriptor metadata
(`global_features_names.json`, for feature schemes), and the recorded metrics
(`final_metrics.json`).

---

## Usage

### 1. Prediction (using the provided model)

Predict half-life for a CSV containing a `smiles` column. The model variant
(global feature / attention) is detected automatically from the checkpoint.

```bash
# RDKit-2D scheme (the best CMPNN variant here)
python predict.py \
    --checkpoint_dir model/ckpt_rdkit2d \
    --smiles_path    my_compounds.csv \
    --output_path    predictions.csv
```

**Output** (`predictions.csv`):

| column               | description                                                       |
|----------------------|-------------------------------------------------------------------|
| `smiles`             | input SMILES                                                      |
| `value`              | true log<sub>10</sub> t<sub>1/2</sub> (only if present in the input) |
| `pred`               | predicted log<sub>10</sub> t<sub>1/2</sub> (ensemble mean)        |
| `pred_std`           | per-fold standard deviation in log<sub>10</sub> space (uncertainty) |
| `pred_half_life_min` | predicted half-life in minutes (10<sup>`pred`</sup>)             |

For the attention schemes, per-atom attention weights are additionally written
to `predictions_attention.json`.

### 2. Training (reproducing or retraining a model)

Reproduce a specific released scheme from its saved hyperparameters (no tuning):

```bash
# rdkit2d scheme
python train.py \
    --train            ../data/training.csv \
    --test             ../data/test.csv \
    --global_features  rdkit2d \
    --best_params_path model/ckpt_rdkit2d/optuna/best_params.json \
    --save_dir         runs/rdkit2d
```

Train from scratch with Optuna (TPE) hyperparameter search:

```bash
# baseline (no feature, no attention)
python train.py --train ../data/training.csv --test ../data/test.csv --tune --save_dir runs/baseline

# attention only
python train.py --train ../data/training.csv --test ../data/test.csv --tune --self_attention \
    --save_dir runs/attn

# global feature only (rdkit2d | mordred | morgan | maccs)
python train.py --train ../data/training.csv --test ../data/test.csv --tune --global_features rdkit2d \
    --save_dir runs/rdkit2d

# attention + feature
python train.py --train ../data/training.csv --test ../data/test.csv --tune --self_attention \
    --global_features rdkit2d --save_dir runs/attn_rdkit2d
```

**Options**

| argument                  | choices / default                                |
|---------------------------|--------------------------------------------------|
| `--num_folds`             | number of CV folds (default 5)                   |
| `--split_seed`            | seed for the shuffled KFold split of the training set (default 1234) |
| `--tune` / `--n_trials`   | enable Optuna tuning / number of trials (off / 40) |
| `--epochs` / `--patience` | training budget / early-stopping patience (150 / 20) |
| `--global_features`       | `rdkit2d`, `mordred`, `morgan`, `maccs` (default none) |
| `--self_attention`        | use the self-attention readout (default off)     |
| `--gpu`                   | GPU index (`-1` for CPU, default 0)              |

Training writes per-fold checkpoints, split CSVs, `test_results.csv`
(`smiles, value, pred, pred_std`), and `final_metrics.json` to `--save_dir`.

---

## Method summary

- **Task** — regression of log<sub>10</sub> half-life (minutes) in HLM.
- **Encoder** — CMPNN (Song et al., 2020), a communicative message-passing
  network over the molecular graph with a BatchGRU aggregation.
- **Readout** — mean pooling, or an optional additive **self-attention** readout
  that assigns a learned weight to each atom.
- **Global features (optional)** — a molecule-level descriptor vector is
  concatenated to the graph embedding before the feed-forward head. Continuous
  descriptors (`rdkit2d`, `mordred`) are standardized on the training split;
  binary fingerprints (`morgan`, `maccs`) are kept as raw 0/1. For `rdkit2d`,
  the numerically unstable `Ipc` descriptor is excluded and any descriptor
  column that is NaN/Inf for any molecule is dropped.
- **Model selection** — Optuna (TPE), where each trial is scored by the mean
  validation MSE across the 5 folds.
- **Final model** — the provided training set is split into 5 folds (seed 1234);
  one model is trained per fold with validation-based early stopping, and
  predictions on the independent test set are averaged across the 5 folds (the
  per-fold spread is reported as `pred_std`).

---

## Acknowledgements and Citation

This code is built on the original **CMPNN** implementation and the
**chemprop / DMPNN** codebase. Please cite the original works if you use this
code:

```bibtex
@inproceedings{song2020cmpnn,
  title     = {Communicative Representation Learning on Attributed Molecular Graphs},
  author    = {Song, Ying and Zheng, Shuangjia and Niu, Zhangming and Fu, Zhang-Hua
               and Lu, Yutong and Yang, Yuedong},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence (IJCAI-20)},
  pages     = {2831--2838},
  year      = {2020},
  doi       = {10.24963/ijcai.2020/392}
}

@article{yang2019dmpnn,
  title   = {Analyzing Learned Molecular Representations for Property Prediction},
  author  = {Yang, Kevin and Swanson, Kyle and Jin, Wengong and Coley, Connor and
             Eiden, Philipp and Gao, Hua and Guzman-Perez, Angel and Hopper, Timothy
             and Kelley, Brian and Mathea, Miriam and others},
  journal = {Journal of Chemical Information and Modeling},
  volume  = {59},
  number  = {8},
  pages   = {3370--3388},
  year    = {2019},
  doi     = {10.1021/acs.jcim.9b00237}
}
```

- Original CMPNN repository: https://github.com/SY575/CMPNN
- chemprop: https://github.com/chemprop/chemprop

The `chemprop/` directory in this folder is derived from the above projects and
modified to add (i) the post-readout global-feature concatenation, (ii) the
optional self-attention readout, and (iii) minor packaging fixes. All original
licenses and credit belong to their respective authors.

---

## License

This project is released under the [MIT License](../LICENSE), the same license
as the parent PredHLM repository. The bundled `chemprop/` code retains the
licenses of its original authors (see Acknowledgements).
