#!/usr/bin/env python
"""
shap_analysis.py - SHAP interpretation of a trained PredHLM ensemble.

Computes SHAP values for the K-fold ensemble and produces global feature-impact
figures and a per-feature summary table. This is the interpretability analysis
behind the "interpretable" model reported in the paper.

K-fold ensemble SHAP
--------------------
The ensemble prediction is the mean of the K fold models:
    f_ens(x) = mean_k f_k(x) = mean(base_k) + sum_feat mean_k(shap_k(feature))
By SHAP's additivity, the ensemble SHAP value of a feature is exactly the mean
of the per-fold SHAP values. SHAP is therefore computed for each fold model (on
that fold's scaled feature space) and averaged.

Scaling vs. interpretation
--------------------------
SHAP values are computed in the model's training space (StandardScaler-scaled),
which is required for correctness. For visualization, the *raw* (unscaled)
feature values are shown so the colour / x-axis are interpretable
(e.g. "MolLogP = 4.2" instead of a z-score).

Outputs (to --output_dir)
-------------------------
  beeswarm.png        feature impact direction & magnitude (top-N)
  bar_importance.png  mean |SHAP| ranking (top-N)
  shap_importance.csv ALL features: rank, mean_abs_shap, mean_shap (signed),
                      correlation(feature value, SHAP) -> direction of effect

Usage
-----
    python shap_analysis.py \
        --input     data/test.csv \
        --model_dir model \
        --feature   rdkit2d \
        --output_dir shap_results
"""
import argparse
import glob
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from mordred import Calculator, descriptors as mordred_descriptors

warnings.filterwarnings('ignore')
RDLogger.logger().setLevel(RDLogger.CRITICAL)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False})

_RDKIT2D_DENY = {'Ipc'}
DPI = 300


# ─────────────────────────────────────────────────────────────────────────────
# Feature generation (mirrors train.py / predict.py)
# ─────────────────────────────────────────────────────────────────────────────
def _rdkit2d_descriptors():
    seen, out = set(), []
    for name, fn in Descriptors._descList:
        if name in _RDKIT2D_DENY or name in seen:
            continue
        seen.add(name)
        out.append((name, fn))
    return out


def remove_abnormal_columns(df):
    drop = [c for c in df.columns
            if df[c].isnull().any()
            or not df[c].apply(lambda x: isinstance(x, (int, float))).all()]
    return df.drop(columns=drop), len(drop)


def build_features(smiles, feature_type, names_in=None):
    """Return (X, feature_names). For fingerprints feature_names are generic."""
    if feature_type == 'rdkit2d':
        desc = _rdkit2d_descriptors()
        rows = [[fn(Chem.MolFromSmiles(s)) for _, fn in desc]
                for s in tqdm(smiles, desc='RDKit2D', ncols=80)]
        df = pd.DataFrame(rows, columns=[n for n, _ in desc])
        df, _ = remove_abnormal_columns(df)
        if names_in is not None:
            df = df[[c for c in names_in if c in df.columns]]
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df.values, list(df.columns)

    if feature_type == 'mordred':
        calc = Calculator(mordred_descriptors, ignore_3D=True)
        df = calc.pandas([Chem.MolFromSmiles(s) for s in smiles])
        df, _ = remove_abnormal_columns(df)
        if names_in is not None:
            df = df[[c for c in names_in if c in df.columns]]
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df.values, list(df.columns)

    if feature_type == 'morgan':
        X = np.stack([np.array(AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(s), radius=2, nBits=2048))
            for s in tqdm(smiles, desc='Morgan', ncols=80)]).astype(float)
        return X, [f'Morgan_{i}' for i in range(X.shape[1])]

    if feature_type == 'maccs':
        out = np.zeros((len(smiles), 166), dtype=np.int8)
        tmp = np.empty(167, dtype=np.int8)
        for i, s in enumerate(tqdm(smiles, desc='MACCS', ncols=80)):
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s)), tmp)
            out[i] = tmp[1:]
        return out.astype(float), [f'MACCS_{i+1}' for i in range(166)]

    raise ValueError(f'Unknown feature type: {feature_type}')


# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────
def make_explainer(model, X_background):
    """TreeExplainer for tree models; KernelExplainer otherwise."""
    tree_models = ('XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor',
                   'RandomForestRegressor')
    if type(model).__name__ in tree_models:
        return shap.TreeExplainer(model)
    bg = shap.kmeans(X_background, min(50, len(X_background)))
    return shap.KernelExplainer(model.predict, bg)


def ensemble_shap(fold_dirs, X_raw):
    """Compute mean SHAP values across fold models (each on its scaled space)."""
    shap_per_fold, base_per_fold = [], []
    for k, fd in enumerate(fold_dirs):
        scaler = joblib.load(os.path.join(fd, 'scaler.pkl'))
        model  = joblib.load(os.path.join(fd, 'model.pkl'))
        Xk = scaler.transform(X_raw)
        expl = make_explainer(model, Xk)
        sv = expl.shap_values(Xk)
        if isinstance(sv, list):           # safety (multi-output)
            sv = sv[-1]
        bv = expl.expected_value
        bv = float(np.asarray(bv).flat[-1]) if isinstance(bv, (list, np.ndarray)) else float(bv)
        shap_per_fold.append(np.asarray(sv))
        base_per_fold.append(bv)
        print(f'  fold {k}: SHAP {np.asarray(sv).shape}, base {bv:+.4f}')
    shap_arr = np.stack(shap_per_fold)                       # (K, n, n_feat)
    return shap_arr.mean(axis=0), shap_arr.std(axis=0), float(np.mean(base_per_fold))


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_beeswarm(shap_values, X_disp, path, top_n):
    plt.figure(figsize=(10, max(6, top_n * 0.38)))
    shap.summary_plot(shap_values, X_disp, max_display=top_n, show=False)
    plt.title('SHAP Beeswarm - feature impact', fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  saved -> {path}')


def plot_bar(shap_values, X_disp, path, top_n):
    plt.figure(figsize=(9, max(5, top_n * 0.33)))
    shap.summary_plot(shap_values, X_disp, plot_type='bar', max_display=top_n, show=False)
    plt.title('SHAP importance - mean |SHAP|', fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f'  saved -> {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='SHAP interpretation of a PredHLM ensemble.')
    ap.add_argument('--input',     required=True, help="CSV with a 'smiles' column")
    ap.add_argument('--model_dir', required=True, help='Model dir with fold_0 .. fold_{K-1}')
    ap.add_argument('--feature',   default='rdkit2d',
                    choices=['rdkit2d', 'mordred', 'morgan', 'maccs'])
    ap.add_argument('--output_dir', default='shap_results')
    ap.add_argument('--top_n', type=int, default=20, help='Top-N features in figures')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fold_dirs = sorted(d for d in glob.glob(os.path.join(args.model_dir, 'fold_*'))
                       if os.path.isfile(os.path.join(d, 'model.pkl')))
    if not fold_dirs:
        raise FileNotFoundError(f'No fold_*/model.pkl found in {args.model_dir}')
    print(f'Ensemble: {len(fold_dirs)} folds')

    smiles = pd.read_csv(args.input)['smiles'].astype(str).tolist()

    names_in = None
    if args.feature in ('rdkit2d', 'mordred'):
        name_file = os.path.join(args.model_dir, f'{args.feature}_names.csv')
        if os.path.isfile(name_file):
            names_in = pd.read_csv(name_file).iloc[:, 0].tolist()

    print('\n[1/3] Building features ...')
    X_raw, feat_names = build_features(smiles, args.feature, names_in=names_in)
    X_disp = pd.DataFrame(X_raw, columns=feat_names)   # raw values for display
    print(f'  feature matrix: {X_raw.shape}')

    print('\n[2/3] Computing SHAP per fold ...')
    shap_values, shap_std, base_value = ensemble_shap(fold_dirs, X_raw)
    print(f'  ensemble base value: {base_value:+.4f}')

    print('\n[3/3] Writing outputs ...')
    # Per-feature summary: magnitude + direction
    mean_abs  = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    n_feat = shap_values.shape[1]
    pear = np.full(n_feat, np.nan)
    spear = np.full(n_feat, np.nan)
    for j in range(n_feat):
        fv, sv = X_raw[:, j], shap_values[:, j]
        if np.std(fv) > 0 and np.std(sv) > 0:
            pear[j]  = pearsonr(fv, sv)[0]
            spear[j] = spearmanr(fv, sv)[0]

    imp = pd.DataFrame({
        'feature': feat_names,
        'mean_abs_shap': mean_abs,
        'mean_shap': mean_shap,
        'corr_value_shap_pearson': pear,
        'corr_value_shap_spearman': spear,
        'fold_std_mean_abs_shap': np.abs(shap_std).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    imp['rank'] = np.arange(1, len(imp) + 1)
    imp['direction'] = np.where(imp['corr_value_shap_spearman'] > 0.05, 'positive',
                          np.where(imp['corr_value_shap_spearman'] < -0.05, 'negative', 'mixed/weak'))
    imp.to_csv(os.path.join(args.output_dir, 'shap_importance.csv'), index=False)
    print(f'  saved -> {os.path.join(args.output_dir, "shap_importance.csv")} '
          f'({len(imp)} features)')

    plot_beeswarm(shap_values, X_disp, os.path.join(args.output_dir, 'beeswarm.png'), args.top_n)
    plot_bar(shap_values, X_disp, os.path.join(args.output_dir, 'bar_importance.png'), args.top_n)

    print('\nTop-10 features by mean |SHAP|:')
    print(imp[['rank', 'feature', 'mean_abs_shap', 'corr_value_shap_spearman',
               'direction']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
