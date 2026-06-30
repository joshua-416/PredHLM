#!/usr/bin/env python
"""
predict.py - Predict HLM half-life with a trained PredHLM ensemble + applicability domain.

For each input molecule this script:
  1. Recreates the molecular features used during training.
  2. Predicts log10 half-life with the K-fold ensemble (mean of K models),
     reporting the inter-model standard deviation as an uncertainty estimate.
  3. Computes an applicability-domain score (SDC) against the training set.
     The raw score is reported; no fixed in/out-of-domain cutoff is imposed
     (users may apply a threshold appropriate for their use case).

Applicability domain - Sum of Distance-weighted Contributions (SDC)
-------------------------------------------------------------------
    SDC = sum_i exp( -3 * TD_i / (1 - TD_i) ),   TD_i = 1 - Tanimoto(query, train_i)

using ECFP4 fingerprints (Morgan radius=2, 2048 bits). A higher SDC means the
query molecule is better supported by structurally similar training compounds.
(Liu & Wallqvist, J. Chem. Inf. Model. 2019.)

Usage
-----
    python predict.py \
        --input   data/test.csv \
        --model_dir model \
        --train   data/training.csv \
        --feature rdkit2d \
        --output  predictions.csv

The input CSV must contain a 'smiles' column. If it also contains 'value'
(log10 half-life), regression metrics are printed.
"""
import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from mordred import Calculator, descriptors as mordred_descriptors

RDLogger.logger().setLevel(RDLogger.CRITICAL)

_RDKIT2D_DENY = {'Ipc'}                 # must match train.py
_FINGERPRINT_TYPES = {'morgan', 'maccs'}


# ─────────────────────────────────────────────────────────────────────────────
# Feature generation (mirrors train.py)
# ─────────────────────────────────────────────────────────────────────────────
def _rdkit2d_descriptors():
    """RDKit 2D descriptor (name, fn) pairs, de-duplicated and Ipc-excluded.
    (RDKit's descriptor list contains one duplicate name, 'SPS'.)"""
    seen, out = set(), []
    for name, fn in Descriptors._descList:
        if name in _RDKIT2D_DENY or name in seen:
            continue
        seen.add(name)
        out.append((name, fn))
    return out


def remove_abnormal_columns(df):
    """Drop columns with NaN or non-numeric values."""
    drop = [c for c in df.columns
            if df[c].isnull().any()
            or not df[c].apply(lambda x: isinstance(x, (int, float))).all()]
    return df.drop(columns=drop), len(drop)


def build_features(smiles, feature_type, names_in=None):
    """Compute the feature matrix for inference. names_in selects/orders the
    descriptor columns saved at training time (rdkit2d / mordred)."""
    if feature_type == 'rdkit2d':
        desc = _rdkit2d_descriptors()
        names = [n for n, _ in desc]
        rows = [[fn(Chem.MolFromSmiles(s)) for _, fn in desc]
                for s in tqdm(smiles, desc='RDKit2D', ncols=80)]
        df = pd.DataFrame(rows, columns=names)
        df, _ = remove_abnormal_columns(df)
        if names_in is not None:
            missing = [c for c in names_in if c not in df.columns]
            if missing:
                print(f'  WARNING: {len(missing)} descriptor(s) missing in input: {missing[:5]}')
            df = df[[c for c in names_in if c in df.columns]]
        return df.replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    if feature_type == 'mordred':
        calc = Calculator(mordred_descriptors, ignore_3D=True)
        df = calc.pandas([Chem.MolFromSmiles(s) for s in smiles])
        df, _ = remove_abnormal_columns(df)
        if names_in is not None:
            df = df[[c for c in names_in if c in df.columns]]
        return df.replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    if feature_type == 'morgan':
        return np.stack([np.array(AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(s), radius=2, nBits=2048))
            for s in tqdm(smiles, desc='Morgan', ncols=80)]).astype(float)

    if feature_type == 'maccs':
        out = np.zeros((len(smiles), 166), dtype=np.int8)
        tmp = np.empty(167, dtype=np.int8)
        for i, s in enumerate(tqdm(smiles, desc='MACCS', ncols=80)):
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s)), tmp)
            out[i] = tmp[1:]
        return out.astype(float)

    raise ValueError(f'Unknown feature type: {feature_type}')


# ─────────────────────────────────────────────────────────────────────────────
# Applicability domain (SDC)
# ─────────────────────────────────────────────────────────────────────────────
def _ecfp4(smiles, nbits=2048):
    mol = Chem.MolFromSmiles(str(smiles))
    return None if mol is None else AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)


def compute_sdc(query_smiles, train_smiles, nbits=2048):
    """SDC applicability-domain score for each query molecule vs the training set."""
    train_fps = [fp for fp in (_ecfp4(s, nbits) for s in train_smiles) if fp is not None]
    sdc = np.full(len(query_smiles), np.nan)
    for i, s in enumerate(tqdm(query_smiles, desc='SDC', ncols=80)):
        fp = _ecfp4(s, nbits)
        if fp is None:
            continue
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps), dtype=np.float64)
        td = 1.0 - sims
        # exp(-3 TD / (1 - TD)); TD = 1 (no shared bits) -> contribution 0
        with np.errstate(divide='ignore', invalid='ignore'):
            expo = np.where(td < 1.0, -3.0 * td / (1.0 - td), -np.inf)
        sdc[i] = float(np.sum(np.exp(expo)))
    return sdc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description='Predict HLM half-life (K-fold ensemble) with SDC applicability domain.')
    ap.add_argument('--input',     required=True, help="Input CSV (must contain 'smiles')")
    ap.add_argument('--model_dir', required=True, help='Model dir containing fold_0 .. fold_{K-1}')
    ap.add_argument('--train',     required=True, help='Training CSV (reference for SDC)')
    ap.add_argument('--feature',   default='rdkit2d',
                    choices=['rdkit2d', 'mordred', 'morgan', 'maccs'])
    ap.add_argument('--output',    default=None, help='Output CSV (default: <input>_pred.csv)')
    args = ap.parse_args()

    out_path = args.output or os.path.splitext(args.input)[0] + '_pred.csv'

    # ── Locate fold models ───────────────────────────────────────────────────
    fold_dirs = sorted(d for d in glob.glob(os.path.join(args.model_dir, 'fold_*'))
                       if os.path.isfile(os.path.join(d, 'model.pkl')))
    if not fold_dirs:
        raise FileNotFoundError(f'No fold_*/model.pkl found in {args.model_dir}')
    print(f'Loaded {len(fold_dirs)}-fold ensemble from {args.model_dir}')

    # ── Load input ───────────────────────────────────────────────────────────
    df_in = pd.read_csv(args.input)
    smiles = df_in['smiles'].astype(str).tolist()
    has_label = 'value' in df_in.columns

    # ── Descriptor name list (rdkit2d / mordred) ─────────────────────────────
    names_in = None
    if args.feature in ('rdkit2d', 'mordred'):
        name_file = os.path.join(args.model_dir, f'{args.feature}_names.csv')
        if os.path.isfile(name_file):
            names_in = pd.read_csv(name_file).iloc[:, 0].tolist()
            print(f'  Descriptor names: {len(names_in)} columns ({name_file})')
        else:
            print(f'  WARNING: {name_file} not found - column alignment may differ.')

    # ── Features ─────────────────────────────────────────────────────────────
    print('\n[1/3] Building features ...')
    X = build_features(smiles, args.feature, names_in=names_in)
    print(f'  feature matrix: {X.shape}')

    # ── Ensemble prediction ──────────────────────────────────────────────────
    print('[2/3] Ensemble prediction ...')
    preds = []
    for fd in fold_dirs:
        scaler = joblib.load(os.path.join(fd, 'scaler.pkl'))
        model  = joblib.load(os.path.join(fd, 'model.pkl'))
        preds.append(model.predict(scaler.transform(X)))
    preds = np.stack(preds)
    pred_mean, pred_std = preds.mean(axis=0), preds.std(axis=0)

    # ── Applicability domain (SDC) ───────────────────────────────────────────
    print('[3/3] Applicability domain (SDC) ...')
    train_smiles = pd.read_csv(args.train)['smiles'].astype(str).tolist()
    sdc = compute_sdc(smiles, train_smiles)

    # ── Assemble output ──────────────────────────────────────────────────────
    # 'pred' / 'pred_std' are in log10 space (the model's native target, matching
    # the reported metrics); 'pred_half_life_min' is the back-transformed value
    # (10**pred) in minutes for convenience. 'sdc' is the raw applicability-domain
    # score (higher = better supported by structurally similar training
    # compounds); no fixed in/out-of-domain cutoff is imposed.
    out = pd.DataFrame({
        'smiles':             df_in['smiles'],
        'pred':               np.round(pred_mean, 4),       # log10 half-life
        'pred_std':           np.round(pred_std, 4),        # ensemble uncertainty (log10)
        'pred_half_life_min': np.round(10 ** pred_mean, 2), # half-life in minutes
        'sdc':                np.round(sdc, 4),
    })
    if has_label:
        out.insert(1, 'value', df_in['value'].astype(float).values)
    out.to_csv(out_path, index=False)
    print(f'\nSaved -> {out_path}  ({len(out)} molecules)')

    # ── Optional metrics ─────────────────────────────────────────────────────
    if has_label:
        y, p = out['value'].values, out['pred'].values
        rmse = np.sqrt(mean_squared_error(y, p))
        print(f'\n  Test metrics (all molecules):')
        print(f'    RMSE {rmse:.4f}   MAE {mean_absolute_error(y, p):.4f}'
              f'   R2 {r2_score(y, p):.4f}')

    # SDC summary (no fixed cutoff; users decide an applicability threshold)
    valid = ~np.isnan(sdc)
    if valid.any():
        s = sdc[valid]
        print(f'\n  SDC (applicability-domain score): '
              f'median {np.median(s):.3f}   '
              f'range [{s.min():.3f}, {s.max():.3f}]')


if __name__ == '__main__':
    main()
