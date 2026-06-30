"""Predict HLM half-life with a trained CMPNN ensemble.

Point ``--checkpoint_dir`` at a directory produced by ``train.py`` (containing
``fold_*/best_model.pt``). The script:
  - auto-detects the model variant (global features / self-attention) from the
    saved checkpoint,
  - recomputes the global descriptor (if used) for the input SMILES and applies
    each fold's saved scaler,
  - averages the per-fold predictions (ensemble), and
  - reports the per-fold standard deviation as a simple uncertainty estimate.

Usage:
    python predict.py --checkpoint_dir model/ckpt_rdkit2d \\
        --smiles_path new_compounds.csv --output_path predictions.csv

Input CSV must have a ``smiles`` column; a ``value`` column, if present, is
passed through to the output. Output columns:
smiles[, value], pred, pred_std, pred_half_life_min (= 10**pred, in minutes).

If the model uses self-attention, per-atom attention weights are also written to
``<output stem>_attention.json`` for visualization.
"""

import argparse
import glob
import json
import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data_from_smiles
from chemprop.utils import load_checkpoint, load_scalers

# Reuse the global-feature generators defined in train.py.
import train


def parse_args():
    p = argparse.ArgumentParser(description='Predict HLM half-life with a CMPNN ensemble.')
    p.add_argument('--checkpoint_dir', type=str, required=True,
                   help='Directory containing fold_*/best_model.pt (a train.py output).')
    p.add_argument('--smiles_path', type=str, required=True,
                   help='Input CSV with a "smiles" column (and optional "value").')
    p.add_argument('--output_path', type=str, required=True, help='Output CSV path.')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--gpu', type=int, default=0, help='GPU index, or -1 for CPU.')
    return p.parse_args()


def discover_fold_checkpoints(checkpoint_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(checkpoint_dir, 'fold_*', 'best_model.pt')))
    if not paths:
        # Fall back to a single-model layout for convenience.
        single = os.path.join(checkpoint_dir, 'best_model.pt')
        if os.path.exists(single):
            return [single]
        raise FileNotFoundError(
            f'No fold_*/best_model.pt (or best_model.pt) found under {checkpoint_dir}.')
    return paths


def predict_batched(model, data, batch_size, target_scaler, want_attn):
    """Run inference; optionally collect per-atom attention weights."""
    model.eval()
    preds_all = []
    attn_all = [] if want_attn else None
    encoder = model.encoder.encoder if want_attn else None  # MPN -> MPNEncoder
    for i in range(0, len(data), batch_size):
        batch = MoleculeDataset(data[i:i + batch_size])
        with torch.no_grad():
            bp = model(batch.smiles(), batch.features())
        bp = bp.detach().cpu().numpy()
        if target_scaler is not None:
            bp = target_scaler.inverse_transform(bp)
        preds_all.extend(bp.reshape(-1).tolist())
        if want_attn:
            for w in encoder.attn_readout.last_attention_weights:
                attn_all.append([float(x) for x in w.tolist()])
    return np.asarray(preds_all, dtype=np.float64), attn_all


def set_features(data: MoleculeDataset, raw: np.ndarray, features_scaler) -> None:
    """Attach scaled (or raw, for fingerprints) global features to the dataset."""
    if features_scaler is None:  # binary fingerprints (morgan/maccs) -> raw 0/1
        scaled = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        scaled = np.nan_to_num(features_scaler.transform(raw), nan=0.0, posinf=0.0, neginf=0.0)
    for i, dp in enumerate(data.data):
        dp.set_features(scaled[i])


def main():
    args = parse_args()
    cuda = args.gpu is not None and args.gpu >= 0 and torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(args.gpu)

    ckpt_paths = discover_fold_checkpoints(args.checkpoint_dir)
    print(f'Found {len(ckpt_paths)} model(s) in {args.checkpoint_dir}')

    # Inspect the first checkpoint to determine the model variant.
    state0 = torch.load(ckpt_paths[0], map_location='cpu')
    use_input_features = bool(getattr(state0['args'], 'use_input_features', False))
    self_attention = bool(getattr(state0['args'], 'self_attention', False))
    print(f'Variant: use_input_features={use_input_features}, self_attention={self_attention}')

    # ---- Load input SMILES ----
    df_in = pd.read_csv(args.smiles_path)
    if 'smiles' not in df_in.columns:
        raise ValueError(f'{args.smiles_path} must contain a "smiles" column.')
    smiles_list = df_in['smiles'].tolist()
    data = get_data_from_smiles(smiles_list)
    valid_smiles = [dp.smiles for dp in data.data]
    if len(valid_smiles) != len(smiles_list):
        print(f'Warning: dropped {len(smiles_list) - len(valid_smiles)} invalid SMILES.')

    # ---- Precompute global descriptor once (only scaling differs per fold) ----
    raw_features = None
    if use_input_features:
        names_path = os.path.join(args.checkpoint_dir, 'global_features_names.json')
        with open(names_path) as f:
            meta = json.load(f)
        generator, kept_names = meta['generator'], meta['kept_names']
        print(f'Computing global features ({generator}, {len(kept_names)} dims)...')
        new_names, new_features = train.compute_global_features(generator, valid_smiles)
        kept_idx = [new_names.index(n) for n in kept_names]
        raw_features = new_features[:, kept_idx]

    # ---- Per-fold inference ----
    all_preds, all_attn = [], []
    for k, ckpt in enumerate(ckpt_paths):
        print(f'[{k + 1}/{len(ckpt_paths)}] {ckpt}')
        model = load_checkpoint(ckpt, cuda=cuda)
        target_scaler, features_scaler = load_scalers(ckpt)
        if use_input_features:
            set_features(data, raw_features, features_scaler)
        preds, attn = predict_batched(model, data, args.batch_size, target_scaler,
                                      want_attn=self_attention)
        all_preds.append(preds)
        if self_attention:
            all_attn.append(attn)

    preds_matrix = np.stack(all_preds, axis=0)             # (K, N_valid)
    pred_mean = preds_matrix.mean(axis=0)
    pred_std = preds_matrix.std(axis=0) if len(ckpt_paths) > 1 else None

    # ---- Write predictions ----
    cols = ['smiles'] + (['value'] if 'value' in df_in.columns else [])
    out = df_in[cols].copy()
    out['pred'] = out['smiles'].map(dict(zip(valid_smiles, pred_mean.tolist())))
    if pred_std is not None:
        out['pred_std'] = out['smiles'].map(dict(zip(valid_smiles, pred_std.tolist())))
    # 'pred' / 'pred_std' are in log10 space (the model's native target, matching
    # the reported metrics); 'pred_half_life_min' is the back-transformed value
    # (10**pred) in minutes for convenience.
    out['pred_half_life_min'] = out['smiles'].map(
        dict(zip(valid_smiles, np.round(10.0 ** pred_mean, 2).tolist())))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or '.', exist_ok=True)
    out.to_csv(args.output_path, index=False)
    print(f'Wrote {int(out["pred"].notna().sum())} predictions to {args.output_path}')

    # ---- Attention weights (fold-averaged) ----
    if self_attention and all_attn:
        attn_path = os.path.splitext(args.output_path)[0] + '_attention.json'
        records = []
        for i in range(len(valid_smiles)):
            per_fold = np.array([all_attn[k][i] for k in range(len(all_attn))])
            rec = {'smiles': valid_smiles[i], 'pred': float(pred_mean[i]),
                   'atom_weights': [float(x) for x in per_fold.mean(axis=0).tolist()]}
            if pred_std is not None:
                rec['pred_std'] = float(pred_std[i])
            records.append(rec)
        with open(attn_path, 'w') as f:
            json.dump(records, f)
        print(f'Wrote per-atom attention weights to {attn_path}')


if __name__ == '__main__':
    main()
