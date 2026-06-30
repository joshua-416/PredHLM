"""Train a CMPNN ensemble for HLM (human liver microsome) half-life regression.

This script trains a 5-fold cross-validation ensemble of CMPNN models and
evaluates the ensemble on an independent test set. It supports two optional
extensions used in our benchmark:

  --self_attention        : replace the default mean readout with an additive
                            self-attention readout (per-atom weights).
  --global_features NAME  : concatenate a global molecular descriptor vector to
                            the molecule embedding before the FFN head.
                            NAME in {rdkit2d, mordred, morgan, maccs}.

Hyperparameters can be tuned with Optuna (--tune), where each trial is scored
by the mean validation MSE across the folds, or loaded from a JSON file
(--best_params_path) to reproduce a specific model.

Workflow
--------
1. The training set (--train) and the independent test set (--test) are provided as
   separate CSVs (the same convention as the main repository).
2. The training set is split into K folds (shuffled KFold, seed 1234).
3. (Optional) Optuna tunes hyperparameters on the K-fold validation MSE.
4. K models are trained (one per fold) with the chosen hyperparameters, each
   with validation-based early stopping.
5. Test predictions are averaged across the K models (ensemble); the per-model
   spread (std) is reported as a simple uncertainty estimate.

Outputs (under --save_dir)
--------------------------
  fold_{k}/best_model.pt        per-fold checkpoint
  training_fold_{k}.csv         per-fold train SMILES/value
  valid_fold_{k}.csv            per-fold validation SMILES/value
  test.csv                      independent test SMILES/value
  test_results.csv              smiles, value, pred (ensemble mean), pred_std
  global_features_names.json    descriptor metadata (if --global_features)
  final_metrics.json            per-fold (train/val/test) mean +/- std + ensemble
"""

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
from argparse import Namespace
from typing import Dict, List, Tuple

import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

import optuna
from optuna.samplers import TPESampler

from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.data.utils import get_data
from chemprop.models import build_model
from chemprop.nn_utils import NoamLR, param_count
from chemprop.train.predict import predict
from chemprop.utils import (build_optimizer, build_lr_scheduler, get_loss_func,
                            load_checkpoint, load_scalers, makedirs, save_checkpoint)


# =========================================================================== #
# Global molecular feature generators
# =========================================================================== #
# Registry maps a name to (generator function, needs_scaling). Continuous
# descriptors are standardized on the train split; binary fingerprints are kept
# as raw 0/1. Add new generators by registering them here.
GLOBAL_FEATURE_REGISTRY: Dict[str, callable] = {}
GLOBAL_FEATURE_NEEDS_SCALING: Dict[str, bool] = {}


def register_global_feature(name: str, needs_scaling: bool = True):
    def deco(fn):
        GLOBAL_FEATURE_REGISTRY[name] = fn
        GLOBAL_FEATURE_NEEDS_SCALING[name] = needs_scaling
        return fn
    return deco


def feature_needs_scaling(name: str) -> bool:
    return GLOBAL_FEATURE_NEEDS_SCALING.get(name, True)


# RDKit descriptors that are numerically unstable for some molecules (Ipc grows
# roughly exponentially with molecular complexity, reaching 1e30+); excluded so
# that standardization and out-of-distribution prediction stay well-behaved.
RDKIT2D_DENYLIST = {'Ipc'}


@register_global_feature('rdkit2d', needs_scaling=True)
def rdkit2d_features(smiles_list: List[str]):
    """RDKit 2D descriptors (Descriptors._descList) minus the denylist.

    NaN/Inf are left as-is; the caller drops offending columns before training.
    """
    desc_list = [(n, f) for n, f in Descriptors._descList if n not in RDKIT2D_DENYLIST]
    names = [n for n, _ in desc_list]
    funcs = [f for _, f in desc_list]
    out = np.full((len(smiles_list), len(funcs)), np.nan, dtype=np.float64)
    for i, smi in enumerate(tqdm(smiles_list, desc='rdkit2d', total=len(smiles_list))):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for j, f in enumerate(funcs):
            try:
                v = f(mol)
                out[i, j] = float(v) if v is not None else np.nan
            except Exception:
                out[i, j] = np.nan
    return names, out


@register_global_feature('mordred', needs_scaling=True)
def mordred_features(smiles_list: List[str]):
    """Mordred 2D descriptors (~1600). Continuous -> scaled like rdkit2d.

    Failed/missing descriptors are coerced to NaN so the offending columns are
    dropped before training.
    """
    from mordred import Calculator, descriptors as mordred_descriptors
    calc = Calculator(mordred_descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    print(f'Computing mordred descriptors for {len(mols)} molecules...')
    df = calc.pandas(mols, nproc=1, quiet=False)
    names = [str(c) for c in df.columns]
    values = df.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    return names, values


@register_global_feature('morgan', needs_scaling=False)
def morgan_features(smiles_list: List[str], radius: int = 2, n_bits: int = 2048):
    """Binary Morgan/ECFP fingerprint. Kept as raw 0/1 (no scaling)."""
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    names = [f'morgan_{i}' for i in range(n_bits)]
    out = np.zeros((len(smiles_list), n_bits), dtype=np.float64)
    for i, smi in enumerate(tqdm(smiles_list, desc='morgan', total=len(smiles_list))):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float64)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr
    return names, out


@register_global_feature('maccs', needs_scaling=False)
def maccs_features(smiles_list: List[str]):
    """MACCS keys (167 bits). Kept as raw 0/1 (no scaling)."""
    from rdkit.Chem import MACCSkeys
    from rdkit import DataStructs
    n_bits = 167
    names = [f'maccs_{i}' for i in range(n_bits)]
    out = np.zeros((len(smiles_list), n_bits), dtype=np.float64)
    for i, smi in enumerate(tqdm(smiles_list, desc='maccs', total=len(smiles_list))):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((n_bits,), dtype=np.float64)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr
    return names, out


def compute_global_features(name: str, smiles_list: List[str]):
    if name not in GLOBAL_FEATURE_REGISTRY:
        raise ValueError(f'Unknown global features generator "{name}". '
                         f'Available: {sorted(GLOBAL_FEATURE_REGISTRY.keys())}')
    return GLOBAL_FEATURE_REGISTRY[name](smiles_list)


def drop_invalid_feature_columns(feat_matrix: np.ndarray, names: List[str]):
    """Drop any column containing NaN or +/-Inf in any row.

    Returns (cleaned_matrix, kept_names, dropped_names).
    """
    keep_mask = np.isfinite(feat_matrix).all(axis=0)
    kept_names = [n for n, k in zip(names, keep_mask) if k]
    dropped_names = [n for n, k in zip(names, keep_mask) if not k]
    return feat_matrix[:, keep_mask], kept_names, dropped_names


# =========================================================================== #
# Data splitting
# =========================================================================== #
def make_cv_folds(pool_idx: List[int], k: int, seed: int
                  ) -> List[Tuple[List[int], List[int]]]:
    """Split pool_idx into k (train_idx, val_idx) pairs via shuffled KFold."""
    pool = np.asarray(pool_idx)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [(pool[tr].tolist(), pool[va].tolist()) for tr, va in kf.split(pool)]


def slice_dataset(data: MoleculeDataset, indices: List[int]) -> MoleculeDataset:
    return MoleculeDataset([data[i] for i in indices])


# =========================================================================== #
# Metrics & logging
# =========================================================================== #
def compute_all_metrics(targets, preds) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    if not np.isfinite(preds).all():
        # Model diverged; report finite "bad" metrics so Optuna records the
        # trial cleanly instead of crashing.
        return {'MSE': float('inf'), 'RMSE': float('inf'),
                'MAE': float('inf'), 'R2': float('-inf')}
    mse = float(mean_squared_error(targets, preds))
    return {'MSE': mse, 'RMSE': float(math.sqrt(mse)),
            'MAE': float(mean_absolute_error(targets, preds)),
            'R2': float(r2_score(targets, preds))}


def fmt_metrics(m: Dict[str, float]) -> str:
    return (f"MSE={m['MSE']:.4f} RMSE={m['RMSE']:.4f} "
            f"MAE={m['MAE']:.4f} R2={m['R2']:.4f}")


def create_logger(save_dir: str) -> logging.Logger:
    makedirs(save_dir)
    logger = logging.getLogger('cmpnn_train')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(save_dir, 'train.log'))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# =========================================================================== #
# Hyperparameters
# =========================================================================== #
DEFAULT_HPARAMS = dict(
    hidden_size=300, depth=3, dropout=0.0, ffn_num_layers=2, ffn_hidden_size=None,
    activation='ReLU', batch_size=50, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4,
    warmup_epochs=2.0, attention_dim=128,
)


def suggest_hparams(trial: optuna.Trial, base_args: Namespace) -> Dict:
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 300, 384, 512])
    # Anchor the schedule on max_lr; init/final are sampled as ratios so the
    # NoamLR schedule is always init < max > final.
    max_lr = trial.suggest_float('max_lr', 1e-4, 1e-2, log=True)
    hp = dict(
        hidden_size=hidden_size,
        depth=trial.suggest_int('depth', 2, 6),
        dropout=trial.suggest_float('dropout', 0.0, 0.5),
        ffn_num_layers=trial.suggest_int('ffn_num_layers', 1, 3),
        ffn_hidden_size=trial.suggest_categorical('ffn_hidden_size',
                                                  [128, 256, 300, 384, 512]),
        activation=trial.suggest_categorical('activation',
                                             ['ReLU', 'LeakyReLU', 'PReLU', 'ELU']),
        batch_size=trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        max_lr=max_lr,
        init_lr=max_lr * trial.suggest_float('init_lr_ratio', 0.01, 0.5, log=True),
        final_lr=max_lr * trial.suggest_float('final_lr_ratio', 0.001, 0.5, log=True),
        warmup_epochs=trial.suggest_categorical('warmup_epochs', [1.0, 2.0, 3.0]),
    )
    if getattr(base_args, 'self_attention', False):
        hp['attention_dim'] = trial.suggest_categorical('attention_dim', [32, 64, 128, 256])
    else:
        hp['attention_dim'] = getattr(base_args, 'attention_dim', 128)
    return hp


def build_args(base_args: Namespace, hparams: Dict, train_size: int) -> Namespace:
    """Materialize a Namespace consumed by the chemprop model builders."""
    args = copy.deepcopy(base_args)
    args.hidden_size = hparams['hidden_size']
    args.depth = hparams['depth']
    args.dropout = hparams['dropout']
    args.ffn_num_layers = hparams['ffn_num_layers']
    args.ffn_hidden_size = hparams['ffn_hidden_size'] or hparams['hidden_size']
    args.activation = hparams['activation']
    args.batch_size = hparams['batch_size']
    args.init_lr = hparams['init_lr']
    args.max_lr = hparams['max_lr']
    args.final_lr = hparams['final_lr']
    args.warmup_epochs = hparams['warmup_epochs']
    args.attention_dim = hparams.get('attention_dim', getattr(base_args, 'attention_dim', 128))
    args.train_data_size = train_size
    return args


# =========================================================================== #
# Training
# =========================================================================== #
def train_one_epoch(model, data, loss_func, optimizer, scheduler, args) -> float:
    model.train()
    # Shuffle by index so the caller's raw-target order stays aligned with
    # data.data for the post-epoch evaluation.
    perm = list(range(len(data)))
    random.shuffle(perm)

    loss_sum, n_seen = 0.0, 0
    num_iters = len(data) // args.batch_size * args.batch_size
    for start in range(0, num_iters, args.batch_size):
        if start + args.batch_size > len(data):
            break
        batch = MoleculeDataset([data[i] for i in perm[start:start + args.batch_size]])
        smiles, features, targets_raw = batch.smiles(), batch.features(), batch.targets()
        mask = torch.Tensor([[t is not None for t in tb] for tb in targets_raw])
        targets = torch.Tensor([[0 if t is None else t for t in tb] for tb in targets_raw])
        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()

        model.zero_grad()
        preds = model(smiles, features)
        loss = (loss_func(preds, targets) * mask).sum() / mask.sum()
        loss_sum += loss.item() * len(batch)
        n_seen += len(batch)
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()
    return loss_sum / max(n_seen, 1)


def predict_and_metric(model, data, raw_targets, batch_size, scaler):
    preds = predict(model=model, data=data, batch_size=batch_size, scaler=scaler)
    return compute_all_metrics(raw_targets, preds), preds


def train_with_early_stopping(args, train_data, val_data, train_raw_targets,
                              val_raw_targets, save_path, logger, patience,
                              max_epochs, features_scaler=None,
                              optuna_trial=None) -> Tuple[float, Dict, int]:
    """Train one model with validation-MSE early stopping; save best checkpoint."""
    # Standardize regression targets on the train split (in place on train_data).
    scaler = StandardScaler().fit(train_raw_targets)
    train_data.set_targets(scaler.transform(train_raw_targets).tolist())

    model = build_model(args)
    if args.cuda:
        model = model.cuda()
    logger.info(f'Model params = {param_count(model):,}')

    loss_func = get_loss_func(args)
    optimizer = build_optimizer(model, args)
    args.num_lrs = 1
    sched_args = copy.deepcopy(args)
    sched_args.epochs = max_epochs
    scheduler = build_lr_scheduler(optimizer, sched_args)

    save_checkpoint(save_path, model, scaler, features_scaler, args)

    best_val_mse, best_val_metrics, best_epoch, since_improve = float('inf'), None, 0, 0
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_data, loss_func, optimizer, scheduler, args)
        if isinstance(scheduler, ExponentialLR):
            scheduler.step()
        train_metrics, _ = predict_and_metric(model, train_data, train_raw_targets,
                                              args.batch_size, scaler)
        val_metrics, _ = predict_and_metric(model, val_data, val_raw_targets,
                                            args.batch_size, scaler)
        logger.info(f'Epoch {epoch:03d} | train_loss={train_loss:.4f} | '
                    f'train {fmt_metrics(train_metrics)} | val {fmt_metrics(val_metrics)}')

        if val_metrics['MSE'] < best_val_mse:
            best_val_mse, best_val_metrics, best_epoch, since_improve = \
                val_metrics['MSE'], val_metrics, epoch, 0
            save_checkpoint(save_path, model, scaler, features_scaler, args)
        else:
            since_improve += 1

        if optuna_trial is not None:
            optuna_trial.report(val_metrics['MSE'], epoch)
        if since_improve >= patience:
            logger.info(f'Early stopping at epoch {epoch} (best epoch={best_epoch}).')
            break

    logger.info(f'Best val MSE={best_val_mse:.4f} at epoch {best_epoch}.')
    return best_val_mse, best_val_metrics, best_epoch


# =========================================================================== #
# Optuna (K-fold CV-scored)
# =========================================================================== #
def run_optuna_cv(base_args, data, fold_pairs, save_dir, logger,
                  n_trials, tune_max_epochs, tune_patience) -> Dict:
    """Tune hyperparameters; each trial is scored as the mean val MSE over folds."""
    K = len(fold_pairs)
    logger.info(f'[optuna] {K}-fold scoring, n_trials={n_trials}')

    def objective(trial: optuna.Trial) -> float:
        hparams = suggest_hparams(trial, base_args)
        fold_mses = []
        for fold_idx, (tr_idx, va_idx) in enumerate(fold_pairs):
            train_data = slice_dataset(data, tr_idx)
            val_data = slice_dataset(data, va_idx)

            features_scaler, snapshot = None, None
            if _use_scaling(base_args, train_data):
                snapshot = _snapshot_features(list(train_data.data) + list(val_data.data))
                features_scaler = train_data.normalize_features(replace_nan_token=0)
                val_data.normalize_features(features_scaler)

            train_raw = [list(t) for t in train_data.targets()]
            val_raw = val_data.targets()
            args = build_args(base_args, hparams, len(train_data))
            ckpt = os.path.join(save_dir, f'trial_{trial.number}_fold_{fold_idx}.pt')
            try:
                val_mse, _, _ = train_with_early_stopping(
                    args, train_data, val_data, train_raw, val_raw,
                    ckpt, logger, tune_patience, tune_max_epochs,
                    features_scaler=features_scaler, optuna_trial=trial)
            except (ValueError, RuntimeError) as e:
                logger.warning(f'Trial {trial.number} fold {fold_idx} failed: {e}')
                val_mse = float('inf')
            finally:
                train_data.set_targets([list(t) for t in train_raw])
                _restore_features(list(train_data.data) + list(val_data.data), snapshot)
                if os.path.exists(ckpt):
                    os.remove(ckpt)
            if not np.isfinite(val_mse):
                return float('inf')
            fold_mses.append(val_mse)
        return float(np.mean(fold_mses))

    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=base_args.seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f'Best trial #{study.best_trial.number} mean val_MSE={study.best_value:.4f}')
    logger.info(f'Best params: {json.dumps(study.best_params, indent=2)}')
    makedirs(save_dir)
    with open(os.path.join(save_dir, 'best_params.json'), 'w') as f:
        json.dump(study.best_params, f, indent=2)

    best = copy.deepcopy(DEFAULT_HPARAMS)
    best.update(study.best_params)
    return best


# ---- feature-scaling helpers (shared) ----
def _use_scaling(base_args, dataset) -> bool:
    return (getattr(base_args, 'use_input_features', False)
            and getattr(base_args, 'features_scaling', True)
            and dataset.data[0].features is not None)


def _snapshot_features(datapoints):
    return {id(dp): dp.features.copy() for dp in datapoints}


def _restore_features(datapoints, snapshot):
    if snapshot is None:
        return
    for dp in datapoints:
        if id(dp) in snapshot:
            dp.set_features(snapshot[id(dp)])


# =========================================================================== #
# Final K-fold ensemble training + test evaluation
# =========================================================================== #
def run_cv_ensemble(base_args, hparams, data, df, fold_pairs, test_idx,
                    save_dir, logger, patience, max_epochs) -> Dict:
    """Train one model per fold, then ensemble their test predictions."""
    K = len(fold_pairs)
    makedirs(save_dir)
    df.loc[test_idx, ['smiles', 'value']].to_csv(
        os.path.join(save_dir, 'test.csv'), index=False)

    test_data = slice_dataset(data, test_idx)
    test_raw = test_data.targets()

    per_fold = {'train': [], 'val': [], 'test': []}
    test_preds_per_fold = []

    for fold_idx, (tr_idx, va_idx) in enumerate(fold_pairs):
        logger.info(f'===== Fold {fold_idx} =====')
        df.loc[tr_idx, ['smiles', 'value']].to_csv(
            os.path.join(save_dir, f'training_fold_{fold_idx}.csv'), index=False)
        df.loc[va_idx, ['smiles', 'value']].to_csv(
            os.path.join(save_dir, f'valid_fold_{fold_idx}.csv'), index=False)

        train_data = slice_dataset(data, tr_idx)
        val_data = slice_dataset(data, va_idx)
        logger.info(f'Sizes: train={len(train_data)} val={len(val_data)} test={len(test_data)}')

        train_raw = train_data.targets()
        val_raw = val_data.targets()

        do_features = _use_scaling(base_args, train_data)
        snapshot = None
        if do_features:
            snapshot = _snapshot_features(
                list(train_data.data) + list(val_data.data) + list(test_data.data))

        args = build_args(base_args, hparams, len(train_data))
        fold_dir = os.path.join(save_dir, f'fold_{fold_idx}')
        makedirs(fold_dir)
        ckpt_path = os.path.join(fold_dir, 'best_model.pt')

        try:
            features_scaler = None
            if do_features:
                features_scaler = train_data.normalize_features(replace_nan_token=0)
                val_data.normalize_features(features_scaler)
                test_data.normalize_features(features_scaler)
                logger.info('Global features standardized using fold-train statistics.')

            train_with_early_stopping(
                args, train_data, val_data, train_raw, val_raw,
                ckpt_path, logger, patience, max_epochs, features_scaler=features_scaler)

            best_model = load_checkpoint(ckpt_path, cuda=args.cuda)
            target_scaler, _ = load_scalers(ckpt_path)

            test_metrics, test_preds = predict_and_metric(
                best_model, test_data, test_raw, args.batch_size, target_scaler)
            train_metrics, _ = predict_and_metric(
                best_model, train_data, train_raw, args.batch_size, target_scaler)
            val_metrics, _ = predict_and_metric(
                best_model, val_data, val_raw, args.batch_size, target_scaler)

            per_fold['train'].append(train_metrics)
            per_fold['val'].append(val_metrics)
            per_fold['test'].append(test_metrics)
            test_preds_per_fold.append(np.asarray(test_preds, dtype=np.float64).reshape(-1))
            logger.info(f'Fold {fold_idx} train {fmt_metrics(train_metrics)}')
            logger.info(f'Fold {fold_idx} val   {fmt_metrics(val_metrics)}')
            logger.info(f'Fold {fold_idx} test  {fmt_metrics(test_metrics)}')
        finally:
            train_data.set_targets([list(t) for t in train_raw])
            _restore_features(
                list(train_data.data) + list(val_data.data) + list(test_data.data), snapshot)

    # ---- Ensemble ----
    preds_matrix = np.stack(test_preds_per_fold, axis=0)   # (K, N_test)
    ensemble_pred = preds_matrix.mean(axis=0)
    ensemble_std = preds_matrix.std(axis=0)
    test_raw_flat = np.asarray(test_raw, dtype=np.float64).reshape(-1)
    ensemble_metrics = compute_all_metrics(test_raw_flat, ensemble_pred)

    keys = ['MSE', 'RMSE', 'MAE', 'R2']
    def mean_std(ml):
        return {k: [float(np.mean([m[k] for m in ml])),
                    float(np.std([m[k] for m in ml]))] for k in keys}
    stats = {s: mean_std(per_fold[s]) for s in ('train', 'val', 'test')}

    logger.info('===== CV per-fold (mean +/- std across folds) =====')
    for s in ('train', 'val', 'test'):
        logger.info('  %-5s: %s' % (s, '  '.join(
            f'{k} {stats[s][k][0]:.4f} +/- {stats[s][k][1]:.4f}' for k in keys)))
    logger.info(f'===== ENSEMBLE TEST: {fmt_metrics(ensemble_metrics)}')

    out = df.loc[test_idx, ['smiles', 'value']].copy()
    out['pred'] = ensemble_pred
    out['pred_std'] = ensemble_std
    out.to_csv(os.path.join(save_dir, 'test_results.csv'), index=False)

    return {
        'ensemble_test': ensemble_metrics,
        'per_fold_train_mean_std': stats['train'],
        'per_fold_val_mean_std': stats['val'],
        'per_fold_test_mean_std': stats['test'],
        'per_fold_train_metrics': per_fold['train'],
        'per_fold_val_metrics': per_fold['val'],
        'per_fold_test_metrics': per_fold['test'],
    }


# =========================================================================== #
# Argument handling
# =========================================================================== #
def parse_args() -> Namespace:
    p = argparse.ArgumentParser(description='Train a CMPNN ensemble for HLM half-life regression.')
    p.add_argument('--train', type=str, required=True,
                   help='Training-set CSV with "smiles" and "value" columns '
                        '(extra columns ignored). Split into K CV folds.')
    p.add_argument('--test', type=str, required=True,
                   help='Independent test-set CSV with "smiles" and "value" columns '
                        '(extra columns ignored). Used only for final evaluation.')
    p.add_argument('--save_dir', type=str, default='./ckpt')
    p.add_argument('--gpu', type=int, default=0, help='GPU index, or -1 for CPU.')

    p.add_argument('--num_folds', type=int, default=5)
    p.add_argument('--split_seed', type=int, default=1234,
                   help='Seed for the shuffled KFold split of the training set.')

    p.add_argument('--tune', action='store_true', default=False,
                   help='Run Optuna hyperparameter tuning before final training.')
    p.add_argument('--n_trials', type=int, default=40)
    p.add_argument('--tune_max_epochs', type=int, default=40)
    p.add_argument('--tune_patience', type=int, default=8)
    p.add_argument('--best_params_path', type=str, default=None,
                   help='Load hyperparameters from this JSON instead of tuning.')

    p.add_argument('--epochs', type=int, default=150, help='Max epochs for final training.')
    p.add_argument('--patience', type=int, default=20, help='Early-stopping patience.')
    p.add_argument('--seed', type=int, default=42, help='Seed for torch/numpy/random.')

    p.add_argument('--global_features', type=str, default=None,
                   choices=['rdkit2d', 'mordred', 'morgan', 'maccs'],
                   help='Global descriptor concatenated to the molecule embedding.')
    p.add_argument('--self_attention', action='store_true', default=False,
                   help='Use an additive self-attention readout instead of mean pooling.')
    p.add_argument('--attention_dim', type=int, default=128)
    return p.parse_args()


def finalize_args(args: Namespace) -> Namespace:
    """Set the fixed model/training fields that the chemprop builders expect."""
    args.dataset_type = 'regression'
    args.use_compound_names = False
    args.features_generator = None
    args.features_path = None
    args.features_only = False
    args.use_input_features = bool(args.global_features)
    args.features_scaling = False
    args.features_dim = 0
    args.max_data_size = None
    args.no_cache = False
    args.bias = False
    args.undirected = False
    args.atom_messages = False
    args.metric = 'rmse'
    args.minimize_score = True
    args.num_tasks = 1
    args.task_names = ['value']
    args.cuda = (args.gpu is not None and args.gpu >= 0 and torch.cuda.is_available())
    if args.cuda:
        torch.cuda.set_device(args.gpu)
    args.num_lrs = 1
    args.train_data_size = 1
    args.ensemble_size = 1
    args.output_size = 1
    args.multiclass_num_classes = 3
    args.show_individual_scores = False
    args.checkpoint_paths = None
    return args


# =========================================================================== #
# Main
# =========================================================================== #
def main():
    args = finalize_args(parse_args())
    makedirs(args.save_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    logger = create_logger(args.save_dir)
    logger.info(f'Args: {vars(args)}')

    # ---- Load data ----
    # Training and test sets are provided as separate CSVs (the same convention
    # as the main repository). They are concatenated into a single dataset
    # (training rows first, then test rows) so the downstream indexing machinery
    # can address both with positional indices.
    def _load_slim(path: str) -> pd.DataFrame:
        d = pd.read_csv(path)
        if 'smiles' not in d.columns or 'value' not in d.columns:
            raise ValueError(f'{path} must contain "smiles" and "value" columns; '
                             f'found {list(d.columns)}')
        return d[['smiles', 'value']]

    train_df = _load_slim(args.train)
    test_df = _load_slim(args.test)
    n_train = len(train_df)
    df = pd.concat([train_df, test_df], ignore_index=True)

    # chemprop.get_data reads from a CSV path; write the combined slim copy so the
    # dataset order matches df exactly (training rows [0:n_train], then test rows).
    slim_path = os.path.join(args.save_dir, 'data_slim.csv')
    df.to_csv(slim_path, index=False)
    data = get_data(path=slim_path, args=args, logger=None)
    logger.info(f'Loaded {len(data)} molecules (train={n_train}, test={len(test_df)}).')

    # ---- Global features (optional) ----
    if args.global_features:
        logger.info(f'Computing global features: {args.global_features}')
        smiles_list = [dp.smiles for dp in data.data]
        names, feats = compute_global_features(args.global_features, smiles_list)
        feats, kept, dropped = drop_invalid_feature_columns(feats, names)
        logger.info(f'Global features: kept {len(kept)} / dropped {len(dropped)}.')
        for i, dp in enumerate(data.data):
            dp.set_features(feats[i])
        args.features_dim = feats.shape[1]
        args.features_size = feats.shape[1]
        args.use_input_features = True
        args.features_scaling = feature_needs_scaling(args.global_features)
        logger.info(f'Feature scaling for "{args.global_features}": {args.features_scaling}')
        with open(os.path.join(args.save_dir, 'global_features_names.json'), 'w') as f:
            json.dump({'generator': args.global_features, 'kept_names': kept,
                       'dropped_names': dropped, 'needs_scaling': args.features_scaling}, f, indent=2)

    # ---- Split: provided test set + K folds on the training set ----
    pool_idx = list(range(n_train))
    test_idx = list(range(n_train, len(df)))
    fold_pairs = make_cv_folds(pool_idx, args.num_folds, args.split_seed)
    logger.info(f'pool={len(pool_idx)} test={len(test_idx)} folds={args.num_folds}')

    # ---- Hyperparameters ----
    if args.tune:
        hparams = run_optuna_cv(args, data, fold_pairs,
                                os.path.join(args.save_dir, 'optuna'), logger,
                                args.n_trials, args.tune_max_epochs, args.tune_patience)
    elif args.best_params_path:
        with open(args.best_params_path) as f:
            params = json.load(f)
        hparams = copy.deepcopy(DEFAULT_HPARAMS)
        hparams.update(params)
    else:
        hparams = copy.deepcopy(DEFAULT_HPARAMS)
    logger.info(f'Using hparams: {json.dumps(hparams, indent=2)}')

    # ---- Train ensemble + evaluate ----
    results = run_cv_ensemble(args, hparams, data, df, fold_pairs, test_idx,
                              args.save_dir, logger, args.patience, args.epochs)

    logger.info('===== FINAL =====')
    for s in ('train', 'val', 'test'):
        ms = results[f'per_fold_{s}_mean_std']
        logger.info('  CV %-5s: %s' % (s, '  '.join(
            f'{k} {ms[k][0]:.4f} +/- {ms[k][1]:.4f}' for k in ('MSE', 'RMSE', 'MAE', 'R2'))))
    logger.info(f'  ENSEMBLE TEST: {fmt_metrics(results["ensemble_test"])}')

    with open(os.path.join(args.save_dir, 'final_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
