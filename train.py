#!/usr/bin/env python
"""
train.py - Train a PredHLM model (K-fold ensemble regression).

This script trains a regression model to predict the log10 half-life (in minutes)
of compounds in human liver microsomes (HLM). It uses Optuna (TPE sampler) with
K-fold cross-validation for hyperparameter optimization, then trains a K-model
ensemble. The independent test set is used only for final evaluation.

Pipeline
--------
1. Generate molecular features from SMILES (rdkit2d / mordred / morgan / maccs).
2. Run K-fold cross-validation; for each Optuna trial the candidate
   hyperparameters are scored as the mean validation MSE across the K folds.
3. Train K final models with the best hyperparameters (one per fold split),
   each with its own StandardScaler fitted on its inner-train portion.
4. Predict the test set as the ensemble (mean) of the K models; report the
   per-fold standard deviation as an uncertainty estimate.

Data leakage is prevented by (i) holding out the test set before any
preprocessing and (ii) fitting the StandardScaler only on the inner-train data
of each fold. Binary fingerprints (morgan / maccs) are not standardized.

Usage
-----
    python train.py \
        --train data/training.csv \
        --test  data/test.csv \
        --feature rdkit2d \
        --model   xgboost \
        --n_trials 100 \
        --n_folds  5 \
        --output_dir results_rdkit2d_xgboost

Input CSVs must contain columns: 'smiles' and 'value' (value = log10 half-life).
"""
import argparse
import os
from copy import deepcopy
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

# Preprocessing / metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cheminformatics
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from mordred import Calculator, descriptors as mordred_descriptors

RDLogger.logger().setLevel(RDLogger.CRITICAL)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_N_ESTIMATORS_ES   = 2000   # fixed; early stopping sets the actual tree count
_EARLY_STOP_ROUNDS = 50
_N_THREADS         = 8
_SPLIT_SEED        = 1234   # train/test split seed (kept for reproducibility)
_KFOLD_SEED        = 42     # K-fold split + Optuna sampler seed
_ES_MODELS         = {'catboost', 'lightgbm', 'xgboost'}
_FINGERPRINT_TYPES = {'morgan', 'maccs'}   # binary -> excluded from scaling

# rdkit2d descriptor to exclude: Ipc grows factorially with molecule size and
# can overflow to ~1e31, which corrupts scaling and distance computations.
_RDKIT2D_DENY = {'Ipc'}


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────
def choose_model(model_name: str, params: dict):
    """Instantiate a regressor by name with the given parameters."""
    params = deepcopy(params) or {}

    if model_name == 'lightgbm':
        params.setdefault('force_row_wise', True)
        params.setdefault('num_threads', _N_THREADS)
        params.setdefault('verbose', -1)
    elif model_name == 'catboost':
        params.setdefault('verbose', 0)
        params.setdefault('allow_writing_files', False)
        params.setdefault('thread_count', _N_THREADS)
    elif model_name == 'random_forest':
        params.setdefault('n_jobs', _N_THREADS)
    elif model_name == 'svm':
        # Cap iterations: libsvm SMO can stall for ill-conditioned (high C,
        # low gamma) combinations, making a single trial run for days.
        params.setdefault('max_iter', 100_000)

    factory = {
        'random_forest': RandomForestRegressor,
        'lightgbm':      LGBMRegressor,
        'xgboost':       XGBRegressor,
        'knn':           KNeighborsRegressor,
        'svm':           SVR,
        'krr':           KernelRidge,
        'catboost':      CatBoostRegressor,
        'ann':           MLPRegressor,
    }
    if model_name not in factory:
        raise ValueError(f'Unknown model: {model_name}')
    return factory[model_name](**params)


def _build_es_model(model_type: str, params: dict):
    """Build a CatBoost / LightGBM / XGBoost model configured for early stopping.

    n_estimators / iterations are fixed at _N_ESTIMATORS_ES; the actual number
    of trees is determined by early stopping at fit time.
    """
    params = deepcopy(params)

    if model_type == 'catboost':
        params.update({
            'iterations':            _N_ESTIMATORS_ES,
            'early_stopping_rounds': _EARLY_STOP_ROUNDS,
            'verbose':               0,
            'allow_writing_files':   False,
            'thread_count':          _N_THREADS,
        })
        return CatBoostRegressor(**params)

    if model_type == 'lightgbm':
        params.update({
            'n_estimators':   _N_ESTIMATORS_ES,
            'force_row_wise':  True,
            'num_threads':     _N_THREADS,
            'verbose':        -1,
        })
        return LGBMRegressor(**params)

    if model_type == 'xgboost':
        params.update({
            'n_estimators':          _N_ESTIMATORS_ES,
            'early_stopping_rounds': _EARLY_STOP_ROUNDS,
        })
        return XGBRegressor(**params)

    raise ValueError(f'Not an early-stopping model: {model_type}')


def _fit_es_model(model_type, model, tr_X, tr_y, va_X, va_y):
    """Fit an early-stopping model using each library's native API."""
    if model_type == 'catboost':
        model.fit(tr_X, tr_y, eval_set=(va_X, va_y), verbose=0)
    elif model_type == 'lightgbm':
        model.fit(tr_X, tr_y, eval_set=[(va_X, va_y)],
                  callbacks=[lgb.early_stopping(_EARLY_STOP_ROUNDS, verbose=False),
                             lgb.log_evaluation(-1)])
    elif model_type == 'xgboost':
        model.fit(tr_X, tr_y, eval_set=[(va_X, va_y)], verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# Feature generation
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


def remove_abnormal_columns(df: pd.DataFrame):
    """Drop descriptor columns containing NaN or non-numeric values."""
    drop = [c for c in df.columns
            if df[c].isnull().any()
            or not df[c].apply(lambda x: isinstance(x, (int, float))).all()]
    return df.drop(columns=drop), len(drop)


def build_features(smiles, feature_type, names_out=None, names_in=None):
    """Compute a feature matrix for a list of SMILES.

    Parameters
    ----------
    feature_type : 'rdkit2d' | 'mordred' | 'morgan' | 'maccs'
    names_out    : if given (rdkit2d / mordred), save the surviving descriptor
                   names to this path (training time).
    names_in     : if given (rdkit2d / mordred), select exactly these descriptor
                   columns in this order (inference time).

    Returns
    -------
    X         : np.ndarray (n_samples, n_features)
    col_names : list[str] or None  (descriptor names; None for fingerprints)
    """
    if feature_type == 'rdkit2d':
        desc = _rdkit2d_descriptors()
        names = [n for n, _ in desc]
        rows = [[fn(Chem.MolFromSmiles(s)) for _, fn in desc]
                for s in tqdm(smiles, desc='RDKit2D', ncols=80)]
        df = pd.DataFrame(rows, columns=names)
        df, _ = remove_abnormal_columns(df)
        if names_in is not None:
            df = df[[c for c in names_in if c in df.columns]]
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if names_out is not None:
            pd.DataFrame(df.columns, columns=['Descriptor']).to_csv(names_out, index=False)
        return df.values, list(df.columns)

    if feature_type == 'mordred':
        calc = Calculator(mordred_descriptors, ignore_3D=True)
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        df = calc.pandas(mols)
        df, _ = remove_abnormal_columns(df)
        if names_in is not None:
            df = df[[c for c in names_in if c in df.columns]]
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if names_out is not None:
            pd.DataFrame(df.columns, columns=['Descriptor']).to_csv(names_out, index=False)
        return df.values, list(df.columns)

    if feature_type == 'morgan':
        arr = []
        for s in tqdm(smiles, desc='Morgan', ncols=80):
            fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),
                                                       radius=2, nBits=2048)
            arr.append(np.array(fp))
        return np.stack(arr), None

    if feature_type == 'maccs':
        out = np.zeros((len(smiles), 166), dtype=np.int8)
        tmp = np.empty(167, dtype=np.int8)
        for i, s in enumerate(tqdm(smiles, desc='MACCS', ncols=80)):
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s)), tmp)
            out[i] = tmp[1:]                # drop bit-0
        return out.astype(float), None

    raise ValueError(f'Unknown feature type: {feature_type}')


def fit_scaler(X_fit, skip_scaling: bool):
    """Fit a StandardScaler on X_fit.

    For binary fingerprints (skip_scaling=True) the scaler is neutralized
    (mean=0, scale=1) so transform() leaves the 0/1 values unchanged, while the
    saved scaler still applies to the full feature matrix.
    """
    sc = StandardScaler().fit(X_fit)
    if skip_scaling:
        sc.mean_[:]  = 0.0
        sc.scale_[:] = 1.0
        if getattr(sc, 'var_', None) is not None:
            sc.var_[:] = 1.0
    return sc


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search space
# ─────────────────────────────────────────────────────────────────────────────
def suggest_params(trial, model_type: str) -> dict:
    """Sample a hyperparameter set for an Optuna trial (regression)."""
    if model_type == 'catboost':
        return {
            'depth':             trial.suggest_int  ('depth',             3, 12),
            'learning_rate':     trial.suggest_float('learning_rate',     5e-3, 0.15, log=True),
            'l2_leaf_reg':       trial.suggest_float('l2_leaf_reg',       1.0, 50.0,  log=True),
            'min_child_samples': trial.suggest_int  ('min_child_samples', 10, 150),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        }
    if model_type == 'lightgbm':
        return {
            'num_leaves':        trial.suggest_int  ('num_leaves',        15, 63),
            'max_depth':         trial.suggest_int  ('max_depth',         3, 12),
            'learning_rate':     trial.suggest_float('learning_rate',     5e-3, 0.2, log=True),
            'min_child_samples': trial.suggest_int  ('min_child_samples', 10, 150),
            'reg_alpha':         trial.suggest_float('reg_alpha',         1e-4, 10.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda',        1e-4, 10.0, log=True),
            'subsample':         trial.suggest_float('subsample',         0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree',  0.5, 1.0),
        }
    if model_type == 'xgboost':
        return {
            'max_depth':        trial.suggest_int  ('max_depth',        3, 12),
            'learning_rate':    trial.suggest_float('learning_rate',    5e-3, 0.2, log=True),
            'min_child_weight': trial.suggest_int  ('min_child_weight', 1, 50),
            'subsample':        trial.suggest_float('subsample',        0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha',        1e-4, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda',       1e-4, 10.0, log=True),
            'gamma':            trial.suggest_float('gamma',            0.0, 0.5),
        }
    if model_type == 'random_forest':
        return {
            'n_estimators':      trial.suggest_int        ('n_estimators',      200, 1500),
            'max_depth':         trial.suggest_int        ('max_depth',         5, 30),
            'min_samples_leaf':  trial.suggest_int        ('min_samples_leaf',  2, 30),
            'min_samples_split': trial.suggest_int        ('min_samples_split', 2, 30),
            'max_features':      trial.suggest_categorical('max_features', ['sqrt', 0.3, 0.5]),
        }
    if model_type == 'knn':
        return {
            'n_neighbors': trial.suggest_int        ('n_neighbors', 3, 25),
            'weights':     trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric':      trial.suggest_categorical('metric',  ['euclidean', 'manhattan']),
        }
    if model_type == 'svm':
        return {
            'C':       trial.suggest_float('C', 0.01, 100.0, log=True),
            'gamma':   trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel':  trial.suggest_categorical('kernel', ['rbf', 'linear']),
            'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
        }
    if model_type == 'krr':
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'polynomial'])
        params = {'alpha': trial.suggest_float('alpha', 0.01, 100.0, log=True),
                  'kernel': kernel}
        if kernel in ('rbf', 'polynomial'):
            params['gamma'] = trial.suggest_float('gamma', 1e-4, 1.0, log=True)
        return params
    if model_type == 'ann':
        n_layers   = trial.suggest_int('n_layers', 1, 3)
        layer_size = trial.suggest_int('layer_size', 64, 512)
        return {
            'hidden_layer_sizes':  tuple([layer_size] * n_layers),
            'activation':          trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha':               trial.suggest_float('alpha', 1e-5, 0.1, log=True),
            'learning_rate_init':  trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
            'max_iter':            1000,
            'early_stopping':      True,
            'validation_fraction': 0.1,
            'n_iter_no_change':    20,
        }
    raise ValueError(f'Unknown model type: {model_type}')


# ─────────────────────────────────────────────────────────────────────────────
# K-fold training helpers
# ─────────────────────────────────────────────────────────────────────────────
def fit_one_fold(model_type, params, tr_X, tr_y, va_X, va_y,
                 *, skip_scaling, predict_X=None):
    """Fit one fold (own scaler) and return (val_mse, val_pred, predict_pred,
    model, scaler)."""
    scaler = fit_scaler(tr_X, skip_scaling)
    tr_Xs, va_Xs = scaler.transform(tr_X), scaler.transform(va_X)

    if model_type in _ES_MODELS:
        model = _build_es_model(model_type, params)
        _fit_es_model(model_type, model, tr_Xs, tr_y, va_Xs, va_y)
    else:
        model = choose_model(model_type, params)
        model.fit(tr_Xs, tr_y)

    va_pred = model.predict(va_Xs)
    val_mse = float(mean_squared_error(va_y, va_pred))
    pred_out = model.predict(scaler.transform(predict_X)) if predict_X is not None else None
    return val_mse, va_pred, pred_out, model, scaler


def optimize_kfold(model_type, X, y, fold_splits, n_trials, skip_scaling):
    """Optuna search; objective = mean validation MSE over the K folds."""
    cache = {}

    def objective(trial):
        params = suggest_params(trial, model_type)
        cache[trial.number] = params
        scores = [fit_one_fold(model_type, params, X[tr], y[tr], X[va], y[va],
                               skip_scaling=skip_scaling)[0]
                  for tr, va in fold_splits]
        return float(np.mean(scores))

    study = optuna.create_study(direction='minimize',
                                sampler=TPESampler(seed=_KFOLD_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = cache.get(study.best_trial.number, study.best_params)
    return best, study


def train_ensemble(model_type, params, X, y, fold_splits, *, skip_scaling, test_X):
    """Train K final models; return (models, scalers, oof_pred, test_pred[K, n])."""
    models, scalers = [], []
    oof = np.zeros(len(y))
    test_preds = []
    for tr, va in fold_splits:
        _, va_pred, te_pred, model, scaler = fit_one_fold(
            model_type, params, X[tr], y[tr], X[va], y[va],
            skip_scaling=skip_scaling, predict_X=test_X)
        models.append(model)
        scalers.append(scaler)
        oof[va] = va_pred
        test_preds.append(te_pred)
    return models, scalers, oof, np.stack(test_preds)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Train a PredHLM K-fold ensemble regressor.')
    ap.add_argument('--train', required=True, help='Training CSV (columns: smiles, value)')
    ap.add_argument('--test',  required=True, help='Test CSV (columns: smiles, value)')
    ap.add_argument('--feature', default='rdkit2d',
                    choices=['rdkit2d', 'mordred', 'morgan', 'maccs'])
    ap.add_argument('--model', default='xgboost',
                    choices=['catboost', 'lightgbm', 'xgboost', 'random_forest',
                             'knn', 'svm', 'krr', 'ann'])
    ap.add_argument('--target_col', default='value', help='Target column name (log10 half-life)')
    ap.add_argument('--n_trials', type=int, default=100, help='Number of Optuna trials')
    ap.add_argument('--n_folds',  type=int, default=5,   help='Number of CV folds')
    ap.add_argument('--output_dir', default='results',  help='Output directory')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir   = os.path.join(args.output_dir, 'model')
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    skip_scaling = args.feature in _FINGERPRINT_TYPES

    print('=' * 64)
    print('  PredHLM training')
    print(f'  feature  : {args.feature}   model : {args.model}')
    print(f'  n_trials : {args.n_trials}   n_folds : {args.n_folds}')
    print(f'  scaling  : {"OFF (binary fingerprint)" if skip_scaling else "StandardScaler"}')
    print('=' * 64)

    # ── Load data ────────────────────────────────────────────────────────────
    df_tr = pd.read_csv(args.train)
    df_te = pd.read_csv(args.test)
    y_tr  = df_tr[args.target_col].astype(float).values
    y_te  = df_te[args.target_col].astype(float).values

    # ── Features ─────────────────────────────────────────────────────────────
    names_out = (os.path.join(model_dir, f'{args.feature}_names.csv')
                 if args.feature in ('rdkit2d', 'mordred') else None)
    print('\n[1/4] Building training features ...')
    X_tr, col_names = build_features(df_tr['smiles'].tolist(), args.feature,
                                     names_out=names_out)
    print('[2/4] Building test features ...')
    X_te, _ = build_features(df_te['smiles'].tolist(), args.feature,
                             names_in=col_names)
    print(f'  train X: {X_tr.shape}   test X: {X_te.shape}')

    # ── K-fold splits (fixed seed) ───────────────────────────────────────────
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=_KFOLD_SEED)
    fold_splits = list(kf.split(X_tr))

    # ── Optuna ───────────────────────────────────────────────────────────────
    print(f'\n[3/4] Hyperparameter optimization '
          f'({args.n_trials} trials x {args.n_folds}-fold CV) ...')
    best_params, study = optimize_kfold(args.model, X_tr, y_tr, fold_splits,
                                        args.n_trials, skip_scaling)
    study.trials_dataframe().to_csv(
        os.path.join(results_dir, 'optuna_trials.csv'), index=False)
    print('  Best params:')
    for k, v in best_params.items():
        print(f'    {k:<20}: {v}')

    # ── Final K-model ensemble ───────────────────────────────────────────────
    print(f'\n[4/4] Training {args.n_folds}-model ensemble ...')
    models, scalers, oof, test_preds = train_ensemble(
        args.model, best_params, X_tr, y_tr, fold_splits,
        skip_scaling=skip_scaling, test_X=X_te)

    ens_pred = test_preds.mean(axis=0)
    ens_std  = test_preds.std(axis=0)

    # ── Metrics: per-fold mean +/- std + ensemble ────────────────────────────
    def reg(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return np.array([mse, np.sqrt(mse), mean_absolute_error(y_true, y_pred),
                         r2_score(y_true, y_pred)])

    val_fold  = np.array([reg(y_tr[va], oof[va]) for _, va in fold_splits])
    test_fold = np.array([reg(y_te, test_preds[k]) for k in range(args.n_folds)])
    ens_m     = reg(y_te, ens_pred)
    keys = ['mse', 'rmse', 'mae', 'r2']

    rows = []
    for name, arr in [('Val_CV', val_fold), ('Test_CV', test_fold)]:
        row = {'dataset': name}
        for i, k in enumerate(keys):
            row[k] = round(float(arr[:, i].mean()), 4)
            row[f'{k}_std'] = round(float(arr[:, i].std()), 4)
        rows.append(row)
    ens_row = {'dataset': 'Test_Ensemble'}
    for i, k in enumerate(keys):
        ens_row[k] = round(float(ens_m[i]), 4)
        ens_row[f'{k}_std'] = ''
    rows.append(ens_row)
    pd.DataFrame(rows).to_csv(os.path.join(results_dir, 'cv_results.csv'), index=False)

    print('\n  Results (per-fold mean +/- std; ensemble = final):')
    print(f'    Val  CV  : RMSE {val_fold[:,1].mean():.4f} +/- {val_fold[:,1].std():.4f}'
          f'   R2 {val_fold[:,3].mean():.4f} +/- {val_fold[:,3].std():.4f}')
    print(f'    Test CV  : RMSE {test_fold[:,1].mean():.4f} +/- {test_fold[:,1].std():.4f}'
          f'   R2 {test_fold[:,3].mean():.4f} +/- {test_fold[:,3].std():.4f}')
    print(f'    Test ENS : RMSE {ens_m[1]:.4f}   MAE {ens_m[2]:.4f}   R2 {ens_m[3]:.4f}')

    # ── Save predictions ─────────────────────────────────────────────────────
    test_out = pd.DataFrame({'smiles': df_te['smiles'], 'value': y_te,
                             'pred': np.round(ens_pred, 4), 'pred_std': np.round(ens_std, 4)})
    for k in range(args.n_folds):
        test_out[f'pred_fold_{k}'] = np.round(test_preds[k], 4)
    test_out.to_csv(os.path.join(results_dir, 'test_results.csv'), index=False)

    pd.DataFrame({'smiles': df_tr['smiles'], 'value': y_tr,
                  'pred_oof': np.round(oof, 4)}).to_csv(
        os.path.join(results_dir, 'oof_predictions.csv'), index=False)

    # ── Save K models + scalers ──────────────────────────────────────────────
    for k, (m, s) in enumerate(zip(models, scalers)):
        fdir = os.path.join(model_dir, f'fold_{k}')
        os.makedirs(fdir, exist_ok=True)
        joblib.dump(m, os.path.join(fdir, 'model.pkl'))
        joblib.dump(s, os.path.join(fdir, 'scaler.pkl'))

    print(f'\nDone. Outputs in: {args.output_dir}')
    print(f'  model/   : {args.n_folds} fold models + scalers'
          + (f' + {args.feature}_names.csv' if names_out else ''))
    print(f'  results/ : cv_results.csv, test_results.csv, '
          f'oof_predictions.csv, optuna_trials.csv')


if __name__ == '__main__':
    main()
