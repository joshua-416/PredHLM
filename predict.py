# predict.py — single-model inference for metabolic stability
# --------------------------------------------------------------------
# Changes from the original:
# - Accepts CSVs that have ONLY a "smiles" column (no "value").
# - If "value" is missing, metrics are automatically skipped even if --metric is set.
# - Output CSV contains only ["smiles","pred"] when there is no ground truth.
# - All bagging/ensemble logic removed; uses a single model.pkl only.
# - Final saved prediction is half-life (minutes), converted from predicted log10 values.
#   * Metrics (if computed) remain on the log10 scale for consistency with training labels.
# - Comments are in English.
# --------------------------------------------------------------------

import argparse, os, glob, pprint, joblib, csv
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score


# RDKit & Mordred
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
from mordred import Calculator, descriptors

# -------------------------------------------------------------------- #
#               ─────────  Feature-engineering helpers  ─────────       #
# -------------------------------------------------------------------- #
def remove_abnormal_columns(df):
    """
    This function takes a pandas DataFrame as input, removes columns containing
    non-float, non-int, or NaN values, and returns the cleaned DataFrame along with
    the number of removed columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The cleaned DataFrame with abnormal columns removed.
    int: The number of removed columns.
    """
    # Initialize a list to hold the names of columns to be removed
    columns_to_remove = []

    # Iterate through each column in the DataFrame
    for column in df.columns:
        # Check if there is NaN
        if df[column].hasnans:
            columns_to_remove.append(column)
            continue

        # Check the column contains any value that is not float, int (abnormal result)
        if not df[column].apply(lambda x: isinstance(x, (float, int))).all():
            columns_to_remove.append(column)

    # Drop the columns with abnormal values
    cleaned_df = df.drop(columns=columns_to_remove)

    # Return the cleaned DataFrame and the number of removed columns
    return cleaned_df, len(columns_to_remove)

def reg_eval(targets, predictions, r=4):
    """
    Evaluate regression performance based on predictions and targets.

    Args:
        targets (array-like): True continuous values.
        predictions (array-like): Predicted continuous values.
        r (int): Rounding digits (default=4).

    Returns:
        mse   : Mean Squared Error
        rmse  : Root Mean Squared Error
        mae   : Mean Absolute Error
        r2    : R^2 score
    """

    targets     = np.array(targets, dtype=float)
    predictions = np.array(predictions, dtype=float)

    mse  = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(targets, predictions)
    r2   = r2_score(targets, predictions)

    return (
        round(mse,  r),
        round(rmse, r),
        round(mae,  r),
        round(r2,   r),
    )

class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w', newline='')
        #self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file, lineterminator='\n')
        #writer = csv.writer(self.csv_file)

        for arg, arg_val in args.items():
            writer.writerow([arg, arg_val])
        # for arg in vars(args):
        #     writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames = fieldnames, lineterminator = '\n')
        #self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def writes_sentence(self, string):
        writer = csv.writer(self.csv_file, lineterminator='\n')
        writer.writerow([string])
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


def smiles_to_morgan(smiles: str) -> np.ndarray:
    """Return 2048-bit Morgan fingerprint (radius=2) as float array."""
    mol = Chem.MolFromSmiles(smiles)
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.array(fp, dtype=float)

def smiles_to_usr(smiles: str) -> np.ndarray:
    """
    Compute USR (Ultrafast Shape Recognition) descriptor.
    Returns empty array if 3D embedding fails; caller will handle masking.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1234)
    try:
        return np.array(rdMolDescriptors.GetUSR(mol), dtype=float)
    except Exception:
        return np.array([])

def smiles_to_rdkit2d(smiles_list):
    """Compute all RDKit 2D descriptors into a DataFrame."""
    desc_names = [d[0] for d in Descriptors._descList]
    rows = []
    for s in tqdm(smiles_list, desc="RDKit2D"):
        mol  = Chem.MolFromSmiles(s)
        rows.append([getattr(Descriptors, n)(mol) for n in desc_names])
    return pd.DataFrame(rows, columns=desc_names)

def smiles_to_maccs(smiles_list):
    """Compute 166-bit MACCS keys for a list of SMILES."""
    fps = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s)) for s in tqdm(smiles_list, desc="MACCS")]
    arr = np.zeros((len(fps), 166), dtype=np.int8)
    tmp = np.empty(167, dtype=np.int8)  # ConvertToNumpyArray writes 167 bits (0..166)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, tmp)
        arr[i] = tmp[1:]                # drop bit-0
    return arr.astype(float)

def load_scaler_and_desc(auto_dir: str | None, scaler_path: str | None,
                         desc_path: str | None, prefix: str):
    """
    Resolve scaler and descriptor-name CSV paths.
    If explicit paths are given, use them; otherwise, try to auto-detect in *auto_dir*.
    """
    # scaler
    if scaler_path:
        scaler_file = scaler_path
    elif auto_dir:
        cand = glob.glob(os.path.join(auto_dir, f"*{prefix}*_scaler*.pkl"))
        scaler_file = cand[0] if cand else None
    else:
        scaler_file = None
    scaler = joblib.load(scaler_file) if scaler_file else None

    # descriptor names
    if desc_path:
        name_file = desc_path
    elif auto_dir:
        for patt in (f"*{prefix}*names_filtered.csv", f"*{prefix}*names.csv"):
            cand = glob.glob(os.path.join(auto_dir, patt))
            if cand:
                name_file = cand[0]
                break
        else:
            name_file = None
    else:
        name_file = None

    names = pd.read_csv(name_file).iloc[:, 0].tolist() if name_file else None
    return scaler, names, scaler_file, name_file

# -------------------------------------------------------------------- #
#                               Main                                   #
# -------------------------------------------------------------------- #
def main():
    # Quiet RDKit warnings
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    parser = argparse.ArgumentParser(
        description="Predict metabolic stability (regression). "
                    "Now accepts CSVs with ['smiles'] or ['smiles','value']."
    )
    # I/O
    parser.add_argument('--test',    type=str, required=True,
                        help='CSV with columns [smiles] or [smiles,value]')
    parser.add_argument('--models',  type=str, required=True,
                        help='Path to folder containing model.pkl OR a direct path to model.pkl')
    parser.add_argument('--output',  type=str, default=None,
                        help='Folder to save prediction files (default=models dir)')

    # Feature options (must match training)
    parser.add_argument('--feature', nargs='+',
                        choices=['mordred','morgan','rdkit2d','maccs','usr','preload'],
                        default=['rdkit2d'],
                        help="Choose feature(s)")
    parser.add_argument('--path',      type=str, default=None,
                        help="Pre-generated feature CSV (required when using 'preload').")
    parser.add_argument('--scaler',    type=str, default=None, help="Explicit scaler .pkl path")
    parser.add_argument('--desc_list', type=str, default=None, help="Explicit descriptor-name CSV path")

    # Misc
    parser.add_argument('--metric', action='store_true',
                        help='Compute regression metrics when ground truth is available')
    args = parser.parse_args().__dict__

    # ---------------------------------------------------------------- #
    #             1)  Load test set & optional ground-truth            #
    # ---------------------------------------------------------------- #
    df_test = pd.read_csv(args['test'])
    if 'smiles' not in df_test.columns:
        raise ValueError("Input CSV must contain a 'smiles' column.")

    test_smiles = df_test['smiles'].astype(str).tolist()
    n_samples   = len(df_test)
    has_value   = 'value' in df_test.columns

    # If no ground truth is present, force-skip metrics even if the flag is set.
    if not has_value and args['metric']:
        print("Note: 'value' column missing; metrics will be skipped.")
        args['metric'] = False

    y_true = (pd.to_numeric(df_test['value'], errors='coerce').values
              if has_value else None)

    # ---------------------------------------------------------------- #
    #             2)  Re-create molecular feature matrix               #
    # ---------------------------------------------------------------- #
    selected = args['feature']
    if isinstance(selected, str):  # guard in case argparse returns a string
        selected = [selected]

    feature_blocks = []
    valid_mask = np.ones(n_samples, dtype=bool)

    # Auto-detect scaler/descriptor list within the model directory when needed
    auto_dir = args['models'] if os.path.isdir(args['models']) else os.path.dirname(args['models'])

    for ftype in selected:
        if ftype == 'morgan':
            feats = np.stack([smiles_to_morgan(s) for s in tqdm(test_smiles, desc='Morgan')])
            feature_blocks.append(feats)

        elif ftype == 'usr':
            # Build a full-sized matrix with placeholders, then mask invalid rows later.
            usr_rows = [smiles_to_usr(s) for s in test_smiles]
            row_len = next((len(r) for r in usr_rows if len(r) > 0), 0)
            if row_len == 0:
                # No valid USR rows at all; append empty block to keep concatenation valid.
                feats = np.zeros((n_samples, 0), dtype=float)
            else:
                feats = np.empty((n_samples, row_len), dtype=float)
                feats[:] = np.nan
                for i, r in enumerate(usr_rows):
                    if len(r) == row_len:
                        feats[i] = r
                    else:
                        valid_mask[i] = False
            feature_blocks.append(feats)

        elif ftype == 'maccs':
            feature_blocks.append(smiles_to_maccs(test_smiles))

        elif ftype == 'rdkit2d':
            df_rdkit = smiles_to_rdkit2d(test_smiles)
            scaler, names, _, _ = load_scaler_and_desc(auto_dir, args['scaler'],
                                                       args['desc_list'], 'rdkit2d')
            if names:
                df_rdkit = df_rdkit[names]
            df_rdkit, _ = remove_abnormal_columns(df_rdkit)
            feats = scaler.transform(df_rdkit) if scaler is not None else df_rdkit.values
            feature_blocks.append(feats)

        elif ftype == 'mordred':
            calc = Calculator(descriptors, ignore_3D=True)
            df_mord = calc.pandas([Chem.MolFromSmiles(s) for s in test_smiles])
            scaler, names, _, _ = load_scaler_and_desc(auto_dir, args['scaler'],
                                                       args['desc_list'], 'mordred')
            if names:
                df_mord = df_mord[names]
            df_mord, _ = remove_abnormal_columns(df_mord)
            feats = scaler.transform(df_mord) if scaler is not None else df_mord.values
            feature_blocks.append(feats)

        elif ftype == 'preload':
            if not args['path']:
                raise ValueError("`--path` must be given when feature=preload")
            feature_blocks.append(pd.read_csv(args['path']).values)

        else:
            raise ValueError(f"Invalid feature: {ftype}")

    # Apply row mask if any featurization failed (rare, mainly for USR)
    if not valid_mask.all():
        test_smiles = [s for s, keep in zip(test_smiles, valid_mask) if keep]
        if y_true is not None:
            y_true = y_true[valid_mask]
        feature_blocks = [blk[valid_mask] for blk in feature_blocks]

    # Concatenate all feature blocks
    X_test = np.concatenate(feature_blocks, axis=1)
    assert X_test.shape[0] == len(test_smiles), "Row count mismatch after featurization."

    # ---------------------------------------------------------------- #
    #                     3)  Load trained single model                #
    # ---------------------------------------------------------------- #
    if os.path.isdir(args['models']):
        model_path = os.path.join(args['models'], 'model.pkl')
    else:
        model_path = args['models']  # allow direct path to the .pkl

    if not os.path.isfile(model_path):
        raise FileNotFoundError("Could not find 'model.pkl'. Provide a folder containing it "
                                "or a direct path to the file via --models.")

    model_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else args['models']
    out_dir   = args['output'] or model_dir
    os.makedirs(out_dir, exist_ok=True)

    # Logger (only when ground truth exists and user asked for metrics)
    csv_logger = None
    if args['metric'] and y_true is not None:
        log_fields = ['model_path', 'params', 'mse', 'rmse', 'mae', 'r2']
        log_file   = os.path.join(out_dir, 'predict_logs.csv')
        csv_logger = CSVLogger(args, fieldnames=log_fields, filename=log_file)

    # ---------------------------------------------------------------- #
    #                             4)  Inference                        #
    # ---------------------------------------------------------------- #
    model        = joblib.load(model_path)
    y_pred_log10 = model.predict(X_test)                 # model outputs log10(half-life in minutes)
    y_pred       = np.power(10.0, y_pred_log10)          # convert to half-life (minutes)

    params = pprint.pformat(getattr(model, "get_params", lambda deep=False: "N/A")(deep=False))

    # Round predictions at the 4th decimal place (keep 3 decimals)
    y_pred = np.round(y_pred.astype(float), 3)

    # Result CSV:
    # - If no ground truth: ["smiles","pred"] where pred is half-life (min)
    # - If ground truth exists: ["smiles","value","pred"] where value is ORIGINAL (likely log10), pred is half-life (min)
    if y_true is None:
        df_res = pd.DataFrame({'smiles': test_smiles, 'pred_min': y_pred})
    else:
        df_res = pd.DataFrame({'smiles': test_smiles, 'value': y_true, 'pred_min': y_pred})

    df_res.to_csv(os.path.join(out_dir, 'results.csv'), index=False)

    # Optional metrics (computed on log10 scale for consistency with labels)
    if csv_logger is not None and y_true is not None:
        mse, rmse, mae, r2 = reg_eval(y_true, y_pred_log10)
        csv_logger.writerow({'model_path': os.path.abspath(model_path),
                             'params': params.replace('\n',' '),
                             'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2})
        csv_logger.close()

    print("✓ Prediction finished. Results saved to:", out_dir)
    if y_true is None:
        print("Saved columns: smiles, pred (half-life in minutes).")
    else:
        print("Saved columns: smiles, value (as given), pred (half-life in minutes).")

# -------------------------------------------------------------------- #
if __name__ == '__main__':
    main()

