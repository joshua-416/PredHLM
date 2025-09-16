# PredHLM
PredHLM is a Python workflow for machine-learning prediction of metabolic stability in human liver microsome. It loads a single pre-trained model and predicts half-life (minutes) for input molecules.

## Developers
Jidon Jang

## Prerequisites
python3<br> numpy<br> pandas<br> tqdm<br> RDKit<br> scikit-learn<br> Mordred<br> (Optional, depends on the model you trained) LightGBM, XGBoost, CatBoost, etc. <br>

## Publication
Jidon Jang, Dokyun Na, Kwang-Seok Oh, "PredHLM: Interpretable prediction of half-life for metabolic stability assessment in human liver microsomes.", (in preparation)

## Usage
### Predict half-life with a pre-trained model (model.pkl), scaler (rdkit2d_minmax_scaler.pkl) and list of descriptor.
`python predict.py --test data_test.csv --models ./model.pkl --feature rdkit2d --scaler rdkit2d_minmax_scaler.pkl --desc_list rdkit2d_names.csv`<br>

The final result.csv file saves their SMILES and predicted half-life for human liver microsomal stability in minutes ('pred_min' column).
