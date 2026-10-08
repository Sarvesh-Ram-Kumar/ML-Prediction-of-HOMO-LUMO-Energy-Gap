from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit import rdBase
import pandas as pd
import numpy as np
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')
rdBase.DisableLog('rdApp.info')
rdBase.DisableLog('rdApp.debug')


# ||RDKIT Functions||

def smiles_to_3d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    try:
        params = AllChem.ETKDG()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            return None
        return mol

    except Exception as e:
        return None


def extract_3d_features(mol):
    return {
        "RadiusOfGyration": rdMolDescriptors.CalcRadiusOfGyration(mol),
        "Asphericity": rdMolDescriptors.CalcAsphericity(mol),
        "Eccentricity": rdMolDescriptors.CalcEccentricity(mol),
        "InertialShapeFactor": rdMolDescriptors.CalcInertialShapeFactor(mol),
        "SpherocityIndex": rdMolDescriptors.CalcSpherocityIndex(mol),

        # supportive physicochemical descriptors
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
    }


#||Test Data Preparation||

def prepare_test_data(feature_columns):
    hopv = pd.read_csv('HOPV_15_revised_2_processed_homo_5fold.csv')

    print("X column names:")
    print(hopv.columns[0])
    print("\ny column name:")
    print(hopv.columns[9])
    X_test = []
    y_test = []
    test_smiles = []
    for _, row in hopv.iterrows():
        if pd.isna(row["electrochemical_gap"]):
            continue
        mol = smiles_to_3d(row["smiles"])
        if mol is None:
            continue

        try:
            features = extract_3d_features(mol)
            X_test.append(features)
            y_test.append(float(row["electrochemical_gap"]))
            test_smiles.append(row["smiles"])

        except Exception:
            continue

    # Convert to DataFrame / NumPy array
    X_test = pd.DataFrame(X_test)
    y_test = np.array(y_test, dtype=float)
    # Ensure same feature order as training
    X_test = X_test[feature_columns]

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    return X_test, y_test, test_smiles

def prepare_test_data_modified(feature_columns):
    hopv = pd.read_csv('HOPV_15_Modified.csv')

    print("X column names:")
    print(hopv.columns[0])
    print("\ny column name:")
    print(hopv.columns[9])
    X_test = []
    y_test = []
    test_smiles = []
    for _, row in hopv.iterrows():
        if pd.isna(row["electrochemical_gap"]):
            continue
        mol = smiles_to_3d(row["smiles"])
        if mol is None:
            continue

        try:
            features = extract_3d_features(mol)
            X_test.append(features)
            y_test.append(float(row["electrochemical_gap"]))
            test_smiles.append(row["smiles"])

        except Exception:
            continue

    # Convert to DataFrame / NumPy array
    X_test = pd.DataFrame(X_test)
    y_test = np.array(y_test, dtype=float)
    # Ensure same feature order as training
    X_test = X_test[feature_columns]

    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    return X_test, y_test, test_smiles