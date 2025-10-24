from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load


class HousePriceNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)
        self.out = nn.Identity()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.out(self.fc3(x))
        return x


@dataclass
class NNArtifacts:
    model: HousePriceNN
    scaler_X: any
    scaler_y: any
    feature_list: List[str]
    cat_maps: Dict[str, Dict[str, int]]


def load_artifacts(folder: str) -> NNArtifacts:
    with open(os.path.join(folder, "feature_list.json"), "r", encoding="utf-8") as f:
        feature_list = json.load(f)
    with open(os.path.join(folder, "cat_mappings.json"), "r", encoding="utf-8") as f:
        cat_maps = json.load(f)
    scaler_X = load(os.path.join(folder, "scaler_X.joblib"))
    scaler_y = load(os.path.join(folder, "scaler_y.joblib"))
    model = HousePriceNN(input_dim=len(feature_list))
    state = torch.load(os.path.join(folder, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return NNArtifacts(model=model, scaler_X=scaler_X, scaler_y=scaler_y, feature_list=feature_list, cat_maps=cat_maps)


def apply_cat_maps(df: pd.DataFrame, cat_maps: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """Apply categorical mappings to a DataFrame.

    Behavior:
    - If a mapping for a column exists in `cat_maps`, use it.
    - If mapping is missing but the column contains strings, try sensible default maps for
      well-known columns (FURNITURE, MATERIAL). If still missing, attempt numeric coercion.
    - As a last resort, create an on-the-fly factorized mapping (prints a warning).

    This function mutates string/categorical columns into numeric columns in-place so the
    feature matrix can be converted to float without TypeErrors.
    """
    df = df.copy()

    # Common sensible default mappings (1-based indexing as in training data)
    # Based on training data: FURNITURE (1=No, 2=Partial, 3=Full)
    # CONDITION (1-5 scale), MATERIAL (1=Panel, 2=Brick, 3=Monolithic, 4=Mixed)
    DEFAULT_MAPS = {
        "FURNITURE": {
            "No": 1, "Partial": 2, "Full": 3,
            "no": 1, "partial": 2, "full": 3,
            "NO": 1, "PARTIAL": 2, "FULL": 3
        },
        "MATERIAL": {
            "Panel": 1, "Brick": 2, "Monolithic": 3, "Mixed": 4,
            "panel": 1, "brick": 2, "monolithic": 3, "mixed": 4,
            "PANEL": 1, "BRICK": 2, "MONOLITHIC": 3, "MIXED": 4
        },
        "CONDITION": {
            "Needs renovation": 1, "Needs Renovation": 1, "needs renovation": 1,
            "Good": 2, "good": 2, "GOOD": 2,
            "Excellent": 3, "excellent": 3, "EXCELLENT": 3,
            # Extended scale if needed (training data has 1-5)
            "Fair": 2, "fair": 2,
            "Very Good": 4, "very good": 4,
            "Perfect": 5, "perfect": 5
        }
    }

    for col in set(list(cat_maps.keys()) + list(df.columns)):
        if col not in df.columns:
            # nothing to do for missing columns
            continue

        # If an explicit mapping exists, apply it
        mapping = cat_maps.get(col) or cat_maps.get(col.upper()) or cat_maps.get(col.lower())
        if mapping:
            # map and coerce; unknowns become -1
            df[col] = df[col].map(mapping).fillna(-1).astype(float)
            continue

        # If column is already numeric, skip
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            continue

        # Try default known mappings
        if col in DEFAULT_MAPS:
            df[col] = df[col].map(DEFAULT_MAPS[col]).fillna(-1).astype(float)
            continue

        # Try to coerce strings that look like numbers (e.g., "1", "2.0")
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().all():
            df[col] = coerced.astype(float)
            continue

        # Last resort: factorize and warn (mapping will be arbitrary and may not match training)
        codes, uniques = pd.factorize(df[col].astype(str))
        df[col] = codes.astype(float)
        try:
            uniq_map = {str(v): int(i) for i, v in enumerate(uniques)}
        except Exception:
            uniq_map = {str(v): int(i) for i, v in enumerate(uniques.tolist())}
        print(f"[nn_inference] Warning: created on-the-fly mapping for '{col}': {uniq_map}")

    return df


def build_feature_matrix(df: pd.DataFrame, feature_list: List[str]) -> np.ndarray:
    # ensure all columns exist
    for c in feature_list:
        if c not in df.columns:
            df[c] = 0
    # Coerce to numeric safely; any remaining non-convertible values will become NaN
    mat = df[feature_list].copy()
    for c in mat.columns:
        if not pd.api.types.is_numeric_dtype(mat[c].dtype):
            mat[c] = pd.to_numeric(mat[c], errors="coerce")
    if mat.isna().any().any():
        # Replace NaNs with column medians (safe fallback) and warn
        for col in mat.columns:
            if mat[col].isna().any():
                med = mat[col].median()
                mat[col].fillna(med, inplace=True)
                print(f"[nn_inference] Warning: filled NaNs in '{col}' with median={med}")
    return mat.astype(float).values


def predict_prices_kzt(art: NNArtifacts, df_features: pd.DataFrame) -> np.ndarray:
    X = build_feature_matrix(df_features, art.feature_list)
    Xs = art.scaler_X.transform(X)
    with torch.no_grad():
        x_tensor = torch.tensor(Xs, dtype=torch.float32)
        y_scaled = art.model(x_tensor).numpy()
    y_ln = art.scaler_y.inverse_transform(y_scaled).flatten()
    return np.exp(y_ln)
