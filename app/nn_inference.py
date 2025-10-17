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
    df = df.copy()
    for col, mapping in cat_maps.items():
        if col in df.columns:
            df[f"{col}_CODE"] = df[col].map(mapping).fillna(-1).astype(int)
        else:
            df[f"{col}_CODE"] = -1
    return df


def build_feature_matrix(df: pd.DataFrame, feature_list: List[str]) -> np.ndarray:
    # ensure all columns exist
    for c in feature_list:
        if c not in df.columns:
            df[c] = 0
    return df[feature_list].astype(float).values


def predict_prices_kzt(art: NNArtifacts, df_features: pd.DataFrame) -> np.ndarray:
    X = build_feature_matrix(df_features, art.feature_list)
    Xs = art.scaler_X.transform(X)
    with torch.no_grad():
        x_tensor = torch.tensor(Xs, dtype=torch.float32)
        y_scaled = art.model(x_tensor).numpy()
    y_ln = art.scaler_y.inverse_transform(y_scaled).flatten()
    return np.exp(y_ln)
