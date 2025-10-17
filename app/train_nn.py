from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump


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


def build_categorical_maps(df: pd.DataFrame, cat_cols):
    maps = {}
    for c in cat_cols:
        vals = [v for v in df[c].astype(str).unique().tolist() if v == v]
        maps[c] = {v: i + 1 for i, v in enumerate(sorted(vals))}  # reserve 0 for missing -> will set -1 later
    return maps


def apply_cat_maps(df: pd.DataFrame, cat_maps):
    df = df.copy()
    for col, mapping in cat_maps.items():
        if col in df.columns:
            df[f"{col}_CODE"] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
        else:
            df[f"{col}_CODE"] = -1
    return df


def main(data_path: str, outdir: str, epochs: int = 100):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_pickle(data_path) if data_path.lower().endswith(".pkl") else pd.read_csv(data_path)

    # Target: ln(price)
    if "PRICE_ln" not in df.columns:
        if "PRICE" in df.columns:
            df["PRICE_ln"] = np.log(df["PRICE"].astype(float))
        else:
            raise ValueError("Dataset must include PRICE_ln or PRICE column.")

    # Normalize column variants
    rename_map = {"TOTAL AREA": "TOTAL_AREA", "Latitude": "LATITUDE", "Longitude": "LONGITUDE"}
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Base features + region stats if present
    base_feats = [
        "ROOMS", "TOTAL_AREA", "FLOOR", "TOTAL_FLOORS", "FURNITURE", "CONDITION", "CEILING", "MATERIAL", "YEAR",
        "LATITUDE", "LONGITUDE",
    ]
    region_feats = [c for c in ["srednmes_zarplata", "chislennost_naseleniya_092025"] if c in df.columns]
    seg_feat = [c for c in ["segment_code"] if c in df.columns]
    used_cols = [c for c in base_feats + region_feats + seg_feat if c in df.columns]
    # If segment_code is present, mirror notebook by removing raw coordinates
    if "segment_code" in used_cols:
        used_cols = [c for c in used_cols if c not in ("LATITUDE", "LONGITUDE")]

    # Categorical columns to code (aligning with notebook categories)
    cat_cols = [c for c in ["FURNITURE", "CONDITION", "MATERIAL"] if c in used_cols]
    num_cols = [c for c in used_cols if c not in cat_cols]

    df_train = df.dropna(subset=["PRICE_ln"]).copy()
    df_train = df_train[used_cols + ["PRICE_ln"]]
    cat_maps = build_categorical_maps(df_train, cat_cols)
    df_enc = apply_cat_maps(df_train, cat_maps)

    # Compose final feature list: numeric + encoded categorical
    feat_list = num_cols + [f"{c}_CODE" for c in cat_cols]

    X = df_enc[feat_list].astype(float).values
    y = df_enc["PRICE_ln"].astype(float).values.reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    y_train_s = scaler_y.fit_transform(y_train)
    y_val_s = scaler_y.transform(y_val)

    model = HousePriceNN(input_dim=X_train_s.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    Xtr = torch.tensor(X_train_s, dtype=torch.float32)
    ytr = torch.tensor(y_train_s, dtype=torch.float32)
    Xva = torch.tensor(X_val_s, dtype=torch.float32)
    yva = torch.tensor(y_val_s, dtype=torch.float32)

    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    best = 1e9
    patience = 10
    bad = 0
    epochs = max(1, int(epochs))
    for epoch in range(epochs):
        model.train()
        run = 0.0
        for xb, yb in dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            run += loss.item()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xva), yva).item()
        scheduler.step(val_loss)
        # light progress logging every 10 epochs and on improvement
        if (epoch + 1) % 10 == 0:
            print({"epoch": epoch + 1, "val_loss": float(val_loss)})
        if val_loss < best:
            best = val_loss
            bad = 0
            torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
        else:
            bad += 1
            if bad >= patience:
                break

    # Save artifacts
    dump(scaler_X, os.path.join(outdir, "scaler_X.joblib"))
    dump(scaler_y, os.path.join(outdir, "scaler_y.joblib"))
    with open(os.path.join(outdir, "feature_list.json"), "w", encoding="utf-8") as f:
        json.dump(feat_list, f)
    with open(os.path.join(outdir, "cat_mappings.json"), "w", encoding="utf-8") as f:
        json.dump(cat_maps, f, ensure_ascii=False)
    print({"best_val_loss_scaled": float(best), "n_features": len(feat_list)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to preprocessed dataset (.pkl or .csv)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "nn_model"))
    ap.add_argument("--epochs", type=int, default=100, help="Max training epochs (default: 100)")
    args = ap.parse_args()
    main(args.data, args.outdir, epochs=args.epochs)
