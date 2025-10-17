import os
import argparse
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump


def build_pipeline(cat_cols, num_cols):
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    pre = ColumnTransformer([
        ("cat", cat, cat_cols),
        ("num", num, num_cols),
    ], remainder="drop")

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    pipe = Pipeline([
        ("pre", pre),
        ("rf", model),
    ])
    return pipe


def main(data_path: str, stats_cols_exist: bool, out_path: str):
    df = pd.read_pickle(data_path) if data_path.lower().endswith(".pkl") else pd.read_csv(data_path)

    # target ln-price must exist
    target_col = "PRICE_ln"
    if target_col not in df.columns:
        if "PRICE" in df.columns:
            df[target_col] = np.log(df["PRICE"].astype(float))
        else:
            raise ValueError("Dataset must include PRICE_ln or PRICE column.")

    # Normalize common column naming variations
    rename_map = {
        "TOTAL AREA": "TOTAL_AREA",
        "Latitude": "LATITUDE",
        "Longitude": "LONGITUDE",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # Minimal feature set aligning with the app inputs and regional stats
    base_feats = [
        "ROOMS", "TOTAL_AREA", "FLOOR", "TOTAL_FLOORS", "FURNITURE", "CONDITION", "CEILING", "MATERIAL", "YEAR",
        "LATITUDE", "LONGITUDE",
    ]
    region_feats = ["srednmes_zarplata", "chislennost_naseleniya_092025"] if stats_cols_exist else []
    feats = [c for c in base_feats + region_feats if c in df.columns]

    df_train = df.dropna(subset=[target_col]).copy()
    X = df_train[feats].copy()
    y = df_train[target_col].astype(float).values

    # Robustly infer categorical vs numeric (handles pandas StringDtype as categorical)
    cat_cols = [c for c in X.columns if not is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
    # Coerce numeric columns to real numbers (non-convertible -> NaN for imputer)
    if num_cols:
        X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = build_pipeline(cat_cols, num_cols)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print({"rmse_ln": float(rmse), "r2_ln": float(r2), "n_test": int(len(y_test))})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dump(pipe, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to preprocessed dataset (.pkl or .csv)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "model.joblib"))
    ap.add_argument("--with-stats", action="store_true", help="Use regional stats columns if present")
    args = ap.parse_args()
    main(args.data, args.with_stats, args.out)
