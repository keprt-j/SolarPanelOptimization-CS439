import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "specific_yield_kwh_per_kwp"
NUMERIC_FEATURES = [
    "temp_c_mean",
    "rh_pct_mean",
    "wind_speed_m_s_mean",
    "wind_direction_deg_mean",
    "ghi_kwh_m2",
    "dni_kwh_m2",
    "dhi_kwh_m2",
    "clearsky_ghi_kwh_m2",
    "hours_ghi_positive",
    "dc_capacity_kW",
    "distance_km",
    "latitude",
    "longitude",
    "elevation_m",
    "tilt",
    "azimuth",
    "day_of_year_sin",
    "day_of_year_cos",
    "wind_dir_sin",
    "wind_dir_cos",
]
CAT_FEATURES = ["kg_climate", "tracking", "type"]


def root_dir():
    return Path.cwd().resolve()


def default_out_dir():
    return root_dir() / "modeling_output"


class ModelingResult:
    def __init__(
        self,
        df,
        train,
        val,
        test,
        num_cols,
        cat_cols,
        feature_cols,
        ridge,
        hgb,
        results,
        predictions,
        best_name,
        best_model,
        importance,
        selected_vars,
    ):
        self.df = df
        self.train = train
        self.val = val
        self.test = test
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.feature_cols = feature_cols
        self.ridge = ridge
        self.hgb = hgb
        self.results = results
        self.predictions = predictions
        self.best_name = best_name
        self.best_model = best_model
        self.importance = importance
        self.selected_vars = selected_vars

    def predict_grid_siting_scores(self, weather_daily, chunk_size=50000):
        from siting_heatmap import grid_siting_scores

        return grid_siting_scores(self, weather_daily, chunk_size=chunk_size)


def load_prep(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"], low_memory=False)
    df["system_id"] = df["system_id"].astype(str)
    return df


def add_engineered_features(df):
    if "date" in df.columns:
        day = df["date"].dt.dayofyear.astype(float)
        angle = (2.0 * np.pi * day) / 365.25
        df["day_of_year_sin"] = np.sin(angle)
        df["day_of_year_cos"] = np.cos(angle)
    if "wind_direction_deg_mean" in df.columns:
        rad = np.radians(pd.to_numeric(df["wind_direction_deg_mean"], errors="coerce"))
        df["wind_dir_sin"] = np.sin(rad)
        df["wind_dir_cos"] = np.cos(rad)
    return df


def pearson_r(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    m = np.isfinite(yt) & np.isfinite(yp)
    if m.sum() < 2:
        return float("nan")
    return float(np.corrcoef(yt[m], yp[m])[0, 1])


def mean_bias(y_true, y_pred):
    return float(np.mean(np.asarray(y_pred, float) - np.asarray(y_true, float)))


def _metric_row(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "r": pearson_r(y_true, y_pred),
        "bias": mean_bias(y_true, y_pred),
    }


def _transform_target(y, mode):
    y_arr = np.asarray(y, dtype=float)
    if mode == "none":
        return y_arr
    if mode == "log1p":
        # Target is non-negative daily yield; guard against tiny negatives from noise.
        return np.log1p(np.clip(y_arr, a_min=0.0, a_max=None))
    raise ValueError(f"Unsupported target_transform={mode}")


def _inverse_transform_target(y, mode):
    y_arr = np.asarray(y, dtype=float)
    if mode == "none":
        return y_arr
    if mode == "log1p":
        return np.expm1(y_arr)
    raise ValueError(f"Unsupported target_transform={mode}")


def _preprocessor(num_cols, cat_cols):
    transformers = [("num", StandardScaler(), num_cols)]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    return ColumnTransformer(transformers=transformers)


def train_model(
    csv_path,
    n_repeats=6,
    top_k=8,
    cv_folds=5,
    target_transform="none",
    fast_mode=False,
    random_seed=1,
):
    df = add_engineered_features(load_prep(csv_path))
    if TARGET not in df.columns or "split" not in df.columns:
        raise ValueError("Prep CSV missing required columns (target/split).")

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError("Expected non-empty train/val/test splits.")

    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CAT_FEATURES if c in df.columns]
    feature_cols = num_cols + cat_cols
    if not feature_cols:
        raise ValueError("No configured feature columns present in prep CSV.")

    x_train, y_train = train[feature_cols], train[TARGET].astype(float)
    x_val, y_val = val[feature_cols], val[TARGET].astype(float)
    x_test, y_test = test[feature_cols], test[TARGET].astype(float)
    train_val = df[df["split"].isin(["train", "val"])].copy()
    x_train_val, y_train_val = train_val[feature_cols], train_val[TARGET].astype(float)
    groups = train_val["system_id"].astype(str).to_numpy()

    if fast_mode:
        cv_folds = min(cv_folds, 3)
        n_repeats = min(n_repeats, 3)

    pre = _preprocessor(num_cols, cat_cols)
    model_templates = {
        "Ridge": Pipeline(steps=[("prep", pre), ("model", Ridge(alpha=1.0, random_state=random_seed))]),
        "HistGradientBoosting": Pipeline(
            steps=[
                ("prep", pre),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="absolute_error",
                        max_depth=6,
                        learning_rate=0.05,
                        max_iter=300,
                        min_samples_leaf=50,
                        l2_regularization=0.1,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
    }

    gkf = GroupKFold(n_splits=cv_folds)
    rows = []
    predictions = {}
    model_map = {}
    importance_model_map = {}

    for name, template in model_templates.items():
        cv_metrics = []
        for train_idx, fold_idx in gkf.split(x_train_val, y_train_val, groups):
            fold_model = clone(template)
            x_fold_train = x_train_val.iloc[train_idx]
            y_fold_train = y_train_val.iloc[train_idx]
            x_fold_val = x_train_val.iloc[fold_idx]
            y_fold_val = y_train_val.iloc[fold_idx]
            y_fold_train_t = _transform_target(y_fold_train.to_numpy(), target_transform)
            fold_model.fit(x_fold_train, y_fold_train_t)
            fold_pred = _inverse_transform_target(fold_model.predict(x_fold_val), target_transform)
            cv_metrics.append(_metric_row(y_fold_val.to_numpy(), fold_pred))

        cv_df = pd.DataFrame(cv_metrics)
        final_model = clone(template)
        y_train_val_t = _transform_target(y_train_val.to_numpy(), target_transform)
        final_model.fit(x_train_val, y_train_val_t)
        test_pred = _inverse_transform_target(final_model.predict(x_test), target_transform)
        test_metrics = _metric_row(y_test.to_numpy(), test_pred)

        predictions[name] = test_pred
        model_map[name] = final_model

        importance_model = clone(template)
        y_train_t = _transform_target(y_train.to_numpy(), target_transform)
        importance_model.fit(x_train, y_train_t)
        importance_model_map[name] = importance_model

        rows.append(
            {
                "model": name,
                "val_MAE": cv_df["MAE"].mean(),
                "val_R2": cv_df["R2"].mean(),
                "val_r": cv_df["r"].mean(),
                "val_bias": cv_df["bias"].mean(),
                "val_MAE_std": cv_df["MAE"].std(ddof=0),
                "val_R2_std": cv_df["R2"].std(ddof=0),
                "val_r_std": cv_df["r"].std(ddof=0),
                "val_bias_std": cv_df["bias"].std(ddof=0),
                "test_MAE": test_metrics["MAE"],
                "test_R2": test_metrics["R2"],
                "test_r": test_metrics["r"],
                "test_bias": test_metrics["bias"],
            }
        )

    results = pd.DataFrame(rows).sort_values("val_MAE", ascending=True)
    best_name = str(results.iloc[0]["model"])
    best_model = model_map[best_name]

    r = permutation_importance(
        importance_model_map[best_name],
        x_val,
        _transform_target(y_val.to_numpy(), target_transform),
        n_repeats=n_repeats,
        random_state=random_seed,
        n_jobs=-1,
    )
    importance = pd.Series(r.importances_mean, index=x_val.columns).sort_values(ascending=False)
    selected_vars = importance.head(top_k).index.tolist()

    return ModelingResult(
        df=df,
        train=train,
        val=val,
        test=test,
        num_cols=num_cols,
        cat_cols=cat_cols,
        feature_cols=feature_cols,
        ridge=model_map["Ridge"],
        hgb=model_map["HistGradientBoosting"],
        results=results,
        predictions=predictions,
        best_name=best_name,
        best_model=best_model,
        importance=importance,
        selected_vars=selected_vars,
    )


def save_model_bundle(fit, out_path, seed, target_transform):
    bundle = {
        "seed": int(seed),
        "target_transform": str(target_transform),
        "best_name": fit.best_name,
        "best_model": fit.best_model,
        "feature_cols": list(fit.feature_cols),
        "num_cols": list(fit.num_cols),
        "cat_cols": list(fit.cat_cols),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(bundle, f)


def parse_args():
    out = default_out_dir()
    p = argparse.ArgumentParser(description="Train models and save summary tables.")
    p.add_argument("--prep-csv", type=Path, default=out / "model_dataset_prep.csv")
    p.add_argument("--metrics-out", type=Path, default=out / "model_metrics.csv")
    p.add_argument("--importance-out", type=Path, default=out / "model_importance.csv")
    p.add_argument("--model-out", type=Path, default=out / "model_bundle.pkl")
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--target-transform", type=str, default="none", choices=["none", "log1p"])
    p.add_argument("--fast-mode", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    fit = train_model(
        args.prep_csv,
        n_repeats=args.repeats,
        top_k=args.top_k,
        cv_folds=args.cv_folds,
        target_transform=args.target_transform,
        fast_mode=args.fast_mode,
        random_seed=args.seed,
    )
    fit.results.to_csv(args.metrics_out, index=False)
    imp_df = fit.importance.rename("importance").reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df.to_csv(args.importance_out, index=False)
    save_model_bundle(
        fit=fit,
        out_path=args.model_out,
        seed=args.seed,
        target_transform=args.target_transform,
    )
    print(fit.results)
    print("Best model:", fit.best_name)
    print("Saved:", args.metrics_out, args.importance_out, args.model_out)


if __name__ == "__main__":
    main()
