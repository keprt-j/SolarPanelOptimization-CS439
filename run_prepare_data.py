import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CONUS_LAT_MIN, CONUS_LAT_MAX = 24.5, 49.5
CONUS_LON_MIN, CONUS_LON_MAX = -124.8, -66.9
MODELING_OUTPUT_DIR = Path.cwd().resolve() / "modeling_output"

def _assign_system_splits(system_ids, random_seed):
    rng = np.random.default_rng(random_seed)
    ids = np.asarray(system_ids, dtype=object).copy()
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val :])
    return train_ids, val_ids, test_ids


def _sample_split_rows(df_split, target_rows, sample_seed):
    if target_rows is None or target_rows <= 0 or len(df_split) <= target_rows:
        return df_split
    rng = np.random.default_rng(sample_seed)
    counts = df_split["system_id"].value_counts()
    weights = (counts / counts.sum()).to_numpy(dtype=float)
    systems = counts.index.to_numpy()
    expected = weights * target_rows
    base = np.floor(expected).astype(int)
    remainder = int(target_rows - base.sum())
    if remainder > 0:
        order = np.argsort(-(expected - base))
        base[order[:remainder]] += 1
    keep_parts = []
    for sid, n_take in zip(systems, base):
        if n_take <= 0:
            continue
        block = df_split[df_split["system_id"] == sid]
        if n_take >= len(block):
            keep_parts.append(block)
            continue
        idx = rng.choice(block.index.to_numpy(), size=n_take, replace=False)
        keep_parts.append(block.loc[idx])
    if not keep_parts:
        return df_split.sample(n=target_rows, random_state=sample_seed)
    out = pd.concat(keep_parts, axis=0)
    if len(out) > target_rows:
        out = out.sample(n=target_rows, random_state=sample_seed)
    return out


def _apply_capped_sampling_by_split(df, max_rows, sample_seed):
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    split_order = ["train", "val", "test"]
    split_counts = df["split"].value_counts()
    total = len(df)
    targets = {
        s: int(np.floor(max_rows * (float(split_counts.get(s, 0)) / total)))
        for s in split_order
    }
    assigned = sum(targets.values())
    remaining = max_rows - assigned
    if remaining > 0:
        fracs = {
            s: (max_rows * (float(split_counts.get(s, 0)) / total)) - targets[s]
            for s in split_order
        }
        for s in sorted(split_order, key=lambda x: fracs[x], reverse=True)[:remaining]:
            targets[s] += 1
    sampled = []
    for i, split in enumerate(split_order):
        split_df = df[df["split"] == split]
        sampled.append(_sample_split_rows(split_df, targets[split], sample_seed + i))
    return pd.concat(sampled, axis=0, ignore_index=True)


def prepare_model_dataset(
    dataset_path,
    static_path,
    out_path,
    random_seed=1,
    max_rows=None,
    sample_seed=None,
    sample_mode="system_stratified",
    contiguous_us_only=True,
):
    system_day_data = pd.read_csv(dataset_path, parse_dates=["date"], low_memory=False)
    system_day_data["system_id"] = system_day_data["system_id"].astype(str)

    systems_static = pd.read_csv(static_path, dtype={"system_id": str}, low_memory=False)
    merge_cols = [
        c
        for c in [
            "system_id",
            "kg_climate",
            "tracking",
            "type",
            "tilt",
            "azimuth",
            "latitude",
            "longitude",
            "elevation_m",
        ]
        if c in systems_static.columns
    ]
    static_features = systems_static[merge_cols].drop_duplicates(subset=["system_id"])
    system_day_data = system_day_data.merge(
        static_features,
        on="system_id",
        how="left",
        suffixes=("", "_site"),
    )

    system_day_data = system_day_data.loc[system_day_data["dc_capacity_kW"] > 0]
    system_day_data = system_day_data.loc[system_day_data["ghi_kwh_m2"] >= 0.05]
    system_day_data = system_day_data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["efficiency_kwh_per_kwp_per_kwh_m2"]
    )
    if contiguous_us_only:
        if "latitude" in system_day_data.columns and "longitude" in system_day_data.columns:
            lat = pd.to_numeric(system_day_data["latitude"], errors="coerce")
            lon = pd.to_numeric(system_day_data["longitude"], errors="coerce")
            conus_mask = lat.between(CONUS_LAT_MIN, CONUS_LAT_MAX) & lon.between(CONUS_LON_MIN, CONUS_LON_MAX)
            system_day_data = system_day_data.loc[conus_mask].copy()
        else:
            raise ValueError("contiguous_us_only=True requires latitude/longitude columns in prep input.")

    unique_system_ids = system_day_data["system_id"].unique()
    train_ids, val_ids, _ = _assign_system_splits(unique_system_ids, random_seed)

    def assign_split(sid):
        if sid in train_ids:
            return "train"
        if sid in val_ids:
            return "val"
        return "test"

    system_day_data["split"] = system_day_data["system_id"].map(assign_split)
    pre_counts = system_day_data["split"].value_counts().to_dict()

    if sample_seed is None:
        sample_seed = random_seed
    if sample_mode == "system_stratified":
        system_day_data = _apply_capped_sampling_by_split(
            system_day_data, max_rows=max_rows, sample_seed=sample_seed
        )
    elif sample_mode == "row_random":
        if max_rows is not None and max_rows > 0 and len(system_day_data) > max_rows:
            system_day_data = system_day_data.sample(n=max_rows, random_state=sample_seed)
    else:
        raise ValueError(f"Unsupported sample_mode={sample_mode}")

    system_day_data.to_csv(out_path, index=False)
    post_counts = system_day_data["split"].value_counts().to_dict()
    print("Split counts before sampling:", pre_counts)
    print("Split counts after sampling:", post_counts)
    return system_day_data


def parse_args():
    out = MODELING_OUTPUT_DIR
    p = argparse.ArgumentParser(description="prepare model_dataset_prep.csv")
    p.add_argument("--dataset-in", type=Path, default=out / "system_day_dataset.csv")
    p.add_argument("--static-in", type=Path, default=out / "systems_static.csv")
    p.add_argument("--prep-out", type=Path, default=out / "model_dataset_prep.csv")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-rows", type=int, default=75000)
    p.add_argument("--sample-seed", type=int)
    p.add_argument(
        "--sample-mode",
        type=str,
        default="system_stratified",
        choices=["system_stratified", "row_random"],
    )
    p.add_argument(
        "--contiguous-us-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only contiguous U.S. rows (exclude Alaska/Hawaii/territories).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = prepare_model_dataset(
        dataset_path=args.dataset_in,
        static_path=args.static_in,
        out_path=args.prep_out,
        random_seed=args.seed,
        max_rows=args.max_rows,
        sample_seed=args.sample_seed,
        sample_mode=args.sample_mode,
        contiguous_us_only=args.contiguous_us_only,
    )
    print(f"Saved {args.prep_out} rows={len(df)}")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()

