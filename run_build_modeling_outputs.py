import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

epsilon = 1e-6

# GHI - Global Horizontal Irradiance
# DNI - Direct Normal Irradiance
# DHI - Diffuse Horizontal Irradiance
# CGHI - Clearsky Global Horizontal Irradiance
# Temp - Temperature
# RH - Relative Humidity
# Wind Speed - Wind Speed
# Wind Direction - Wind Direction

def root_dir():
    return Path.cwd().resolve()


def default_out_dir():
    return root_dir() / "modeling_output"


def _nsrdb_catalog(nsrdb_dir):
    
    pattern = re.compile(
        r"^(?P<location_id>.+)_(?P<lat>-?\d+\.?\d*)_(?P<lon>-?\d+\.?\d*)_2024\.csv$"
    )
    rows = []
    for file_path in nsrdb_dir.glob("*.csv"):
        match = pattern.match(file_path.name)
        if not match:
            continue
        rows.append(
            {
                "nsrdb_file": file_path.name,
                "nsrdb_location_id": match.group("location_id"),
                "nsrdb_latitude": float(match.group("lat")),
                "nsrdb_longitude": float(match.group("lon")),
            }
        )
    return pd.DataFrame(rows)


def _pv_daily_from_file(file_path):
    match = re.match(r"^(?P<sid>\d+)_ac_", file_path.name)
    if not match:
        return None

    system_id = match.group("sid")
    raw = pd.read_csv(file_path, encoding="utf-8")
    if raw.empty:
        return None

    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    energy_cols = [c for c in raw.columns if "ac_energy" in c.lower() and "daily_sum" in c.lower()]
    if not energy_cols:
        return None

    max_power_cols = [c for c in raw.columns if "ac_power" in c.lower() and "daily_max" in c.lower()]
    mean_power_cols = [c for c in raw.columns if "ac_power" in c.lower() and "daily_mean" in c.lower()]

    system_daily_energy = pd.DataFrame(
        {
            "date": raw["date"],
            "ac_energy_kwh": raw[energy_cols].sum(axis=1, min_count=1),
            "ac_power_daily_max_kw": raw[max_power_cols].mean(axis=1) if max_power_cols else np.nan,
            "ac_power_daily_mean_kw": raw[mean_power_cols].mean(axis=1) if mean_power_cols else np.nan,
            "system_id": system_id,
        }
    )
    return system_daily_energy.dropna(subset=["date"])


def _weather_daily_from_file(file_path):
    hourly_weather = pd.read_csv(file_path, skiprows=2, encoding="utf-8")
    hourly_weather["date"] = pd.to_datetime(
        {"year": hourly_weather["Year"], "month": hourly_weather["Month"], "day": hourly_weather["Day"]},
        errors="coerce",
    )
    
    expected_weather_cols = (
        "GHI",
        "Clearsky GHI",
        "DNI",
        "DHI",
        "Temperature",
        "Relative Humidity",
        "Wind Speed",
        "Wind Direction",
    )
    
    # GHI - Global Horizontal Irradiance
    # DNI - Direct Normal Irradiance
    # DHI - Diffuse Horizontal Irradiance
    # CGHI - Clearsky Global Horizontal Irradiance
    # Temp - Temperature
    # RH - Relative Humidity
    # Wind Speed - Wind Speed
    # Wind Direction - Wind Direction`

    for col in expected_weather_cols:
        if col not in hourly_weather.columns:
            hourly_weather[col] = np.nan

    hourly_weather["_ghi"] = pd.to_numeric(hourly_weather["GHI"], errors="coerce").fillna(0.0)
    hourly_weather["_cghi"] = pd.to_numeric(hourly_weather["Clearsky GHI"], errors="coerce").fillna(0.0)
    hourly_weather["_dni"] = pd.to_numeric(hourly_weather["DNI"], errors="coerce").fillna(0.0)
    hourly_weather["_dhi"] = pd.to_numeric(hourly_weather["DHI"], errors="coerce").fillna(0.0)
    hourly_weather["_temp"] = pd.to_numeric(hourly_weather["Temperature"], errors="coerce")
    hourly_weather["_rh"] = pd.to_numeric(hourly_weather["Relative Humidity"], errors="coerce")
    hourly_weather["_wind"] = pd.to_numeric(hourly_weather["Wind Speed"], errors="coerce")
    hourly_weather["_dir"] = pd.to_numeric(hourly_weather["Wind Direction"], errors="coerce")

    def circular_mean_degrees(series):
        radians = np.radians(series.dropna().to_numpy())
        if radians.size == 0:
            return np.nan
        return float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360.0)

    return hourly_weather.groupby("date", as_index=False).agg(
        ghi_kwh_m2=("_ghi", lambda s: s.sum() / 1000.0),
        clearsky_ghi_kwh_m2=("_cghi", lambda s: s.sum() / 1000.0),
        dni_kwh_m2=("_dni", lambda s: s.sum() / 1000.0),
        dhi_kwh_m2=("_dhi", lambda s: s.sum() / 1000.0),
        temp_c_mean=("_temp", "mean"),
        rh_pct_mean=("_rh", "mean"),
        wind_speed_m_s_mean=("_wind", "mean"),
        wind_direction_deg_mean=("_dir", circular_mean_degrees),
        hours_ghi_positive=("_ghi", lambda s: float((s > 0).sum())),
    )


def _print_csv_summary(label, csv_path, df):
    print(f"\n[{label}] -> {csv_path}")
    print(f"rows={len(df):,} cols={len(df.columns)}")
    print("columns:", ", ".join(df.columns))
    if not df.empty:
        print(df.head(3).to_string(index=False))


def build_modeling_outputs(systems_url, pv_dir, nsrdb_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    systems_catalog = pd.read_csv(systems_url, dtype={"system_id": str}, low_memory=False)
    systems_catalog["system_id"] = systems_catalog["system_id"].astype(str).str.strip()
    systems_catalog["latitude"] = pd.to_numeric(systems_catalog["latitude"], errors="coerce")
    systems_catalog["longitude"] = pd.to_numeric(systems_catalog["longitude"], errors="coerce")
    systems_catalog = systems_catalog.drop_duplicates(subset=["system_id"], keep="first")

    lat = systems_catalog["latitude"]
    lon = systems_catalog["longitude"]
    us_mask = (
        (lat.between(24.0, 49.5) & lon.between(-124.9, -66.0))
        | (lat.between(51.0, 71.5) & lon.between(-169.5, -129.0))
        | (lat.between(18.5, 22.8) & lon.between(-161.0, -154.5))
    )

    static_cols = [
        "system_id",
        "system_public_name",
        "site_location",
        "timezone_or_utc_offset",
        "latitude",
        "longitude",
        "elevation_m",
        "dc_capacity_kW",
        "kg_climate",
        "pvcz_composite",
        "pvcz_t_rack",
        "pvcz_t_roof",
        "pvcz_humidity",
        "pvcz_wind",
        "tracking",
        "type",
        "azimuth",
        "tilt",
        "first_timestamp",
        "last_timestamp",
        "years",
        "qa_status",
        "qa_issue",
    ]
    present_static_cols = [c for c in static_cols if c in systems_catalog.columns]
    systems_static = systems_catalog.loc[us_mask, present_static_cols].copy()

    nsrdb_catalog = _nsrdb_catalog(nsrdb_dir)
    if nsrdb_catalog.empty:
        raise ValueError("No NSRDB files matched expected naming pattern.")

    system_coords = np.radians(systems_static[["latitude", "longitude"]].astype(float).to_numpy())
    weather_coords = np.radians(nsrdb_catalog[["nsrdb_latitude", "nsrdb_longitude"]].astype(float).to_numpy())
    sys_lat, sys_lon = system_coords[:, 0][:, None], system_coords[:, 1][:, None]
    w_lat, w_lon = weather_coords[:, 0][None, :], weather_coords[:, 1][None, :]
    dlat, dlon = w_lat - sys_lat, w_lon - sys_lon
    a = np.sin(dlat / 2.0) ** 2 + np.cos(sys_lat) * np.cos(w_lat) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    distance_km_matrix = 6371.0 * c
    nearest_idx = np.argmin(distance_km_matrix, axis=1)

    system_weather_map = systems_static[["system_id"]].copy()
    system_weather_map["nsrdb_file"] = nsrdb_catalog.iloc[nearest_idx]["nsrdb_file"].to_numpy()
    system_weather_map["nsrdb_location_id"] = nsrdb_catalog.iloc[nearest_idx]["nsrdb_location_id"].to_numpy()
    system_weather_map["distance_km"] = distance_km_matrix[np.arange(distance_km_matrix.shape[0]), nearest_idx]
    system_weather_map["match_method"] = "nearest_haversine"

    systems_static_path = out_dir / "systems_static.csv"
    system_weather_map_path = out_dir / "system_weather_map.csv"
    systems_static.to_csv(systems_static_path, index=False)
    system_weather_map.to_csv(system_weather_map_path, index=False)
    _print_csv_summary("systems_static", systems_static_path, systems_static)
    _print_csv_summary("system_weather_map", system_weather_map_path, system_weather_map)

    us_system_ids = set(systems_static["system_id"])
    pv_system_daily_parts = []
    for file_path in sorted(pv_dir.glob("*.csv")):
        daily_block = _pv_daily_from_file(file_path)
        if daily_block is None or daily_block.empty:
            continue
        if daily_block["system_id"].iat[0] not in us_system_ids:
            continue
        daily_block["pv_source_file"] = file_path.name
        pv_system_daily_parts.append(daily_block)

    pv_system_daily = pd.concat(pv_system_daily_parts, ignore_index=True) if pv_system_daily_parts else pd.DataFrame()
    pv_system_daily_path = out_dir / "pv_system_daily.csv"
    pv_system_daily.to_csv(pv_system_daily_path, index=False)
    _print_csv_summary("pv_system_daily", pv_system_daily_path, pv_system_daily)

    file_to_location = (
        system_weather_map[["nsrdb_file", "nsrdb_location_id"]]
        .drop_duplicates("nsrdb_file")
        .set_index("nsrdb_file")["nsrdb_location_id"]
        .to_dict()
    )

    weather_location_daily_parts = []
    for nsrdb_file in sorted(file_to_location):
        file_path = nsrdb_dir / nsrdb_file
        if not file_path.is_file():
            continue
        weather_daily_block = _weather_daily_from_file(file_path)
        weather_daily_block["nsrdb_location_id"] = str(file_to_location[nsrdb_file])
        weather_daily_block["nsrdb_file"] = nsrdb_file
        weather_location_daily_parts.append(weather_daily_block)

    weather_location_daily = pd.concat(weather_location_daily_parts, ignore_index=True) if weather_location_daily_parts else pd.DataFrame()

    weather_location_daily_path = out_dir / "weather_location_daily.csv"

    weather_location_daily.to_csv(weather_location_daily_path, index=False)
    _print_csv_summary("weather_location_daily", weather_location_daily_path, weather_location_daily)

    dc_capacity_by_system = systems_static.set_index("system_id")["dc_capacity_kW"].astype(float)
    weather_for_join = weather_location_daily.drop(columns=["nsrdb_file"], errors="ignore")
    
    system_day_dataset = pv_system_daily.merge(system_weather_map, on="system_id", how="inner").merge(weather_for_join, on=["nsrdb_location_id", "date"], how="inner")

    system_day_dataset["dc_capacity_kW"] = system_day_dataset["system_id"].map(dc_capacity_by_system)

    system_day_dataset["specific_yield_kwh_per_kwp"] = system_day_dataset["ac_energy_kwh"] / (system_day_dataset["dc_capacity_kW"] + epsilon)
    
    system_day_dataset["ghi_resource_kwh_m2"] = system_day_dataset["ghi_kwh_m2"]

    system_day_dataset["efficiency_kwh_per_kwp_per_kwh_m2"] = (system_day_dataset["specific_yield_kwh_per_kwp"] / (system_day_dataset["ghi_resource_kwh_m2"] + epsilon))

    system_day_dataset["efficiency_vs_clearsky_kwh_per_kwp_per_kwh_m2"] = (system_day_dataset["specific_yield_kwh_per_kwp"] / (system_day_dataset["clearsky_ghi_kwh_m2"] + epsilon))

    system_day_dataset_path = out_dir / "system_day_dataset.csv"
    system_day_dataset.to_csv(system_day_dataset_path, index=False)
    _print_csv_summary("system_day_dataset", system_day_dataset_path, system_day_dataset)

    return {
        "systems_static": len(systems_static),
        "system_weather_map": len(system_weather_map),
        "pv_system_daily": len(pv_system_daily),
        "weather_location_daily": len(weather_location_daily),
        "system_day_dataset": len(system_day_dataset),
    }


def parse_args():
    root = root_dir()
    p = argparse.ArgumentParser(description="build modeling_output CSV artifacts from raw files")
    p.add_argument(
        "--systems-url",
        default="https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/systems_20250729.csv",
        help="PVDAQ systems catalog CSV URL",
    )
    p.add_argument("--pv-dir", type=Path, default=root / "pvdaq_downloads_2024")
    p.add_argument("--nsrdb-dir", type=Path, default=root / "NSRDB_2024_Data")
    p.add_argument("--out-dir", type=Path, default=default_out_dir())
    return p.parse_args()


def main():
    args = parse_args()

    counts = build_modeling_outputs(
        systems_url=args.systems_url,
        pv_dir=args.pv_dir,
        nsrdb_dir=args.nsrdb_dir,
        out_dir=args.out_dir,
    )

    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

