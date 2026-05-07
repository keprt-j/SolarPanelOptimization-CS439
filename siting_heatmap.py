"""NSRDB grid loading + siting predictions from a trained ModelingResult."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.tri import LinearTriInterpolator, Triangulation

from run_train_model import ModelingResult

_ROOT = Path(__file__).resolve().parent

US_LAT_MIN, US_LAT_MAX = 24.5, 49.5
US_LON_MIN, US_LON_MAX = -124.8, -66.9
NSRDB_US_DIRNAME = "NSRDB_2024_Data"
CONUS_COVERAGE_LAT_MAX_MIN = 48.5
US_BASEMAP_IMAGE = _ROOT / "heatmaps" / "USA_location_map.svg.png"


def build_nsrdb_file_signature(nsrdb_dir):
    files = sorted(Path(nsrdb_dir).glob("*.csv"))
    return tuple((f.name, int(f.stat().st_mtime), int(f.stat().st_size)) for f in files)


def _parse_lat_lon_from_stem(stem):
    parts = stem.split("_")
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None
    if len(parts) >= 3:
        try:
            return float(parts[-3]), float(parts[-2])
        except ValueError:
            return None
    return None


def nsrdb_coordinate_coverage(nsrdb_dir):
    lats, lons = [], []
    for file_path in sorted(Path(nsrdb_dir).glob("*.csv")):
        parsed = _parse_lat_lon_from_stem(file_path.stem)
        if parsed is None:
            continue
        lat, lon = parsed
        lats.append(lat)
        lons.append(lon)
    if not lats:
        return None
    return {
        "lat_min": float(min(lats)),
        "lat_max": float(max(lats)),
        "lon_min": float(min(lons)),
        "lon_max": float(max(lons)),
        "count": len(lats),
    }


def _circular_mean_degrees(values):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    radians = np.radians(arr)
    return float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360.0)


def load_nsrdb_us_annual_weather(nsrdb_dir):
    nsrdb_dir = Path(nsrdb_dir)
    files = sorted(nsrdb_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame()

    rows = []
    for i, file_path in enumerate(files):
        parsed = _parse_lat_lon_from_stem(file_path.stem)
        if parsed is None:
            continue
        lat, lon = parsed
        if not (US_LAT_MIN <= lat <= US_LAT_MAX and US_LON_MIN <= lon <= US_LON_MAX):
            continue

        sample_cols = pd.read_csv(file_path, skiprows=2, nrows=1).columns.tolist()
        daily_compact_cols = {
            "Year",
            "Month",
            "Day",
            "ghi_kwh_m2",
            "clearsky_ghi_kwh_m2",
            "dni_kwh_m2",
            "dhi_kwh_m2",
            "temp_c_mean",
            "rh_pct_mean",
            "wind_speed_m_s_mean",
            "wind_direction_deg_mean",
            "hours_ghi_positive",
        }
        if daily_compact_cols.issubset(set(sample_cols)):
            daily_agg = pd.read_csv(
                file_path,
                skiprows=2,
                usecols=list(daily_compact_cols),
                low_memory=False,
            )
            daily_agg["date"] = pd.to_datetime(
                {"year": daily_agg["Year"], "month": daily_agg["Month"], "day": daily_agg["Day"]},
                errors="coerce",
            )
            daily_agg = daily_agg[
                [
                    "date",
                    "ghi_kwh_m2",
                    "clearsky_ghi_kwh_m2",
                    "dni_kwh_m2",
                    "dhi_kwh_m2",
                    "temp_c_mean",
                    "rh_pct_mean",
                    "wind_speed_m_s_mean",
                    "wind_direction_deg_mean",
                    "hours_ghi_positive",
                ]
            ].copy()
        else:
            usecols = [
                "Year",
                "Month",
                "Day",
                "GHI",
                "Clearsky GHI",
                "DNI",
                "DHI",
                "Temperature",
                "Relative Humidity",
                "Wind Speed",
                "Wind Direction",
            ]
            hourly = pd.read_csv(file_path, skiprows=2, usecols=usecols, low_memory=False)
            daily = pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        {"year": hourly["Year"], "month": hourly["Month"], "day": hourly["Day"]},
                        errors="coerce",
                    ),
                    "_ghi": pd.to_numeric(hourly["GHI"], errors="coerce").fillna(0.0),
                    "_cghi": pd.to_numeric(hourly["Clearsky GHI"], errors="coerce").fillna(0.0),
                    "_dni": pd.to_numeric(hourly["DNI"], errors="coerce").fillna(0.0),
                    "_dhi": pd.to_numeric(hourly["DHI"], errors="coerce").fillna(0.0),
                    "_temp": pd.to_numeric(hourly["Temperature"], errors="coerce"),
                    "_rh": pd.to_numeric(hourly["Relative Humidity"], errors="coerce"),
                    "_wind": pd.to_numeric(hourly["Wind Speed"], errors="coerce"),
                    "_dir": pd.to_numeric(hourly["Wind Direction"], errors="coerce"),
                }
            )

            daily_agg = daily.groupby("date", as_index=False).agg(
                ghi_kwh_m2=("_ghi", lambda s: s.sum() / 1000.0),
                clearsky_ghi_kwh_m2=("_cghi", lambda s: s.sum() / 1000.0),
                dni_kwh_m2=("_dni", lambda s: s.sum() / 1000.0),
                dhi_kwh_m2=("_dhi", lambda s: s.sum() / 1000.0),
                temp_c_mean=("_temp", "mean"),
                rh_pct_mean=("_rh", "mean"),
                wind_speed_m_s_mean=("_wind", "mean"),
                wind_direction_deg_mean=("_dir", _circular_mean_degrees),
                hours_ghi_positive=("_ghi", lambda s: float((s > 0).sum())),
            )
        daily_agg = daily_agg[daily_agg["date"].notna()].copy()
        if daily_agg.empty:
            continue

        daily_agg["latitude"] = lat
        daily_agg["longitude"] = lon
        daily_agg["grid_id"] = i
        rows.append(daily_agg)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _choose_default(series):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        return float(s.median())
    mode = series.dropna().mode()
    return mode.iat[0] if len(mode) else np.nan


def build_heatmap_inference_frame(fit: ModelingResult, weather_daily):
    if weather_daily.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_defaults = fit.df.copy()
    defaults = {col: _choose_default(df_defaults[col]) for col in fit.feature_cols if col in df_defaults.columns}
    defaults.setdefault("distance_km", 0.0)
    defaults.setdefault("dc_capacity_kW", 1.0)

    infer = weather_daily.copy()
    day = infer["date"].dt.dayofyear.astype(float)
    angle = (2.0 * np.pi * day) / 365.25
    infer["day_of_year_sin"] = np.sin(angle)
    infer["day_of_year_cos"] = np.cos(angle)
    rad = np.radians(pd.to_numeric(infer["wind_direction_deg_mean"], errors="coerce"))
    infer["wind_dir_sin"] = np.sin(rad)
    infer["wind_dir_cos"] = np.cos(rad)

    for col in fit.feature_cols:
        if col not in infer.columns:
            infer[col] = defaults.get(col, np.nan)

    infer_x = infer[fit.feature_cols].copy()
    for col in fit.num_cols:
        if col in infer_x.columns:
            infer_x[col] = pd.to_numeric(infer_x[col], errors="coerce")
    for col in fit.cat_cols:
        if col in infer_x.columns:
            infer_x[col] = infer_x[col].astype(str).fillna("unknown")

    geo = infer[["grid_id", "latitude", "longitude", "date"]].copy()
    return infer_x, geo


def predict_in_chunks(model, x, chunk_size=50000):
    preds = np.empty(len(x), dtype=float)
    if len(x) == 0:
        return preds
    for start in range(0, len(x), chunk_size):
        end = min(start + chunk_size, len(x))
        preds[start:end] = model.predict(x.iloc[start:end])
    return preds


def grid_siting_scores(fit: ModelingResult, weather_daily, chunk_size=50000):
    """Best-model predictions aggregated per NSRDB grid cell (daily yield → cell mean)."""
    if weather_daily.empty:
        raise ValueError("No usable NSRDB weather rows were parsed from the selected folder.")

    infer_x, infer_geo = build_heatmap_inference_frame(fit, weather_daily)
    missing = [c for c in fit.feature_cols if c not in infer_x.columns]
    if missing:
        raise ValueError(f"Heatmap inference schema is missing required model features: {missing}")

    row_valid = np.isfinite(infer_x.select_dtypes(include=[np.number]).to_numpy()).all(axis=1)
    infer_x = infer_x.loc[row_valid].reset_index(drop=True)
    infer_geo = infer_geo.loc[row_valid].reset_index(drop=True)
    if infer_x.empty:
        raise ValueError("All inference rows were invalid after numeric sanitization.")

    daily_preds = predict_in_chunks(fit.best_model, infer_x, chunk_size=chunk_size)
    daily_preds = np.maximum(daily_preds, 0.0)
    if not np.isfinite(daily_preds).any():
        raise ValueError("Model returned non-finite predictions for all grid rows.")

    pred_daily_df = infer_geo.copy()
    pred_daily_df["predicted_specific_yield_kwh_per_kwp_day"] = daily_preds
    pred_daily_df["predicted_annual_specific_yield_kwh_per_kwp"] = daily_preds * 365.25
    return (
        pred_daily_df.groupby(["grid_id", "latitude", "longitude"], as_index=False)
        .agg(
            predicted_specific_yield_kwh_per_kwp_day=("predicted_specific_yield_kwh_per_kwp_day", "mean"),
            predicted_annual_specific_yield_kwh_per_kwp=("predicted_annual_specific_yield_kwh_per_kwp", "mean"),
        )
        .query("@US_LAT_MIN <= latitude <= @US_LAT_MAX and @US_LON_MIN <= longitude <= @US_LON_MAX")
        .sort_values("predicted_specific_yield_kwh_per_kwp_day", ascending=False)
        .reset_index(drop=True)
    )


def _load_basemap():
    if US_BASEMAP_IMAGE.is_file():
        return plt.imread(US_BASEMAP_IMAGE)
    return None


def draw_us_siting_heatmap(site_scores, clip_low, clip_high):
    values = site_scores["predicted_specific_yield_kwh_per_kwp_day"].to_numpy(dtype=float)
    vmin, vmax = np.nanpercentile(values, [clip_low, clip_high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))

    lons = site_scores["longitude"].to_numpy(dtype=float)
    lats = site_scores["latitude"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6.2), facecolor="black")
    ax.set_facecolor("black")

    if len(site_scores) >= 3:
        tri = Triangulation(lons, lats)
        interp = LinearTriInterpolator(tri, values)
        gx = np.linspace(US_LON_MIN, US_LON_MAX, 900)
        gy = np.linspace(US_LAT_MIN, US_LAT_MAX, 520)
        gx_mesh, gy_mesh = np.meshgrid(gx, gy)
        gz = np.asarray(interp(gx_mesh, gy_mesh), dtype=float)

        basemap = _load_basemap()
        if basemap is not None:
            ax.imshow(
                basemap,
                extent=(US_LON_MIN, US_LON_MAX, US_LAT_MIN, US_LAT_MAX),
                aspect="auto",
                alpha=0.88,
                zorder=1,
            )

        cf = ax.contourf(
            gx_mesh,
            gy_mesh,
            gz,
            levels=24,
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            alpha=0.62,
            zorder=2,
        )
        sc = ax.scatter(
            lons,
            lats,
            c=values,
            s=16,
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            alpha=0.58,
            linewidths=0,
            zorder=3,
            rasterized=True,
        )
        cb = fig.colorbar(cf, ax=ax, fraction=0.03, pad=0.02)
    else:
        sc = ax.scatter(
            lons,
            lats,
            c=values,
            s=20,
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            alpha=0.62,
            linewidths=0,
            rasterized=True,
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Predicted specific yield (kWh/kW/day)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.get_yticklabels(), color="white")
    ax.set_xlim(US_LON_MIN, US_LON_MAX)
    ax.set_ylim(US_LAT_MIN, US_LAT_MAX)
    ax.set_xlabel("Longitude", color="white")
    ax.set_ylabel("Latitude", color="white")
    ax.set_title("Continental U.S. Solar Siting Heatmap", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
