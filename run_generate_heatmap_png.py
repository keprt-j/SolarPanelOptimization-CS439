from pathlib import Path
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_train_model import train_model
from siting_heatmap import (
    CONUS_COVERAGE_LAT_MAX_MIN,
    draw_us_siting_heatmap,
    load_nsrdb_us_annual_weather,
    nsrdb_coordinate_coverage,
)


ROOT_DIR = Path.cwd()
PREP_CSV = ROOT_DIR / "modeling_output" / "model_dataset_prep.csv"
NSRDB_DIR = ROOT_DIR / "NSRDB_2024_Data"
OUT_DIR = ROOT_DIR / "heatmaps"
SEED = 1


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    coverage = nsrdb_coordinate_coverage(NSRDB_DIR)
    if coverage is None:
        raise SystemExit("Could not parse lat/lon from NSRDB filenames.")
    print("Coverage:", coverage)
    if coverage["lat_max"] < CONUS_COVERAGE_LAT_MAX_MIN:
        raise SystemExit(
            f"Coverage too low for full CONUS (lat_max={coverage['lat_max']:.2f}). "
            "Provide a northern-U.S. complete weather grid."
        )

    fit = train_model(
        PREP_CSV,
        n_repeats=3,
        cv_folds=3,
        target_transform="log1p",
        fast_mode=True,
        random_seed=SEED,
    )
    weather_daily = load_nsrdb_us_annual_weather(NSRDB_DIR)
    site_scores = fit.predict_grid_siting_scores(weather_daily)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_png = OUT_DIR / f"conus_heatmap_log1p_{stamp}.png"
    out_top = OUT_DIR / f"conus_top25_log1p_{stamp}.csv"
    fig = draw_us_siting_heatmap(site_scores, clip_low=2, clip_high=98)
    fig.savefig(out_png, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    site_scores.head(25).to_csv(out_top, index=False)

    print("Saved PNG:", out_png)
    print("Saved Top-25 CSV:", out_top)
    print("Pred min/max:", float(site_scores["predicted_specific_yield_kwh_per_kwp_day"].min()), float(site_scores["predicted_specific_yield_kwh_per_kwp_day"].max()))


if __name__ == "__main__":
    main()
