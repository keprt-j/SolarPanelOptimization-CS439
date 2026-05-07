# Solar Panel Optimization

This project builds a solar yield modeling pipeline from PVDAQ + NSRDB CSV data, compares two regressors, and serves results in Streamlit with a U.S. siting heatmap.

## Project Structure

- `run_build_modeling_outputs.py` - builds intermediate modeling CSVs from raw PV/weather data.
- `run_prepare_data.py` - filters, splits, and samples the final training dataset.
- `run_train_model.py` - trains models, saves metrics/feature importance, and saves a model bundle.
- `run_generate_heatmap_png.py` - generates offline heatmap PNG + top-25 site CSV.
- `streamlit_results_app.py` - interactive app for model diagnostics + siting results.
- `modeling_output/` - generated datasets + training artifacts.
- `heatmaps/` - generated heatmap outputs.

## Requirements

- Python 3.10+ (3.11 recommended)
- Packages used by the scripts/app:
  - `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `streamlit`

Install dependencies (if needed):

```bash
pip install numpy pandas scikit-learn matplotlib streamlit
```

## Data Inputs

Expected default folders in project root:

- `pvdaq_downloads_2024/` - PV system daily CSV files.
- `NSRDB_2024_Data/` - NSRDB weather CSV grid files for solar farm locations.
- `NSRDB_2024_US_Data/` - NSRDB weather CSV grid files for across the US.

Expected files in project root:
- `systems_20250729.csv` - solar farm system catalog.
- `NSRDB_location_file_map.json` - dictionary file that links NSRDB weather data to solar farm sites.

The build script also fetches a systems catalog from (csv file also available in the root directory):

- `https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/systems_20250729.csv`

## Run Pipeline (End-to-End)

From the project root:

1) Build modeling CSV artifacts

```bash
python run_build_modeling_outputs.py
```

2) Prepare final model dataset

```bash
python run_prepare_data.py
```

3) Train models and save artifacts

```bash
python run_train_model.py
```

4) (Optional) Generate static heatmap outputs

```bash
python run_generate_heatmap_png.py
```

5) Launch Streamlit app

```bash
streamlit run streamlit_results_app.py
```

## What Gets Generated

After running steps 1-3, `modeling_output/` contains:

- `systems_static.csv`
- `system_weather_map.csv`
- `pv_system_daily.csv`
- `weather_location_daily.csv`
- `system_day_dataset.csv`
- `model_dataset_prep.csv`
- `model_metrics.csv`
- `model_importance.csv`
- `model_bundle.pkl` (includes saved seed + best model metadata)

Heatmap script outputs to `heatmaps/`:

- `conus_heatmap_log1p_<timestamp>.png`
- `conus_top25_log1p_<timestamp>.csv`

## Models Used

- `Ridge` (linear baseline, regularized)
- `HistGradientBoostingRegressor` (nonlinear ensemble)

Evaluation includes grouped CV by `system_id`, holdout test metrics, and permutation feature importance.
