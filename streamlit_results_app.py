from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from run_train_model import MODELING_OUTPUT_DIR, TARGET, train_model
from siting_heatmap import (
    CONUS_COVERAGE_LAT_MAX_MIN,
    NSRDB_US_DIRNAME,
    build_nsrdb_file_signature,
    draw_us_siting_heatmap,
    load_nsrdb_us_annual_weather,
    nsrdb_coordinate_coverage,
)


@st.cache_data(show_spinner=False)
def cached_nsrdb_weather(nsrdb_dir_str, file_signature):
    del file_signature
    return load_nsrdb_us_annual_weather(nsrdb_dir_str)


def load_saved_seed_default():
    bundle_path = MODELING_OUTPUT_DIR / "model_bundle.pkl"
    if not bundle_path.is_file():
        return 1
    try:
        with bundle_path.open("rb") as f:
            bundle = pickle.load(f)
        return int(bundle.get("seed", 1))
    except Exception:
        return 1


def draw_actual_vs_pred(y_test, preds, title):
    y = y_test.to_numpy(dtype=float)
    stack = np.concatenate([y, preds.astype(float)])
    vmin, vmax = np.nanpercentile(stack, [0.5, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(stack)), float(np.nanmax(stack))

    rng = np.random.default_rng(42)
    n = len(y)
    idx = rng.choice(n, size=min(25000, n), replace=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y[idx], preds[idx], s=6, alpha=0.25, linewidths=0)
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('Actual yield (kWh/kW/day)')
    ax.set_ylabel('Predicted yield')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def draw_residual_panels(x_test, y_test, preds, feature):
    residual = y_test.to_numpy(dtype=float) - preds

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].scatter(preds, residual, s=6, alpha=0.25, linewidths=0)
    axs[0].axhline(0.0, color='red', linestyle='--', linewidth=1)
    axs[0].set_title('Residual vs Predicted')
    axs[0].set_xlabel('Predicted')
    axs[0].set_ylabel('Residual (actual - predicted)')
    axs[0].grid(alpha=0.2)

    axs[1].scatter(x_test[feature], residual, s=6, alpha=0.25, linewidths=0)
    axs[1].axhline(0.0, color='red', linestyle='--', linewidth=1)
    axs[1].set_title(f'Residual vs {feature}')
    axs[1].set_xlabel(feature)
    axs[1].set_ylabel('Residual')
    axs[1].grid(alpha=0.2)

    cal_df = pd.DataFrame({'pred': preds, 'actual': y_test.to_numpy(dtype=float)})
    cal_df = cal_df[np.isfinite(cal_df['pred']) & np.isfinite(cal_df['actual'])].copy()
    q = min(20, max(5, len(cal_df) // 3000))
    cal_df['bin'] = pd.qcut(cal_df['pred'], q=q, duplicates='drop')
    cal = cal_df.groupby('bin', observed=True).agg(pred_mean=('pred', 'mean'), actual_mean=('actual', 'mean'))
    mn = float(min(cal['pred_mean'].min(), cal['actual_mean'].min()))
    mx = float(max(cal['pred_mean'].max(), cal['actual_mean'].max()))
    axs[2].plot(cal['pred_mean'], cal['actual_mean'], marker='o', linestyle='-')
    axs[2].plot([mn, mx], [mn, mx], 'r--', linewidth=1)
    axs[2].set_title('Binned actual vs predicted')
    axs[2].set_xlabel('Mean predicted')
    axs[2].set_ylabel('Mean actual')
    axs[2].grid(alpha=0.2)

    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title='Solar Yield Results', layout='wide')
    st.title('Solar Farm Yield Model: Results')
    with st.sidebar:
        st.header('Run Settings')
        csv_path = st.text_input('Prepared CSV path', str(MODELING_OUTPUT_DIR / 'model_dataset_prep.csv'))
        nsrdb_us_dir = st.text_input(
            'NSRDB US grid folder',
            str(Path.cwd() / NSRDB_US_DIRNAME),
        )
        n_repeats = st.slider('Permutation repeats', 3, 12, 6, 1)
        cv_folds = st.slider('Group CV folds', 3, 8, 5, 1)
        saved_seed_default = load_saved_seed_default()
        seed_base = st.number_input('Base seed', min_value=0, max_value=1_000_000, value=saved_seed_default, step=1)
        st.caption(
            'Permutation repeats: controls stability of feature-importance ranking. '
        )
        st.caption(
            'More folds give more reliable validation metrics, but take longer.'
        )

    if not Path(csv_path).is_file():
        st.stop()

    with st.spinner('training models'):
        fit = train_model(
            Path(csv_path),
            n_repeats=n_repeats,
            top_k=8,
            cv_folds=cv_folds,
            random_seed=int(seed_base),
        )

    st.subheader('Dataset Statistics')
    c1, c2, c3 = st.columns(3)
    c1.metric('Rows', f'{len(fit.df):,}')
    c2.metric('Features Used', len(fit.feature_cols))
    c3.metric('Best Validation Model', fit.best_name)
    st.dataframe(pd.DataFrame({'feature': fit.feature_cols}), use_container_width=True, height=220)

    st.subheader('Heatmap of Best Solar Farm Regions in Continental US')
    nsrdb_path = Path(nsrdb_us_dir)
    if not nsrdb_path.is_dir():
        st.stop()

    with st.spinner('loading NSRDB US grid weather'):
        signature = build_nsrdb_file_signature(nsrdb_path)
        if not signature:
            st.stop()
        coverage = nsrdb_coordinate_coverage(nsrdb_path)
        if coverage is None:
            st.stop()
        if coverage['lat_max'] < CONUS_COVERAGE_LAT_MAX_MIN:
            st.stop()
        weather_daily = cached_nsrdb_weather(str(nsrdb_path), signature)
        try:
            site_scores = fit.predict_grid_siting_scores(weather_daily)
        except ValueError:
            st.stop()

    cclip1, cclip2 = st.columns(2)
    with cclip1:
        clip_low = st.slider('Color clip low percentile', 0, 20, 2, 1)
    with cclip2:
        clip_high = st.slider('Color clip high percentile', 80, 100, 98, 1)
    if clip_low >= clip_high:
        clip_low, clip_high = 2, 98

    st.pyplot(draw_us_siting_heatmap(site_scores, clip_low=clip_low, clip_high=clip_high), clear_figure=True)

    st.markdown('**Top candidate coordinates (annual average prediction)**')
    st.dataframe(
        site_scores.head(25).rename(
            columns={
                'latitude': 'lat',
                'longitude': 'lon',
                'predicted_specific_yield_kwh_per_kwp_day': 'pred_yield_kwh_per_kwp_day',
                'predicted_annual_specific_yield_kwh_per_kwp': 'pred_annual_kwh_per_kwp',
            }
        ),
        use_container_width=True,
        height=360,
    )

    top_numeric = [c for c in fit.selected_vars if c in fit.num_cols][:4]
    st.subheader('Ridge Regression vs. Gradient Boosting')
    st.dataframe(fit.results, use_container_width=True)

    st.subheader('Predicted vs Actual (Test)')
    p1, p2 = st.columns(2)
    ridge_fig = draw_actual_vs_pred(
        fit.test[TARGET],
        fit.predictions['Ridge'],
        'Ridge Regression: Actual vs Predicted',
    )
    hgb_fig = draw_actual_vs_pred(
        fit.test[TARGET],
        fit.predictions['HistGradientBoosting'],
        'Gradient Boosting: Actual vs Predicted',
    )
    p1.pyplot(ridge_fig, clear_figure=True)
    p2.pyplot(hgb_fig, clear_figure=True)

    st.subheader(f'Permutation Importance of {fit.best_name} on Validation Set')
    imp_df = fit.importance.rename('importance').reset_index()
    imp_df.columns = ['feature', 'importance']
    st.dataframe(imp_df, use_container_width=True, height=360)

    focus_feature = top_numeric[0] if top_numeric else fit.num_cols[0]
    st.subheader('Residual Diagnostics')
    st.pyplot(
        draw_residual_panels(
            fit.test[fit.feature_cols],
            fit.test[TARGET],
            fit.predictions[fit.best_name],
            focus_feature,
        ),
        clear_figure=True,
    )


if __name__ == '__main__':
    main()
