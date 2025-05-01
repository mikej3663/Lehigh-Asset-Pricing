import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# --- Streamlit File Upload ---
st.title("Neural Network Dashboard - Asset Pricing")

file_path = 'prediction_output.csv'
try:
    your_df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error(f"Error: File not found at {file_path}")
    st.stop()

# --- Convert 'date' column to datetime and sort ---
try:
    your_df['date'] = pd.to_datetime(your_df['date'])
    your_df = your_df.sort_values("date")
except KeyError:
    st.error("Error: 'date' column not found in the CSV file.")
    st.stop()

# --- Available Prediction Columns ---
available_columns = ['pred_mlp_32', 'pred_mlp_64_32', 'pred_mlp_128_64_32', 'pred_hgbr', 'pred_Lasso', 'pred_Ridge']

# --- Group by date using median ---
try:
    your_df = your_df.groupby('date')[['ret'] + available_columns].median().reset_index()
except KeyError as e:
    st.error(f"Error: Column(s) {e} not found after grouping.")
    st.stop()

# --- Sidebar Selectors ---
selected_model = st.sidebar.selectbox("Model Type", available_columns)

# --- Check for required columns ---
if 'date' not in your_df.columns or 'ret' not in your_df.columns:
    st.error("Error: Your DataFrame must contain 'date' and 'ret' columns.")
    st.stop()

# --- Tolerance for outlier removal ---
outlier_tolerance = st.sidebar.slider("Outlier Tolerance (%)", 10, 100, 20) / 100.0

# --- Function to identify outlier indices based on median and tolerance ---
def identify_outlier_indices(series, tolerance):
    median_val = series.median()
    lower_bound = median_val - tolerance * abs(median_val)
    upper_bound = median_val + tolerance * abs(median_val)
    return series[~((series >= lower_bound) & (series <= upper_bound))].index

# --- Identify outlier indices for actual returns and predictions ---
actual_outlier_indices = identify_outlier_indices(your_df['ret'], outlier_tolerance)
predictions_outlier_indices = identify_outlier_indices(your_df[selected_model], outlier_tolerance)

# --- Combine all outlier indices ---
all_outlier_indices = actual_outlier_indices.union(predictions_outlier_indices)
st.write(f"Number of all outlier indices: {len(all_outlier_indices)}") # DEBUG

# --- Filter the original DataFrame to exclude rows with outliers in either series ---
df_cleaned_combined = your_df[~your_df.index.isin(all_outlier_indices)].copy()
st.write(f"Shape of df_cleaned_combined after combined outlier removal: {df_cleaned_combined.shape}") # DEBUG

# --- Prepare DataFrame for Cleaned Data (Combined) ---
if not df_cleaned_combined.empty:
    dates_cleaned = df_cleaned_combined['date']
    actual_returns_plot_cleaned = df_cleaned_combined['ret'].clip(lower=-15, upper=15)
    predictions_plot_cleaned = df_cleaned_combined[selected_model].clip(lower=-15, upper=15)

    df_prices_cleaned = pd.DataFrame({
        "Date": dates_cleaned,
        "Actual": actual_returns_plot_cleaned,
        "Predictions": predictions_plot_cleaned
    })
    st.write("Shape of df_prices_cleaned (combined) before dropping NaNs:", df_prices_cleaned.shape) # DEBUG
    df_prices_cleaned = df_prices_cleaned.dropna(subset=['Actual', 'Predictions'])
    st.write("Shape of df_prices_cleaned (combined) after dropping NaNs:", df_prices_cleaned.shape) # DEBUG

    # --- Plot: Actual vs Predictions (Outliers Removed - Combined) ---
    st.subheader(f"Predictions vs Actual (Outliers Removed - Combined +/- {outlier_tolerance * 100:.0f}%)")
    if not df_prices_cleaned.empty:
        fig_prices_cleaned, ax_prices_cleaned = plt.subplots(figsize=(10, 6))
        ax_prices_cleaned.plot(df_prices_cleaned['Date'], df_prices_cleaned['Predictions'], label="Predictions", color='blue')
        ax_prices_cleaned.plot(df_prices_cleaned['Date'], df_prices_cleaned['Actual'], label="Actual", color='orange')
        ax_prices_cleaned.set_xlabel('Date')
        ax_prices_cleaned.set_ylabel('Returns')
        ax_prices_cleaned.set_title('Predictions vs Actual (Outliers Removed - Combined)')
        ax_prices_cleaned.legend()
        st.pyplot(fig_prices_cleaned)

        # --- R² Score (Outliers Removed - Combined) ---
        if df_prices_cleaned['Actual'].notna().any() and df_prices_cleaned['Predictions'].notna().any():
            r2_val_cleaned = r2_score(
                df_prices_cleaned['Actual'].values.reshape(-1, 1),
                df_prices_cleaned['Predictions'].values.reshape(-1, 1)
            )
            st.markdown(f"### R² Score (Outliers Removed - Combined): {r2_val_cleaned:.4f}")
        else:
            st.warning("Cannot calculate R² score after combined outlier removal (NaNs present).")

        # --- Summary statistics (prediction column with outliers removed - Combined) ---
        st.markdown("### Prediction Summary Statistics (Outliers Removed - Combined)")
        st.write(df_prices_cleaned['Predictions'].describe())
    else:
        st.warning("No data to plot after combined outlier removal.")
        st.warning("Cannot calculate R² score or summary statistics for cleaned data (combined).")
else:
    st.warning("DataFrame is empty after combined outlier removal.")

# --- Prepare DataFrame for Original Data (for consistent plotting) ---
dates_original = your_df['date']
actual_returns_plot_original = your_df['ret'].clip(lower=-15, upper=15)
predictions_plot_original = your_df[selected_model].clip(lower=-15, upper=15)

df_prices_original = pd.DataFrame({
    "Date": dates_original,
    "Actual": actual_returns_plot_original,
    "Predictions": predictions_plot_original
})
st.write("Shape of df_prices_original before dropping NaNs:", df_prices_original.shape) # DEBUG
df_prices_original = df_prices_original.dropna(subset=['Actual', 'Predictions'])
st.write("Shape of df_prices_original after dropping NaNs:", df_prices_original.shape) # DEBUG

# --- Plot: Actual vs Predictions (Original - No Outlier Removal) ---
st.subheader("Predictions vs Actual (Original - No Outlier Removal)")
if not df_prices_original.empty:
    fig_prices, ax_prices = plt.subplots(figsize=(10, 6))
    ax_prices.plot(df_prices_original['Date'], df_prices_original['Predictions'], label="Predictions", color='blue')
    ax_prices.plot(df_prices_original['Date'], df_prices_original['Actual'], label="Actual", color='orange')
    ax_prices.set_xlabel('Date')
    ax_prices.set_ylabel('Returns')
    ax_prices.set_title('Predictions vs Actual (Original)')
    ax_prices.legend()
    st.pyplot(fig_prices)

    # --- R² Score (Original) ---
    if df_prices_original['Actual'].notna().any() and df_prices_original['Predictions'].notna().any():
        r2_val = r2_score(
            df_prices_original['Actual'].values.reshape(-1, 1),
            df_prices_original['Predictions'].values.reshape(-1, 1)
        )
        st.markdown(f"### R² Score (Original): {r2_val:.4f}")
    else:
        st.warning("Cannot calculate original R² score (NaNs present).")

    # --- Summary statistics (original prediction column) ---
    st.markdown("### Prediction Summary Statistics (Original)")
    st.write(df_prices_original['Predictions'].describe())
else:
    st.warning("No data to plot for original data.")
    st.warning("Cannot calculate original R² score or summary statistics.")
