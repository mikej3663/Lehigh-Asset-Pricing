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

# --- Function to remove outliers within +/- 10% range of the median ---
def remove_close_outliers(series, tolerance=0.10): # Default tolerance is +/- 10%
    median_val = series.median()
    lower_bound = median_val - tolerance * abs(median_val)
    upper_bound = median_val + tolerance * abs(median_val)
    return series[(series >= lower_bound) & (series <= upper_bound)]

# --- Apply outlier removal to actual returns ---
cleaned_actual_returns = remove_close_outliers(your_df['ret'])
st.write(f"Shape of cleaned_actual_returns: {cleaned_actual_returns.shape}") # DEBUG

# --- Apply outlier removal to the selected prediction model ---
cleaned_predictions = remove_close_outliers(your_df[selected_model])
st.write(f"Shape of cleaned_predictions: {cleaned_predictions.shape}") # DEBUG

# --- Find intersection of indices ---
common_indices = cleaned_actual_returns.index.intersection(cleaned_predictions.index)
st.write(f"Number of common indices after outlier removal: {len(common_indices)}") # DEBUG

# --- Prepare DataFrame for Cleaned Data ---
dates_cleaned = your_df['date'].iloc[common_indices]
actual_returns_plot_cleaned = cleaned_actual_returns.loc[common_indices].clip(lower=-15, upper=15)
predictions_plot_cleaned = cleaned_predictions.loc[common_indices].clip(lower=-15, upper=15)

df_prices_cleaned = pd.DataFrame({
    "Date": dates_cleaned,
    "Actual": actual_returns_plot_cleaned,
    "Predictions": predictions_plot_cleaned
})
st.write("Shape of df_prices_cleaned before dropping NaNs:", df_prices_cleaned.shape) # DEBUG
df_prices_cleaned = df_prices_cleaned.dropna(subset=['Actual', 'Predictions'])
st.write("Shape of df_prices_cleaned after dropping NaNs:", df_prices_cleaned.shape) # DEBUG

# --- Plot: Actual vs Predictions (Outliers Removed within +/- 10%) ---
st.subheader("Predictions vs Actual (Outliers Removed within +/- 10%)")
if not df_prices_cleaned.empty:
    fig_prices_cleaned, ax_prices_cleaned = plt.subplots(figsize=(10, 6))
    ax_prices_cleaned.plot(df_prices_cleaned['Date'], df_prices_cleaned['Predictions'], label="Predictions", color='blue')
    ax_prices_cleaned.plot(df_prices_cleaned['Date'], df_prices_cleaned['Actual'], label="Actual", color='orange')
    ax_prices_cleaned.set_xlabel('Date')
    ax_prices_cleaned.set_ylabel('Returns')
    ax_prices_cleaned.set_title('Predictions vs Actual (Outliers Removed)')
    ax_prices_cleaned.legend()
    st.pyplot(fig_prices_cleaned)

    # --- R² Score (Outliers Removed) ---
    if df_prices_cleaned['Actual'].notna().any() and df_prices_cleaned['Predictions'].notna().any():
        r2_val_cleaned = r2_score(
            df_prices_cleaned['Actual'].values.reshape(-1, 1),
            df_prices_cleaned['Predictions'].values.reshape(-1, 1)
        )
        st.markdown(f"### R² Score (Outliers Removed): {r2_val_cleaned:.4f}")
    else:
        st.warning("Cannot calculate R² score after outlier removal (NaNs present).")

    # --- Summary statistics (prediction column with outliers removed) ---
    st.markdown("### Prediction Summary Statistics (Outliers Removed)")
    st.write(df_prices_cleaned['Predictions'].describe())
else:
    st.warning("No data to plot after outlier removal.")
    st.warning("Cannot calculate R² score or summary statistics for cleaned data.")

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
    ax_prices_set_title = ax_prices.set_title('Predictions vs Actual (Original)')
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
