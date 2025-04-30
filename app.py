import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# --- Streamlit File Upload ---
st.title("Neural Network Dashboard - Asset Pricing")

file_path = 'prediction_output.csv'
your_df = pd.read_csv(file_path)

# --- Convert 'date' column to datetime and sort ---
your_df['date'] = pd.to_datetime(your_df['date'])
your_df = your_df.sort_values("date")

# --- Available Prediction Columns ---
available_columns = ['pred_mlp_32', 'pred_mlp_64_32', 'pred_mlp_128_64_32', 'pred_hgbr', 'pred_Lasso', 'pred_Ridge']

# --- Group by date using median ---
your_df = your_df.groupby('date')[['ret'] + available_columns].median().reset_index()

# --- Sidebar Selectors ---
selected_model = st.sidebar.selectbox("Model Type", available_columns)

# --- Check for required columns ---
if 'date' not in your_df.columns or 'ret' not in your_df.columns:
    st.error("Error: Your DataFrame must contain 'date' and 'ret' columns.")
    st.stop()

# --- Function to remove outliers within +/- 10% range ---
def remove_close_outliers(series):
    median_val = series.median()
    lower_bound = median_val - 0.10 * abs(median_val)
    upper_bound = median_val + 0.10 * abs(median_val)
    return series[(series >= lower_bound) & (series <= upper_bound)]

# --- Apply outlier removal to actual returns ---
cleaned_actual_returns = remove_close_outliers(your_df['ret'])

# --- Apply outlier removal to the selected prediction model ---
cleaned_predictions = remove_close_outliers(your_df[selected_model])

# --- Prepare DataFrame ---
dates = your_df['date'][cleaned_actual_returns.index.intersection(cleaned_predictions.index)]
actual_returns_plot = cleaned_actual_returns[cleaned_actual_returns.index.isin(dates.index)].clip(lower=-15, upper=15)
predictions_plot = cleaned_predictions[cleaned_predictions.index.isin(dates.index)].clip(lower=-15, upper=15)


df_prices_cleaned = pd.DataFrame({
    "Date": dates,
    "Actual": actual_returns_plot,
    "Predictions": predictions_plot
})

# --- Plot: Actual vs Predictions (Outliers Removed) ---
st.subheader("Predictions vs Actual (Outliers Removed within +/- 10%)")
fig_prices_cleaned, ax_prices_cleaned = plt.subplots(figsize=(10, 6))
ax_prices_cleaned.plot(df_prices_cleaned['Date'], df_prices_cleaned['Predictions'], label="Predictions", color='blue')
ax_prices_cleaned.plot(df_prices_cleaned['Date'], df_prices_cleaned['Actual'], label="Actual", color='orange')
ax_prices_cleaned.set_xlabel('Date')
ax_prices_cleaned.set_ylabel('Returns')
ax_prices_cleaned.set_title('Predictions vs Actual (Outliers Removed)')
ax_prices_cleaned.legend()
st.pyplot(fig_prices_cleaned)

# --- R² Score (Outliers Removed) ---
df_prices_cleaned = df_prices_cleaned.dropna(subset=['Actual', 'Predictions'])

r2_val_cleaned = r2_score(
    df_prices_cleaned['Actual'],
    df_prices_cleaned['Predictions']
)

st.markdown(f"### R² Score (Outliers Removed): {r2_val_cleaned:.4f}")

# --- Summary statistics (prediction column with outliers removed) ---
st.markdown("### Prediction Summary Statistics (Outliers Removed)")
st.write(df_prices_cleaned['Predictions'].describe())

# --- Original Plots and Stats for Comparison ---
st.subheader("Predictions vs Actual (Original - No Outlier Removal)")
st.pyplot(fig_prices)
st.markdown(f"### R² Score (Original): {r2_val:.4f}")
st.markdown("### Prediction Summary Statistics (Original)")
st.write(df_prices['Predictions'].describe())
