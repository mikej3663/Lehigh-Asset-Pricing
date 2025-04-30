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
fold_types = ["Rolling", "Expanding"]
selected_fold = st.sidebar.selectbox("Fold Type (CV)", fold_types)

# --- Check for required columns ---
if 'date' not in your_df.columns or 'ret' not in your_df.columns:
    st.error("Error: Your DataFrame must contain 'date' and 'ret' columns.")
    st.stop()

# --- Prepare DataFrame ---
dates = your_df['date']
actual_returns = your_df['ret'].clip(lower=-15, upper=15)
predictions = your_df[selected_model].clip(lower=-15, upper=15)

df_prices = pd.DataFrame({
    "Date": dates,
    "Actual": actual_returns,
    "Predictions": predictions
})

# --- Smoothing based on fold type ---
if selected_fold == "Rolling":
    window_size = 5
    df_prices['Predictions_Smoothed'] = df_prices['Predictions'].rolling(window=window_size).mean()
elif selected_fold == "Expanding":
    df_prices['Predictions_Smoothed'] = df_prices['Predictions'].expanding().mean()

# --- Plot: Smoothed Predictions vs Actual ---
st.subheader("Predictions vs Actual")
fig_prices, ax_prices = plt.subplots(figsize=(10, 6))
ax_prices.plot(df_prices['Date'], df_prices['Predictions_Smoothed'], label="Predictions (Smoothed)", color='blue')
ax_prices.plot(df_prices['Date'], df_prices['Actual'], label="Actual", color='orange')  # No smoothing on Actual
ax_prices.set_xlabel('Date')
ax_prices.set_ylabel('Returns')
ax_prices.set_title('Predictions vs Actual')
ax_prices.legend()
st.pyplot(fig_prices)

# --- R² Score ---
# Drop NaNs from smoothed columns (due to rolling/expanding window)
df_prices = df_prices.dropna(subset=['Actual', 'Predictions_Smoothed'])

r2_val = r2_score(
    df_prices['Actual'],
    df_prices['Predictions_Smoothed']
)

st.markdown(f"### R² Score: {r2_val:.4f}")

# --- Summary statistics (original prediction column) ---
st.markdown("### Prediction Summary Statistics")
st.write(df_prices['Predictions'].describe())
