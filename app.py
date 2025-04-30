import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- File Upload ---
file_path = 'prediction_output.csv'  # Adjust this path if needed
your_df = pd.read_csv(file_path)

# --- Convert 'date' column to datetime ---
your_df['date'] = pd.to_datetime(your_df['date'])

# --- Available Prediction Columns ---
available_columns = ['pred_mlp_32', 'pred_mlp_64_32', 'pred_mlp_128_64_32', 'pred_hgbr', 'pred_Lasso', 'pred_Ridge']

# --- Use median to reduce distortion from outliers ---
your_df = your_df.groupby('date')[['ret'] + available_columns].median().reset_index()

# --- Sidebar Selectors ---
selected_model = st.sidebar.selectbox("Model Type", available_columns)
fold_types = ["Rolling", "Expanding"]
selected_fold = st.sidebar.selectbox("Fold Type (CV)", fold_types)

# --- Get Data from your DataFrame ---
if 'date' not in your_df.columns or 'ret' not in your_df.columns:
    st.error("Error: Your DataFrame must contain 'date' and 'ret' columns.")
    st.stop()

dates = your_df['date']
actual_returns = your_df['ret'].values

# --- Get Predictions ---
prediction_column = selected_model
if prediction_column not in your_df.columns:
    st.error(f"Error: The column '{prediction_column}' is not found in your DataFrame.")
    st.stop()

predictions = your_df[prediction_column].values

# --- Create DataFrame for plotting ---
df_prices = pd.DataFrame({"Date": dates, "Actual": actual_returns, "Predictions": predictions})

# Optional: Clip extreme values for visualization clarity (adjust bounds as needed)
df_prices[['Actual', 'Predictions']] = df_prices[['Actual', 'Predictions']].clip(lower=-0.5, upper=0.5)

# --- Dashboard Layout ---
st.title("Neural Network Dashboard - Asset Pricing")

# Show basic stats
st.write("Prediction Summary Statistics:")
st.write(df_prices['Predictions'].describe())

# --- Plot: Predictions vs Actual ---
st.subheader("Predictions vs Actual")
fig_prices, ax_prices = plt.subplots(figsize=(10, 6))
ax_prices.plot(df_prices['Date'], df_prices['Predictions'], label="Predictions", color='blue')
ax_prices.plot(df_prices['Date'], df_prices['Actual'], label="Actual", color='orange')
ax_prices.set_xlabel('Date')
ax_prices.set_ylabel('Returns')
ax_prices.set_title('Predictions vs Actual')
ax_prices.legend()
st.pyplot(fig_prices)
