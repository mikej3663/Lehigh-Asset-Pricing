import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# --- File Upload ---
file_path = 'prediction_output.csv'  # Update this path accordingly


your_df = pd.read_csv(file_path)

your_df = your_df.rename(columns={'date': 'date'})

# --- Available Prediction Columns ---
available_columns = ['pred_mlp_32', 'pred_mlp_64_32', 'pred_mlp_128_64_32', 'pred_hgbr', 'pred_Lasso', 'pred_Ridge']

your_df = your_df.groupby('date')[['pred_mlp_32', 'pred_mlp_64_32', 'pred_mlp_128_64_32', 'pred_hgbr', 'pred_Lasso', 'pred_Ridge']].mean()

# --- Sidebar Selectors ---
selected_model = st.sidebar.selectbox("Model Type", available_columns)
fold_types = ["Rolling", "Expanding"]
selected_fold = st.sidebar.selectbox("Fold Type (CV)", fold_types)

# --- Get Data from your DataFrame ---
# Assuming your DataFrame has a 'date' column


if 'date' not in your_df.columns:
    st.error("Error: Your DataFrame must contain a column named 'date'.")
    st.stop()

dates = your_df['date']

# Assuming your actual values are in 'ret' column
if 'ret' not in your_df.columns:
    st.error("Error: Your DataFrame must contain a column named 'ret'.")
    st.stop()

actual_returns = your_df['ret'].values

# Get Predictions
prediction_column = selected_model  # Dynamically set the selected prediction column
if prediction_column not in your_df.columns:
    st.error(f"Error: The column '{prediction_column}' is not found in your DataFrame.")
    st.stop()

predictions = your_df[prediction_column].values
df_prices = pd.DataFrame({"Date": dates, "Actual": actual_returns, "Predictions": predictions})

# --- Dashboard Layout ---
st.title("Neural Network Dashboard - Asset Pricing")

# Row 1: Predictions vs Actual
st.subheader("Predictions vs Actual")
fig_prices, ax_prices = plt.subplots(figsize=(10, 6))
ax_prices.plot(df_prices['Date'], df_prices['Predictions'], label="Predictions", color='blue')
ax_prices.plot(df_prices['Date'], df_prices['Actual'], label="Actual", color='orange')
ax_prices.set_xlabel('Date')
ax_prices.set_ylabel('Returns')
ax_prices.set_title('Predictions vs Actual')
ax_prices.legend()
st.pyplot(fig_prices)
