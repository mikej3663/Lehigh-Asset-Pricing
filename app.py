import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Load Your DataFrame ---
try:
    your_df = pd.read_csv('prediction_output.csv')  # Updated path to the uploaded file
except FileNotFoundError:
    st.error("Error: 'prediction_output.csv' not found. Please make sure your data file is in the correct path.")
    st.stop()

# --- Define Model Types and Fold Types ---
model_types = ["OLS", "Lasso", "Ridge", "RandomForest", "HistGradientBoostingRegressor",
               "Neural Net 1 Layer", "Neural Net 2 Layers", "Neural Net 3 Layers",
               "Neural Net 4 Layers", "Neural Net 5 Layers"]
fold_types = ["Rolling", "Expanding"]

# --- Sidebar Selectors ---
selected_model = st.sidebar.selectbox("Model Type", model_types)
selected_fold = st.sidebar.selectbox("Fold Type (CV)", fold_types)

# --- Construct the Prediction Column Name ---
prediction_column = f"pred_{selected_model}"

# --- Get Data from your DataFrame ---
# Assuming your DataFrame has a 'date' column
if 'date' not in your_df.columns:
    st.error("Error: Your DataFrame must contain a column named 'date'.")
    st.stop()

dates = pd.to_datetime(your_df['date'])

# Assuming your actual values are in 'ret' column
if 'ret' not in your_df.columns:
    st.error("Error: Your DataFrame must contain a column named 'ret'.")
    st.stop()

actual_returns = your_df['ret'].values

# Get Predictions
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
