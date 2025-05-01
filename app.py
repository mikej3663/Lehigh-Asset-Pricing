import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# --- Streamlit File Upload ---
st.title("Neural Network Dashboard - Asset Pricing")

file_path = 'prediction_output3.csv'
your_df = pd.read_csv(file_path)

# --- Convert 'date' column to datetime and sort ---
your_df['date'] = pd.to_datetime(your_df['date'])
your_df = your_df.sort_values("date")

# --- Available Prediction Columns ---
available_columns = [
    'pred_mlp_64_32','pred_mlp_128_64_32','pred_mlp_256_128_64_32','pred_hgbr','pred_Lasso', 'pred_Ridge'
]

# Mapping for human-readable names
name_mapping = {
    'pred_mlp_64_32': 'Neural Network (1 Fold)',
    'pred_mlp_128_64_32': 'Neural Network (2 Folds)',
    'pred_mlp_256_128_64_32': 'Neural Network (3 Folds)',
    'pred_hgbr': 'HistGradientBoostingRegressor()',
    'pred_Lasso': 'Lasso',
    'pred_Ridge': 'Ridge'
}

# Reverse the mapping to map selected model back to original column names
reverse_mapping = {v: k for k, v in name_mapping.items()}

# Replace the column names using the mapping
new_available_columns = list(name_mapping.values())

# --- Sidebar Selectors ---
selected_model = st.sidebar.selectbox("Model Type", new_available_columns)

# Map selected model back to original column name for DataFrame access
selected_model_column = reverse_mapping[selected_model]

# --- Group by date using median ---
your_df = your_df.groupby('date')[['ret'] + available_columns].median().reset_index()

# --- Check for required columns ---
if 'date' not in your_df.columns or 'ret' not in your_df.columns:
    st.error("Error: Your DataFrame must contain 'date' and 'ret' columns.")
    st.stop()

# --- Prepare DataFrame ---
dates = your_df['date']
actual_returns = your_df['ret'].clip(lower=-15, upper=15)
predictions = your_df[selected_model_column].clip(lower=-15, upper=15)

df_prices = pd.DataFrame({
    "Date": dates,
    "Actual": actual_returns,
    "Predictions": predictions
})

# --- Plot: Actual vs Predictions (No smoothing) ---
st.subheader("
