import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np





# --- Load Your DataFrame ---
try:
    your_df = pd.read_csv('your_data.csv')
except FileNotFoundError:
    st.error("Error: 'your_data.csv' not found. Please make sure your data file is in the same directory.")
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
prediction_column = f"{selected_model}_{selected_fold}"

# --- Get Data from your DataFrame ---
# Assuming your DataFrame has a 'Date' column
if 'Date' not in your_df.columns:
    st.error("Error: Your DataFrame must contain a column named 'Date'.")
    st.stop()

dates = pd.to_datetime(your_df['Date'])

# Assuming your actual prices are in a column named 'Actual'
if 'Actual' not in your_df.columns:
    st.error("Error: Your DataFrame must contain a column named 'Actual'.")
    st.stop()

actual_prices = your_df['Actual'].values

# Get Predictions
if prediction_column not in your_df.columns:
    st.error(f"Error: The column '{prediction_column}' is not found in your DataFrame.")
    st.stop()

predictions = your_df[prediction_column].values
df_prices = pd.DataFrame({"Date": dates, "Actual": actual_prices, "Predictions": predictions})

# --- S&P 500 Data from your DataFrame ---
# Adjust these column names if they are different
sp500_actual_col = 'SP500_Actual'
sp500_predictions_col = 'SP500_Predictions'

if sp500_actual_col not in your_df.columns or sp500_predictions_col not in your_df.columns:
    st.warning(f"Warning: '{sp500_actual_col}' or '{sp500_predictions_col}' columns not found. S&P 500 comparison will not be displayed.")
    df_sp500 = pd.DataFrame({"Date": dates})
else:
    sp500_actual = your_df[sp500_actual_col].values
    sp500_predictions = your_df[sp500_predictions_col].values
    df_sp500 = pd.DataFrame({"Date": dates, "S&P 500 Actual": sp500_actual, "S&P 500 Predictions": sp500_predictions})

# --- Metrics Data (Adapt to your actual metrics columns) ---
def get_metric_value(df, model, fold, metric_name='R2'):
    metric_column = f"{model}_{fold}_{metric_name}"
    if metric_column in df.columns:
        return df[metric_column].iloc[0] # Assuming one row of metrics
    return np.nan

r_squared = get_metric_value(your_df, selected_model, selected_fold, metric_name='R2')
df_metrics = pd.DataFrame({"Metrics": ["R²"], "Value": [r_squared]})

# --- Dashboard Layout ---
st.title("Neural Network Dashboard - Asset Pricing")

# Row 1: Predictions vs Actual
st.subheader("Predictions vs Actual")
fig_prices, ax_prices = plt.subplots(figsize=(10, 6))
ax_prices.plot(df_prices['Date'], df_prices['Predictions'], label="Predictions", color='blue')
ax_prices.plot(df_prices['Date'], df_prices['Actual'], label="Actual", color='orange')
ax_prices.set_xlabel('Date')
ax_prices.set_ylabel('Price')
ax_prices.set_title('Predictions vs Actual')
ax_prices.legend()
st.pyplot(fig_prices)

# Row 2: Predictions vs Actual vs S&P 500
st.subheader("Predictions vs Actual vs S&P 500")
df_comparison = df_prices.merge(df_sp500, on="Date", how='left')

fig_comparison, ax_comparison = plt.subplots(figsize=(10, 6))
ax_comparison.plot(df_comparison['Date'], df_comparison['Predictions'], label="Predictions", color='blue')
ax_comparison.plot(df_comparison['Date'], df_comparison['Actual'], label="Actual", color='orange')
if 'S&P 500 Actual' in df_comparison.columns:
    ax_comparison.plot(df_comparison['Date'], df_comparison['S&P 500 Actual'], label="S&P 500 Actual", color='green')
if 'S&P 500 Predictions' in df_comparison.columns:
    ax_comparison.plot(df_comparison['Date'], df_comparison['S&P 500 Predictions'], label="S&P 500 Predictions", color='red')

ax_comparison.set_xlabel('Date')
ax_comparison.set_ylabel('Value')
ax_comparison.set_title('Predictions vs Actual vs S&P 500')
ax_comparison.legend()
st.pyplot(fig_comparison)

# Row 3: Metric Table
st.subheader("Metrics")
st.dataframe(df_metrics, hide_index=True)
