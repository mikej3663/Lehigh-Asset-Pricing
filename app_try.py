import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# --- File Upload ---
file_path = 'prediction_output.csv'  # Update this path accordingly

try:
    your_df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error(f"Error: The file '{file_path}' was not found in the directory.")
    st.stop()

# --- Checking function to verify if 'date' column exists ---
def check_date_column(df):
    if 'date' not in df.columns:
        st.error("Error: Your DataFrame must contain a column named 'date'.")
        print(f"The 'date' column is not found in the DataFrame.")
        print("Here are the column names in the DataFrame:", df.columns.tolist())
        st.stop()
    else:
        date_index = df.columns.get_loc('date')  # Get the index of the 'date' column
        st.write(f"The 'date' column is found at index {date_index} in the DataFrame.")
        print(f"The 'date' column is located at index {date_index} in the DataFrame.")

# Call the checking function
check_date_column(your_df)

# --- Available Prediction Columns ---
available_columns = ['pred_mlp_32', 'pred_mlp_64_32', 'pred_mlp_128_64_32', 'pred_hgbr', 'pred_Lasso', 'pred_Ridge']

# Group by 'date' and calculate the mean of prediction columns
your_df = your_df.groupby('date')[available_columns].mean()

# --- Sidebar Selectors ---
selected_model = st.sidebar.selectbox("Model Type", available_columns)
fold_types = ["Rolling", "Expanding"]
selected_fold = st.sidebar.selectbox("Fold Type (CV)", fold_types)

# --- Get Data from your DataFrame ---
dates = pd.to_datetime(your_df['date'])

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
