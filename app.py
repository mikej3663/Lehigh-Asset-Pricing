import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- File Upload ---
#file_path = 'prediction_output.csv'  # Adjust this path if needed
uploaded_file = st.file_uploader("Upload your prediction_output.csv", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Load CSV from uploaded file
your_df = pd.read_csv(uploaded_file)

your_df = pd.read_csv(file_path)

# --- Convert 'date' column to datetime and sort ---
your_df['date'] = pd.to_datetime(your_df['date'])
your_df = your_df.sort_values("date")

# --- Available Prediction Columns ---
available_columns = ['pred_mlp_32', 'pred_mlp_64_32', 'pred_mlp_128_64_32', 'pred_hgbr', 'pred_Lasso', 'pred_Ridge']

# --- Group by date using median to reduce distortion ---
your_df = your_df.groupby('date')[['ret'] + available_columns].median().reset_index()

# --- Sidebar Selectors ---
selected_model = st.sidebar.selectbox("Model Type", available_columns)
fold_types = ["Rolling", "Expanding"]
selected_fold = st.sidebar.selectbox("Fold Type (CV)", fold_types)

# --- Check for required columns ---
if 'date' not in your_df.columns or 'ret' not in your_df.columns:
    st.error("Error: Your DataFrame must contain 'date' and 'ret' columns.")
    st.stop()

# --- Prepare plotting data ---
dates = your_df['date']
actual_returns = your_df['ret'].values
predictions = your_df[selected_model].values

df_prices = pd.DataFrame({
    "Date": dates,
    "Actual": actual_returns,
    "Predictions": predictions
})

# --- Optional: Clip extreme values before smoothing for readability ---
df_prices[['Actual', 'Predictions']] = df_prices[['Actual', 'Predictions']].clip(lower=-15, upper=15)

# --- Apply rolling average smoothing ---
window_size = 5  # You can adjust this for more or less smoothing
df_prices['Actual_Smoothed'] = df_prices['Actual'].rolling(window=window_size).mean()
df_prices['Predictions_Smoothed'] = df_prices['Predictions'].rolling(window=window_size).mean()

# --- Dashboard Layout ---
st.title("Neural Network Dashboard - Asset Pricing")

# Show basic stats
st.write("Prediction Summary Statistics:")
st.write(df_prices['Predictions'].describe())

# --- Plot: Smoothed Predictions vs Actual ---
st.subheader("Predictions vs Actual")
fig_prices, ax_prices = plt.subplots(figsize=(10, 6))
ax_prices.plot(df_prices['Date'], df_prices['Predictions_Smoothed'], label="Predictions (Smoothed)", color='blue')
ax_prices.plot(df_prices['Date'], df_prices['Actual_Smoothed'], label="Actual (Smoothed)", color='orange')
ax_prices.set_xlabel('Date')
ax_prices.set_ylabel('Returns')
ax_prices.set_title('Predictions vs Actual')
ax_prices.legend()
st.pyplot(fig_prices)
