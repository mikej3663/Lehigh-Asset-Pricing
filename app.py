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

# --- Group by date using median ---
your_df = your_df.groupby('date')[['ret'] + available_columns].median().reset_index()

# --- Sidebar Selectors ---
selected_model = st.sidebar.selectbox("Model Type", available_columns)

# --- Check for required columns ---
if 'date' not in your_df.columns or 'ret' not in your_df.columns:
    st.error("Error: Your DataFrame must contain 'date' and 'ret' columns.")
    st.stop()

# --- Prepare DataFrame ---
dates = your_df['date']
actual_returns = your_df['ret'].clip(lower=-0.15, upper=0.15)
predictions = your_df[selected_model].clip(lower=-0.15, upper=0.15)

df_prices = pd.DataFrame({
    "Date": dates,
    "Actual": actual_returns,
    "Predictions": predictions
})

# --- Calculate Cumulative Returns ---
df_prices['Actual_Cumulative'] = (1 + df_prices['Actual']).cumprod() - 1
df_prices['Predictions_Cumulative'] = (1 + df_prices['Predictions']).cumprod() - 1

# --- Plot: Cumulative Actual vs Cumulative Predictions ---
st.subheader("Cumulative Predictions vs Actual")
fig_prices, ax_prices = plt.subplots(figsize=(10, 6))
ax_prices.plot(df_prices['Date'], df_prices['Predictions_Cumulative'], label="Predictions (Cumulative)", color='blue')
ax_prices.plot(df_prices['Date'], df_prices['Actual_Cumulative'], label="Actual (Cumulative)", color='orange')  
ax_prices.set_xlabel('Date')
ax_prices.set_ylabel('Cumulative Returns')
ax_prices.set_title('Cumulative Predictions vs Actual')
ax_prices.legend()
st.pyplot(fig_prices)

# --- R² Score ---
# Drop NaNs from columns to avoid mismatch
df_prices = df_prices.dropna(subset=['Actual_Cumulative', 'Predictions_Cumulative'])

r2_val = r2_score(
    df_prices['Actual_Cumulative'],
    df_prices['Predictions_Cumulative']
)

st.markdown(f"### R² Score: {r2_val:.4f}")

# --- Summary statistics (original prediction column) ---
st.markdown("### Prediction Summary Statistics")
st.write(df_prices['Predictions_Cumulative'].describe())
