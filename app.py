import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid
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

# Mapping for human-readable names (for dropdown display only)
name_mapping = {
    'pred_mlp_64_32': 'Neural Network (1 Fold)',
    'pred_mlp_128_64_32': 'Neural Network (2 Folds)',
    'pred_mlp_256_128_64_32': 'Neural Network (3 Folds)',
    'pred_hgbr': 'HistGradientBoostingRegressor()',
    'pred_Lasso': 'Lasso',
    'pred_Ridge': 'Ridge'
}

# --- Group by date using median ---
your_df = your_df.groupby('date')[['ret'] + available_columns].median().reset_index()

# --- Sidebar Selectors ---
# Dropdown for Model Matrix (None or Model)
model_matrix_options = ['None'] + list(name_mapping.values())
selected_model_matrix = st.sidebar.selectbox("Model Matrix", model_matrix_options, index=0)

# Dropdown for Confusion Matrices (None or Models)
confusion_matrix_options = ['None', 'NN1', 'NN2', 'NN3', 'HGBR', 'Lasso', 'Ridge']
selected_conf_matrix = st.sidebar.selectbox("Confusion Matrices", confusion_matrix_options, index=0)

# Display the introductory message
st.markdown("### Welcome")

# --- Check for required columns ---
if 'date' not in your_df.columns or 'ret' not in your_df.columns:
    st.error("Error: Your DataFrame must contain 'date' and 'ret' columns.")
    st.stop()

# --- Prepare DataFrame ---
dates = your_df['date']
actual_returns = your_df['ret'].clip(lower=-15, upper=15)
predictions = your_df[selected_model_matrix].clip(lower=-15, upper=15) if selected_model_matrix != 'None' else None

df_prices = pd.DataFrame({
    "Date": dates,
    "Actual": actual_returns,
    "Predictions": predictions
})

# --- Plot: Actual vs Predictions (No smoothing) ---
if selected_model_matrix != 'None':
    st.subheader(f"Predictions vs Actual ({selected_model_matrix})")
    fig_prices, ax_prices = plt.subplots(figsize=(10, 6))
    ax_prices.plot(df_prices['Date'], df_prices['Predictions'], label="Predictions", color='blue')
    ax_prices.plot(df_prices['Date'], df_prices['Actual'], label="Actual", color='orange')  # No smoothing on Actual
    ax_prices.set_xlabel('Date')
    ax_prices.set_ylabel('Returns')
    ax_prices.set_title('Predictions vs Actual')
    ax_prices.legend()
    st.pyplot(fig_prices)

# --- R² Score ---
# Drop NaNs from columns to avoid mismatch
df_prices = df_prices.dropna(subset=['Actual', 'Predictions'])

r2_val = r2_score(
    df_prices['Actual'],
    df_prices['Predictions']
)

st.markdown(f"### R² Score: {r2_val:.4f}")

# --- Summary statistics (original prediction column) ---
st.markdown("### Prediction Summary Statistics")
st.write(df_prices['Predictions'].describe())

# --- Confusion Matrices (Generate only when selected model is chosen) ---
if selected_conf_matrix != 'None':
    # Assuming bigresults is loaded from a file for confusion matrix (adjust file path accordingly)
    bigresults = pd.read_csv('predictions/portfolio4.csv')

    # Define model_dict and params (adjust according to your needs)
    model_dict = {'NN1': 'pred_mlp_64_32', 'NN2': 'pred_mlp_128_64_32', 'NN3': 'pred_mlp_256_128_64_32', 'HGBR': 'pred_hgbr', 'Lasso': 'pred_Lasso', 'Ridge': 'pred_Ridge'}
    params = {}  # Assume params are defined elsewhere or imported as needed
    nport = 5

    model_name = model_dict[selected_conf_matrix]

    # True portfolio: based on actual returns
    bigresults['true_port'] = bigresults.groupby('date')['ret'].transform(
        lambda x: pd.qcut(x, nport, labels=False, duplicates='drop') + 1
    )

    # Predicted portfolio: based on model signal
    port_col = f'port_{model_name}'

    if port_col not in bigresults.columns:
        st.error(f"Skipping {model_name}: {port_col} not found.")
    else:
        # Create portfolios based on predictions
        cm = confusion_matrix(bigresults['true_port'], bigresults[port_col])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'P{p+1}' for p in range(nport)],
                    yticklabels=[f'P{p+1}' for p in range(nport)])
        plt.xlabel('Predicted Portfolio')
        plt.ylabel('True Portfolio')
        plt.title(f'Confusion Matrix: True vs Predicted Portfolios for {model_name}')
        plt.tight_layout()
        st.pyplot(plt)
