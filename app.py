import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score

# --- Streamlit File Upload ---
st.title("Neural Network Dashboard - Asset Pricing")

# Load the portfolio data
file_path = 'portfolio4.csv'
bigresults = pd.read_csv(file_path)

# Check the columns in the loaded dataset
st.write("Bigresults columns:", bigresults.columns)

# --- Available Prediction Columns ---
available_columns = [
    'pred_mlp_64_32', 'pred_mlp_128_64_32', 'pred_mlp_256_128_64_32', 'pred_hgbr', 'pred_Lasso', 'pred_Ridge'
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

# Reverse the mapping to map selected model back to original column names
reverse_mapping = {v: k for k, v in name_mapping.items()}

# --- Group by date using median ---
your_df = bigresults.groupby('date')[['ret'] + available_columns].median().reset_index()

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

# If a model is selected, get the predictions, otherwise set them to None
if selected_model_matrix != 'None':
    # Map the selected model name to the original column name
    selected_model_column = reverse_mapping[selected_model_matrix]
    predictions = your_df[selected_model_column].clip(lower=-15, upper=15)
else:
    predictions = None

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

# Check if the Predictions column has data
if df_prices['Predictions'].isnull().all():
    st.error("No predictions available. Please select a valid model.")
else:
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
    bigresults = pd.read_csv('portfolio4.csv')

    # Check the columns of bigresults
    st.write("Bigresults columns:", bigresults.columns)

    # Use the model name to get the portfolio column and predictions column
    model_name = model_dict[selected_conf_matrix]
    port_col = f'port_{model_name}'

    # Ensure the portfolio column exists
    if port_col not in bigresults.columns:
        st.error(f"Skipping {model_name}: {port_col} not found.")
    else:
        # True portfolio: based on actual returns (we'll use the first portfolio, e.g., port_1, as the true values)
        bigresults['true_port'] = bigresults.groupby('date')['port_1'].transform(
            lambda x: pd.qcut(x, nport, labels=False, duplicates='drop') + 1
        )

        # Create confusion matrix based on true vs predicted portfolios
        cm = confusion_matrix(bigresults['true_port'], bigresults[port_col])

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'P{p+1}' for p in range(nport)],
                    yticklabels=[f'P{p+1}' for p in range(nport)])
        plt.xlabel('Predicted Portfolio')
        plt.ylabel('True Portfolio')
        plt.title(f'Confusion Matrix: True vs Predicted Portfolios for {model_name}')
        plt.tight_layout()
        st.pyplot(plt)
