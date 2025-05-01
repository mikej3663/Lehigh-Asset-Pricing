import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, r2_score

# --- Streamlit File Upload ---
st.title("Neural Network Dashboard - Asset Pricing")

file_path = 'prediction_output3.csv'
your_df = pd.read_csv(file_path)

# --- Convert 'date' column to datetime and sort ---
your_df['date'] = pd.to_datetime(your_df['date'])
your_df = your_df.sort_values("date")

# --- Available Prediction Columns & Mappings ---
available_columns = [
    'pred_mlp_64_32', 'pred_mlp_128_64_32', 'pred_mlp_256_128_64_32',
    'pred_hgbr', 'pred_Lasso', 'pred_Ridge'
]
name_mapping = {
    'pred_mlp_64_32': 'Neural Network (2 Folds)',
    'pred_mlp_128_64_32': 'Neural Network (3 Folds)',
    'pred_mlp_256_128_64_32': 'Neural Network (4 Folds)',
    'pred_hgbr': 'HistGradientBoostingRegressor()',
    'pred_Lasso': 'Lasso',
    'pred_Ridge': 'Ridge'
}
reverse_mapping = {v: k for k, v in name_mapping.items()}

# --- Pre-aggregate ---
your_df = your_df.groupby('date')[['ret'] + available_columns].median().reset_index()

# --- Sidebar Selectors ---
model_matrix_options = ['None'] + list(name_mapping.values())
selected_model_matrix = st.sidebar.selectbox("Model Matrix", model_matrix_options, index=0)

confusion_matrix_options = ['None', 'NN1', 'NN2', 'NN3', 'HGBR', 'Lasso', 'Ridge']
selected_conf_matrix = st.sidebar.selectbox("Confusion Matrices", confusion_matrix_options, index=0)

# --- Show welcome page if nothing is selected, then stop ---
if selected_model_matrix == 'None' and selected_conf_matrix == 'None':
    st.markdown("### Welcome")
    st.write("Use the sidebar to select a model or a confusion matrix to begin.")
    st.stop()

# --- At least one visualization will render below ---

# Check for required columns
if 'date' not in your_df.columns or 'ret' not in your_df.columns:
    st.error("Error: Your DataFrame must contain 'date' and 'ret' columns.")
    st.stop()

# Prepare core DataFrame
dates = your_df['date']
actual_returns = your_df['ret'].clip(lower=-15, upper=15)

# Handle model‐matrix dropdown
if selected_model_matrix != 'None':
    col = reverse_mapping[selected_model_matrix]
    preds = your_df[col].clip(lower=-15, upper=15)
    df_prices = pd.DataFrame({
        "Date": dates,
        "Actual": actual_returns,
        "Predictions": preds
    })

    st.subheader(f"Predictions vs Actual ({selected_model_matrix})")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(df_prices['Date'], df_prices['Predictions'], label="Predictions")
    ax.plot(df_prices['Date'], df_prices['Actual'], label="Actual")
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    ax.legend()
    st.pyplot(fig)

    # R²
    df_clean = df_prices.dropna(subset=['Actual','Predictions'])
    r2 = r2_score(df_clean['Actual'], df_clean['Predictions'])
    st.markdown(f"### R² Score: {r2:.4f}")

    st.markdown("### Prediction Summary Statistics")
    st.write(df_clean['Predictions'].describe())

# Handle confusion‐matrix dropdown
if selected_conf_matrix != 'None':
    bigresults = pd.read_csv('prediction_output3.csv')
    # ... (your renaming/reset_index steps) ...

    model_dict = {
      'NN2':'pred_mlp_64_32','NN3':'pred_mlp_128_64_32','NN4':'pred_mlp_256_128_64_32',
      'HGBR':'pred_hgbr','Lasso':'pred_Lasso','Ridge':'pred_Ridge'
    }
    nport = 5
    mr = model_dict[selected_conf_matrix]
    port_col = f'port_{mr}'

    # build true portfolios
    bigresults['true_port'] = bigresults.groupby('date')['ret']\
        .transform(lambda x: pd.qcut(x, nport, labels=False, duplicates='drop') + 1)

    # build predicted portfolios
    if mr not in bigresults.columns:
        st.error(f"Predictions column {mr} not found in your CSV.")
    else:
        bigresults[port_col] = bigresults.groupby('date')[mr]\
            .transform(lambda x: pd.qcut(x, nport, labels=False, duplicates='drop') + 1)

        # now the port_col exists, compute confusion matrix
        cm = confusion_matrix(bigresults['true_port'], bigresults[port_col])
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'P{i+1}' for i in range(nport)],
                    yticklabels=[f'P{i+1}' for i in range(nport)],
                    ax=ax)
        ax.set_xlabel('Predicted Portfolio')
        ax.set_ylabel('True Portfolio')
        ax.set_title(f'Confusion Matrix ({selected_conf_matrix})')
        st.pyplot(fig)
