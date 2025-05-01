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
    'pred_hgbr': 'HistGradientBoostingRegressor',
    'pred_Lasso': 'Lasso',
    'pred_Ridge': 'Ridge'
}
reverse_mapping = {v: k for k, v in name_mapping.items()}

# --- Pre-aggregate ---
your_df = your_df.groupby('date')[['ret'] + available_columns].median().reset_index()

# --- Sidebar Selectors ---
model_matrix_options = ['None'] + list(name_mapping.values())
confusion_matrix_options = ['None', 'NN2', 'NN3', 'NN4', 'HGBR', 'Lasso', 'Ridge']
selected_model_matrix = st.sidebar.selectbox("Model Matrix", model_matrix_options, index=0)
selected_conf_matrix = st.sidebar.selectbox("Additional Graphs", confusion_matrix_options, index=0)

# --- Show welcome page if nothing is selected, then stop ---
if selected_model_matrix == 'None' and selected_conf_matrix == 'None':

    st.markdown("**By:** Joseph Carruth, Jay Geneve, Michael Jamesley, and Evan Trock")


    st.image("Market.png", use_container_width=True)
    
    st.markdown("""
    
    ### Purpose  
    To fit varying regressors and neural networks to predict stock returns to ultimately create a Zero-Cost Long-Short Portfolio.
    
    ### Data  
    - WRDS CRSP Data  
    - OpenAP Data  
    - Fama-French Five Factors + Momentum
    
    ### Methodology  
    Import packages and data, lagging signals to effectively predict stock returns in the next year. We selected only the top 5000 firms by their net operating assets, since the model took too long to run on the entire firms list. We split the training dataset into 2020 with our holdout being 2021–2024. We preprocessed the data in a custom ML pipeline, using cross-sectional means for all continuous OpenAP features. We then trained on an expanding window each year after validation, fitting Lasso, Ridge, HistGradientBoostingRegressor, and three MLPRegressors. Finally, we binned predictions into portfolios and compared against true returns to evaluate trading performance.
    
    ### Results  
    Our Long-Short portfolio saw significant gains—outperforming the S&P 500 net of the risk-free rate.
    """)
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
        st.markdown(f"### Confusion Matrix: {selected_conf_matrix}")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'P{i+1}' for i in range(nport)],
                    yticklabels=[f'P{i+1}' for i in range(nport)],
                    ax=ax)
        ax.set_xlabel('Predicted Portfolio')
        ax.set_ylabel('True Portfolio')
        ax.set_title(f'Confusion Matrix ({selected_conf_matrix})')
        st.pyplot(fig)

    if selected_model_matrix != 'None':
        model_col = reverse_mapping[selected_model_matrix]
        if selected_conf_matrix in model_dict and model_dict[selected_conf_matrix] == model_col:
            preds = your_df[model_col].clip(lower=-15, upper=15)
            df_prices = pd.DataFrame({
                "Date": dates,
                "Actual": actual_returns,
                "Predictions": preds
            })
            df_clean = df_prices.dropna(subset=['Actual','Predictions'])
            st.markdown("### Histogram of Prediction Errors")
            df_clean['Residual'] = df_clean['Actual'] - df_clean['Predictions']
            fig_hist, ax_hist = plt.subplots()
            sns.histplot(df_clean['Residual'], bins=30, kde=True, ax=ax_hist)
            ax_hist.set_title("Distribution of Residuals")
            ax_hist.set_xlabel("Prediction Error (Residual)")
            ax_hist.set_ylabel("Frequency")
            st.pyplot(fig_hist)
