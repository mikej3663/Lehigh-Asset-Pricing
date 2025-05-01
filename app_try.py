import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import zscore
from data_loader import DataLoader
from evaluator import Evaluator
from config import AVAILABLE_MODELS, WINDOW_SIZE, MODEL_NAME_MAPPING
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.title("Neural Network Dashboard - Asset Pricing")

 # Load and preprocess
loader = DataLoader("portfolio4.csv")
your_df = loader.load_and_process()

selected_model = st.sidebar.selectbox("Model Type", AVAILABLE_MODELS)

if 'date' not in your_df.columns or 'ret' not in your_df.columns:
    st.error("Your DataFrame must contain 'date' and 'ret' columns.")
    st.stop()

# Evaluation
evaluator = Evaluator(your_df, selected_model, window=WINDOW_SIZE)
df_prices = evaluator.get_raw_df()
y_actual_label = "Actual"
y_pred_label = "Predictions"

# Remove global anomalies in predictions before error calculation for selected models
for model_col in ["pred_mlp_32", "pred_mlp_64_32", "pred_mlp_128_64_32"]:
    if model_col in df_prices.columns:
        z_scores = zscore(df_prices[model_col])
        pred_outliers = abs(z_scores) > 3
        df_prices[model_col] = df_prices[model_col].mask(pred_outliers, df_prices[model_col].median())

# Plotting
st.subheader(f"Predictions vs Actual - {MODEL_NAME_MAPPING.get(selected_model, selected_model)}")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_prices["Date"], df_prices[y_actual_label], label="Actual", color="orange")
ax.plot(df_prices["Date"], df_prices[y_pred_label], label="Predictions", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Returns")
ax.set_title("Predictions vs Actual")
ax.legend()
st.pyplot(fig)

# Overlay R² and adjusted R² on predictions plot
st.subheader(f"Predictions vs Actual with R² Overlay - {MODEL_NAME_MAPPING.get(selected_model, selected_model)}")

# R² Score
r2_val = evaluator.compute_r2(df_prices[[y_actual_label, y_pred_label]])
st.markdown(f"### R² Score: {r2_val:.4f}")

# Outlier Regressor (Highlighting large prediction errors)
st.markdown("### Outlier Analysis")

df_prices["Error"] = (df_prices["Actual"] - df_prices["Predictions"]).abs()
outlier_threshold = df_prices["Error"].quantile(0.90)
outliers = df_prices[df_prices["Error"] > outlier_threshold]

df_prices["RollingMean"] = df_prices["Predictions"].rolling(window=5, min_periods=1).mean()
df_prices.loc[df_prices["Error"] > outlier_threshold, "Predictions"] = df_prices["RollingMean"]

r2_val = evaluator.compute_r2(df_prices[[y_actual_label, y_pred_label]])

st.write(f"Replaced {len(outliers)} outlier predictions with rolling mean of previous predictions.")

# Adjusted R² calculation after outlier replacement
n = len(df_prices)
p = 1  # Only one predictor (model prediction)
adjusted_r2 = 1 - (1 - r2_val) * ((n - 1) / (n - p - 1))
st.markdown(f"### Adjusted R² Score (after outlier replacement): {adjusted_r2:.4f}")


# Plot: Predictions before and after outlier adjustment
df_prices["Predictions_Adjusted"] = df_prices["Predictions"].copy()
df_prices.loc[df_prices["Error"] > outlier_threshold, "Predictions_Adjusted"] = df_prices["RollingMean"]

st.subheader(f"Predictions vs Actual (with Adjusted Predictions) - {MODEL_NAME_MAPPING.get(selected_model, selected_model)}")

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(df_prices["Date"], df_prices["Actual"], label="Actual", color="orange")
ax2.plot(df_prices["Date"], df_prices["Predictions_Adjusted"], label="Adjusted Predictions", color="green")
ax2.set_xlabel("Date")
ax2.set_ylabel("Returns")
ax2.set_title("Predictions vs Actual (After Outlier Adjustment)")
ax2.legend()
st.pyplot(fig2)

# Summary stats
st.markdown("### Prediction Summary Statistics")
st.write(df_prices["Predictions"].describe())

# Confusion Matrix
st.subheader(f"Confusion Matrix - {MODEL_NAME_MAPPING.get(selected_model, selected_model)}")

# Define a classification threshold
threshold = 0  # You can adjust this threshold as needed

# Convert predictions and actuals into binary labels
y_true = (df_prices["Actual"] > threshold).astype(int)
y_pred = (df_prices["Predictions_Adjusted"] > threshold).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])

# Plot confusion matrix using matplotlib and Streamlit
fig_cm, ax_cm = plt.subplots()
disp.plot(ax=ax_cm)
st.pyplot(fig_cm)