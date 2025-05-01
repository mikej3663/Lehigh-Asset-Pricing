from data_loader import DataLoader
from evaluator import Evaluator
from config import AVAILABLE_MODELS

if __name__ == "__main__":
    loader = DataLoader("prediction_output.csv")
    df = loader.load_and_process()

    for model in AVAILABLE_MODELS:
        evaluator = Evaluator(df, model)
        raw = evaluator.get_raw_df()
        r2 = evaluator.compute_r2(raw)
        print(f"{model} - R²: {r2:.4f}")