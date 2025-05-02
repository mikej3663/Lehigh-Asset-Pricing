import pandas as pd
from sklearn.metrics import r2_score

class Evaluator:
    def __init__(self, df, selected_model, window=5):
        self.df = df.copy()
        self.model = selected_model
        self.window = window

    def compute_r2(self, df):
        return r2_score(
            df['Actual'].dropna(),
            df['Predictions'].dropna()
        )
    def get_raw_df(self):
        df = pd.DataFrame({
            "Date": self.df['date'],
            "Actual": self.df['ret'].clip(lower=-15, upper=15),
            "Predictions": self.df[self.model].clip(lower=-15, upper=15)
        })
        return df