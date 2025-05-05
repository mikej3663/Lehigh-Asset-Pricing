import pandas as pd
from config import AVAILABLE_MODELS

class DataLoader:
    def __init__(self, filepath_or_buffer):
        self.file = filepath_or_buffer
        self.df = None

    def load_and_process(self):
        df = pd.read_csv(self.file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values("date")
        df = df.groupby('date')[['ret'] + AVAILABLE_MODELS].median().reset_index()
        self.df = df
        return df