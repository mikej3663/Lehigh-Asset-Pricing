class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.0025, upper=0.9975):
        self.lower = lower
        self.upper = upper
    
    def fit(self, X, y=None):
        # Store quantiles for each column
        X_df = pd.DataFrame(X)
        self.lower_bounds_ = X_df.quantile(self.lower)
        self.upper_bounds_ = X_df.quantile(self.upper)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].clip(lower=self.lower_bounds_[col],
                                       upper=self.upper_bounds_[col])
        return X_df.values  # return as numpy array for sklearn compatibility
    
    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else []
