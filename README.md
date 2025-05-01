# Lehigh Asset Pricing
By: Joseph Carruth, Jay Geneve, Micahel Jamesley, and Evan Trock

## Purpose
To fit varying regressors and neural networks to predict stock returns to ultimately create a Zero-Cost Long-Short Portfolio. 

## Data
1. WRDS CRSP Data
2. OpenAP Data
3. Fama French Five Factors + Momentum

## Methodology

Import packages and data, lagging signals to effectively predict stock returns in the next year. We selected only the top 5000 firms by their net operating assets, since the model took too long to run on the entire firms list. We split the training dataset into 2020 with our holdout being 2021-2024. We preprocessed the data in Custom ML Pipeline. Cross Sectional Means for all continuous data in the OpenAP dataset. We used an expanding window to train our model every year after validation. We used Lasso, Ridge, HistGradientBoostingRegressor and 3 MLPRegressors to try to fit the data to returns. We then binned the predictions to portfolios and analyzed the true returns to see how our model did trading. 

## Results 

Our Long-Short portfolio saw significant gains. Some over the S&P500 minus the risk free rate.
