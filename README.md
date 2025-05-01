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

Our Long-Short portfolio saw significant gains. With the Ridge model seeing inflated gains from Penny stocks. The Multi-Layer-Perceptron Models had strong performance and as we see from the outcome, the 3 layers performed the best, while four layers showed diminishing marginal returns. 


## Citations

Gu, Shihao and Kelly, Bryan T. and Xiu, Dacheng, Empirical Asset Pricing via Machine Learning (September 13, 2019). Chicago Booth Research Paper No. 18-04, 31st Australasian Finance and Banking Conference 2018, Yale ICF Working Paper No. 2018-09, Available at SSRN: https://ssrn.com/abstract=3159577 or http://dx.doi.org/10.2139/ssrn.3159577 

Voigt, S. (2024, June 17). Replicating Gu, Kelly & Xiu (2020). Tidy Finance. https://www.tidy-finance.org/blog/gu-kelly-xiu-replication/
Senevirathne, K. (2024, March 4). How I’m using machine learning to trade in the stock market. Medium. https://medium.com/analytics-vidhya/how-im-using-machine-learning-to-trade-in-the-stock-market-3ba981a2ffc2
 

