# **Research Proposal: Fitting Neural Networks to Predict Asset Price**

# By Joseph Carruth, Jay Geneve, Michael Jamesley, and Evan Trock

## **Research Question**

To address the problem of fitting various neural networks to predict monthly asset returns, we pose our general question as follows:

*Can a neural network effectively predict future stock prices based on historical data and other relevant market indicators?*

Some questions we specifically would like to answer include the following:

* What input features yield the best predictive power for stock price forecasting?  
  * Examples include historical prices, volume, macroeconomic variables, etc.  
* Can the model generate returns above the S\&P 500 when used as part of a trading strategy?   
* Can we obtain a Sharpe Ratio (an equity’s risk premium over its standard deviation) or R-Squared value close to or better than the NN used in the prompt paper, *"Empirical Asset Pricing via Machine Learning"* ? Better than a buy-and-hold investor? 

## **Needed Data**

Our final data set will be permno-month observations, spanning from 1975 to the present. The dataset will include information about the company (e.g., name, PERMNO, ticker, and industry), volume traded (liquidity), and monthly return. We hope to get firm returns from Wharton Research Data Services (WRDS), because its returns include dividends and delisting information.

Additionally, we will collect a suite of predictor variables. This will include but not be limited to

* S\&P returns and other, macroeconomic time-series data like DPS, EPS, and treasury bill rate (which can be collected via FRED)  
* Firm accounting data, via Compustat in the WRDS dataset   
* Known asset pricing ***signals (i.e.possible predictors)*** via the Open Asset Pricing dataset

Using these signals, we will build various models that view the data differently as they produce asset return predictions from the training dataset. 

Our goal is to use different pattern recognition algorithms that use functions like EMA and RSI to train our model on the patterns that the value of an asset follows. We will then use data the model hasn’t seen to test its forecasting accuracy in order to iterate and improve our model over time. 

## Resources

1. [Data (October 2024 Release) – Open Source Asset Pricing](https://www.openassetpricing.com/data/) This dataset contains two kinds of data:   
   1. Portfolio returns, where the portfolio is the return earned by trading long minus short “anomaly” portfolios. This isn’t relevant to our project.  
   2. Stock-level Signals. This is the data we need. They have a point-and-click version that contains most signals, except a few that require WRDS access.  
2. [9.7. Open Asset Pricing](https://ledatascifi.github.io/ledatascifi-2025/content/05/05e_OpenAP_anomaly_plot.html) This includes a walkthrough of obtaining and downloading data on “anomaly portfolio returns”. More importantly, it includes three examples we need to understand:  
   1. A [quick tour](https://github.com/mk0417/open-asset-pricing-download/blob/master/examples/quick_tour.ipynb)  
   2. How to c[ombine it with stock price returns from CRSP](https://github.com/mk0417/open-asset-pricing-download/blob/master/examples/merge_signals_with_crsp.ipynb). The key point here is to lag the predictor variables, so we use, e.g., January data to predict February returns.  
   3. [Machine Learning Exampl](https://github.com/mk0417/open-asset-pricing-download/blob/master/examples/ML_portfolio_example.ipynb)e.   
3. [5.4. Coding in Teams](https://ledatascifi.github.io/ledatascifi-2025/content/05/01c_teams.html) \- Branching on our work repo will be very useful, so we can try different experiments.   
4. [5.5. Sharing large files](https://ledatascifi.github.io/ledatascifi-2025/content/05/01d_sharingBigFiles.html) \- We can all download the stock-level signal file individually, or share it this way. The former is probably easier. 

## Process

**The goal of this project is to upgrade the final part of the Machine Learning Example above: To try many kinds of models, many different set ups of models, and various sets of predictor variables.**  
Data we will construct:

* **Dataset \#1: We should save, for each model, all of the stock level predictions** (which then are used to sort stocks into portfolios). We can then use this to see what kinds of stocks a given strategy recommends buying on the long side and what it recommends shorting. This is an important check, because some “anomalies” require you to short small firms that are very hard to short.

Think of this as building a dataset like the midterm. For a PERMNO-month (row), a given column refers to the model that made it, and values are the prediction the model made for that firm-month. We can make it so that this dataset can be saved, reloaded, and built up as we go, because this project will not end up with code you run in one go. For this, Prof. Bowen’s midterm answer code has features we can use\! (Load data if it exists, only do a model if we don’t have predictions for it already, etc.)

* **Dataset \#2: Contains the monthly performance for each model in Dataset \#1. This is actually the more important dataset.**

  For a given model and month, we take Dataset \#1 and sort the stocks into 5 buckets (“portfolios “), and get the average return. The long-short return for a model-month is the return of the 5th bucket minus the 1st bucket.

## Outputs

* Display our results on a dashboard. We want to communicate clearly with the reader the most favorable models for their zero-cost portfolio. The mock dashboard shows for individual stocks the returns predicted under different NNs. Our hope is to incorporate some individual stock returns down the line to give the reader a better understanding of their portfolio.

<img src="Stock Prediction Dashboard.png?raw=true"/>

**Gemini Canvas Dashboard**  
<img src="Stock Return Prediction Dashboard.png?raw=true"/>  
Canvas’ dashboard provides clear model performance metrics, which we like for its simplicity and ease of use. But we will be seeking more specificity in terms of models, returns, and opportunity cost of investors. 

* From Dataset \#2:

  Tables and figures that show how different models compare in terms of out of sample performance. 

  Using these, we can assess what kinds of model choices increase and decrease our out-of-sample performance? We can compare this to [Gu, Kelly & Xiu (2020)](https://www.tidy-finance.org/blog/gu-kelly-xiu-replication/). 

  A plot of the cumulative returns to each possible model (the long-short portfolio), as though we traded it out of sample from 2000-2024.

<img src="Total Value of Investment.png?raw=true"/> 
This cumulative returns plot taken from another asset pricing model shows a comparison between two model parameters and the S\&P 500\. A performance indicator like this would be very useful for an investor looking to maximize returns using our dashboard. 

* We will pick our favorite model, and then use the stock-level predictions to make the **Canonical Asset Pricing Table 1**.

<img src="Canonical Table 1.png?raw=true"/>

* Finally, we will try to upload our stock prediction signals (Dataset \#1) to [Assaying Anomalies](https://sites.psu.edu/assayinganomalies/upload/) and get a report back. 

**Bibliography**  
Gu, Shihao and Kelly, Bryan T. and Xiu, Dacheng, *Empirical Asset Pricing via Machine Learning* (September 13, 2019). Chicago Booth Research Paper No. 18-04, 31st Australasian Finance and Banking Conference 2018, Yale ICF Working Paper No. 2018-09, Available at SSRN: [https://ssrn.com/abstract=3159577](https://ssrn.com/abstract=3159577) or [http://dx.doi.org/10.2139/ssrn.3159577](https://dx.doi.org/10.2139/ssrn.3159577) 

Voigt, S. (2024, June 17). *Replicating Gu, Kelly & Xiu (2020)*. Tidy Finance. [https://www.tidy-finance.org/blog/gu-kelly-xiu-replication/](https://www.tidy-finance.org/blog/gu-kelly-xiu-replication/)

Senevirathne, K. (2024, March 4). *How I’m using machine learning to trade in the stock market*. Medium. [https://medium.com/analytics-vidhya/how-im-using-machine-learning-to-trade-in-the-stock-market-3ba981a2ffc2](https://medium.com/analytics-vidhya/how-im-using-machine-learning-to-trade-in-the-stock-market-3ba981a2ffc2)

