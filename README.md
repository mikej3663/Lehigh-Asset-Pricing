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


| Model | Sharpe Ratio |
| -- | -- |
| mkt | 0.25561435483589773 |
| mlp_128_64_32 | 0.4029660847272751 |
|mlp_256_128_64_32 | 0.09799322755044002 |
|mlp_64_32 | -0.3998369946074465 |
|hgbr | 1.4319704242917604 |
|Ridge | 0.0672141449653041 |




## Citations

Gu, Shihao and Kelly, Bryan T. and Xiu, Dacheng, Empirical Asset Pricing via Machine Learning (September 13, 2019). Chicago Booth Research Paper No. 18-04, 31st Australasian Finance and Banking Conference 2018, Yale ICF Working Paper No. 2018-09, Available at SSRN: https://ssrn.com/abstract=3159577 or http://dx.doi.org/10.2139/ssrn.3159577 

Voigt, S. (2024, June 17). Replicating Gu, Kelly & Xiu (2020). Tidy Finance. https://www.tidy-finance.org/blog/gu-kelly-xiu-replication/
Senevirathne, K. (2024, March 4). How I’m using machine learning to trade in the stock market. Medium. https://medium.com/analytics-vidhya/how-im-using-machine-learning-to-trade-in-the-stock-market-3ba981a2ffc2
 

## Imports
``` python
import pandas as pd
import polars as pl

import numpy as np

import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from df_after_transform import df_after_transform
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, cross_val_score

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
```

## Get Data (Fama French & OpenAP)

``` python
import pandas_datareader as pdr
# load Fama-French factors

start = '1957-03-01' # start date for Fama-French data based on the s&p500 inception date
start_date = datetime.strptime(start, '%Y-%m-%d')

# load Fama-French 5 factors
ff_5 = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', start=start_date)[0]
ff_mom = pdr.get_data_famafrench('F-F_Momentum_Factor', start=start_date)[0]

import gdown
if not os.path.exists('signals_wide.zip'):
    # Download the file from Google Drive using gdown
    # Make sure to replace 'FILE_ID' with the actual file ID from your shareable link
    # Example: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
# From a shareable link like: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    file_id = '1T-nogu88A4hcFXijjftSO41K5P4Hj27y'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'signals_wide.zip'
    gdown.download(url, output, quiet=False)
    
else:
    print("signals_wide.zip already exists. Skipping download.")
```

## Runtime Help --> Convert to Float

``` python
float_cols = X.select_dtypes('float64').columns
X.loc[:, float_cols] = X.loc[:, float_cols].astype(np.float32, copy=False)

X.info()

y= y.astype(np.float32)
y_permno = y_permno.astype(np.float32)
```


## Main Loops for Training and Validation 
``` python
prediction_output_fname = 'predictions/prediction_output3.csv'
force_rerun = False # set to True to force run all models again
    
# Create a Loop that goes by year instead of doing cv split function, trains up to year, validates and then expands trains and validates continued until end of training
model_dict = { 
               'Lasso': Lasso(),
               'Ridge': Ridge(), 
               'hgbr': HistGradientBoostingRegressor(),
               'mlp': mlp,
              }

# Create a dictionary to store the parameters for the mlp regressor (hidden layer sizes from 1 to 5)
params = {
    'mlp': { 'hidden_layer_sizes' : [(64, 32)]#, (128, 64, 32), (256, 128, 64, 32)] #, #(256, 128, 64, 32), (512, 256, 128, 64, 32)],
}}

######## END: PARAMETERS #########

os.makedirs(os.path.dirname(prediction_output_fname), exist_ok=True)
if os.path.exists(prediction_output_fname):
    prediction_output = pd.read_csv(prediction_output_fname)
    prediction_output['date'] = pd.to_datetime(prediction_output['date'])
    already_run_models = [c[5:] for c in prediction_output.columns if c.startswith('pred_')]
else:
    already_run_models = []
results = {}

for model in model_dict.keys():
    print(f"\nTraining model: {model}")
    param_grid = params.get(model, {})
    for param in ParameterGrid(param_grid):
  # Get parameters for the model or an empty dict if none
        print(f"Using parameters: {param}")
        if model == 'mlp':
                # Assume param is a dictionary like {'hidden_layer_sizes': (64, 32)}
                param_suffix = "_".join(str(x) for x in param['hidden_layer_sizes'])
                model_name = f"{model}_{param_suffix}"
        else:
                model_name = model  # Ridge, RF, etc, no param differentiation
                
        if model_name in already_run_models and not force_rerun:
            print(f"Model {model_name} already run. Skipping.")
            continue
                
        if model_name not in results:
            results[model_name] = []  # Initialize results for this model variant

        # Train and validate for each year
   
        for year in tqdm(range(2000, 2020), desc="Training Years", unit="year"):
            # print(f"Training to predict year: {year}")
            
            # Split the data into train and test sets based on year
            X_train_year = X_train[X_train.index.get_level_values('date').year < year]
            y_train_year = y_train[X_train.index.get_level_values('date').year < year]
            
            X_val_year = X_train[X_train.index.get_level_values('date').year == year]
            y_val_year = y_train[X_train.index.get_level_values('date').year == year]
            
            # Fit the model
            model_pipe = make_pipeline(preproc_pipe, model_dict[model].set_params(**param))
            model_pipe.fit(X_train_year, y_train_year)
            
            y_pred = model_pipe.predict(X_val_year) 
            
            # Validate the model
            
            
            # Make predictions
            
            # Make a nice model name
            
            results[model_name].append(pd.DataFrame({
                'permno': X_val_year.index.get_level_values('permno'),
                'date': X_val_year.index.get_level_values('date'),
                'ret': y_val_year.values,
                f'pred_{model_name}': y_pred,
    
            }))
            # combine into a dataframe
```


