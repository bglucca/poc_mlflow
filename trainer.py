import pandas as pd
import numpy as np
from utils import load_data_for_model

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow

import warnings
warnings.filterwarnings("ignore")

mlflow.autolog()

# Data Ingest
df_model = load_data_for_model()

X = df_model.drop(columns = ['sales','category'])
y = df_model['sales']

# Model Instances
lr = LinearRegression()
en = ElasticNet()
dt = DecisionTreeRegressor()
rf = RandomForestRegressor()
models = [lr,en,dt,rf]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 123)

# Parameters for Gridsearch
en_params = {
    'alpha': [0.1,0.3,1],
    'l1_ratio': [0.25, 0.5,0.75],
}

dt_params = {
                'max_depth': [3, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 3, 5]
            }

rf_params = {
                'n_estimators': [100, 200],
                'max_depth': [2,3],
                'min_samples_split': [2, 5, 10],
            }


# Baseline Model - Linear Regression
with mlflow.start_run() as run:

    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    mse = -1 * mean_squared_error(y_test, y_pred, squared = False)
    mae = -1 * mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('Logging Baseline Linear Regression Results')
    mlflow.log_metrics({'Neg. RMSE':mse,
                        'Neg. MAE':mae,
                        'R2':r2})

print('Running Other models and performing GridSearch')

# Other Models and Hyperparameter Optimization
model_list = [(en, en_params),
              (dt, dt_params),
              (rf, rf_params)]

for model, pg in model_list:

    with mlflow.start_run() as run:
            
            cv = GridSearchCV(model,
                            param_grid = pg,
                            n_jobs = -1,
                            verbose=1,
                            scoring=['r2','neg_mean_absolute_error','neg_root_mean_squared_error'],
                            refit = 'neg_root_mean_squared_error')
            
            cv.fit(X_train, y_train)

            y_pred = cv.predict(X_test)

            mse = -1 * mean_squared_error(y_test, y_pred, squared = False)
            mae = -1 * mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mlflow.log_metrics({'Neg. RMSE':mse,
                                'Neg. MAE':mae,
                                'R2':r2})