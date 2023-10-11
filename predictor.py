import mlflow
import pandas as pd
import os
from utils import load_data_for_model

MODEL_REL_PATH = 'models:/dt_amz_prices/staging'

df_model = load_data_for_model()

X = df_model.sample(10, random_state=123)

y = X['sales'].values

X = X.drop(columns = ['sales','category'])

print(X.head())

model = mlflow.sklearn.load_model(MODEL_REL_PATH)

y_pred = pd.Series(model.predict(X))

print(pd.DataFrame(list(zip(y,y_pred)), columns=['sales','pred_sales']))