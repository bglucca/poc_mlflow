# MLflow 101
## What is this repo about?
This repo is a results of my initial studies of MLflow.
The main idea is to use ML Flow to manage a simple ML problem to get a grasp on how the tool works.

## Methodology
This repo solves a supervised learning problem, trying to predict the sales for amazon groceries in the UK.

An initial ETL is performed to clean big problems in data. This can be found on the `00_etl.ipynb` notebook.

This is followed by the logging and training of 4 models: Linear Regression, ElasticNet, Decision Tree and Random Forest. This happens in the `trainer.py` script.

The Linear Regression is a baseline model used to establish a minimal performance for the other models. All other models are tuned using `GridSearchCV`.

The models were logged and compared using the MLflow UI. This led to the registration of the best perfoming model.

This model is then called on the `predictor.py`.

## Repo Structure
```
data
│   ├───raw
│   └───refined
├─── .gitignore
├─── 00_etl.ipynb -> Data ETL from raw to refined
├─── predictor.py -> script to run predictions
├─── README.md
├─── trainer.py -> script to train and log models
├─── utils.py -> Util functions
```
## Running the project
First, make sure MLflow is installed on the env you are using. There are conda and pip env files in this repo.

1. Run `00_etl.ipynb` to preprocess the data
2. Run `trainer.py` to train and log the models
3. Compare the models using the mlflow ui. To call the UI just type `mlflow ui` in a terminal.
4. To use the model you want to predict, write the model's URI path to `predictor.py` and run it.

## Acknowledgements and links
Check out my post on [Medium](https://medium.com/@luccagomes/mlflow-101-how-i-did-it-f451e787d916) going more in detail on the usage of MLflow and why I'm probably going to use it more often.

The data used is publicly available on [Kaggle](https://www.kaggle.com/datasets/dalmacyali1905/amazon-uk-grocery-dataset-unsupervised-learning/data)

[MLflow docs](https://mlflow.org/docs/latest/index.html)

Also, check out my [personal page](https://bglucca.github.io) with a little more about myself

