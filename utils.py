import pandas as pd

def load_data_for_model(path:str = 'data/refined/dataset.parquet') -> pd.DataFrame:

    df = pd.read_parquet(path)
    df_model = df.drop(columns = ['asin','brand'])
    df_model = pd.concat([df_model,pd.get_dummies(df_model['category'])], axis = 1)

    return df_model