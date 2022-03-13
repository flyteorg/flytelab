from copyreg import pickle
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import requests
import io
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
imputer_cat = SimpleImputer(strategy="most_frequent")
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from flytekit import task, workflow
from joblib import dump
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple
import pickle
hi=None

@task
def get_dataset() -> pd.DataFrame:
    #return load_digits(as_frame=True).frame
   
    url = "https://github.com/smadarab/flytelab/raw/main/census.csv" # Make sure the url is the raw version of the file on GitHub
    download = requests.get(url).content
    df = pd.read_csv(io.StringIO(download.decode('utf-8')),sep=',')
    print("df is created",df.columns)
    #df.dropna(inplace=True)
    #df = df.reset_index()
    print(df.columns)
    return(df)


    

@task
def train_model(train: pd.DataFrame) -> Tuple[AdaBoostClassifier,OneHotEncoder]:
    num_cols = ['age', 'education-num', 'capital-gain',
            'capital-loos', 'hour-per-week']
    cat_cols = ['workclass', 
            'marital-status', 'occupation', 
            'relationship', 'race', 
            'sex', 'native-country']
    log_transform_cols = ['capital-loos', 'capital-gain']    
    def get_cat_cols(X):
        return X[cat_cols]
    def get_num_cols(X):
        return X[num_cols]
    def get_log_transform_cols(X):
        return X[log_transform_cols]
    def get_dummies(X):
        print('\n \n',type(X))
        return pd.get_dummies(pd.DataFrame(X))
    def cat_imputer(X):
        print(X.shape)
        return(imputer_cat.fit_transform(X))
        #return X.apply(lambda col: imputer_cat.fit_transform(col))  
    def one_hot_encode(X):
        print("one hot encode")
        ohe = OneHotEncoder(handle_unknown = 'ignore')
        ohe.fit(pd.DataFrame(X))
        global hi
        hi=ohe
        dump(ohe, 'onehot.joblib') 
        return ohe.transform(pd.DataFrame(X)).toarray()

    log_transform_pipeline = Pipeline([
    ('get_log_transform_cols', FunctionTransformer(get_log_transform_cols, validate=False)),
    ('imputer', SimpleImputer(strategy='mean')),   
    ('log_transform', FunctionTransformer(np.log1p))
    ])

    num_cols_pipeline = Pipeline([
    ('get_num_cols', FunctionTransformer(get_num_cols, validate=False)),
    ('imputer', SimpleImputer(strategy='mean')),
    ('min_max_scaler', MinMaxScaler())
    ])

    cat_cols_pipeline = Pipeline([
    ('get_cat_cols', FunctionTransformer(get_cat_cols, validate=False)),
    ('imputer', SimpleImputer(strategy="most_frequent")),
#    ('get_dummies', FunctionTransformer(get_dummies, validate=False))
    ('one_hot_encode', FunctionTransformer(one_hot_encode, validate=False))
    ])       

    steps_ = FeatureUnion([
    ('log_transform', log_transform_pipeline),
    ('num_cols', num_cols_pipeline),
    ('cat_cols', cat_cols_pipeline)
])
    full_pipeline = Pipeline([('steps_', steps_)])
    y = train['income'].map({'<=50K': 0, '>50K': 1})
    X = full_pipeline.fit_transform(train)
    model = AdaBoostClassifier(n_estimators=300)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = np.nan_to_num(X_train)
    y_train=np.nan_to_num(y_train)
    print("X_train dimensiona",X_train)
    return model.fit(X_train, y_train),hi


@workflow
def main() -> tuple[AdaBoostClassifier,OneHotEncoder]:
    return train_model(train=get_dataset())


if __name__ == "__main__":
    print(f"trained model: {main()}")
