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




import os
if os.environ.get('https_proxy'):
 del os.environ['https_proxy']
if os.environ.get('http_proxy'):
 del os.environ['http_proxy']
import numpy as np 
import pip
package_names=['flytekit', 'sklearn', 'grpc'] #packages to install
pip.main(['install'] + package_names + ['--upgrade'])

import grpc
channel = grpc.insecure_channel('localhost:8501', options=(('grpc.enable_http_proxy', 0),))

from argparse import ArgumentParser
from pathlib import Path

import streamlit as st

import requests
import io
import pandas as pd

from flytekit.remote import FlyteRemote
from flytekit.models import filters
from flytekit.models.admin.common import Sort
from joblib import load

#from sklearn.datasets import load_digits


PROJECT_NAME = "flytelab-my_project".replace("_", "-")
WORKFLOW_NAME = "my_project.workflows.main"


parser = ArgumentParser()
parser.add_argument("--remote", action="store_true")
args = parser.parse_args()

backend = os.getenv("FLYTE_BACKEND", 'remote' if args.remote else 'sandbox')

# configuration for accessing a Flyte cluster backend
remote = FlyteRemote.from_config(
    default_project=PROJECT_NAME,
    default_domain="development",
    config_file_path=Path(__file__).parent / f"{backend}.config",
)

# get the latest workflow execution
[latest_execution, *_], _ = remote.client.list_executions_paginated(
    PROJECT_NAME,
    "development",
    limit=1,
    filters=[
        filters.Equal("launch_plan.name", WORKFLOW_NAME),
        filters.Equal("phase", "SUCCEEDED"),
    ],
    sort_by=Sort.from_python_std("desc(execution_created_at)"),
)

wf_execution = remote.fetch_workflow_execution(name=latest_execution.id.name)
remote.sync(wf_execution, sync_nodes=False)
model = wf_execution.outputs["o0"]
print(model)
encoder = wf_execution.outputs["o1"]
print("\n one encoder \n",encoder)


############
# App Code #
############

#data = load_digits(as_frame=True)
url = "https://github.com/smadarab/flytelab/raw/main/census.csv" # Make sure the url is the raw version of the file on GitHub
download = requests.get(url).content
df = pd.read_csv(io.StringIO(download.decode('utf-8')),sep=',')
print("df is created",df.columns)
df.dropna(inplace=True)
df = df.reset_index()
data=df

st.write("# Flytelab: my_project")
st.write("### Demo project")
st.write(f"Model: `{model}`")

age = st.text_input("age", 50)
education_num = st.text_input("education-num", 13.0)
capital_gain = st.text_input("capital-gain", 0.0)
capital_loos = st.text_input("capital-loos", 0.0)
hour_per_week = st.text_input("hour-per-week", 13.0)
workclass = st.text_input("workclass", "Self-emp-not-inc")
marital_status = st.text_input("marital-status", "Married-civ-spouse")
occupation = st.text_input("occupation", "Exec-managerial")
relationship = st.text_input("relationship", "Husband")
race = st.text_input("race", "White")
sex = st.text_input("sex", "Male")
native_country = st.text_input("native-country", "United-States")

#st.write("Use the slider below to select a sample for prediction")

X_train = [age, education_num, capital_gain, capital_loos, hour_per_week, workclass,
            marital_status, occupation, relationship, race, sex, native_country]

X_train = pd.DataFrame({'age': age, 'education_num': education_num,'capital-gain':capital_gain,'capital_loos':capital_loos,'hour_per_week':hour_per_week,'workclass':workclass,'martial_status':marital_status,'occupation':occupation,'relationship':relationship,'race':race,'sex':sex,'native_country':native_country},index=[0])

num_cols = ['age', 'education_num', 'capital-gain',
            'capital_loos', 'hour_per_week']
cat_cols = ['workclass', 
            'martial_status', 'occupation', 
            'relationship', 'race', 
            'sex', 'native_country']
log_transform_cols = ['capital_loos', 'capital-gain']    
def get_cat_cols(X):
    return X[cat_cols]
def get_num_cols(X):
    return X[num_cols]
def get_log_transform_cols(X):
    print("in function",X.columns)
    return X[log_transform_cols]
def get_dummies(X):
    print('\n \n',type(X))
    return pd.get_dummies(pd.DataFrame(X))
def cat_imputer(X):
    print(X.shape)
    return(imputer_cat.fit_transform(X))
    #return X.apply(lambda col: imputer_cat.fit_transform(col))  
def one_hot_encode(X):
    print(X.shape)
    print("current wd",os.getcwd())
    ohe = load('onehot.joblib')
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
#('get_dummies', FunctionTransformer(get_dummies, validate=False))
('one_hot_encode', FunctionTransformer(one_hot_encode, validate=False))
])       

steps_ = FeatureUnion([
('log_transform', log_transform_pipeline),
('num_cols', num_cols_pipeline),
('cat_cols', cat_cols_pipeline)
])
full_pipeline = Pipeline([('steps_', steps_)])
X_train = full_pipeline.fit_transform(X_train)


#X_train=np.array(X_train)
#st.image(data.images[sample_index], clamp=True, width=300)
#st.write(f"Ground Truth: {data.target[sample_index]}")
st.write(f"Prediction: {model.predict(X_train)}")
