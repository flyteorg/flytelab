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
from sklearn.preprocessing import MinMaxScaler




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
scaler = wf_execution.outputs["o2"]
print("\n one encoder \n",encoder)

############
# App Code #
############

#data = load_digits(as_frame=True)


st.write("# Flytelab: Predict Potential Donator")
st.write(f"## Team: Cubits")

#st.write(f"Model: `{model}`")

with st.form(key='my_form'):
    age = st.number_input("age")
    education_num = st.number_input("education-num")
    capital_gain = st.number_input("capital-gain")
    capital_loos = st.number_input("capital-loos")
    hour_per_week = st.number_input("hour-per-week")
    workclass = st.text_input("workclass", "Self-emp-not-inc")
    marital_status = st.text_input("marital-status", "Married-civ-spouse")
    occupation = st.text_input("occupation", "Exec-managerial")
    relationship = st.text_input("relationship", "Husband")
    race = st.text_input("race", "White")
    sex = st.text_input("sex", "Male")
    native_country = st.text_input("native-country", "United-States")
    submit_button = st.form_submit_button(label='Submit')


       
X_train = pd.DataFrame({'age': age, 'education-num': education_num,'capital-gain':capital_gain,'capital-loss':capital_loos,'hours-per-week':hour_per_week,'workclass':workclass,'marital-status':marital_status,'occupation':occupation,'relationship':relationship,'race':race,'sex':sex,'native-country':native_country},index=[0])


#X_train = pd.DataFrame(dict_val,index=[0])
num_cols = ['age', 'education-num', 'capital-gain',
        'capital-loss', 'hours-per-week']
cat_cols = ['workclass', 
        'marital-status', 'occupation', 
        'relationship', 'race', 
        'sex', 'native-country']
log_transform_cols = ['capital-loss', 'capital-gain']    
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
    #dump(ohe, 'onehot.joblib') 
    print(X)
    return encoder.transform(pd.DataFrame(X)).toarray()
def min_max_scaling(X):
    return scaler.transform(pd.DataFrame(X)).tolist()

log_transform_pipeline = Pipeline([
('get_log_transform_cols', FunctionTransformer(get_log_transform_cols, validate=False)),
('imputer', SimpleImputer(strategy='mean')),   
('log_transform', FunctionTransformer(np.log1p))
])

num_cols_pipeline = Pipeline([
('get_num_cols', FunctionTransformer(get_num_cols, validate=False)),
('imputer', SimpleImputer(strategy='mean')),
('min_max_scaler', FunctionTransformer(min_max_scaling, validate=False))
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
X = full_pipeline.fit_transform(X_train)
   
y_pred=model.predict_proba(X)
final = y_pred
if np.argmax(final)==1:
    st.write(f"Can make donation")
elif np.argmax(final)==0:
    st.write(f"Cannot make donation")    


#X_train=np.array(X_train)
#st.image(data.images[sample_index], clamp=True, width=300)
#st.write(f"Ground Truth: {data.target[sample_index]}")
st.write(f"Prediction: {np.argmax(final)}")
