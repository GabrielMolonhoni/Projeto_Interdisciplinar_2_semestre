import argparse, os
import boto3
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC



if __name__ == "__main__":

    # Pass in environment variables and hyperparameters
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--estimators", type=int, default=15)

    # sm_model_dir: model artifacts stored here after training
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

    args, _ = parser.parse_known_args()
    model_dir = args.model_dir
    sm_model_dir = args.sm_model_dir
    training_dir = args.train

    # Read in data
    df = pd.read_csv(training_dir + "/train.csv", sep=",")
    print(df.shape)
    print(df.columns)

    # Preprocess data
    
    # numerical variables
    num_vars = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
            'DiabetesPedigreeFunction', 'Age']

    # categorical variables
    cat_vars = ['Pregnancie_Stratified', 'Age_Stratified', 'BMI_Stratified']
  
    num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('robust_scaler', RobustScaler())
])

    cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # as the categories are numbers, we can use the SimpleImputer
    ('one-hot-encoding', OneHotEncoder(handle_unknown='ignore'))
])

    # (name, transformer, columns)
    preprocessed_pipeline = ColumnTransformer([
    ('numerical', num_pipeline, num_vars),
    ('categorical', cat_pipeline, cat_vars)
])
    y_column = 'Outcome'
    y_train = df[y_column]
    print(y_train.shape)
    
    data_train = df[df.columns.drop(y_column)]
    X_train = preprocessed_pipeline.fit_transform(data_train)

    # Build model
    
    svm_rbf = SVC(kernel='rbf', C = 100, gamma = 0.01)
    svm_rbf.fit(X_train, y_train)

    # Save model
    joblib.dump(svm_rbf, os.path.join(args.sm_model_dir, "model.joblib"))

# Model serving
"""
Deserialize fitted model
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
"""
def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        request_body = json.loads(request_body)
        inpVar = request_body["Input"]
        return inpVar
    else:
        raise ValueError("This model only supports application/json input")

"""
predict_fn
    input_data: returned array from input_fn above
    model (sklearn model) returned model loaded from model_fn above
"""
def predict_fn(input_data, model):
    return model.predict(input_data)

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: the content type the endpoint expects to be returned. Ex: JSON, string
"""
def output_fn(prediction, content_type):
    res = int(prediction[0])
    respJSON = {"Output": res}
    return respJSON
