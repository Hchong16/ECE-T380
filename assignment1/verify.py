# Harry Chong
# Driver code to verify results using the model pkl and pipeline pkl files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Helper Function
class DropColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
            
    def fit(self, X, y = None):
        return self # nothing else to do
    
    def transform(self, X):
        # Check if x is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise KeyError("Input is not a pandas DataFrame")
        return X.drop([self.column], axis=1)

##################################################################################
# Driver code for Linear Regression (pipeline1.pkl & model1.pkl)
# Load dataset and pipeline
dataset = pd.read_pickle("appml-assignment1-dataset.pkl")
x_data = pd.DataFrame(dataset["X"])
y_data = pd.DataFrame(dataset["y"])

dataPipeline = pickle.load(open("pipeline1.pkl", "rb"))
transformedData = dataPipeline.fit_transform(x_data)

# Split data into training and test sets
x_train, x_test = train_test_split(transformedData, test_size = 0.25, random_state=0)
y_train, y_test = train_test_split(y_data, test_size = 0.25, random_state=0)

# Load model
lin_reg = pickle.load(open("model1.pkl", "rb"))

# Predict result
y_pred = lin_reg.predict(x_test)

# Calculate MSE and Accuracy
lin_mse = np.sqrt(mean_squared_error(y_test, y_pred))
lin_accuracy = lin_reg.score(x_test, y_test)
print("Linear Regression MSE: {}".format(lin_mse))
print("Linear Regression Accuracy: {}".format(lin_accuracy))

##################################################################################
# Driver code for Random Forest (pipeline2.pkl & model2.pkl)
# Load dataset and pipeline
dataset = pd.read_pickle("appml-assignment1-dataset.pkl")
x_data = pd.DataFrame(dataset["X"])
y_data = pd.DataFrame(dataset["y"])

dataPipeline = pickle.load(open("pipeline2.pkl", "rb"))
transformedData = dataPipeline.fit_transform(x_data)

# Split data into training and test sets
x_train, x_test = train_test_split(transformedData, test_size = 0.25, random_state=0)
y_train, y_test = train_test_split(y_data, test_size = 0.25, random_state=0)

# Load model
forest_reg = pickle.load(open("model2.pkl", "rb"))

# Predict result
y_pred = forest_reg.predict(x_test)

# Calculate MSE and Accuracy
forest_mse = np.sqrt(mean_squared_error(y_test, y_pred))
forest_accuracy = forest_reg.score(x_test, y_test)
print("Random Forest MSE: {}".format(forest_mse))
print("Random Forest Accuracy: {}".format(forest_accuracy))