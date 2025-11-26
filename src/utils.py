# Utils contains common functionality which all the entire project can use
import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        # write in binary mode
        # The with statement ensures the file is automatically closed after writing.
        # open(file_path, "wb") opens or creates the file exactly at the location given in file_path.

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # ** → Python’s “dictionary unpacking” operator. It expands the dictionary into keyword arguments.
            # ** = “Take all key-value pairs from a dictionary and pass them as named parameters.”
            # If gs.best_params_ is: {"max_depth": 10, "n_estimators": 200}
            # eg python does model.set_params(max_depth=10, n_estimators=200)
            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)
            # test score are model prediction
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    