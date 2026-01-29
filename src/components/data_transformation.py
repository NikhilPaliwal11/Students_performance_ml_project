import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()           #Initializing
     
    # this function is used to create a pickle file which is responsible for converting categorical into numerical and to perform standard scaler
    def get_data_transformer_object(self):
        '''This function is responsible for data transformation'''
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            # Create a pipeline and we need to handle missing values too
            #Pipeline allows you to sequentially apply a list of transformers to preprocess the data
            num_pipeline = Pipeline(
                steps = [("imputer",SimpleImputer(strategy="median")),
                        ("scaler",StandardScaler())])
            
            # We handling missing value and use median because there are outlier in data.
            # We do standard scaler for numerical variables.
            

            # Categorical pipleine handling Simple Imputer for missing values , one hot encoder for encoding because of less categories in each variable.
            # Standardscaler for converting it standard normal varibale between 0 and 1
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy= "most_frequent")),    #mode
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            # Dont subtract by mean. Just scale by standard deviation and keep zeros as zeors because sparse matrix contains [0,0,0,1,0] one hot encoding
            # if mean 0.05 all zero would become non - zeros , your sparse matrix become dense huge memory problem.
            # Scikit-learn prevented your sparse data from getting converted into a massive dense matrix and eating your memory.
            
            logging.info(f"Categorical columns:{categorical_columns}")
            logging.info(f"Numerical columns:{numerical_columns}")
        
            # Now we combine both numerical pipeline and categorical pipeline together and for that we use preprocessing.
            #List of (name, transformer, columns) tuples specifying the transformer objects to be applied to subsets of the data
            #ColumnTransformer applies different preprocessing pipelines to different sets of columns and then combines the results into a single feature matrix.
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        # Whatever we put after return is what gets sent back to whoever calls this function.
        except Exception as e:
            raise CustomException(e,sys)
        
    # Starting our data transformation inside this function
    def initiate_data_transformation(self,train_path,test_path):

        try:
            # We first read our train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")

            # You are calling one method of a class from another method of the same class.
            # the object that contains your entire data preprocessing logic.

            preprocessor_obj = self.get_data_transformer_object()
            # Python is basically doing this:  preprocessor_obj = preprocessor
            # preprocessor_obj now holds only the ColumnTransformer object you created
            # Nothing else gets included because you didn’t return anything else.
            
            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            input_feature_train_df = train_df.drop(columns = [target_column_name],axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name],axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying the preprocessing object on training dataframe and testing dataframe"
            )

            # Applying the preprocessing on train and test     
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            '''The preprocessor learns:
               Median values for missing numbers
               Most frequent category values
               All categories for OneHotEncoder
               Scaling parameters (mean & std)'''

            # np.c_ means:Column wise concatenation (join array side by side) X(features)  y(target)   after np.c_  [X(features),y(target)]
            # model now gets inputs + target combined in one array.

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            # preprocessing object need to be converted into pickle file . we have the path and save this pickle file in same path
            # we have written this function into utils.

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj)
            
            # obj = Your trained ColumnTransformer -- obj is your ColumnTransformer object
            # obj is your trained preprocessing pipeline (ColumnTransformer), and you are saving it into preprocessor.pkl 
            # so that you can reuse it during prediction or deployment.
            # Because you called: preprocessor_obj.fit_transform(input_feature_train_df)
            '''Your preprocessor_obj now contains learned values, like:
                Median values for missing numericals
                Most frequent values for categorical imputation
                Encoded category mappings
                Mean & standard deviation for scalers
                So youre saving:
            ✅ The full fitted preprocessing pipeline
            ✅ Along with everything it learned from training data

            Yes — even after you called it and assigned the output to input_feature_train_arr, the preprocessor is still “trained”. Let me explain why.

            1. What happens when you call fit_transform()
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            .fit_transform() does two things internally:
             fit → Learns parameters from the training data
                   Median values for SimpleImpute
                   Most frequent categories for categorical columns
                   Category mappings for OneHotEncoder
                   Mean & std for StandardScaler
            transform → Applies those learned parameters to convert your data into a numerical array
            The returned value is stored in input_feature_train_arr, which is just the transformed array.'''

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                #Accessing a variable stored inside another object which is stored inside your object
            )

        except Exception as e:
            raise CustomException(e,sys)