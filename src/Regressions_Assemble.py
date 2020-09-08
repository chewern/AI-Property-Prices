# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:24:30 2020
@author: CHEW ERN
"""
import datetime as dt
import sqlite3
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# Custom Classes and Functions
def clean_data(df):
    """this function basically does all the data filtering processes described in the EDA"""
    #dropping rows with missing price data, or rather picking up rows with valid price data
    df = df[df.price.notnull()]
        #picking up rows with not null lot_size data
    df = df[df.lot_size.notnull()]
        #fill up missing review_score with 8.0
    df.review_score = df.review_score.fillna(8.0)
        #remove rows with null in zipcode column
    df = df[df.zipcode.notnull()]    
        #remove rows with null in bedrooms column
    df = df[df.bedrooms.notnull()]
        #remove rows with null in bathrooms column
    df = df[df.bathrooms.notnull()] 
        #remove rows with null in date column
    df = df[df.date.notnull()]
        #remove rows with null in living_room_size column
    df = df[df.living_room_size.notnull()]
        #remove last row of missing data
    df = df[df.waterfront.notnull()]
        #convert to int types
    to_int_list = ['price','bedrooms','view','review_score','basement_size','waterfront',
                   'built','renovation','zipcode','living_room_size','lot_size']
    df[to_int_list] = df[to_int_list].astype(int)
        #tidy up condition column to lower case and then map to 5 categories
    df.condition = df.condition.str.lower().map({'terrible':0,'poor':1,'fair':2,'good':3,'excellent':4})

    return df      

def get_age(df):
    """Changing built year to age of house at point of transaction then rename column to age.
    transaction_year = df['date'].dt.year"""
    df['built'] = df['date'].dt.year - df['built']
        #renaming built to age
    df = df.rename(columns={'built':'age'})
        
    #converting renovation into age of renovation at transaction, 
    #with 0 renovation converted to age of property    
    for i, yr in df.renovation.items(): 
        if yr == 0:
            df.loc[i,'renovation'] = df.loc[i,'age'] 
        else: 
            df.loc[i,'renovation'] = df.loc[i,'date'].year - df.loc[i, 'renovation']
        
    #renaming renovation to reno_age
    df = df.rename(columns={'renovation':'reno_age'})
        
    return df

def display_result(model_name, rmse, r2):
    """A simple print out function"""
    print("===============================")
    print(model_name)
    print("\tRMSE:     {:.0f}".format(rmse))
    print("\tR2 Score: {:.4f}".format(r2))
        
    return

def save_result_to_file(txt_filename, model_name, rmse, r2, compare):
    """A simple save-to-file function"""
    now = dt.datetime.now()
    with open("data/" + txt_filename, "a") as f:  #'a' for append, 'w' to write
        print("############################################", file=f)
        print(model_name, file=f)
        print("Ran on: " + now.strftime("%Y-%m-%d %H:%M hr"), file=f)
        print("############################################", file=f)
        print("RMSE:     {:.0f}".format(rmse), file=f)
        print("R2 Score: {:.4f}\n".format(r2), file=f)
        print("Actual vs Predicted Comparison", file=f)
        print("===============================", file=f)
        print(compare.head(), file=f)
        print('\n\n', file=f)
        
    return

    
def main():
    
    # DATA INPUT
    #######################################################
    try:                     #format provided in AIAP instructions and works in BASH
        con = sqlite3.connect('data/home_sales.db')
    except:                  #relative path that works for my home pc spyder
        con = sqlite3.connect('../data/home_sales.db')

    cursor = con.cursor()
        #look for names of tables in database and assigned the name to table_name
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_name = str(cursor.fetchall()[0][0])

    #writing database into raw_data with parsing of date column, then display shape and head()
    raw_data = pd.read_sql_query("SELECT * FROM " + table_name, con, parse_dates=['date'])

    # cleaning data in steps already outlined in EDA
    input_data = clean_data(raw_data)
    
    # Seperate out the outcome variable from the loaded dataframe
    output_var_name = 'price'
    output_var = input_data[output_var_name]
    input_data.drop(output_var_name, axis=1, inplace=True)


    # DATA ENGINEERING
    ########################################################
    # Subsetting the input_data: define features to keep
    feature_names = ['date','bedrooms','bathrooms','floors','waterfront',
                     'view','condition','review_score','basement_size',
                     'built','renovation','zipcode','living_room_size','lot_size']
    features = input_data[feature_names]

    # changing the built and renovation columns to age and reno_age.
    features = get_age(features)

    #converting the date to month only string
    features['date'] = features['date'].apply(lambda x: x.strftime('%B'))
    
    #update the feature_names to reflect new names: age and reno_age
    feature_names = list(features.columns)

    # Unskewing data by applying log function to output_var and selected features to unskew
    output_var_log = np.log(output_var)
#    col_to_unskew = ['lot_size','basement_size','living_room_size']
#    features[col_to_unskew] = features[col_to_unskew].apply(np.log1p)

    # Create the pipeline ...
    # 1. Pre-processing
    # Define variables made up of lists. Each list is a set of columns that will go through the same data transformations.
    numerical_features = ['bedrooms','bathrooms','floors',
                          'view','condition','review_score','basement_size',
                          'age','reno_age','living_room_size','lot_size'] # TASK: Define numerical column names
    categorical_features = ['zipcode','date']
    binary_features = ['waterfront']
    
    preprocess = make_column_transformer(
        (StandardScaler(),numerical_features), 
        (OneHotEncoder(), categorical_features), 
        remainder='passthrough'
    )


    # MODELS ASSEMBLY with optimised hyper-parameters
    ######################################################
    model_OLS = make_pipeline(preprocess,
        LinearRegression()
    )
    
    model_Ridge = make_pipeline(preprocess,
        Ridge(alpha=1)
    )
    
    model_Lasso = make_pipeline(preprocess,
        Lasso(alpha=0.0001)
    )
    
    model_DT = make_pipeline(preprocess,
        DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=38)
    )
    
    model_KN = make_pipeline(preprocess,
        KNeighborsRegressor(n_neighbors=9, weights='distance')
    )
    
    regressors = [('OLS',model_OLS), ('Ridge',model_Ridge), ('Lasso',model_Lasso),
                  ('Decision Tree',model_DT), ('KNeighbors',model_KN)]


    # TRAIN & SCORE REGRESSORS, printout and save to file
    #####################################################
    # Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(features, output_var_log, 
                                                        random_state=42, test_size=0.25) 
    
    #inverse log to get the price, round to whole numbers
    y_test = np.round(np.exp(y_test), decimals=0)

    for model_name, regressor in regressors:
        # fitting each model and then make prediction
        regressor.fit(x_train, y_train)
        pred_test = regressor.predict(x_test)
        
        #inverse log to get the price, round to whole numbers
        pred_test = np.round(np.exp(pred_test), decimals=0)
        
        # calculate the scores and difference
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        r2 = r2_score(y_test, pred_test)
        difference = y_test - pred_test
        
        ### printout on screen ###
        display_result(model_name, rmse, r2)
        
        #####################
        #  output to file
        #####################
        txt_filename = 'Regressions_Assemble.txt'
        compare = pd.DataFrame(list(zip(y_test, pred_test, difference)),
                               columns=['Actual Sale Price','Predicted Sale Price','Difference'])

        #output_filename = "data/Regression " + now.strftime("%Y-%m-%d %H:%M") + "\.txt"
        save_result_to_file(txt_filename, model_name, rmse, r2, compare)
        
    print("\nRegression result summaries are saved in data/" + txt_filename)


#### executing main()
if __name__ == '__main__':
    main()
