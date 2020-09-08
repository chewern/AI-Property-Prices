# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:24:30 2020
@author: CHEW ERN
"""
import datetime as dt
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, classification_report, roc_curve, auc

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

def classify_into_bins(prices):
    '''this function does the bin sorting for the prices'''
    bins_edges = [8000000, 2000000, 1500000, 1000000, 900000, 800000, 
                  700000, 650000, 600000, 550000, 525000, 500000, 
                  475000, 450000, 425000, 400000, 375000, 350000, 
                  325000, 300000, 275000, 250000, 225000, 200000, 50000]
    
    #making a list of bin labels
    labels = list(range(len(bins_edges)-1))
    
    price_bin_index = [] #initialize the bin index
    for price in prices:
        for i, bin_edge in enumerate(bins_edges[1:]):  #first bin edge need not check
            if (price >= bin_edge):
                price_bin_index.append(i)  #found the bin, and then break loop
                break
    
    assert len(prices) == len(price_bin_index), "Bin Error!" 
    
    return price_bin_index, labels

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

def results_print(model_name, accuracy, precision, recall, f1):
    """function for printing out the results"""
    print("###############################################################")
    print(model_name + ' 24 bins')
    print("========================")
    print("Accuracy: {:.5f}".format(accuracy))
    print("Precision: {:.5f}".format(precision))
    print("Recall: {:.5f}".format(recall))
    print("F1 Score: {:.5f}\n".format(f1))
    return

def confusion_matrix_plot(labels, model_name, cm):
    """function to plot confusion heatmap and save plot to file"""   
    
    #setting filepath and print to inform user of file location
    filepath = "data/_" + model_name + "_confusion.png"
    print('Confusion matrix plot is saved in: ' + filepath)
    
    #plotting
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt="d"); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels (bin index)');ax.set_ylabel('True labels (bin index)'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
    
    #saving to file
    plt.savefig(fname=filepath)
    #plt.show(block=False)
    plt.close()
    return

def plot_class_error(y_test, pred_test, model_name):
    """function plots and save the classification error plot"""
    
    #setting filepath and print to inform user of file location
    filepath = "data/_" + model_name + "_class_error.png"
    print('Classification error plot is saved in: ' + filepath + '\n')

    #calculating data for the plot
    difference = y_test - pred_test
    d = Counter(difference)

    #plotting
    plt.bar(d.keys(), d.values())
    plt.xlim(-20,20)
    plt.ylim(0,1400)
    plt.xlabel('Mis-classification Distance (Actual Class - Predicted Class)')
    plt.ylabel('Frequency')
    plt.title('Plot of ' + model_name + ' "Classification Error" Distribution')
    
    #saving to file
    plt.savefig(fname=filepath)

    #plt.show(block=False)
    plt.close()
    return

def save_to_file(txt_filename, model_name, accuracy, precision, recall, f1):
    """function saves output to file"""
    
    #sets the filepath and print to inform user of file location
    filepath = "data/" + txt_filename
    print('Results are saved in: ' + filepath +'\n')
        
    now = dt.datetime.now()
    #saving to file the following:
    with open("data/" + txt_filename, "a") as f:  #'a' for append, 'w' to write
        print("############################################", file=f)
        print(model_name + ' using 24 bins', file=f)
        print("\t\tRan on: " + now.strftime("%Y-%m-%d %H:%M hr"), file=f)
        print("############################################", file=f)
        print("Accuracy: {:.5f}".format(accuracy), file=f)
        print("Precision: {:.5f}".format(precision), file=f)
        print("Recall: {:.5f}".format(recall), file=f)
        print("F1 Score: {:.5f}\n\n".format(f1), file=f)
    return

def main():
    
    # DATA INPUT
    #######################################
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
    target_name = 'price'
    prices_list = input_data[target_name]
    input_data.drop(target_name, axis=1, inplace=True)
    
    # classifying the prices into bins, returning bin index and bin labels
    price_bin_index, labels = classify_into_bins(prices_list)

    # DATA ENGINEERING / MODEL DEFINITION
    ########################################
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

    # MODELS ASSEMBLY
    ######################################################
    model_KNN = make_pipeline(preprocess,
        KNeighborsClassifier(n_neighbors=30)
    )

    model_MLP = make_pipeline(preprocess,
        MLPClassifier(hidden_layer_sizes=100, alpha=0.1, random_state=1)
    )

    model_SVC = make_pipeline(preprocess,
        SVC(C=1000, gamma=10)
    )
    
    model_GBC = make_pipeline(preprocess,
        GradientBoostingClassifier(n_estimators=120, random_state=1)
    )
    
    clfs = [('KNeighborsClassifier',model_KNN), ('MLP_Classifier',model_MLP),
            ('SVM',model_SVC), ('Gradient_Boosting',model_GBC)]
    
    
    # TRAIN & SCORE Classifiers, printout and save to file
    #######################################################
    # Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(features, price_bin_index, random_state=42, test_size=0.25) 

    for model_name, clf in clfs:
        # fitting each model and then make prediction
        clf.fit(x_train, y_train)
        pred_test = clf.predict(x_test)

        # SCORING/EVALUATION of Classifier: Accuracy, Precision, Recall and F1
        accuracy = accuracy_score(y_test, pred_test)
        precision = precision_score(y_test, pred_test, average='weighted')
        recall = recall_score(y_test, pred_test, average='weighted')
        f1 = f1_score(y_test, pred_test, average='weighted')
        
        # Compute, plot and save the Confusion Matrix to data folder
        cm = confusion_matrix(y_test, pred_test)    
    
        # print evaluation results, confusion matrix and classification error
        results_print(model_name, accuracy, precision, recall, f1)

        #saving scores to file
        txt_filename = 'Classifications.txt'
        save_to_file(txt_filename, model_name, accuracy, precision, recall, f1)
    
        #saving confusion matrix and classification error plot to files
        #confusion_matrix_plot(labels, model_name, cm)
        #plot_class_error(y_test, pred_test, model_name)

#### executing main()
if __name__ == '__main__':
    main()
