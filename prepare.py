import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import scipy as sp
from pydataset import data
from env import user, password, host

import warnings
warnings.filterwarnings("ignore")

import acquire as acq




def prep_iris(df):
    '''
    Takes in a iris dataframe and returns a cleaned dataframe
    Arguments: df - a pandas dataframe with the expected feature names and columns
    Return: prep_iris - a dataframe with the cleaning operations performed on it
    '''
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    # Drop columns 
    columns_to_drop = ['species_id']
    df = df.drop(columns = columns_to_drop)
    df = df.rename(columns={'species_name': 'species'})
    return df



def prep_titanic(df):
    '''
    Takes in a titanic dataframe and returns a cleaned dataframe
    Arguments: df - a pandas dataframe with the expected feature names and columns
    Return: clean_df - a dataframe with the cleaning operations performed on it
    '''
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    # Drop columns 
    columns_to_drop = ['embarked', 'class', 'passenger_id', 'deck', "Unnamed: 0"]
    df = df.drop(columns = columns_to_drop)
    # encoded categorical variables
    dummy_df = pd.get_dummies(df[['sex', 'class', 'embark_town']], dummy_na=False, drop_first=[True, True])
    df = pd.concat([df, dummy_df], axis=1)
    return df



def prep_telco(df):
    '''
    Takes in a telco dataframe and returns a cleaned dataframe
    Arguments: df - a pandas dataframe with the expected feature names and columns
    Return: clean_df - a dataframe with the cleaning operations performed on it
    '''
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    # Drop columns 
    columns_to_drop = ['Unnamed: 0', 'contract_type_id', 'internet_service_type_id', 'payment_type_id']
    df = df.drop(columns = columns_to_drop)
    
    # df
    
    return df




def split_data(df, dataset=None):
    target_cols = {
        'telco': 'churn',
        'titanic': 'survived',
        'iris': 'species'
    }
    if dataset:
        if dataset not in target_cols.keys():
            print('please choose a real dataset tho')

        else:
            target = target_cols[dataset]
            train_val, test = train_test_split(
                df,
                train_size = 0.8
                stratify=df[target],
                random_state=seed)
            train, validate = train_test_split(
                train_val,
                train_size = 0.75,
                stratify = train_val[target],
                random_state = seed)
            return train, validate, test
    else:
        print('please specify what df we are splitting.')
        
        