## IMPORTS ##

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
import os
directory = os.getcwd()
seed = 3333
## __________________________________________________________________________________________________________________________________
## VARIABLES ##
directory = os.getcwd()


## __________________________________________________________________________________________________________________________________
## FUNCTIONS ##


##-------------------------------------------------------------------##

def get_db_url(database_name):
    """
    this function will:
    - take in a string database_name 
    - return a string connection url to be used with sqlalchemy later.
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{database_name}'

##-------------------------------------------------------------------##

def remove_columns(df, cols_to_remove):  
    """
    This function removes columns listed in arguement
    - cols_to_remove = ["col1", "col2", "col3", ...]
    returns DF w/o the columns.
    """
    df = df.drop(columns=cols_to_remove)
    return df

##-------------------------------------------------------------------##
def new_titanic_data(sql_query):
    """
    This function will:
    - take in the SQL query
    - create connection_url to mySQL
    - return a df off of the query from the titanic_db
    """
    url = get_db_url('titanic_db')
    
    sql_query = 'select * from passengers'
    
    return pd.read_sql(sql_query, url)

##-------------------------------------------------------------------##
def get_titanic_data():
    """
    This function will:
    - Check local directory for csv file
        - return csv if exists
    - if csv doesn't exist:
        - creates df from sql query
        - write df to csv
    - outputs titanic df
    """
    filename = 'titanic.csv'
    
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        return df
    else:
        df = new_titanic_data(sql_query)
        df.to_csv(filename)
        return df
    
##-------------------------------------------------------------------##
def new_iris_data():
    '''
    This function reads the iris data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT 
                    species_id,
                    species_name,
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                FROM measurements
                JOIN species USING(species_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('iris_db'))
    
    return df

##-------------------------------------------------------------------##
def get_iris_data():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('iris_df.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('iris_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_iris_data()
        
        # Cache data
        df.to_csv('iris_df.csv')
        
    return df

##-------------------------------------------------------------------##
def new_telco_data(sql_query):
    """
    This function will:
    - take in a SQL_query
    - create a connection_url to mySQL
    - return a df of the given query from the telco_db
    """
    
    sql_query = """
        SELECT * FROM customers
        JOIN contract_types USING (contract_type_id)
        JOIN internet_service_types USING (internet_service_type_id)
        JOIN payment_types USING (payment_type_id)
        """
    
    url = get_db_url('telco_churn')
    
    return pd.read_sql(sql_query, url)

##-------------------------------------------------------------------##
def get_telco_data():
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - if csv doesn't exist:
        - creates df of sql query
        - writes df to csv
    - outputs telco df
    """
    filename = 'telco.csv'
    
    if os.path.isfile(filename): 
        df = pd.read_csv(filename)
        return df
    else:
        df = new_telco_data(sql_query)

        df.to_csv(filename)
        return df

##-------------------------------------------------------------------##
def split_data_iris(df):
    seed = 3333
    train, test = train_test_split(df,
                               train_size = 0.8,
                               stratify = df.species,
                               random_state=seed)
    train, validate = train_test_split(train,
                                  train_size = 0.75,
                                  stratify = train.species,
                                  random_state=seed)
    return train, validate, test

##-------------------------------------------------------------------##
def prep_iris(iris) -> pd.DataFrame:
    '''
    prep_iris will take a single positional argument,
    a single pandas DataFrame,
    and will output a cleaned version of the dataframe
    this is expected to receive the data output by 
    get_iris_data from acquire module, see documentation
    for acquire.py for further details
    return: pd.DataFrame
    '''
    # drop that species_id column:
    iris = iris.drop(columns='species_id')
    # rename that species_name column into species for cleanliness:
    iris = iris.rename(columns={'species_name':'species'})
    return iris

##-------------------------------------------------------------------##
def split_data_titanic(df):
    seed = 3333
    train, test = train_test_split(df,
                               train_size = 0.8,
                               stratify = df.survived,
                               random_state=seed)
    train, validate = train_test_split(train,
                                  train_size = 0.75,
                                  stratify = train.survived,
                                  random_state=seed)
    return train, validate, test

##-------------------------------------------------------------------##
def prep_titanic(titanic) -> pd.DataFrame:
    '''
    prep_titanic will take in a single pandas DataFrame, titanic
    as expected from the acquire.py return of get_titanic_data
    it will return a single cleaned pandas dataframe
    of this titanic data, ready for analysis.
    '''
    titanic = titanic.drop(columns=[
        'Unnamed: 0',
        'passenger_id',
        'embarked',
        'deck',
        'class'
    ])
    titanic.loc[:,'age'] = titanic.age.fillna(round(titanic.age.mode())).values
    titanic.loc[:, 'embark_town'] = titanic.embark_town.fillna('Southampton')
    return titanic

##-------------------------------------------------------------------##
def split_data_telco(df):
    seed = 3333
    train, test = train_test_split(df,
                               train_size = 0.8,
                               stratify = df.churn,
                               random_state=seed)
    train, validate = train_test_split(train,
                                  train_size = 0.75,
                                  stratify = train.churn,
                                  random_state=seed)
    return train, validate, test

##-------------------------------------------------------------------##
def prep_telco(telco) -> pd.DataFrame:
    '''
    prep_telco will take in a a single pandas dataframe
    presumed of the same structure as presented from 
    the acquire module's get_telco_data function (refer to acquire docs)
    returns a single pandas dataframe with redudant columns
    removed and missing values filled.
    '''
    telco = telco.drop(
    columns=[
        'Unnamed: 0',
        'internet_service_type_id',
        'payment_type_id',
        'contract_type_id',   
    ])
    telco.loc[:,'internet_service_type'] = telco.internet_service_type.\
    fillna('no internet')
    telco = telco.set_index('customer_id')
    telco.loc[:,'total_charges'] = (telco.total_charges + '0')
    telco.total_charges = telco.total_charges.astype(float)
    return telco

##-------------------------------------------------------------------##

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
                train_size=0.8,
                stratify=df[target],
                random_state=seed)
            train, validate = train_test_split(
                train_val,
                train_size=0.75,
                stratify=train_val[target],
                random_state=seed)
            return train, validate, test
    else:
        print('please specify what df we are splitting.')
        

##__________________________________________________________________________________________________________________________________##
## SCRIPTS ##



##__________________________________________________________________________________________________________________________________##