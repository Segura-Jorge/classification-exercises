## IMPORTS ##

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy as sp
from pydataset import data
from env import user, password, host
from env import get_db_url

import os

import warnings
warnings.filterwarnings("ignore")

## __________________________________________________________________________________________________________________________________
## VARIABLES ##



## __________________________________________________________________________________________________________________________________
## FUNCTIONS ##

directory = os.getcwd()

def get_db_url(database_name):
    """
    this function will:
    - take in a string database_name 
    - return a string connection url to be used with sqlalchemy later.
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{database_name}'
    
##-------------------------------------------------------------------##

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

##__________________________________________________________________________________________________________________________________##
## SCRIPTS ##



##__________________________________________________________________________________________________________________________________##