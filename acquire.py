## IMPORTS ##

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from pydataset import data
from env import user, password, host
from env import get_db_url


__________________________________________________________________________________________________________________________________
## VARIABLES ##



__________________________________________________________________________________________________________________________________
## FUNCTIONS ##

def get_db_url(database_name):
    """
    this function will:
    - take in a string database_name 
    - return a string connection url to be used with sqlalchemy later.
    """
    return f'mysql+pymysql://{user}:{password}@{host}/{database_name}'
    
##-------------------------------------------------------------------##

def get_titanic_data1(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a connection url (get_db_url)
    - return a df of the given query from the titanic_db
    """
    # create the connection url
    url = get_db_url('titanic_db')
    
    return pd.read_sql(SQL_query, url)

##-------------------------------------------------------------------##

def get_iris_data1(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a connection url (get_db_url)
    - return a df of the given query from the iris_db
    """
    # create the connection to db
    url = get_db_url('iris_db')
    
    return pd.read_sql(SQL_query, url)

##-------------------------------------------------------------------##

def get_telco_data1(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a connection url (get_db_url)
    - return a df of the given query from the telco_churn
    """
    # create the connection to db
    url = get_db_url('telco_churn')
    
    return pd.read_sql(SQL_query, url)

##-------------------------------------------------------------------##

def get_titanic_data(SQL_query, directory, filename="titanic.csv"):
    """
    This function will:
    - Check local directory for csv file
        - return df if file exists
        - If csv doesn't exist:
            - create a df of the sql query
            - write the df to a csv file
    - return titanic df
    """
    # Checks if csv exists
    if os.path.exists(directory + filename):
        #if YES
        df = pd.read_csv(filename)
        return df
    
    # if NOT
    else:
        # obtaining new data from sql
        df = get_titanic_data1(SQL_query)
        
        # convert to csv
        df.to_csv(filename)
        return df
    
##-------------------------------------------------------------------##

def get_iris_data(SQL_query, directory, filename="iris.csv"):
    """
    This function will:
    - Check local directory for csv file
        - return df if file exists
        - If csv doesn't exist:
            - create a df of the sql query
            - write the df to a csv file
    - return titanic df
    """
    # Checks if csv exists
    if os.path.exists(directory + filename):
        #if YES
        df = pd.read_csv(filename)
        return df
    
    # if NOT
    else:
        # obtaining new data from sql
        df = get_iris_data1(SQL_query)
        
        # convert to csv
        df.to_csv(filename)
        return df
    
##-------------------------------------------------------------------##
    
def get_telco_data(SQL_query, directory, filename="telco.csv"):
    """
    This function will:
    - Check local directory for csv file
        - return df if file exists
        - If csv doesn't exist:
            - create a df of the sql query
            - write the df to a csv file
    - return titanic df
    """
    # Checks if csv exists
    if os.path.exists(directory + filename):
        #if YES
        df = pd.read_csv(filename)
        return df
    
    # if NOT
    else:
        # obtaining new data from sql
        df = get_telco_data1(SQL_query)
        
        # convert to csv
        df.to_csv(filename)
        return df
    
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


__________________________________________________________________________________________________________________________________
## SCRIPTS ##



__________________________________________________________________________________________________________________________________