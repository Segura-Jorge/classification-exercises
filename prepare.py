import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import scipy as sp
from pydataset import data
from env import user, password, host

import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import acquire
import pandas as pd
import numpy as np
import os
from env import host, user, password



def get_connection(db, user=user, host=host, password=password):
    
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

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



def new_titanic_data():
    '''
    This function reads the titanic data from the Codeup db into a df.
    '''
    # Create SQL query.
    sql_query = 'SELECT * FROM passengers'
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    
    return df



def get_titanic_data():
    '''
    This function reads in titanic data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('titanic_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('titanic_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_titanic_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('titanic_df.csv')
        
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
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    
    return df



def new_telco_data():
    '''
    This function reads the telco data from the Codeup db into a df.
    '''
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    
    return df



def get_telco_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_telco_data()
        
        # Cache data
        df.to_csv('telco.csv')
        
    return df




# ############################################################################




import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ------------------- BASIC SPLIT FUNCTION  -------------------

def my_train_test_split(df, target):
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train[target])
    return train, validate, test
    

# ------------------- IRIS DATA -------------------

def clean_iris(df):

    '''Prepares acquired Iris data for exploration'''
    
    # drop column using .drop(columns=column_name)
    df = df.drop(columns='species_id')
    
    # remame column using .rename(columns={current_column_name : replacement_column_name})
    df = df.rename(columns={'species_name':'species'})
    
    # create dummies dataframe using .get_dummies(column_name,not dropping any of the dummy columns)
    dummy_df = pd.get_dummies(df['species'], drop_first=False)
    
    # join original df with dummies df using .concat([original_df,dummy_df], join along the index)
    df = pd.concat([df, dummy_df], axis=1)
    
    return df


def split_iris_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on species.
    return train, validate, test DataFrames.
    '''
    
    # splits df into train_validate and test using train_test_split() stratifying on species to get an even mix of each species
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    
    # splits train_validate into train and validate using train_test_split() stratifying on species to get an even mix of each species
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.species)
    return train, validate, test


def prep_iris(df):
    '''Prepares acquired Iris data for exploration'''
    
    # drop column using .drop(columns=column_name)
    df = df.drop(columns='species_id')
    
    # remame column using .rename(columns={current_column_name : replacement_column_name})
    df = df.rename(columns={'species_name':'species'})
    
    # create dummies dataframe using .get_dummies(column_name,not dropping any of the dummy columns)
    dummy_df = pd.get_dummies(df['species'], drop_first=False)
    
    # join original df with dummies df using .concat([original_df,dummy_df], join along the index)
    df = pd.concat([df, dummy_df], axis=1)
    
    # split data into train/validate/test using split_data function
    train, validate, test = split_iris_data(df)
    
    return train, validate, test

# ------------------- TITANIC DATA -------------------

def clean_titanic_data(df):
    '''
    This function will clean the data prior to splitting.
    '''
    # Drops any duplicate values
    df = df.drop_duplicates()

    # Drops columns that are already represented by other columns
    cols_to_drop = ['deck', 'embarked', 'class', 'passenger_id']
    df = df.drop(columns=cols_to_drop)

    # Fills the small number of null values for embark_town with the mode
    df['embark_town'] = df.embark_town.fillna(value='Southampton')

    # Uses one-hot encoding to create dummies of string columns for future modeling 
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na=False, drop_first=[True])
    df = pd.concat([df, dummy_df], axis=1)

    return df

def split_titanic_data(df):
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    # Creates the test set
    train, test = train_test_split(df, test_size = .2, random_state=123, stratify=df.survived)
    
    # Creates the final train and validate set
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.survived)
    
    return train, validate, test

def impute_titanic_mode(train, validate, test):
    '''
    Takes in train, validate, and test, and uses train to identify the best value to replace nulls in embark_town
    Imputes that value into all three sets and returns all three sets
    '''
    imputer = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
    train[['embark_town']] = imputer.fit_transform(train[['embark_town']])
    validate[['embark_town']] = imputer.transform(validate[['embark_town']])
    test[['embark_town']] = imputer.transform(test[['embark_town']])
    return train, validate, test

def impute_mean_age(train, validate, test):
    '''
    This function imputes the mean of the age column for
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test

def prep_titanic_data(df):
    '''
    Combines the clean_titanic_data, split_titanic_data, and impute_mean_age functions.
    '''
    df = clean_titanic_data(df)
    df = df.drop(columns='sex')
    df = df.drop(columns='embark_town')

    train, validate, test = split_titanic_data(df)
    
    train, validate, test = impute_mean_age(train, validate, test)

    return train, validate, test

# ------------------- TELCO DATA -------------------
def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=1349, 
                                        stratify=df.Churn)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=1349, 
                                   stratify=train_validate.Churn)
    return train, validate, test



def prep_telco(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.drop(columns=['InternetService', 'Contract', 'PaymentMethod'])
    df['gender'] = df.gender.map({'Female': 1, 'Male': 0})
    df['Partner'] = df.Partner.map({'Yes': 1, 'No': 0})
    df['Dependents'] = df.Dependents.map({'Yes': 1, 'No': 0})
    df['PhoneService'] = df.PhoneService.map({'Yes': 1, 'No': 0})
    df['PaperlessBilling'] = df.PaperlessBilling.map({'Yes': 1, 'No': 0})
    df.TotalCharges.fillna('20.20',inplace=True)
    df['Churn'] = df.Churn.map({'Yes': 1, 'No': 0})
    dummy_df = pd.get_dummies(df[['MultipleLines', 'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']],
                              drop_first=True)
    df = df.drop(columns=['MultipleLines', 'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'])
    df = pd.concat([df, dummy_df], axis=1)
    
    # split the data
    train, validate, test = split_telco_data(df)
    
    return train, validate, test


#-------------#####--------