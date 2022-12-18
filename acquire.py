import os
import pandas as pd
import numpy as np
from pydataset import data
from scipy import stats 
from env import username, host, password

# ---------------------------Get Connection Function-----------------------------------------------------
def get_connection(db, user=username, host=host, password=password):
    '''This will return the databases in sequeul ace
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'     # sequel login 
# -----------------------------------Get Tidy DataFrame-------------------------------------------------------------------
def get_tidy_data():
    filename = "tidy_data_attendence.csv"       # Tidy Data CSV
    if os.path.isfile(filename):           
        return pd.read_csv(filename)        # filename returned in system files
    else:
        df = pd.read_sql('SELECT * FROM attendance', get_connection('tidy_data'))       # read the SQL query into a dataframe
        df.to_csv(filename)         # Write that dataframe to save csv file. //"caching" the data for later.
        return df       # return df
#-------------------------------------Get Titanic DataFrame------------------------------------------------------------------------------------------
def get_titanic():
    filename = "titanic.csv"        # Titanic Data CSV
    if os.path.isfile(filename):
        return pd.read_csv(filename)        # filename returned in system files
    else:
        df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))      # read the SQL query into a dataframe
        df.to_csv(filename)     # Write that dataframe to save csv file. //"caching" the data for later.
        return df        # Return the dataframe to the calling code
#-------------------------------------Get Iris Datafram-----------------------------------------------------------------------------------------    
def get_iris():
    filename = "iris.csv"       #iris CSV
    if os.path.isfile(filename):
        return pd.read_csv(filename)        # filename returned in system files
    else:
        df = pd.read_sql('''
        SELECT * FROM measurements as a 
        JOIN species as b
        USING (species_id);''', get_connection('iris_db'))      # Read the SQL query into a dataframe
        df.to_csv(filename)        # Write that dataframe to save csv file. //"caching" the data for later.
        return df       # Return the dataframe to the calling code
#-------------------------------------Get Telco Datafram-----------------------------------------------------------------------------------------    
def get_telco_data():
    filename = "telco.csv"      #iris CSV

    if os.path.isfile(filename):
        return pd.read_csv(filename)        # filename returned in system files
    else:
        df = pd.read_sql('''
        SELECT *
        FROM customers AS a
        JOIN contract_types as b
        USING (contract_type_id)
        JOIN internet_service_types as c
        ON a.internet_service_type_id = c.internet_service_type_id
        lEFT JOIN payment_types as d
        ON a.payment_type_id = d.payment_type_id;''', get_connection('telco_churn'))        # Read the SQL query into a dataframe
        df.to_csv(filename)          # Write that dataframe to save csv file. //"caching" the data for later.
        return df        # Return the dataframe to the calling code