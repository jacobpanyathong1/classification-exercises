import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ------------------------------------------------------------------------------------------------

def my_train_test_split(df, target):
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train[target])
    return train, validate, test
    

#-------------------------------------------------------------------------------------------------

def clean_iris(df): 
    to_drop = df.columns[1:3]
    df.drop(columns=to_drop, inplace=True)
    species = df.rename(columns={'species_name': 'species'}, inplace=True)
    dummies = pd.get_dummies(df[['species']])
    df= pd.concat([df, dummies], axis=1)
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
                                        random_state=123, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test

def prep_telco(df):
    '''
     This function cleans the telco data to prepare for splitting
    '''
    df.drop(columns=['internet_service_type_id.1', 'payment_type_id.1'], inplace=True) #Drop duplicate columns
    encoded_vars = pd.get_dummies(df[['contract_type', 'internet_service_type', 'payment_type']])
    pd.set_option('display.max_columns', 500) # Display all columns
    df = pd.concat([telco_df, encoded_vars], axis=1)
    df['total_charges'] = (df.total_charges + '0').astype('float') # Change total charges to float
    df['total_charges'].dtype
    df['gender'] = df.gender.map({'Female': 1, 'Male': 0})
    df['partner'] = df.partner.map({'Yes': 1, 'No': 0})
    df['dependents'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['phone_service'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['churn'] = df.churn.map({'Yes': 1, 'No': 0})
    return df

    # split the data
    #train, validate, test = split_telco_data(df)
    
    #return train, validate, test