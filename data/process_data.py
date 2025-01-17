# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load data from 2 main csv files messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id',how='inner')
    return df


def clean_data(df):
    """
    Preprocess the data doing basic cleaning
    Transforming concatinated data into proper columns with binary indicators
    """
    # create a dataframe of the 36 individual category columns
    categories= df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.head(1)
    # use this row to extract a list of new column names for categories.
    # up to the second to last character of each string with slicing
    category_colnames = []
    for index, value in row.items():
        category_colnames.append(value.values[0][0:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[-1])
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
        # Cap results to 1
        categories[column] = categories[column].apply(lambda x : 1 if x >=1 else 0)
    # Replace categories column in df with new category columns
    df = df.drop(['categories'], axis = 1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # Removing duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Saves data into sqlite format using the database_filename parameter 
    """
    # Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages_categories', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()