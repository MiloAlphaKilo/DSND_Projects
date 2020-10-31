import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    column_names = []
    categories_col_list = ['id', 'message', 'original', 'genre', 'categories']

    print(df.columns)
    print(df.head())

    df = df.join(df['categories'].str.split(';', expand=True).add_prefix('cat_'))

    row = df.iloc[0]

    for k, v in row.items():
        if k in categories_col_list:
            v = k
            column_names.append(v)
        else:
            column_names.append(v[:-2])

    df.columns = column_names

    for column in df.iloc[:, 5:]:

        df[column] = df[column].astype(str).str.slice(start=-1)
        df[column] = df[column].astype(int)

    df = df.drop('categories', axis=1)
    df.sort_values('id', inplace=True)
    df.drop_duplicates(keep='first', inplace=True)

    print(df.columns)
    print(df.head())

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_messages', engine, index=False)


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
