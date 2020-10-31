import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline

from sqlalchemy import create_engine


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', engine)

    X = df.message.values
    Y = df.iloc[:, 4:]
    category_names = df.iloc[:, 4:].columns.tolist()

    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # pipeline = Pipeline([
    #     ('vect', CountVectorizer(tokenizer=tokenize, min_df=5)),
    #     ('tfidf', TfidfTransformer(use_idf=True)),
    #     ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10,
    #                                                          min_samples_split=10)))
    # ])

    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))
                         ])

    # Create parameters dictionary
    # parameters = {'vect__min_df': [1, 5],
    #               'tfidf__use_idf': [True, False],
    #               'clf__estimator__n_estimators': [10, 25],
    #               'clf__estimator__min_samples_split': [2, 5, 10]}

    parameters = {'clf__estimator__class_weight': [None],
                  'clf__estimator__n_estimators': [10, 20],
                  'clf__estimator__max_depth': [2, None],
                  'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__min_df': [1, 5],
                  'tfidf__use_idf': [True, False],
                  }

    model = GridSearchCV(estimator=pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath+'.sav', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
