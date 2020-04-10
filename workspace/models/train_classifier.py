import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    """Load data from database file
    Input: database filepath
    Output: data split into X, Y and categories for training later.
        - X: training data with message column only
        - Y: 36 categories
        - category_names = column name of Y"""
    # load data from database
    con = sqlite3.connect(database_filepath)
    df = pd.read_sql(sql = 'select * from InsertTableName', con = con)
    df.dropna(axis = 0, inplace = True)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names
    

def tokenize(text):
    """Clean and tokenize the text in messages.
    Input: 
        - Text in X
    Output:
        - Cleaned tokens"""
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Build model with pipeline and grid search.
    Input:
        None
    Content:
        - Model pipeline with 2 transformer and 1 classifier
        - parameters for grid search
    Output:
        grid search on pipeline"""
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()), 
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {

        'vect__max_features': (None, 5000),
        'tfidf__use_idf': (True, False)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Using classification_report to evalute the model"""
    y_pred = model.predict(X_test)
    reports = []
    for i in range(36):
        reports.append(classification_report(Y_test.values[:, i], y_pred[:, i], target_names=category_names))
    return reports


def save_model(model, model_filepath):
    """Output the model into pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


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