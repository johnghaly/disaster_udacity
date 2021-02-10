# import libraries
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import sys


def load_data(database_filepath):
    """
    Load data using a sqllite datatbase filepath
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('messages_categories', engine)
    X =  df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])
    return X,Y,category_names


def tokenize(text):
    """
    Tokenize function, does basic tokenization such as
    Converting text into tokens, removing stop words, convert to lower case
    and applying a lemmatizer
    """
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a classification model (RandomForest) within a pipeline
    containing the preprocessing step
    The classifier is tuned through GridSearch with 5 fold cross-validation
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('clf', RandomForestClassifier())
    ])
# Commenting since grid search is very slow
#     parameters = {'clf__max_depth': [2, 4, 6, 8, 10, None],
#      'clf__max_features': ['auto', 'sqrt', None],
#      'clf__n_estimators': [100, 200, 300, 400, 500]}
    # Simplified param list to speed up running
    print("Grid search")
    parameters = {'clf__n_estimators': [10, 100]}
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1, scoring='f1_micro', cv=5)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model performance by calculating basic accuracy and
    the full metrics provided by the classification_report from sklearn
    """
    # predict on test data
    Y_pred = model.predict(X_test)
    accuracy = (Y_pred == Y_test).mean()
    print('Accuracy: ')
    print(accuracy)
    print('Classification report: ')
    for i in range(0,36):
        print(classification_report(Y_test[category_names[i]].values, Y_pred[:,i], target_names=category_names[i]))


def save_model(model, model_filepath):
    """
    Save the model into the .pkl format for later deployement/use
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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