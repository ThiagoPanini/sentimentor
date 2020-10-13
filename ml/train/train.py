"""
This python script is responsible for reading, preparing and training a sentiment classification model from
e-commerce reviews taken from brazilian web-sites

* Metadata can be find at: https://www.kaggle.com/olistbr/brazilian-ecommerce
* Reference notebook: ../notebooks/EDA_BrazilianECommerce.ipynb

--- SUMMARY ---

1. Project Preparation
    1.1 Project Variables
    1.2 Logging Object
2. Reading Data
3. Prep Pipelines
    3.1 Initial Preparation
    3.2 Text Transformers
4. Modeling
    4.1 Model Training
    4.2 Evaluating Metrics
    4.3 Complete Solution
    4.4 Final Model Performance
    4.5 Saving pkl Files

---------------------------------------------------------------
Written by Thiago Panini - Latest version: October 12th 2020
---------------------------------------------------------------
"""


# Importing libs
import logging
from log.log_config import logger_config, generic_exception_logging
import os
import numpy as np
import pandas as pd
from ml.train.custom_transformers import *
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve


"""
-----------------------------------
----- 1. PROJECT PREPARATION ------
       1.1 Project Variables
-----------------------------------
"""

# Variables for address paths
DATA_PATH = 'ml/data'
PIPELINES_PATH = 'ml/pipelines'
MODELS_PATH = 'ml/models'

# Variables for reading the data
FILENAME = 'olist_order_reviews_dataset.csv'
COLS_READ = ['review_comment_message', 'review_score']
CORPUS_COL = 'review_comment_message'
TARGET_COL = 'target'

# Defining stopwords
PT_STOPWORDS = stopwords.words('portuguese')

# Variables for saving data
METRICS_FILEPATH = 'ml/train/performance.csv'

# Variables for retrieving model
MODEL_KEY = 'LogisticRegression'


"""
-----------------------------------
----- 1. PROJECT PREPARATION ------
       1.2. Logging Object
-----------------------------------
"""

# Creating a logging object
logger = logging.getLogger(__name__)
logger = logger_config(logger, level=logging.DEBUG, filemode='w')


"""
-----------------------------------
-------- 2. READING DATA ----------
-----------------------------------
"""

# Reading the data with text corpus and score - Handling possible errors: OK
logger.debug('Reading the data')
try:
    df = import_data(os.path.join(DATA_PATH, FILENAME), usecols=COLS_READ, verbose=False)
except Exception as e:
    generic_exception_logging(e, logger=logger, exit_flag=True)


"""
-----------------------------------
------- 3. PREP PIPELINES ---------
     3.1 Initial Preparation
-----------------------------------
"""

# Creating a dictionary for mapping the target column based on review score
score_map = {
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 1
}

# Creating a pipeline for the initial prep on the data
initial_prep_pipeline = Pipeline([
    ('mapper', ColumnMapping(old_col_name='review_score', mapping_dict=score_map, new_col_name=TARGET_COL)),
    ('null_dropper', DropNullData()),
    ('dup_dropper', DropDuplicates())
])

# Applying initial prep pipeline
logger.debug('Applying the initial_prep_pipeline on raw data')
try:
    df_prep = initial_prep_pipeline.fit_transform(df)
except Exception as e:
    generic_exception_logging(e, logger=logger, exit_flag=True)


"""
-----------------------------------
------- 3. PREP PIPELINES ---------
      3.2 Text Transformers
-----------------------------------
"""

# Defining regex transformers to be applied
regex_transformers = {
    'break_line': re_breakline,
    'hiperlinks': re_hiperlinks,
    'dates': re_dates,
    'money': re_money,
    'numbers': re_numbers,
    'negation': re_negation,
    'special_chars': re_special_chars,
    'whitespaces': re_whitespaces
}

# Building a text prep pipeline
text_prep_pipeline = Pipeline([
    ('regex', ApplyRegex(regex_transformers)),
    ('stopwords', StopWordsRemoval(PT_STOPWORDS)),
    ('stemming', StemmingProcess(RSLPStemmer())),
    ('vectorizer', TfidfVectorizer(max_features=300, min_df=7, max_df=0.8, stop_words=PT_STOPWORDS))
])

# Extracgint X and y
logger.debug('Extracting X and y variables for training')
try:
    X = df_prep[CORPUS_COL].tolist()
    y = df_prep[TARGET_COL]
except Exception as e:
    generic_exception_logging(e, logger=logger, exit_flag=True)

# Applying pipeline
logger.debug('Applying text_prep_pipeline on X data')
try:
    X_prep = text_prep_pipeline.fit_transform(X)
except Exception as e:
    generic_exception_logging(e, logger=logger, exit_flag=True)

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=.20, random_state=42)

# Saving states before prep pipeline
logger.debug('Saving X and y data into project folder')
try:
    df_prep[CORPUS_COL].to_csv(os.path.join(DATA_PATH, 'X_data.csv'), index=False)
    df_prep[TARGET_COL].to_csv(os.path.join(DATA_PATH, 'y_data.csv'), index=False)
except Exception as e:
    generic_exception_logging(e, logger=logger, exit_flag=True)


"""
-----------------------------------
---------- 4. MODELING  -----------
        4.1 Model Training
-----------------------------------
"""

# Specifing a Logistic Regression model for sentiment classification
logreg_param_grid = {
    'C': np.linspace(0.1, 10, 20),
    'penalty': ['l1', 'l2'],
    'class_weight': ['balanced', None],
    'random_state': [42],
    'solver': ['liblinear']
}

# Setting up the classifiers
set_classifiers = {
    'LogisticRegression': {
        'model': LogisticRegression(),
        'params': logreg_param_grid
    }
}

# Creating an object and training the classifiers
logger.debug('Training a sentiment classification model')
try:
    trainer = BinaryClassification()
    trainer.fit(set_classifiers, X_train, y_train, random_search=True, scoring='accuracy', verbose=0)
except Exception as e:
    generic_exception_logging(e, logger=logger, exit_flag=True)


"""
-----------------------------------
--------- 4. MODELING  -----------
    4.2 Evaluating Metrics
-----------------------------------
"""

# Evaluating metrics
logger.debug('Evaluating models performance')
try:
    performance = trainer.evaluate_performance(X_train, y_train, X_test, y_test, cv=5, save=True, overwrite=True, 
                                                performances_filepath=METRICS_FILEPATH)
except Exception as e:
    generic_exception_logging(e, logger=logger, exit_flag=True)


"""
-----------------------------------
--------- 4. MODELING  -----------
    4.3. Complete Solution
-----------------------------------
"""

# Returning the model to be saved
model = trainer.classifiers_info[MODEL_KEY]['estimator']

# Creating a complete pipeline for prep and predict
e2e_pipeline = Pipeline([
    ('text_prep', text_prep_pipeline),
    ('model', model)
])

# Defining a param grid for searching best pipelines options
"""param_grid = [{
    'text_prep__vectorizer__max_features': np.arange(550, 851, 50),
    'text_prep__vectorizer__min_df': [7, 9, 12, 15],
    'text_prep__vectorizer__max_df': [.4, .5, .6, .7]
}]
"""
param_grid = [{
    'text_prep__vectorizer__max_features': np.arange(600, 601, 50),
    'text_prep__vectorizer__min_df': [7],
    'text_prep__vectorizer__max_df': [.6]
}]


# Searching for best options
logger.debug('Searching for the best hyperparams combination')
try:
    grid_search_prep = GridSearchCV(e2e_pipeline, param_grid, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
    grid_search_prep.fit(X, y)
    logger.info(f'Done searching. The set of new hyperparams are: {grid_search_prep.best_params_}')
except Exception as e:
    generic_exception_logging(e, logger=logger, exit_flag=True)

# Returning the best options
logger.debug('Updating model hyperparams')
try:
    vectorizer_max_features = grid_search_prep.best_params_['text_prep__vectorizer__max_features']
    vectorizer_min_df = grid_search_prep.best_params_['text_prep__vectorizer__min_df']
    vectorizer_max_df = grid_search_prep.best_params_['text_prep__vectorizer__max_df']

    # Updating the e2e pipeline with the best options found on search
    e2e_pipeline.named_steps['text_prep'].named_steps['vectorizer'].max_features = vectorizer_max_features
    e2e_pipeline.named_steps['text_prep'].named_steps['vectorizer'].min_df = vectorizer_min_df
    e2e_pipeline.named_steps['text_prep'].named_steps['vectorizer'].max_df = vectorizer_max_df
except Exception as e:
    generic_exception_logging(e, logger=logger, exit_flag=True)

# Fitting the model again
logger.debug('Fitting the final model using the final pipeline')
try:
    e2e_pipeline.fit(X, y)
except Exception as e:
    generic_exception_logging(e, logger=logger, exit_flag=True)


"""
-----------------------------------
--------- 4. MODELING  -----------
    4.4 Final Model Performance
-----------------------------------
"""

# Retrieving performance for te final model after hyperparam updating
logger.debug('Evaluating final performance')
try:
    final_model = e2e_pipeline.named_steps['model']
    final_performance = cross_val_performance(final_model, X_prep, y, cv=5)
    final_performance = final_performance.append(performance)
    final_performance.to_csv(METRICS_FILEPATH, index=False)
except Exception as e:
    generic_exception_logging(e, logger=logger, exit_flag=True)


"""
-----------------------------------
--------- 4. MODELING  -----------
      4.5 Saving pkl files
-----------------------------------
"""

logger.debug('Saving pkl files')
# Creating folders for saving pkl files (if not exists)
try:
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    if not os.path.exists(PIPELINES_PATH):
        os.makedirs(PIPELINES_PATH)

    # Saving pkl files
    dump(initial_prep_pipeline, os.path.join(PIPELINES_PATH, 'initial_prep_pipeline.pkl'))
    dump(text_prep_pipeline, os.path.join(PIPELINES_PATH, 'text_prep_pipeline.pkl'))
    dump(e2e_pipeline, os.path.join(PIPELINES_PATH, 'e2e_pipeline.pkl'))
    dump(final_model, os.path.join(MODELS_PATH, 'sentiment_clf_model.pkl'))
    logger.info('Finished the module')
except Exception as e:
    generic_exception_logging(e, logger=logger, exit_flag=True)
