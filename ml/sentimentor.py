"""
---------------------------------------------------------
                    Resumo do Módulo
---------------------------------------------------------


---------------------------------------------------------
                          FAQ
---------------------------------------------------------

"""
# Third
from log.log_config import logger_config
from ml.custom_transformers import *

# Python
import logging
from os import getenv
from os.path import isfile
from dotenv import load_dotenv
from pandas import read_csv, DataFrame
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from joblib import load, dump
from json import dumps


"""
-----------------------------------
------ 1. PROJECT VARIABLES -------
-----------------------------------
"""

# Creating a logger object
logger = logger_config(logger=logging.getLogger(__file__), filemode='a')

# Variables for reading the data
COLS_READ = ['review_comment_message', 'review_score']
CORPUS_COL = 'review_comment_message'
TARGET_COL = 'target'

# Defining stopwords
PT_STOPWORDS = stopwords.words('portuguese')

# Messages
WARNING_MESSAGE = f'Module {__file__} finished with ERROR status'

# .env variables
_ENV_FILE = '.env'
if isfile(_ENV_FILE):
    load_dotenv(_ENV_FILE)
else:
    logger.warning('Error on loading Env variables: .env file not found')


"""
-----------------------------------
------ 2. SENTIMENTOR CLASS -------
-----------------------------------
"""

class Sentimentor():

    def load_pkl(self):
        self.pipeline = load(getenv('TEXT_PIPELINE'))
        self.model = load(getenv('MODEL'))

    def train(self):
        """
        This method is used for training a sentiment classification model

        Parameters
        ----------
        None

        Return
        ------
        This method returns nothing except the prep pipeline and model pkl 
        files set as class attributes
        """

        logger.info('Started model training')
        
        """
        -----------------------------------
        -------- 2.1 Reading Data ---------
        -----------------------------------
        """

        # Raw data
        logger.debug('Reading raw data')
        try:
            raw_data = read_csv(getenv('RAW_DATA'), usecols=COLS_READ)
        except Exception as e:
            logger.error(e)
            logger.warning(WARNING_MESSAGE)
            exit()


        """
        -----------------------------------
        ------- 2.2 Prep Pipelines --------
             2.2.1 Initial Preparation
        -----------------------------------
        """

        # Mapping review score into classes
        score_map = {
            1: 0,
            2: 0,
            3: 0,
            4: 1,
            5: 1
        }

        # Building initial prep pipeline
        initial_prep_pipeline = Pipeline([
            ('mapper', ColumnMapping(old_col_name='review_score', mapping_dict=score_map, new_col_name=TARGET_COL)),
            ('null_dropper', DropNullData()),
            ('dup_dropper', DropDuplicates())
        ])

        logger.debug('Applying initial_prep_pipeline on raw data')
        try:
            df_prep = initial_prep_pipeline.fit_transform(raw_data)
        except Exception as e:
            logger.error(e)
            logger.warning(WARNING_MESSAGE)
            exit()
            

        """
        -----------------------------------
        ------- 2.2 Prep Pipelines --------
              2.2.2 Text Transformers
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

        # Extracting X and y
        X = df_prep[CORPUS_COL].tolist()
        y = df_prep[TARGET_COL]

        # Applying pipeline
        logger.debug('Applying text_prep_pipeline on X data')
        try:
            X_prep = text_prep_pipeline.fit_transform(X)
        except Exception as e:
            logger.error(e)
            logger.warning(WARNING_MESSAGE)
            exit()

        # Splitting the data into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=.20, random_state=42)


        """
        -----------------------------------
        ---------- 2.3 Modeling -----------
               2.3.1 Model Training
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
        logger.debug('Training the model')
        try:
            trainer = BinaryClassification()
            trainer.fit(set_classifiers, X_train, y_train, random_search=True, 
                        scoring='accuracy', verbose=0)
        except Exception as e:
            logger.error(e)
            logger.warning(WARNING_MESSAGE)
            exit()


        """
        -----------------------------------
        ---------- 2.3 Modeling -----------
             2.3.2 Evaluating Metrics
        -----------------------------------
        """

        logger.debug('Evaluating metrics')
        try:
            performance = trainer.evaluate_performance(X_train, y_train, X_test, y_test, cv=5, save=True, overwrite=True, 
                                                        performances_filepath=getenv('MODEL_METRICS'))
        except Exception as e:
            logger.error(e)
            logger.warning(WARNING_MESSAGE)
            exit()


        """
        -----------------------------------
        ---------- 2.3 Modeling -----------
             2.3.3 Complete Solution
        -----------------------------------
        """

        # Returning the model to be saved
        model = trainer.classifiers_info[getenv('MODEL_KEY')]['estimator']

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
            logger.error(e)
            logger.warning(WARNING_MESSAGE)
            exit()

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
            logger.error(e)
            logger.warning(WARNING_MESSAGE)
            exit()

        # Fitting the model again
        logger.debug('Fitting the final model')
        try:
            e2e_pipeline.fit(X, y)
        except Exception as e:
            logger.error(e)
            logger.warning(WARNING_MESSAGE)
            exit()


        """
        -----------------------------------
        ---------- 2.3 Modeling -----------
             2.3.4 Final Performance
        -----------------------------------
        """

        # Retrieving performance for te final model after hyperparam updating
        logger.debug('Evaluating final performance')
        try:
            final_model = e2e_pipeline.named_steps['model']
            final_performance = cross_val_performance(final_model, X_prep, y, cv=5)
            final_performance = final_performance.append(performance)
            final_performance.to_csv(getenv('MODEL_METRICS'), index=False)
        except Exception as e:
            logger.error(e)
            logger.warning(WARNING_MESSAGE)
            exit()


        """
        -----------------------------------
        ---------- 2.3 Modeling -----------
               2.3.5 Saving pkl files
        -----------------------------------
        """

        logger.debug('Saving pkl files')
        try:
            dump(initial_prep_pipeline, getenv('INITIAL_PIPELINE'))
            dump(text_prep_pipeline, getenv('TEXT_PIPELINE'))
            dump(e2e_pipeline, getenv('E2E_PIPELINE'))
            dump(final_model, getenv('MODEL'))
            logger.info('Finished training')
        except Exception as e:
            logger.error(e)
            logger.warning(WARNING_MESSAGE)
            exit()

        # Loading pkl files as attributes of sentimentor object after training
        self.load_pkl()

    def prep_input_data(self, input_data):
        """
        This method is used for preparing the input data for making predictions

        Parameters
        ----------
        :param input_data: input data that comes from API user [type: str or DataFrame]

        Returns
        -------
        :returns prep_data: input data after text prep pipeline [type: np.array]
        """

        if type(input_data) is str:
            input_data = [input_data]
        elif type(input_data) is DataFrame:
            input_data = list(input_data.iloc[:, 0].values)

        return self.pipeline.transform(input_data)

    def make_predictions(self, input_data):
        """
        This method is used for making predictions given the user input trough the API

        Parameters
        ----------
        :param input_data: input data that comes from API user [type: str or DataFrame]

        Returns
        -------
        :returns json_pred: input data after text prep pipeline [type: json]
        """

        # Loading pkl files
        self.load_pkl()

        # Preparing the data and calling the classifier for making predictions
        logger.debug('Making predictions')
        try:
            text_list = self.prep_input_data(input_data)
            pred = self.model.predict(text_list)
            proba = self.model.predict_proba(text_list)[:, 1]

            # Analyzing the results and preparing the output
            class_sentiment = ['Positive' if c == 1 else 'Negative' for c in pred][0]
            class_proba = [p if c == 1 else 1 - p for c, p in zip(pred, proba)][0]

            # Building up a pandas DataFrame to delivery the results
            results = {
                "datetime_prediction": str(datetime.now().strftime('%d-%m-%Y %H:%M:%S')),
                "text_input": str(input_data),
                "class_sentiment": str(class_sentiment),
                "class_probability": float(round(class_proba, 4))
            }
            logger.info('Sucessfuly returned predictions for received data')
        except Exception as e:
            logger.error(e)
            logger.warning('Error on making predictions')
            

        return dumps(results)
