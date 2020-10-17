"""
This python script puts together all the transformations needed for receiving a text input (comment or review) and
returning the sentiment associated. For this to be done, we load the pkl files for pipelines built on train.py script
and handle some common exceptions that we can face on production

* Metadata can be find at: https://www.kaggle.com/olistbr/brazilian-ecommerce
* Reference notebook: ../notebooks/EDA_BrazilianECommerce.ipynb

--- SUMMARY ---

1. Project Variables
2. Sentimentor Class
3. Main Program
    3.1 Making Predictions on Text

---------------------------------------------------------------
Written by Thiago Panini - Latest version: September 26th 2020
---------------------------------------------------------------
"""

# Python
from datetime import datetime
from pandas import DataFrame
from joblib import load
from os import getenv
from os.path import isfile
from dotenv import load_dotenv

# Reading env variables
_ENV_FILE = '.env'
if isfile(_ENV_FILE):
    load_dotenv(_ENV_FILE)


"""
-----------------------------------
------ 2. SENTIMENTOR CLASS -------
-----------------------------------
"""

class Sentimentor():

    def __init__(self, input_data):
        self.input_data = input_data
        self.pipeline = load(getenv('TEXT_PIPELINE'))
        self.model = load(getenv('MODEL'))

    def prep_input(self):
        """
        This method uses the self.data attribute to make modifications on input text dtype in order to make it
        applicable to the text prep pipeline (self.pipeline attribute)

        Parameters
        ----------
        None

        Returns
        -------
        :return: text_prep: input data after pipeline transform method [type: depends on input]

        Application
        -----------
        # Preparing the input data for making predictions
        text_input = 'some review or comment extracted online'
        sentimentor = Sentimentor(data=text_input)
        text_prep = sentimentor.prep_input()
        """

        # Verify if the type of input data
        if type(self.input_data) is str:
            self.input_data = [self.input_data]
        elif type(self.input_data) is DataFrame:
            self.input_data = list(self.input_data.iloc[:, 0].values)

        # Apply the pipeline to prepare the input data
        return self.pipeline.transform(self.input_data)

    def make_predictions(self):
        """
        This method is used for consuming the sentiment classification model and export results on demand

        Parameters
        ----------
        :param export_results: flag that guides exporting a csv file with predictions [type: bool, default: False]
        :param export_path: referente for storing the results [type: string, default: '../log_results']

        Returns
        -------
        :return: df_results: DataFrame with sentiment predictions for the given input [type: pd.DataFrame]
        """

        # Preparing the data and calling the classifier for making predictions
        text_list = self.prep_input()
        pred = self.model.predict(text_list)
        proba = self.model.predict_proba(text_list)[:, 1]

        # Analyzing the results and preparing the output
        class_sentiment = ['Positive' if c == 1 else 'Negative' for c in pred]
        class_proba = [p if c == 1 else 1 - p for c, p in zip(pred, proba)]

        # Building up a pandas DataFrame to delivery the results
        results = {
            'datetime_prediction': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'text_input': self.input_data,
            'prediction': pred,
            'class_sentiment': class_sentiment,
            'class_probability': class_proba
        }
        #df_results = DataFrame(results)

        # Exporting results
        """if export_results:
            now = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = str(getuser()) + '_prediction_' + str(now) + '.csv'
            df_results.to_csv(os.path.join(export_path, filename), index=False, sep=';', encoding='UTF-16')

            # Storing results in an unique file
            try:
                sentimentor_predictions = pd.read_csv(os.path.join(export_path, ALL_LOG_FILE))
                sentimentor_predictions = sentimentor_predictions.append(df_results)
                sentimentor_predictions.to_csv(os.path.join(export_path, ALL_LOG_FILE))
            except FileNotFoundError:
                # File log doesn't exists, creating one
                df_results.to_csv(os.path.join(export_path, ALL_LOG_FILE), index=False)"""

        return results