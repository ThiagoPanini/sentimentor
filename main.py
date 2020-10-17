"""
This main.py module is responsible for encapsulating all other modules
needed for receiving an input from an API call and return the request
"""

# Libraries
import logging
from apps.sentimentor import Sentimentor
from log.log_config import logger_config, generic_exception_logging
from joblib import load
from os.path import join


"""
-----------------------------------
------ 1. PROJECT VARIABLES -------
-----------------------------------
"""

# Creating a logging object
logger = logging.getLogger(__name__)
logger = logger_config(logger, level=logging.DEBUG, filemode='a')

# Variables for path address
PIPE_PATH = 'ml/pipelines'
MODEL_PATH = 'ml/models'
LOG_PATH = 'log/application_log.log'

# Variables for pkl files
TEXT_PIPE = 'text_prep_pipeline.pkl'
MODEL = 'sentiment_clf_model.pkl'
TRAIN = False

"""
-----------------------------------
-------- 3. MAIN PROGRAM ----------
  3.1 Making Predictions on Text
-----------------------------------
"""

if __name__ == '__main__':

    # Training the model (if applicable)
    if TRAIN:
        from ml.train import train
    
    # Reading pkl files
    logger.debug('Reading pkl files')
    try:
        pipeline = load(join(PIPE_PATH, TEXT_PIPE))
        model = load(join(MODEL_PATH, MODEL))
    except Exception as e:
        generic_exception_logging(e, logger=logger, exit_flag=True)

    # Fake input
    text_input = 'Não gostei deste produto. Não me atendeu e custou muito caro'
    #text_input = pd.read_csv('test/test_data.csv', sep=';')

    # Instancing an object and executing predictions
    sentimentor = Sentimentor(data=text_input, pipeline=pipeline, model=model)
    logger.debug('Creating a sentimentor object and making predictions')
    try:
        output = sentimentor.make_predictions()
        logger.info('Module finished with success status')
        exit()
    except Exception as e:
        generic_exception_logging(e, logger=logger, exit_flag=True)