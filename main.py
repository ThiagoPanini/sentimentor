"""
This main.py module is responsible for encapsulating all other modules
needed for receiving an input from an API call and return the request
"""

# Libraries
import logging
from dotenv import load_dotenv
from os import getenv
from os.path import join, dirname, isfile
from apps.sentimentor import Sentimentor
from log.log_config import logger_config, generic_exception_logging
from joblib import load
from os.path import join


"""
-----------------------------------
------ 1. PROJECT VARIABLES -------
-----------------------------------
"""

# Reading env variables from .env
_ENV_FILE = join(dirname(__file__), '.env')
if isfile(_ENV_FILE):
    load_dotenv(dotenv_path=_ENV_FILE)

# Creating a logging object
logger = logging.getLogger(__name__)
logger = logger_config(logger, level=logging.DEBUG, filemode='a')

# Messages
WARNING_MESSAGE = f'Module {__file__} finished with ERROR status'


"""
-----------------------------------
-------- 3. MAIN PROGRAM ----------
  3.1 Making Predictions on Text
-----------------------------------
"""

if __name__ == '__main__':

    # Training the model (if applicable)
    if bool(getenv('TRAIN')):
        logger.info('Starting train.py script')
        from ml import train
    
    # Reading pkl files
    logger.debug('Reading pkl files')
    try:
        pipeline = load(getenv('TEXT_PIPELINE'))
        model = load(getenv('MODEL'))
    except Exception as e:
        logger.error(e)
        logger.warning(WARNING_MESSAGE)
        exit()

    # Fake input
    text_input = 'Não gostei deste produto. Não me atendeu e custou muito caro'
    #text_input = pd.read_csv('test/test_data.csv', sep=';')

    # Instancing an object and executing predictions
    logger.debug('Creating a sentimentor object and making predictions')
    sentimentor = Sentimentor(data=text_input, pipeline=pipeline, model=model)
    try:
        output = sentimentor.make_predictions()
        logger.info('Module finished with success status')
        exit()
    except Exception as e:
        logger.error(e)
        logger.warning(WARNING_MESSAGE)
        exit()