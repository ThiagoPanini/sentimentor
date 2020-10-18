"""
This main.py module is responsible for encapsulating all other modules
needed for receiving an input from an API call and return the request
"""

# Python
from dotenv import load_dotenv
from os import getenv
from os.path import join, dirname, isfile

# Third
from apps import create_app
from ml.sentimentor import Sentimentor


"""
-----------------------------------
------ 1. PROJECT VARIABLES -------
-----------------------------------
"""

# Reading env variables from .env
_ENV_FILE = join(dirname(__file__), '.env')
if isfile(_ENV_FILE):
    load_dotenv(dotenv_path=_ENV_FILE)

# Messages
WARNING_MESSAGE = f'Module {__file__} finished with ERROR status'


"""
-----------------------------------
----- 2. PYTHON APPLICATION -------
-----------------------------------
"""

# Creating application
app = create_app(getenv('FLASK_ENV') or 'default')

# Running application
if __name__ == '__main__':

    # Application env variables
    ip = '0.0.0.0'
    port = app.config['APP_PORT']
    debug = app.config['DEBUG']

    # Initializing application from flask web server
    app.run(host=ip, debug=debug, port=port, use_reloader=debug)