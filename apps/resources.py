"""
---------------------------------------------------------
                    Resumo do Módulo
---------------------------------------------------------
    O módulo api.py é responsável por definir as rotas
(resources) da API a partir de uma aplicação já instanciada

---------------------------------------------------------
                          FAQ
---------------------------------------------------------
"""

# Python
from json import dumps
import logging

# Third
from ml.sentimentor import Sentimentor
from flask import make_response
from log.log_config import logger_config

# Creating a logger object
logger = logger_config(logging.getLogger(__file__), filemode='a')

def associate_resources(app):
    """
    Cria recursos e define rotas para a aplicação

    Parâmetros
    ----------
    :param app: aplicação flask instanciada e configurada

    Retorno
    -------
    Essa função não possui retorno, além das definições intrínsecas dos recursos

    Exemplo
    -------
    # Instanciando aplicação do flask
    app = Flask(__name__)
    create_resources(app)
    """

    # Sentimentor object
    sentimentor = Sentimentor()

    # Definição de rota '/'
    @app.route('/')
    def index():
        logger.info('Hello World test')
        return make_response(dumps({'Hello': 'World'}))

    # Definição de rota '/train'
    @app.route('/train')
    def train():
        # Executing train() method from Sentimentor class
        try:
            sentimentor.train()
            sentimentor.trained = True
        except Exception as e:
            logger.error(e)
            logger.warning('Error on training the model')

        # Preparing response
        dict_return = {
            "Operation": "Model training",
            "Status": "Success"
        }

        return make_response(dumps(dict_return))

    # Definição de rota '/teste'
    @app.route('/predict')
    def predict():
        fake_input = 'Não gostei do produto. Péssimo atendimento e não compraria novamente. Muito ruim'
        json_predictions = sentimentor.make_predictions(input_data=fake_input)
        return make_response(json_predictions)