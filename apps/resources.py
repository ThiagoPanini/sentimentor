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

# Third
from apps.sentimentor import Sentimentor


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

    # Definição de rota '/'
    @app.route('/')
    def index():
        return {'hello': 'World Hiro'}

    # Definição de rota '/home'
    @app.route('/home')
    def home():

        # Fake input
        input_data = 'Não gostei deste produto. Não me atendeu e custou muito caro'

        # Creating a Sentimentor object and making predictions
        sentimentor = Sentimentor(input_data=input_data)
        output = sentimentor.make_predictions()
        dict_return = {
            "input_data": input_data,
            "output": output['class_sentiment']
        }
        return dumps(dict_return)