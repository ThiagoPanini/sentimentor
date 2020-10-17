"""
---------------------------------------------------------
                    Resumo do Módulo
---------------------------------------------------------
    O módulo __init__.py dentro da pasta apps/ serve como
um inicializador em tempo de execução para scripts que
criam uma aplicação Flask, como applications.py

---------------------------------------------------------
                          FAQ
---------------------------------------------------------

1. Qual seu papel na execução da aplicação?
    R: Em resumo, o arquivo __init__.py permite marcar um 
diretório no disco como um pacote Python para futuras 
importações*

Ref[1.1]: https://stackoverflow.com/questions/448271/what-is-init-py-for
Em um dos exemplos do link acima, é mostrado algo 
relacionado a inicialização de sessões (no caso, apps)

Ref[1.2]: https://stackoverflow.com/questions/6323860/sibling-package-imports

*Obs: Nas versões mais recentes do Python, não é necessário 
colocar o arquivo __init__.py no diretório criado, uma vez 
que o python automaticamente identifica que a pasta pode 
ser um pacote.
"""

# Python
from flask import Flask
from pandas import DataFrame
from datetime import datetime

# Third
from apps.resources import associate_resources
from config import config


def create_app(config_name):
    """
    Cria uma aplicação Flask e aplica as configurações necessárias

    Parameters
    ----------
    :param config_name: chave da configuração a ser aplicada na aplicação

    Returns
    -------
    :return app: objeto Flask inicializado e configurado

    Application
    -----------
    app = create_app(config_name=os.getenv('FLASK_ENV') or 'default')
    """

    # Inicializa objeto Flask
    app = Flask('api-user')

    # Aplica a configuração de acordo com parâmetro da função
    app.config.from_object(config[config_name])

    # Aplica rotas na aplicação instanciada
    associate_resources(app)

    return app
