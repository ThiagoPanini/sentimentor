"""
---------------------------------------------------------
                    Resumo do Módulo
---------------------------------------------------------
    Arquivo para setagem das configurações da aplicação a 
partir da criação de classes específicas para cada setup 
de utilização. Nele, as variáveis de ambientes são lidas
e utilizadas na configuração de cada classe.

---------------------------------------------------------
                          FAQ
---------------------------------------------------------

1. Como funciona o módulo getenv da biblioteca os?

    R: O módulo getenv(key) recebe um parâmetro chave e
realiza uma procura, dentro das variáveis de ambiente 
configuradas para o projeto, o valor respectivo da chave

Ref[1.1]: https://docs.python.org/3/library/os.html
Ref[1.2]: https://www.geeksforgeeks.org/python-os-getenv-method/

---------------------------------------------------------

2. Qual a usabilidade do arquivo config.py?

    R: O módulo config.py é responsável pela definição de
configurações específicas a serem utilizadas no
desenvolvimento da aplicação

---------------------------------------------------------

3. Por que são criadas diferentes classes no arquivo 
config.py?

    R: Diferentes configurações são utilizadas para 
determinadas ações dentro do desenvolvimento da aplicação.
Exemplos:
    - Necessidade de enviar arquivos do usuário à AWS 
    (apenas prod)
    - Envio de e-mails durante a execução (apenas prod)

---------------------------------------------------------

4. Em um teste simples, os retornos da função getenv() são 
nulos pras chaves 'SECRET_KEY', 'APP_PORT' e 'DEBUG'. 
Como contornar?

    R: Na verdade, os.getenv(key) é um método que 
simplesmente busca os valores das variáveis de ambientes 
passadas no argumento "key"
    - Se essa variável não existe, None é retornado
    - O argumento "default" pode ser uma alternativa

---------------------------------------------------------

5. Como são setadas as variáveis de ambiente antes de serem 
coletadas pelo os.getenv(key)?

    R: Ao longo do desenvolvimento da aplicação, o arquivo 
.env (localizado na raíz do diretório do projeto) será 
populado com as informações relevantes dentro das variáveis 
de ambiente sensíveis ao os.getenv(key)
    * O pacote python-dotenv é o responsável por permitir 
a leitura das variáveis de ambiente automaticamente no 
arquivo application.py
"""

# Python
from os import getenv


class Config:
    SECRET_KEY = getenv('SECRET_KEY') or '8QAzMJCmwETRvGKBxoZw' # Normalmente utilizada para autenticação de usuários
    APP_PORT = getenv('APP_PORT')
    DEBUG = getenv('DEBUG')

class DevelopmentConfig(Config):
    FLASK_ENV = 'development'
    DEBUG = True

class TestingConfig(Config):
    FLASK_ENV = 'testing'
    TESTING = True

# Armazena todas as configurações em uma estrutura chave-valor
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}