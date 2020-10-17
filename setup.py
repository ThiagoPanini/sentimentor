"""
The setup.py module can be used to consolidate useful
informations about the project and also for making
package installations easier on project virtual env
"""

# Libraries
from setuptools import find_packages, setup

# Project variables
__version__ = '0.1.0'
__description__ = 'Python Sentimentor Project'
__long_description__ = 'Sentimentor project for giving back the sentiment of a given input string'
__author__ = 'Thiago Panini, Rodrigo Hiroaki'

# Building setup
setup(
    name='Sentimentor',
    version=__version__,
    author=__author__,
    packages=find_packages(),
    license='MIT',
    description=__description__,
    long_description=__long_description__,
    url='https://github.com/ThiagoPanini/sentimentor',
    keywords='Sentiment, Machine Learning, Classification, API, Python, Flask',
    include_package_data=True,
    zip_safe=False
)
