"""
Code that sets up logging features and helps building up logging objects
with pre-configured parameters
"""

# Libraries
import logging

def logger_config(logger, level=logging.DEBUG, log_format='%(levelname)s;%(asctime)s;%(filename)s;%(module)s;%(lineno)d;%(message)s',
                  log_filepath='log/application_log.log', filemode='w'):
    """
    Function that creates a logger object with pre-configured params

    Parameters
    ----------
    :param logger: logging object criated within module scope [type: logging.getLogger()]

    Returns
    -------
    :return logger: logger object already configured
    """

    # Setting level for the logger object
    logger.setLevel(level)

    # Creating a formatter
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    # Creating handlers
    file_handler = logging.FileHandler(log_filepath, mode=filemode)
    stream_handler = logging.StreamHandler()

    # Setting up formatter on handlers
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Adding handlers on logger object
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def generic_exception_logging(e, logger, exit_flag=True):
    """
    Generic function for logging exceptions

    Parameters
    ----------
    :param e: python exception [type: Exception]
    :param logger: logger object already created and configurated [type: logging.getLogger()]
    :param exit_flag: flag that guides the exit of a module [type: bool, default: True]
    """

    # Logging an error and a warning
    logger.error(f'Error on trying to execute the code with stack traceback: {e}')
    
    # Finishing program if applicable
    if exit_flag:
        logger.warning(f'Module {__file__} finished with ERROR status')
        exit()