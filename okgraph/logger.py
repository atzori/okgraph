import logging
from logging.config import fileConfig
from os import path

# Create a logger that can be used in every module
LOG_CONFIG_FILE = path.dirname(path.realpath(__file__)) + '/logging.ini'
if not path.isfile(LOG_CONFIG_FILE):
    LOG_CONFIG_FILE = path.dirname(path.dirname(LOG_CONFIG_FILE)) + '/logging.ini'
fileConfig(LOG_CONFIG_FILE)
logger = logging.getLogger()
logger.info(f'Logger configuration file is: {LOG_CONFIG_FILE}')
