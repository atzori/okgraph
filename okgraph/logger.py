import logging
from logging.config import fileConfig
from os import path

module_path = str.upper(__name__).replace("OKGRAPH.", "")

# Create a logger that can be used in every module
LOG_CONFIG_FILE = path.join(path.dirname(path.realpath(__file__)), "logging.ini")
if not path.isfile(LOG_CONFIG_FILE):
    LOG_CONFIG_FILE = path.join(path.dirname(path.dirname(LOG_CONFIG_FILE)), "logging.ini")
fileConfig(LOG_CONFIG_FILE)
logger = logging.getLogger()
logger.info(f"{module_path}: Logger configuration file is: {LOG_CONFIG_FILE}")
