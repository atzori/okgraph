import logging
import operator
import re
import string
from os import path
from logging.config import fileConfig

import numpy as np

LOG_CONFIG_FILE = path.dirname(path.realpath(__file__)) + '/logging.ini'
if not path.isfile(LOG_CONFIG_FILE):
    LOG_CONFIG_FILE = path.dirname(path.dirname(LOG_CONFIG_FILE)) + '/logging.ini'

fileConfig(LOG_CONFIG_FILE)
logger = logging.getLogger()
logger.info(f'Log config file is: {LOG_CONFIG_FILE}')


# generatore parola per parola
def get_word(filename):
    """generatore parola per parola"""
    regex = re.compile('[%s]' % re.escape(string.punctuation))  # espressione regolare che rimuove la punteggiatura
    with open(filename) as file:
        for line in file:
            line = regex.sub(' ', line)  # sostituisci la punteggiatura con uno spazio
            listaParole = line.lower().split()

            for w in listaParole:
                yield w


def creation(path, name='dictTotal.npy', save=True):
    g = get_word(path)
    dict_creation = {}
    logger.info('Start creation dictionary')

    for word in g:
        dict_creation[word] = dict_creation.get(word, 0) + 1

    if save == True:
        np.save(name, dict_creation)

    dict_creation = dict(sorted(dict_creation.items(), key=operator.itemgetter(1), reverse=True))
    logger.info('End creation')
    return dict_creation


def head(path, n=0, file=None, gzip=False):
    from os import system
    system('head -c ' + str(n) + ' ' + path + ' > ' + file)
    if gzip is True:
        system('gzip ' + file)


def tail(path, n=0, file=None, gzip=False):
    from os import system
    system('tail -c ' + str(n) + ' ' + path + ' > ' + file)
    if gzip is True:
        system('gzip ' + file)


def download(url, path_name='text8.zip', name='Text8 Dataset'):
    from urllib.request import urlretrieve
    from tqdm import tqdm

    class ProgressBar(tqdm):
        last_block = 0

        def hook(self, block_num=1, block_size=1, total_size=None):
            self.total = total_size
            self.update((block_num - self.last_block) * block_size)
            self.last_block = block_num

    with ProgressBar(unit='B', unit_scale=True, miniters=1, desc=name) as pbar:
        urlretrieve(url, path_name, pbar.hook)
