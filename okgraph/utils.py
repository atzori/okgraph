"""
The 'utils' module contains generic utilities that support the other modules of
the library.
"""
import logging
from logging.config import fileConfig
import numpy as np
import operator
from os import path
import re
import string
from typing import Dict, Iterator

LOG_CONFIG_FILE = path.join(
    path.dirname(path.realpath(__file__)),
    "logging.ini")
if not path.isfile(LOG_CONFIG_FILE):
    LOG_CONFIG_FILE = path.join(
        path.dirname(path.dirname(LOG_CONFIG_FILE)),
        "logging.ini")
fileConfig(LOG_CONFIG_FILE)
logger = logging.getLogger()
"""RootLogger: Logger for run time documentation.

Check the 'logging.ini' file for the logger configuration.
"""

logger.info(f"Logger configuration file is: {LOG_CONFIG_FILE}")


def get_words(file_path: str) -> Iterator[str]:
    """
    Reads a text file and allows to scroll over it word by word. The text is
    formatted so that the words are lowercase and the punctuation is removed.

    Args:
        file_path (str): path of the corpus file.

    Returns:
        Iterator[str]: an iterator of words.
    """
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    with open(file_path, encoding="utf-8") as file_path:
        for line in file_path:
            line = regex.sub(' ', line)
            words = line.lower().split()

            for w in words:
                yield w


def create_dictionary(corpus: str, dictionary: str = "dictTotal.npy",
                      save_dictionary: bool = True) -> Dict[str, int]:
    """
    Creates a dictionary to summarize the distribution of the words in the
    corpus.

    Args:
        corpus (str): path of the corpus file.
        dictionary (str): path of the dictionary file in which store the created
            dictionary.
        save_dictionary (bool): True to store the created dictionary in a file,
            False otherwise.

    Returns:
        Dict[str, int]: the corpus dictionary structured as {word: occurrences}.
    """
    words = get_words(corpus)

    logger.info(f"Started dictionary creation")

    occurrence_dict = {}
    for word in words:
        occurrence_dict[word] = occurrence_dict.get(word, 0) + 1

    occurrence_dict = dict(sorted(occurrence_dict.items(), key=operator.itemgetter(1), reverse=True))

    if save_dictionary is True:
        np.save(dictionary, occurrence_dict)

    logger.info(f"Ended dictionary creation")

    return occurrence_dict


def head(path, n=0, file=None, gzip=False):
    """
    TODO: add documentation
    Args:
        path:
        n:
        file:
        gzip:

    Returns:

    """
    from os import system
    system('head -c ' + str(n) + ' ' + path + ' > ' + file)
    if gzip is True:
        system('gzip ' + file)


def tail(path, n=0, file=None, gzip=False):
    """
    TODO: add documentation
    Args:
        path:
        n:
        file:
        gzip:

    Returns:

    """
    from os import system
    system('tail -c ' + str(n) + ' ' + path + ' > ' + file)
    if gzip is True:
        system('gzip ' + file)


def download(url, path_name='text8.zip', name='Text8 Dataset'):
    """
    TODO: add documentation
    Args:
        url:
        path_name:
        name:

    Returns:

    """
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
