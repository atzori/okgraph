import numpy as np
import operator
import re
import string
from okgraph.logger import logger

module_path = str.upper(__name__).replace("OKGRAPH.", "")


def get_words(file_path: str):
    """
    Reads a text file and convert it to a list of its words in lowercase format.
    :param file_path: path (with name) of the file
    :return: a list of words (strings)
    """
    # Regular expression that identifies the punctuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    # Read the file line by line and extract all the words
    with open(file_path, encoding="utf-8") as file_path:
        for line in file_path:
            line = regex.sub(' ', line)
            words = line.lower().split()

            for w in words:
                yield w


def create_dictionary(corpus: str, dictionary_name: str = "dictTotal.npy", dictionary_save: bool = True):
    """
    Creates a dictionary of the words in the file using the structure {word in file: occurrences of word in file}
    :param corpus: path of the corpus
    :param dictionary_name: path of the dictionary
    :param dictionary_save: save or not the created dictionary into a file whose name is specified by "dictionary_path"
    :return: the created dictionary
    """
    # Get all the words from the file
    words = get_words(corpus)

    logger.info(f"{module_path}: Started dictionary creation")

    # Create a dictionary of the type {word in file: occurrences of word in file}
    dictionary = {}
    for word in words:
        dictionary[word] = dictionary.get(word, 0) + 1

    # Order the dictionary by the words occurrences (from greater to smaller)
    dictionary = dict(sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True))

    # If wanted, save the dictionary into a file with the specified path
    if dictionary_save is True:
        np.save(dictionary_name, dictionary)

    logger.info(f"{module_path}: Ended dictionary creation")

    # Return the created dictionary
    return dictionary


def head(path, n=0, file=None, gzip=False):
    # TODO: add documentation
    from os import system
    system('head -c ' + str(n) + ' ' + path + ' > ' + file)
    if gzip is True:
        system('gzip ' + file)


def tail(path, n=0, file=None, gzip=False):
    # TODO: add documentation
    from os import system
    system('tail -c ' + str(n) + ' ' + path + ' > ' + file)
    if gzip is True:
        system('gzip ' + file)


def download(url, path_name='text8.zip', name='Text8 Dataset'):
    # TODO: add documentation
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
