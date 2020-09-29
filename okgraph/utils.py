"""The 'utils' module contains generic utilities supporting the other modules of
the library.
"""
from itertools import chain
import logging
from logging.config import fileConfig
import numpy as np
import operator
from os import makedirs, path
import re
import string
from typing import Dict, Iterator, List, Tuple

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


def check_extension(file: str,
                    default_extension: str,
                    allowed_extensions: List[str]) -> str:
    """Checks the validity of a file name.
    If the file has no extension, the default extension is appended to the file
    name. If the file extension is one of the allowed extensions, the file
    name is returned, otherwise a ValueError is raised.

    Args:
        file (str): name of the file.
        default_extension (str): default extension for the file names with no
            specified extension. The starting '.' should be included in the
            extension name.
        allowed_extensions (List[str]): list of allowed extensions for the
            file. The starting '.' should be included in the extension name.

    Returns:
        str: the name of the file. If the specified name had no extension, the
            default extension has been added.

    """
    # Check the validity of the specified extensions (should start with '.')
    for extension in allowed_extensions + [default_extension]:
        if extension[0] != ".":
            raise ValueError(
                f"{extension} is not a valid extension. Extension should start"
                f" with '.'"
            )

    (file_basename, file_extension) = path.splitext(file)

    if file_extension == "":
        file = file_basename + default_extension
        file_extension = default_extension

    if file_extension in allowed_extensions:
        return file
    else:
        raise ValueError(
            f"file {file} has not a valid extension. Valid extension are"
            f" {allowed_extensions}"
        )


def get_words(file_path: str) -> Iterator[str]:
    """Reads a text file and allows to scroll over it word by word. The text is
    formatted so that the words are lowercase and the punctuation is removed.

    Args:
        file_path (str): path of the corpus file.

    Yields:
        str: the next word in the file.

    """
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    with open(file_path, encoding="utf-8") as file_path:
        for line in file_path:
            line = regex.sub(' ', line)
            words = line.lower().split()

            for w in words:
                yield w


def generate_dictionary(corpus: str, dictionary: str = "dictTotal.npy",
                        save_dictionary: bool = True) -> Dict[str, int]:
    """Creates a dictionary representing the distribution of the words in the
    corpus.

    Args:
        corpus (str): path of the corpus file.
        dictionary (str): path of the file where the dictionary is saved.
        save_dictionary (bool): True to save the created dictionary in a file,
            False otherwise.

    Returns:
        Dict[str, int]: the corpus dictionary structured as {word: occurrences}.

    """
    words = get_words(corpus)

    logger.info(f"Started dictionary generation")

    occurrence_dict = {}
    for word in words:
        occurrence_dict[word] = occurrence_dict.get(word, 0) + 1

    sorted_occurrence_dict = \
        {k: v for k, v in
         sorted(occurrence_dict.items(),
                key=operator.itemgetter(1),
                reverse=True)
         }

    if save_dictionary is True:
        parent_dir = path.dirname(dictionary)
        if not path.exists(parent_dir):
            makedirs(parent_dir)
        np.save(dictionary, sorted_occurrence_dict)

    logger.info(f"Dictionary generated")

    return sorted_occurrence_dict


def list_flatten(l: List) -> List:
    """Converts a multidimensional or nested list into a one-dimensional list.

    Args:
        l (List): multidimensional or nested list.

    Returns:
        List: flattened list.

    Example:
        >>> l = [[1, 2], [[3, 4], [5, 6, 7]]]
        >>> list_flatten(l)
        [1, 2, [3, 4], [5, 6, 7]]

    """
    return list(chain.from_iterable(l))


def tuple_combinations(tuple_elements: Tuple[List, ...]) -> List[Tuple]:
    """Given a tuple of lists containing some elements, evaluates all the
    possible combinations of those elements.

    Args:
        tuple_elements (Tuple[List, ...]): the original tuple containing all
            the possible elements that can occupy the respective position in
            new tuples.

    Returns:
        List[Tuple]: the list of tuples obtained from the combination of the
            given elements.

    Example:
        >>> elements = (["a", "b"], ["1"], ["$", "£"])
        >>> tuple_combinations(elements)
        [('a', '1', '$'), ('a', '1', '£'), ('b', '1', '$'), ('b', '1', '£')]

    """
    combinations: List[Tuple] = []
    tuple_size = len(tuple_elements)

    # List of indexes used to pick the elements in the different positions of
    # the input tuple. Start from the first elements in every position.
    actual_indexes = [0 for _ in range(tuple_size)]
    # Max number of elements available for every position of the tuples
    max_indexes = [len(l) for l in tuple_elements]

    # Creates new tuples until the actual index for the first position of the
    # tuple indicates all the elements has been selected
    while actual_indexes[0] < max_indexes[0]:
        # Create a new tuple picking the items from all the positions of the
        # tuple indexed through the list of actual indexes
        new_tuple = ()
        for j in range(tuple_size):
            new_tuple = \
                new_tuple.__add__( (tuple_elements[j][actual_indexes[j]],) )

        # Add the tuple as a new combination
        combinations += [new_tuple]

        # Scroll the actual indexes backwards
        j = -1
        while j >= -tuple_size:
            # Pass to the next element in the actual position
            actual_indexes[j] += 1

            # If the last element for the actual position has not been passed
            if actual_indexes[j] < max_indexes[j]:
                # Proceed creating a new combination
                break
            # If the last element for the actual position has just been used
            else:
                # If the element was not from the first position of the tuple
                if j > -tuple_size:
                    # Start again from the first element for the same position
                    actual_indexes[j] = 0
                # Check the index in the previous position of the tuple
                j -= 1

    return combinations
