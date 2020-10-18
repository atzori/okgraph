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
    path.dirname(path.normpath(__file__)),
    "logging.ini")
"""str: name of the file containing the logger configuration."""
fileConfig(LOG_CONFIG_FILE)

logger = logging.getLogger()
"""RootLogger: logger for run time documentation.

Check the 'logging.ini' file for the logger configuration.
"""


def download_file(url: str,
                  save_file: str,
                  chunk_size: int = 1024,
                  bar_length: int = 50
                  ) -> None:
    """Download a file from a specified URL.

    Uses the 'requests' library: check its documentation for more info.
    https://requests.readthedocs.io/en/master/

    FIXME: may be very slow downloading.

    Args:
        url (str): URL from which download the content.
        save_file (str): name of the file in which save the downloaded content.
        chunk_size (int): download the streamed file by chunks of the specified
         size.
        bar_length (int): length (in chars) of the shown progression bar.

    Raises:
        ConnectionError: if the size of the download file is not known and it
         is not possible to retrieve it correctly.

    """
    import requests

    bar_char = "\u2588"

    session = requests.session()

    # Check if the URL is for a downloadable file
    pass

    # Try to get the size of the downloadable file
    header = session.head(url)
    total_bytes = header.headers.get("content-length")
    if total_bytes is None or int(total_bytes) <= 0:
        print("Impossible to correctly retrieve the file without knowing its"
              " size.")
        raise ConnectionError
    total_bytes = int(total_bytes)

    # Create a file in which store the downloaded bytes
    with open(save_file, "wb") as wb_file:
        downloaded_bytes = 0

        # Until the file is not totally downloaded
        while downloaded_bytes < total_bytes:
            # Request to download the file, resuming the download from where it
            # has been previously left
            resume_header = {"Range": f"bytes={downloaded_bytes}-"}
            request = requests.get(url,
                                   headers=resume_header,
                                   stream=True,
                                   verify=False,
                                   allow_redirects=True)

            # Start downloading chunk by chunk
            for data_chunk in request.iter_content(chunk_size=chunk_size):
                # Write the data and count the written bytes
                wb_file.write(data_chunk)
                downloaded_bytes += len(data_chunk)
                # Print the download progression
                done_percentage = downloaded_bytes / total_bytes
                done_bar = int(bar_length * done_percentage)
                print(f"\r{done_percentage * 100:4.1f}% "
                      f"|{bar_char * done_bar}{' ' * (bar_length - done_bar)}|"
                      f" = {downloaded_bytes / (2**20):.2f}MB"
                      f"/{total_bytes / (2**20):.2f}MB",
                      end='')
        print()

    session.close()


def check_extension(file_name: str,
                    default_extension: str,
                    allowed_extensions: List[str]) -> str:
    """Checks the validity of a file name.
    If the file has no extension, the default extension is appended to the file
    name. If the file extension is one of the allowed extensions, the file
    name is returned, otherwise a ValueError is raised.

    Args:
        file_name (str): name of the file.
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

    (file_basename, file_extension) = path.splitext(file_name)

    if file_extension == "":
        file_name = file_basename + default_extension
        file_extension = default_extension

    if file_extension in allowed_extensions:
        return file_name
    else:
        raise ValueError(
            f"file {file_name} has not a valid extension. Valid extension are"
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
