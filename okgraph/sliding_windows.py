"""The 'sliding_windows' module contains the utilities to search for semantic
similarities between the words in a corpus.
"""
import itertools
import math
import numpy
from okgraph.indexing import DEFAULT_INDEX_DIR, FIELD_ID, FIELD_CONTENT
from okgraph.utils import logger
import operator
from typing import Dict, List, Tuple
from whoosh import index
from whoosh.qparser import QueryParser

DEFAULT_DICTIONARY_DIR: str = "dictTotal.npy"
"""str: Default path for the corpus dictionary file."""


class SlidingWindows:
    """A class used to inspect the context of the target words.

    The words that appear close to the target words inside the corpus are the
    words with similar meaning to the target words (according to the
    distributional hypothesis) and are going to be referenced as labels. The
    labels appear in the same context of the target words. The context is
    represented by windows of text with specified size containing the target
    words. The size of a text window is measured in 'words'. The text windows
    are extracted from the corpus and used to identify the labels. The
    congruence of the labels with the target words is evaluated through a
    TF-IDF statistic.
    """

    def __init__(self,
                 target_words: Tuple[str, ...],
                 corpus_index_path: str = DEFAULT_INDEX_DIR,
                 corpus_dictionary_path: str = DEFAULT_DICTIONARY_DIR,
                 window_size: int = 14,
                 noise_threshold: float = 0.10,
                 ):
        """The constructor creates a SlidingWindows object.

        Inspects the indexed corpus to build the windows containing the target
        words. The windows are analyzed to calculate numerical statistics of all
        the words in the windows. The most characteristic words, according to
        the TF-IDF statistic, are the most valuable labels.

        Args:
            target_words (Tuple[str, ...]): tuple of word/words whose context
                has/have to be inspected.
            corpus_index_path (str): path of the indexed corpus.
            corpus_dictionary_path (str): path of the corpus dictionary.
            window_size (int): size of the windows containing the target
                word/words.
            noise_threshold (float): upper bound of the noise statistic value
                to accept a word as a valid label. The word noise is evaluated
                through the 'word corpus frequency / word window frequency'
                ratio (a logarithm function is used to scale the ratio). Words
                appearing in the corpus with a higher frequency than the
                windows are high rated and probably not characteristic for the
                windows. The noise and TF-IDF statistics are not bound, so that
                a word can score a high TF-IDF and yet a high noise.

        """
        if not isinstance(window_size, int):
            raise TypeError(f"windows_size must be an int")
        if not isinstance(target_words, tuple):
            raise TypeError(f"target_words must be a tuple")
        for target_word in target_words:
            if not isinstance(target_word, str):
                raise TypeError(f"target_words must be a tuple of strings")
        if len(target_words) == 0:
            raise ValueError(f"target_words must contain at least one word")
        if len(target_words) > window_size:
            raise ValueError(f"target_words contains {len(target_words)} words "
                             f"that cannot fit in windows of the specified "
                             f"size ({window_size})")

        logger.info(f"{target_words}: "
                    f"Start windowing of target words")

        self._target_words = list(target_words)
        self._window_size = window_size
        self._corpus_index_path = corpus_index_path

        logger.info(f"{self._target_words}: "
                    f"Loading corpus dictionary")
        self._corpus_dict = dict(
            (numpy.load(corpus_dictionary_path, allow_pickle=True)).item()
        )
        self._corpus_total_occurrences = \
            self._total_occurrences(self._corpus_dict)

        logger.debug(f"{self._target_words}: "
                     f"Building corpus inverse frequency dictionary")
        self.corpus_inverse_frequency_dict = \
            self._inverse_frequency_dictionary(self._corpus_dict,
                                               self._corpus_total_occurrences)

        logger.info(f"{self._target_words}: "
                    f"Creating windows")
        (self._windows_list, self._windows_dict) = self._create_windows()

        logger.info(f"{self._target_words}: "
                    f"Processing windows data")
        logger.debug(f"{self._target_words}: "
                     f"Number of windows: {len(self._windows_list)}")

        if len(self._windows_list) > 0:
            logger.debug(f"{self._target_words}: "
                         f"Removing target words from windows dictionary")
            for word in self._target_words:
                self._windows_dict.pop(word)
            logger.debug(f"{self._target_words}: "
                         f"Total unique windows words: "
                         f"{len(self._windows_dict)}")

            self._windows_total_occurrences = \
                self._total_occurrences(self._windows_dict)
            logger.debug(f"{self._target_words}: "
                         f"Total windows occurrences: "
                         f"{self._windows_total_occurrences}")

            logger.debug(f"{self._target_words}: "
                         f"Building windows occurrence dictionary")
            self._windows_occurrence_dict = \
                self._windows_occurrence_dictionary(self._windows_dict,
                                                    self._windows_list)

            logger.debug(f"{self._target_words}: "
                         f"Building windows frequency dictionary")
            self._windows_frequency_dict = \
                self._frequency_dictionary(self._windows_dict,
                                           self._windows_total_occurrences)

            logger.debug(f"{self._target_words}: "
                         f"Building windows noise dictionary")
            self._noise_dict = \
                self._noise_dictionary(self._windows_dict,
                                       self._windows_total_occurrences,
                                       self._corpus_dict,
                                       self._corpus_total_occurrences)

            logger.debug(f"{self._target_words}: "
                         f"Building windows TF-IDF dictionary")
            self._tf_idf_dict = \
                self._tf_idf_dictionary(self._windows_occurrence_dict,
                                        self._windows_frequency_dict,
                                        self.corpus_inverse_frequency_dict,
                                        self._windows_list)

            logger.debug(f"{self._target_words}: "
                         f"Cleaning windows TF-IDF dictionary")
            cleaned_tf_idf_dict = \
                self._clean_results(self._target_words, self._windows_dict,
                                    self._tf_idf_dict, self._noise_dict,
                                    noise_threshold)

            logger.debug(f"{self._target_words}: "
                         f"Sorting cleaned TF-IDF dictionary to obtain labels")
            self._results_dict = \
                {k: v for k, v in
                 sorted(cleaned_tf_idf_dict.items(),
                        key=operator.itemgetter(1),
                        reverse=True)
                 }
        else:
            logger.info(f"{self._target_words}: "
                        f"No windows found in corpus")
            self._windows_total_occurrences = 0
            self._windows_occurrence_dict = {}
            self._windows_frequency_dict = {}
            self._noise_dict = {}
            self._results_dict = {}

    def __str__(self) -> str:
        return "The windows of " + " and ".join(self._target_words)

    def _create_windows(self) -> Tuple[List[List[str]], Dict[str, int]]:
        """Extracts the windows from the corpus to represent the context of the
        target words. Target words are centered in the windows.

        Returns:
            Tuple[List[List[str]], Dict[str, int]]: the list of windows
                containing the target words and the dictionary
                {word: occurrences} related to those windows.

        """
        # Limit of parsed documents.
        limit = None

        # Open the indexed corpus to search for the documents containing the
        # target words.
        ix = index.open_dir(self._corpus_index_path)

        # Define all the possible sequences of target words.
        target_words_permutations = \
            list(itertools.permutations(self._target_words,
                                        len(self._target_words)))

        # Define the queries to search the target words in the indexed text:
        # the query will match the documents containing the words in the same
        # order of the permutation at a max distance defined by the integer
        # following the '~' character.
        text_queries = \
            ['"'+' '.join(permutation)+'"~'+str(self._window_size)
             for permutation in target_words_permutations]
        parsed_queries = \
            [QueryParser(FIELD_CONTENT, ix.schema).parse(u''+text_query)
             for text_query in text_queries]

        # Find the documents containing the target words.
        # Save the documents content in a dictionary using their id as a key:
        # this way the documents matching more than one permutation of the
        # target words are counted just one time.
        documents_results = {}
        with ix.searcher() as searcher:
            queries_results = \
                [searcher.search(parsed_query, limit=limit)
                 for parsed_query in parsed_queries]
            for query_results in queries_results:
                for result in query_results:
                    if result[FIELD_ID] not in documents_results:
                        documents_results[result[FIELD_ID]] = \
                            result[FIELD_CONTENT]
        # Convert the dictionary into a list of windows
        documents_results = list(documents_results.values())
        # Filter the content of every matched document:
        # search and extract from every document the window/windows in which the
        # target words appear closely.
        windows_list, windows_dict = self._window_extraction(documents_results)
        return windows_list, windows_dict

    def _window_extraction(self,
                           documents_results: List[str]
                           ) -> Tuple[List[List[str]], Dict[str, int]]:
        """Extracts the windows from the list of documents of the indexed
        corpus. Windows are created centering the target words in a sequence of
        words with fixed maximum size.

        Args:
            documents_results (List[str]): list of matched documents of
                the indexed corpus.

        Returns:
            Tuple[List[List[str]], Dict[str, int]]: the list of windows (a 
                window is a list of words) and their occurrence dictionary.

        """
        windows_list = []
        windows_dict = {}
        for content in documents_results:
            content = content.split()
            new_window = []
            matched_target_words = []
            # Index of the first target word match
            i_first_match = None
            # Index of the last target word match
            i_last_match = None
            # Distance between the first and last target words (target words
            # included)
            center_size = None
            # Size of the window borders
            offset_size = None

            for i, word in list(enumerate(content)):
                new_window.append(word)

                # Matched a new target word
                if word in self._target_words and \
                        word not in matched_target_words:
                    matched_target_words.append(word)

                    if len(matched_target_words) == 1:
                        i_first_match = i
                    if len(matched_target_words) == len(self._target_words):
                        i_last_match = i
                        center_size = i_last_match - i_first_match + 1
                        offset_size = \
                            math.floor((self._window_size - center_size) / 2)

                # Exceeded maximum distance between two target words: reset the
                # previous matches
                if matched_target_words and \
                        i - i_first_match + 1 > self._window_size:
                    matched_target_words = []
                    i_first_match = None
                    i_last_match = None
                    center_size = None
                    offset_size = None

                # New window completed: save it and reset previous matches
                elif len(matched_target_words) == len(self._target_words) and \
                        i - offset_size == i_last_match:
                    # Add the last 'window_size' words to the windows list
                    windows_list.append(
                        new_window[-(center_size+2*offset_size):]
                    )
                    # Update the dictionary with the words in the new window
                    for window_word in windows_list[-1]:
                        windows_dict[window_word] = \
                            windows_dict.get(window_word, 0) + 1

                    matched_target_words = []
                    i_first_match = None
                    i_last_match = None
                    center_size = None
                    offset_size = None

        return windows_list, windows_dict

    @staticmethod
    def _total_occurrences(dictionary: Dict[str, int]) -> int:
        """Get the total amount of words in the dictionary, or rather the sum 
        of the occurrences of every unique word in the dictionary.

        Args:
            dictionary (Dict[str, int]): dictionary {word: occurrences}.

        Returns:
            int: the total amount of words in the dictionary.

        """
        total_occurrences = 0
        for occurrences in dictionary.values():
            total_occurrences += occurrences

        return total_occurrences

    @staticmethod
    def _windows_occurrence_dictionary(
            dictionary: Dict[str, int],
            windows: List[List[str]]
    ) -> Dict[str, int]:
        """Evaluates the presences of the words in the windows.
        
        Args:
            dictionary (Dict[str, int]): dictionary {word: occurrences}.
            windows (List[List[str]]): list of all the matched windows of the 
                corpus.

        Returns:
            Dict[str, int]: dictionary {word: number of windows containing the 
            word}

        """
        dictionary_keys = list(dictionary.keys())
        windows_occurrence_dict = {}
        for word in dictionary_keys:
            for window in windows:
                if word in window:
                    windows_occurrence_dict[word] = \
                        windows_occurrence_dict.get(word, 0) + 1

        return windows_occurrence_dict

    @staticmethod
    def _frequency_dictionary(
            occurrence_dictionary: Dict[str, int], 
            total_occurrences: int
    ) -> Dict[str, float]:
        """Evaluates the frequency of every word in the dictionary.
        The frequency of a word is obtained dividing the word occurrences by
        the total number of word occurrences in the dictionary.

        Args:
            occurrence_dictionary (Dict[str, int]): dictionary {word:
                occurrences}.
            total_occurrences (int): total occurrences of the dictionary.

        Returns:
            Dict[str, float]: dictionary {word: words frequency}

        """
        occurrence_dictionary_keys = list(occurrence_dictionary.keys())
        f_dict = {}
        for word in occurrence_dictionary_keys:
            f_dict[word] = occurrence_dictionary.get(word) / total_occurrences

        return f_dict

    @staticmethod
    def _inverse_frequency_dictionary(
            occurrence_dictionary: Dict[str, int],
            total_occurrences: int
    ) -> Dict[str, float]:
        """Evaluates the inverse frequency of every word in the dictionary.
        The inverse frequency of a word is obtained from the division of the
        total number of word occurrences in the dictionary by the word
        occurrences.

        Args:
            occurrence_dictionary (Dict[str, int]): dictionary {word:
                occurrences}.
            total_occurrences (int): total occurrences of the dictionary.

        Returns:
            Dict[str, float]: dictionary {word: words inverse frequency}.

        """
        occurrence_dictionary_keys = list(occurrence_dictionary.keys())
        if_dict = {}
        for word in occurrence_dictionary_keys:
            if_dict[word] = total_occurrences / occurrence_dictionary.get(word)
        return if_dict

    @staticmethod
    def _noise_dictionary(
            windows_dictionary: Dict[str, int],
            windows_total_occurrences: int,
            corpus_dictionary: Dict[str, int],
            corpus_total_occurrences: int
    ) -> Dict[str, float]:
        """Evaluates the importance of a word in the defined context, or rather
        how much that word is characteristic for the defined windows. The
        importance is evaluated through the 'word corpus frequency / word
        window frequency' ratio, so that words appearing mainly in the windows
        are low rated, while words appearing often in all the corpus are high
        rated. A logarithm function is used to scale the ratio.

        Args:
            windows_dictionary (Dict[str, int]): dictionary {word: occurrences
                of word in windows}.
            windows_total_occurrences (int): total occurrences of the windows
                dictionary.
            corpus_dictionary (Dict[str, int]): dictionary {word: occurrences
                of word in corpus}.
            corpus_total_occurrences (int):  total occurrences of the corpus
                dictionary.

        Returns:
            Dict[str, float]: the noise dictionary
                {word: log(1 + word corpus frequency / word window frequency)}.

        """
        windows_dictionary_keys = \
            list(windows_dictionary.keys())

        noise_dict = {}
        for word in windows_dictionary_keys:
            # QSTN: all the words in the windows dictionary surely are in the
            #  corpus dictionary, 'cause windows have been extracted from the
            #  corpus. We are scrolling through the windows, so the if
            #  statement will always be False, right?
            if word not in corpus_dictionary:
                # QSTN: possibly unreachable
                noise_dict[word] = 0
            else:
                window_frequency = \
                    windows_dictionary.get(word) / windows_total_occurrences
                corpus_frequency = \
                    corpus_dictionary.get(word) / corpus_total_occurrences
                noise_dict[word] = \
                    numpy.log10(1+corpus_frequency / window_frequency)

        return noise_dict

    @staticmethod
    def _tf_idf_dictionary(
            windows_occurrence_dictionary: Dict[str, int],
            windows_frequency_dictionary: Dict[str, float],
            corpus_inverse_frequency_dictionary: Dict[str, float],
            windows_list: List[List[str]]
    ) -> Dict[str, float]:
        """Evaluates the TF-IDF statistic for every word in the windows.

        Args:
            windows_occurrence_dictionary (Dict[str, int]): dictionary {word in
                windows: occurrences}.
            windows_frequency_dictionary (Dict[str, float]): dictionary {word
                in windows: frequency}.
            corpus_inverse_frequency_dictionary (Dict[str, float]): dictionary
                {word in corpus: inverse frequency}.
            windows_list (List[List[str]]): list of matched windows.

        Returns:
            Dict[str, float]: the TF-IDF dictionary {word in window: TF-IDF}.

        """
        windows_frequency_dictionary_keys = \
            list(windows_frequency_dictionary.keys())

        tf_idf_dict = {}
        for word in windows_frequency_dictionary_keys:
            # QSTN: all the words in the windows dictionary surely are in the
            #  corpus dictionary, 'cause windows have been extracted from the
            #  corpus. We are scrolling through the windows, so none of the
            #  conditions in the if statement will always be True, right?
            if word not in windows_frequency_dictionary or \
               word not in windows_occurrence_dictionary or \
               word not in corpus_inverse_frequency_dictionary:
                # QSTN: possibly unreachable
                tf_idf_dict[word] = 0
            else:
                tf = windows_occurrence_dictionary.get(word) / len(windows_list)
                idf = numpy.log10(corpus_inverse_frequency_dictionary.get(word))
                tf_idf_dict[word] = tf * idf

        return tf_idf_dict

    @staticmethod
    def _clean_results(
            target_words: List[str],
            windows_dictionary: Dict[str, int],
            tf_idf_dictionary: Dict[str, float],
            noise_dictionary: Dict[str, float],
            noise_threshold: float
    ) -> Dict[str, float]:
        """Clean the result dictionary removing the less significant words.
        These words as treated as not significant:
            - words whose occurrences are higher than the total amount of
               unique words;
            - words shorter than 3 characters;
            - words with a noise higher than the specified threshold;
            - digits;
        So these words are not present in the cleaned dictionary.

        Args:
            target_words (List[str]): list of the target words.
            windows_dictionary (Dict[str, int]): dictionary {word in window:
                occurrences}.
            tf_idf_dictionary (Dict[str, float]): dictionary {word in window:
                TF-IDF}.
            noise_dictionary (Dict[str, float]): dictionary
                {word: log(1 + word corpus frequency / word window frequency)}.
            noise_threshold (float): maximum 'noise' value to accept the word
                as significant.

        Returns:
            Dict[str, float]: the cleaned TF-IDF dictionary {significant word
                in window: TF-IDF}.

        """
        cleaned_dictionary = dict(tf_idf_dictionary)

        # Remove the words whose occurrences are higher than the total amount
        # of unique words
        count_removed_words = 0
        logger.debug(
            f"{target_words}:"
            f" Removing words whose occurrences are higher than the total"
            f" amount of unique words")
        windows_dictionary_keys = list(windows_dictionary.keys())
        for key in windows_dictionary_keys:
            if windows_dictionary.get(key) > len(cleaned_dictionary):
                count_removed_words += 1
                logger.debug(
                    f"{target_words}: "
                    f"Removing '{key}': "
                    f"{windows_dictionary.get(key)} < {len(cleaned_dictionary)}"
                )
                cleaned_dictionary.pop(key)
        logger.debug(f"{target_words}: Removed {count_removed_words} words")

        # Remove words shorter than 3 characters
        count_removed_words = 0
        logger.debug(
            f"{target_words}:"
            f" Removing words shorter than 3 characters")
        clean_dictionary_keys = list(cleaned_dictionary.keys())
        for key in clean_dictionary_keys:
            if len(key) < 3:
                count_removed_words += 1
                logger.debug(
                    f"{target_words}:"
                    f" Removing '{key}': len={len(key)} < 3")
                cleaned_dictionary.pop(key)
        logger.debug(f"{target_words}: Removed {count_removed_words} words")

        # Remove words with a noise higher than the specified threshold
        count_removed_words = 0
        logger.debug(
            f"{target_words}:"
            f" Removing words with a noise higher than the specified threshold")
        clean_dictionary_keys = list(cleaned_dictionary.keys())
        for key in clean_dictionary_keys:
            if noise_dictionary.get(key) > noise_threshold:
                count_removed_words += 1
                logger.debug(
                    f"{target_words}:"
                    f" Removing '{key}':"
                    f" {noise_dictionary.get(key):.4f} > {noise_threshold:.2f}")
                cleaned_dictionary.pop(key)
        logger.debug(f"{target_words}: Removed {count_removed_words} words")

        # Remove digits
        count_removed_words = 0
        logger.debug(f"{target_words}: Removing digits")
        clean_dictionary_keys = list(cleaned_dictionary.keys())
        for key in clean_dictionary_keys:
            if str.isnumeric(key):
                count_removed_words += 1
                logger.debug(f"{target_words}: Removing '{key}'")
                cleaned_dictionary.pop(key)
        logger.debug(f"{target_words}: Removed {count_removed_words} words")

        logger.debug(
            f"{target_words}:"
            f" Cleaned dictionary counts {len(cleaned_dictionary)} words,"
            f" against the {len(tf_idf_dictionary)} words of the starting"
            f" dictionary")

        return cleaned_dictionary

    def get_target_words(self) -> List[str]:
        """Returns the list of target words.

        Returns:
            List[str]: the list of target words.
        """
        return self._target_words

    def get_results_dict(self) -> Dict[str, float]:
        """Return the dictionary of labels and their TF-IDF value, related to
        the target words.

        Returns:
            Dict[str, float]: the dictionary of labels and their TF-IDF value,
                ordered for congruence through the TF-IDF statistic.

        """
        return self._results_dict

    def get_results_list(self, k: int = None) -> List[str]:
        """Returns the list of labels related to the target words.

        Args:
            k (int): sets the limit to the number of labels to return. If None,
                no limit is set and all the labels are returned.

        Returns:
            List[str]: the list of labels, ordered for congruence through the
                TF-IDF statistic and limited to 'k' values.

        """
        if k is not None:
            if not isinstance(k, int):
                raise TypeError(f"k must be an int")

        return list(self._results_dict)[:k]
