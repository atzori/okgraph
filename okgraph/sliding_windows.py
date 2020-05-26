import operator
import re
import itertools
import numpy
import math
from whoosh import index
from whoosh.qparser import QueryParser
import numpy as np
from okgraph.logger import logger
from okgraph.indexing import FIELD_TITLE, FIELD_CONTENT

module_path = str.upper(__name__).replace("OKGRAPH.", "")


class SlidingWindows:
    """
    A class used to inspect the context of one or two target words.
    The words that appear close to the target words inside the corpus are the words with similar meaning to the target
     words (according to the distributional hypothesis) and are going to be referenced as labels. The labels appear in
     the same context of the target words.
    The context is represented by windows of text with specified size containing the target words. The size of a text
     window is measured in "words". The text windows are extracted from the corpus and used to identify the labels.
    Attributes:
        target_words: list of word/words whose context has/have to be inspected
        corpus_index_path: path of the stored and indexed corpus
        corpus_dict: dictionary of the corpus {word in corpus: occurrences in corpus}
        corpus_total_occurrences: total number of occurrences in the corpus
        corpus_inverse_frequency_dict: dictionary {word in corpus: inverse frequency}
        windows_max_size: maximum size of the windows (in 'words') that can contain the target word/words
        windows_list: list of context windows
        windows_dict: dictionary of the windows {word in windows: occurrences in windows}
        windows_total_occurrences: total number of occurrences in all of the windows
        windows_occurrence_dict: dictionary {word in windows: number of windows containing the word}
        windows_frequency_dict: dictionary {word in windows: frequency}
        noise_dict: dictionary {word: log(1 + word window frequency / word corpus frequency)}
        tf_idf_dict: dictionary {word in window: TF-IDF}
        results_dict: dictionary {significant word: TF-IDF}
    """

    def __init__(self,
                 target_words: (str,),
                 corpus_index_path: str = "indexdir",
                 corpus_dictionary_path: str = "dictTotal.npy",
                 windows_max_size: int = 14,
                 noise_threshold: float = 0.10,
                 ):
        """
        Creates a SlidingWindows object.
        Load the data and find the labels related to the target words. The labels of the target words are highly related
         to the windows size: little changes in this value could bring wide changes in the results.
        :param target_words: tuple of word/words whose context has/have to be inspected
        :param corpus_index_path: path of the stored and indexed corpus
        :param corpus_dictionary_path: path of the stored dictionary of the corpus {word: occurrences}
        :param windows_max_size: size of the windows (in 'words') that can contain the target word/words
        :param noise_threshold: maximum "noise" value to accept the label as significant
        """

        # Input check
        if not isinstance(target_words, tuple):
            raise TypeError(f"target_words must be a tuple")
        for target_word in target_words:
            if not isinstance(target_word, str):
                raise TypeError(f"target_words must be a tuple of strings")
        if len(target_words) == 0:
            raise ValueError(f"target_words must contain at least one word")
        if len(target_words) > windows_max_size:
            raise ValueError(f"target_words contains {len(target_words)} words that cannot fit"
                             f" in windows of the specified size ({windows_max_size})")
        if not isinstance(windows_max_size, int):
            raise TypeError(f"windows_size must be an int")

        self.target_words = list(target_words)
        self.windows_size = windows_max_size
        logger.info(f"{module_path} {self.target_words}: Start windowing of target words")

        # Get the corpus data
        self.corpus_index_path = corpus_index_path
        logger.info(f"{module_path} {self.target_words}: Loading corpus dictionary")
        self.corpus_dict = dict(np.load(corpus_dictionary_path, allow_pickle=True).item())

        # Processing the corpus data
        logger.info(f"{module_path} {self.target_words}: Processing corpus data")
        logger.debug(f"{module_path} {self.target_words}: Total unique corpus words: {len(self.corpus_dict)}")
        self.corpus_total_occurrences = self.__total_occurrences(self.corpus_dict)
        logger.debug(f"{module_path} {self.target_words}: Total corpus occurrences: {self.corpus_total_occurrences}")
        logger.debug(f"{module_path} {self.target_words}: Building corpus inverse frequency dictionary")
        self.corpus_inverse_frequency_dict = self.__inverse_frequency_dictionary(
            self.corpus_dict, self.corpus_total_occurrences)

        # Create the windows
        logger.info(f"{module_path} {self.target_words}: Creating windows")
        (self.windows_list, self.windows_dict) = self.__create_windows()

        # Processing windows data
        logger.info(f"{module_path} {self.target_words}: Processing windows data")
        logger.debug(f"{module_path} {self.target_words}: Number of windows: {len(self.windows_list)}")
        if len(self.windows_list) > 0:
            logger.debug(f"{module_path} {self.target_words}: Removing target words from windows dictionary")
            for word in self.target_words:
                self.windows_dict.pop(word)
            logger.debug(f"{module_path} {self.target_words}: Total unique windows words: {len(self.windows_dict)}")

            self.windows_total_occurrences = self.__total_occurrences(self.windows_dict)
            logger.debug(f"{module_path} {self.target_words}: Total windows occurrences: {self.windows_total_occurrences}")

            logger.debug(f"{module_path} {self.target_words}: Building windows occurrence dictionary")
            self.windows_occurrence_dict = self.__windows_occurrence_dictionary(self.windows_dict, self.windows_list)

            logger.debug(f"{module_path} {self.target_words}: Building windows frequency dictionary")
            self.windows_frequency_dict = self.__frequency_dictionary(self.windows_dict, self.windows_total_occurrences)

            logger.debug(f"{module_path} {self.target_words}: Building windows noise dictionary")
            self.noise_dict = self.__noise_dictionary(self.windows_dict, self.windows_total_occurrences,
                                                      self.corpus_dict, self.corpus_total_occurrences)

            logger.debug(f"{module_path} {self.target_words}: Building windows TF-IDF dictionary")
            self.tf_idf_dict = self.__tf_idf_dictionary(self.windows_occurrence_dict, self.windows_frequency_dict,
                                                        self.corpus_inverse_frequency_dict, self.windows_list)

            logger.debug(f"{module_path} {self.target_words}: Cleaning windows TF-IDF dictionary")
            cleaned_tf_idf_dict = self.__clean_results(self.target_words, self.windows_dict, self.tf_idf_dict,
                                                       self.noise_dict, noise_threshold)

            logger.debug(f"{module_path} {self.target_words}: Sorting cleaned TF-IDF dictionary to obtain labels")
            self.results_dict = dict(sorted(cleaned_tf_idf_dict.items(), key=operator.itemgetter(1), reverse=True))
        else:
            self.windows_total_occurrences = 0
            self.windows_occurrence_dict = {}
            self.windows_frequency_dict = {}
            self.noise_dict = {}
            self.results_dict = {}

    def __str__(self) -> str:
        """
        Returns the object as a string.
        """
        string = "The windows of " + self.target_words[0]
        for e in self.target_words[1:]:
            string = string + "and " + e
        return string

    def __create_windows(self) -> ([[str]], {str: int}):
        """
        Extracts the windows from the corpus to represent the context of the target words.
        :return: the list of the windows (list of words) containing the target words and the dictionary {word: occurrences}
          related to these windows
        """
        # Limit of parsed documents
        limit = None

        # Open the indexed corpus to search for the documents containing the target words
        query_results = None
        ix = index.open_dir(self.corpus_index_path)

        # Define all the possible sequences of target words
        target_words_permutations = list(itertools.permutations(self.target_words, len(self.target_words)))

        # Define the queries to search the target words in the indexed text:
        #  the query will match the documents containing the words in the same order of the permutation at a max
        #  distance defined by the integer following the '~' character.
        text_queries = ['"'+' '.join(permutation)+'"~'+str(self.windows_size)
                        for permutation in target_words_permutations]
        parsed_queries = [QueryParser(FIELD_CONTENT, ix.schema).parse(u''+text_query)
                          for text_query in text_queries]

        # Find the documents containing the target words.
        #  Save the documents content in a dictionary using their title as a key: this way the documents matching
        #  more than one permutation of the target words are counted one single time
        documents_results = {}
        with ix.searcher() as searcher:
            search_results = [searcher.search(parsed_query, limit=limit) for parsed_query in parsed_queries]
            for results in search_results:
                for result in results:
                    if result[FIELD_TITLE] not in documents_results:
                        documents_results[result[FIELD_TITLE]] = result[FIELD_CONTENT]

        # Filter the content of every matched document:
        #  search and extract from every document the window/windows in which the target words appear closely
        windows_list, windows_dict = self.__window_extraction(documents_results)
        return windows_list, windows_dict

    def __window_extraction(self, documents_results: [str]) -> ([[str]], {str: int}):
        """
        Extracts the windows from the list of matched indexed documents.
         Windows are created centering the target words in a sequence of words with fixed maximum size.
        :param documents_results: list of matched indexed documents
        :return: the list of windows and their occurrence dictionary
        """
        windows_list = []
        windows_dict = {}
        for (title, content) in documents_results.items():
            content = content.split()
            new_window = []
            matched_target_words = []
            i_first_match = None  # index of the first target word match
            i_last_match = None   # index of the last target word match
            center_size = None    # distance between the first and last target words (target words included)
            offset_size = None    # size of the window borders

            for word, i in zip(content, range(len(content))):
                new_window.append(word)

                # Matched a new target word
                if word in self.target_words and word not in matched_target_words:
                    matched_target_words.append(word)

                    if len(matched_target_words) == 1:
                        i_first_match = i
                    if len(matched_target_words) == len(self.target_words):
                        i_last_match = i
                        center_size = i - i_first_match + 1
                        offset_size = math.ceil((self.windows_size - center_size) / 2)

                # Exceeded maximum distance between two target words: reset the previous matches
                if matched_target_words and i - i_first_match + 1 > self.windows_size:
                    matched_target_words = []
                    i_first_match, i_last_match, center_size, offset_size = None, None, None, None

                # New window completed: save it and reset previous matches
                elif len(matched_target_words) == len(self.target_words) and i - offset_size == i_last_match:
                    windows_list.append(new_window[-(center_size+2*offset_size):])
                    for window_word in windows_list[-1]:
                        windows_dict[window_word] = windows_dict.get(window_word, 0) + 1
                    new_window, matched_target_words = [], []
                    i_first_match, i_last_match, center_size, offset_size = None, None, None, None

        return windows_list, windows_dict

    @staticmethod
    def __total_occurrences(dictionary: {str: int}) -> int:
        """
        Get the total amount of words in the dictionary, or rather the sum of the occurrences of every unique word in
         the dictionary
        :param dictionary: {word: occurrences}
        :return: the total amount of words in the dictionary
        """
        total_occurrences = 0
        for occurrences in dictionary.values():
            total_occurrences += occurrences

        return total_occurrences

    @staticmethod
    def __windows_occurrence_dictionary(dictionary: {str: int}, windows: [[str]]) -> {str: int}:
        """
        Evaluates the presences of the words in the windows.
        :param dictionary: {word: occurrences}
        :param windows: list of all the windows from the corpus
        :return: dictionary {word: number of windows containing the word}
        """
        dictionary_keys = list(dictionary.keys())
        windows_occurrence_dict = {}
        for word in dictionary_keys:
            for window in windows:
                if word in window:
                    windows_occurrence_dict[word] = windows_occurrence_dict.get(word, 0) + 1

        return windows_occurrence_dict

    @staticmethod
    def __frequency_dictionary(occurrence_dictionary: {str: int}, total_occurrences: int) -> {str: float}:
        """
        Evaluates the frequency of every word in the dictionary.
        The frequency of a word is obtained dividing the word occurrences by the total number of word occurrences in
         the dictionary.
        :param occurrence_dictionary: dictionary {word: occurrences}
        :param total_occurrences: total amount of occurrences in the dictionary
        :return: dictionary {word: words frequency}
        """
        occurrence_dictionary_keys = list(occurrence_dictionary.keys())
        f_dict = {}
        for word in occurrence_dictionary_keys:
            f_dict[word] = occurrence_dictionary.get(word) / total_occurrences

        return f_dict

    @staticmethod
    def __inverse_frequency_dictionary(occurrence_dictionary: {str: int}, total_occurrences: int) -> {str: float}:
        """
        Evaluates the inverse frequency of every word in the dictionary.
        The inverse frequency of a word is obtained from the division of the total number of word occurrences in the
         dictionary by the words occurrences.
        :param occurrence_dictionary: dictionary {word: occurrences}
        :param total_occurrences: total amount of occurrences in the dictionary
        :return: dictionary {word: words inverse frequency}
        """
        occurrence_dictionary_keys = list(occurrence_dictionary.keys())
        if_dict = {}
        for word in occurrence_dictionary_keys:
            if_dict[word] = total_occurrences / occurrence_dictionary.get(word)
        return if_dict

    @staticmethod
    def __noise_dictionary(windows_dictionary: {str: int}, windows_total_occurrences: {str: int},
                           corpus_dictionary: {str: int}, corpus_total_occurrences: {str: int}) -> {str: float}:
        """
        Evaluates the importance of a word in the defined context, or rather how much that word is characteristic for
         the defined windows. The importance is evaluated through the "word corpus frequency / word window frequency"
         ratio, so that words appearing mainly in the windows are low rated, while words appearing often in all the
         corpus are high rated. A logarithm function is used to scale the ratio.
        :param windows_dictionary: dictionary {word: occurrences of word in windows}
        :param corpus_dictionary: dictionary {word: occurrences of word in corpus}
        :return: the noise dictionary {word: log(1 + word corpus frequency / word window frequency)}
        """
        windows_dictionary_keys = list(windows_dictionary.keys())

        noise_dict = {}
        for word in windows_dictionary_keys:
            # QSTN: all the words in the windows dictionary surely are in the corpus dictionary, 'cause windows have
            #  been extracted from the corpus. We are scrolling through the windows, so the if statement will always be
            #  False, right?
            if word not in corpus_dictionary:
                # QSTN: probably unreachable
                noise = 0
            else:
                window_frequency = windows_dictionary.get(word) / windows_total_occurrences
                corpus_frequency = corpus_dictionary.get(word) / corpus_total_occurrences
                noise = numpy.log10(1+corpus_frequency / window_frequency)

            noise_dict[word] = noise

        return noise_dict

    @staticmethod
    def __tf_idf_dictionary(windows_occurrence_dictionary: {str: int}, windows_frequency_dictionary: {str: float},
                            corpus_inverse_frequency_dictionary: {str: float}, windows_list: [[str]]) -> {str: float}:
        """
        Evaluates the TF-IDF statistic for every word in the windows.
        :param windows_occurrence_dictionary: dictionary {word in windows: occurrences}
        :param windows_frequency_dictionary: dictionary {word in windows: frequency}
        :param corpus_inverse_frequency_dictionary: dictionary {word in corpus: inverse frequency}
        :param windows_list: list of windows
        :return: the TF-IDF dictionary {word in window: TF-IDF}
        """
        windows_frequency_dictionary_keys = list(windows_frequency_dictionary.keys())
        tf_idf_dict = {}
        for word in windows_frequency_dictionary_keys:
            # QSTN: all the words in the windows dictionary surely are in the corpus dictionary, 'cause windows have
            #  been extracted from the corpus. We are scrolling through the windows, so none of the conditions in the if
            #  statement will always be True, right?
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
    def __clean_results(target_words: [str], windows_dictionary: {str: int}, tf_idf_dictionary: {str: float},
                        noise_dictionary: {str: float}, noise_threshold: float) -> {str: float}:
        """
        Clean the result dictionary removing the not so significant words.
        :param windows_dictionary: dictionary {word in window: occurrences}
        :param tf_idf_dictionary: dictionary {word in window: TF-IDF}
        :param noise_dictionary: dictionary {word: log(1 + word window frequency / word corpus frequency)}
        :param noise_threshold: minimum "noise" value to accept the word as significant
        :return: the cleaned TF-IDF dictionary {significant word in window: TF-IDF}
        """

        cleaned_dictionary = dict(tf_idf_dictionary)

        # Remove the words whose occurrences are higher than the total amount of unique words
        count_removed_words = 0
        logger.debug(f"{module_path} {target_words}:"
                     f" Removing words whose occurrences are higher than the total amount of unique words")
        windows_dictionary_keys = list(windows_dictionary.keys())
        for key in windows_dictionary_keys:
            if windows_dictionary.get(key) >= len(cleaned_dictionary):
                count_removed_words += 1
                logger.debug(f"{module_path} {target_words}:"
                             f" Removing '{key}': {windows_dictionary.get(key)} < {len(cleaned_dictionary)}")
                cleaned_dictionary.pop(key)
        logger.debug(f"{module_path} {target_words}: Removed {count_removed_words} words")

        # Remove words shorter than 3 characters
        count_removed_words = 0
        logger.debug(f"{module_path} {target_words}: Removing words shorter than 3 characters")
        clean_dictionary_keys = list(cleaned_dictionary.keys())
        for key in clean_dictionary_keys:
            if len(key) < 3:
                count_removed_words += 1
                logger.debug(f"{module_path} {target_words}: Removing '{key}': len={len(key)} < 3")
                cleaned_dictionary.pop(key)
        logger.debug(f"{module_path} {target_words}: Removed {count_removed_words} words")

        # Remove words with a noise higher than the specified threshold
        count_removed_words = 0
        logger.debug(f"{module_path} {target_words}:"
                     f" Removing words with a noise higher than the specified threshold")
        clean_dictionary_keys = list(cleaned_dictionary.keys())
        for key in clean_dictionary_keys:
            if noise_dictionary.get(key) > noise_threshold:
                count_removed_words += 1
                logger.debug(f"{module_path} {target_words}:"
                             f" Removing '{key}': noise={noise_dictionary.get(key):.4f} > {noise_threshold:.2f}")
                cleaned_dictionary.pop(key)
        logger.debug(f"{module_path} {target_words}: Removed {count_removed_words} words")

        # Remove digits
        count_removed_words = 0
        logger.debug(f"{module_path} {target_words}: Removing digits")
        clean_dictionary_keys = list(cleaned_dictionary.keys())
        for key in clean_dictionary_keys:
            if str.isnumeric(key):
                count_removed_words += 1
                logger.debug(f"{module_path} {target_words}: Removing '{key}'")
                cleaned_dictionary.pop(key)
        logger.debug(f"{module_path} {target_words}: Removed {count_removed_words} words")

        clean_dictionary_keys = list(cleaned_dictionary.keys())
        for key in clean_dictionary_keys:
            cleaned_dictionary[key] = cleaned_dictionary[key] #/ numpy.log10(1+noise_dictionary.get(key))

        logger.debug(f"{module_path} {target_words}:"
                     f" Cleaned dictionary counts {len(cleaned_dictionary)} words,"
                     f" against the {len(tf_idf_dictionary)} words of the starting dictionary")
        logger.debug(f"{module_path} {target_words}: Final cleaned dictionary keys: {cleaned_dictionary}")

        # Returns the cleaned dictionary
        return cleaned_dictionary

    def get_results(self):
        return self.results_dict
