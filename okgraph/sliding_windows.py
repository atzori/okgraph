import operator
import re
import numpy
import logging
from os import path
from logging.config import fileConfig
from whoosh import index
from whoosh.qparser import QueryParser
import numpy as np

# Create a logger using the specified configuration
LOG_CONFIG_FILE = path.dirname(path.realpath(__file__)) + '/logging.ini'
if not path.isfile(LOG_CONFIG_FILE):
    LOG_CONFIG_FILE = path.dirname(path.dirname(LOG_CONFIG_FILE)) + '/logging.ini'
fileConfig(LOG_CONFIG_FILE)
logger = logging.getLogger()


class SlidingWindows:
    """
    A class used to inspect the context of one or two target words.
    The words that appear close to the target words inside the corpus are the words with similar meaning to the target
     words (according to the distributional hypothesis) and are going to be referenced as labels. The labels appear in
     the same context of the target words.
    The context is represented by windows of text with specified size containing the target words. The size of a text
     window is measured in 'words'. The text windows are extracted from the corpus and used to identify the labels.
    Attributes:
        target_words: list of one or two words whose context has to be inspected
        corpus_index_path: path of the stored and indexed corpus
        corpus_dictionary: dictionary of the corpus {word in corpus: occurrences in corpus}
        corpus_total_occurrences: total number of occurrences in the corpus
        corpus_inverse_frequency_dictionary: dictionary {word in corpus: inverse frequency}
        window_center_size: max distance (in 'words') between two target words sharing the context
        window_offset_size: number of additional words surrounding the window's center
        windows_list: list of context windows
        windows_dictionary: dictionary of the windows {word in windows: occurrences in windows}
        windows_total_occurrences: total number of occurrences in all of the windows
        windows_occurrence_dict: dictionary {word in windows: number of windows containing the word}
        windows_frequency_dict: dictionary {word in windows: frequency}
        noise_dict: dictionary {word: log(1 + word window frequency / word corpus frequency)}
        tf_idf_dict: dictionary {word in window: TF-IDF}
        results_dict: dictionary {significant word: TF-IDF}
        info: to print or not messages in the log
    """

    def __init__(self,
                 target_words: [str],
                 corpus_index_path: str = 'indexdir',
                 corpus_dictionary_path: str = 'dictTotal.npy',
                 window_center_size: int = 7,
                 window_offset_size: int = 2,
                 noise_threshold: float = 0.70,
                 tf_idf_threshold: float = 0.02,
                 info: bool = True
                 ):
        """
        Creates a SlidingWindows object.
        Load the data and find the labels related to the target words. The labels related to the target words are highly
         related to the windows' size: little changes in those values could bring wide changes in the results.
        :param target_words: list of one or two words whose context has to be inspected
        :param corpus_index_path: path of the stored and indexed corpus
        :param corpus_dictionary_path: path of the stored dictionary of the corpus {word: occurrences}
        :param window_center_size: max distance between the two words of interest to be considered close
        :param window_offset_size: number of additional words surrounding the window's center
        :param noise_threshold: minimum 'noise' value to accept the word as significant
        :param tf_idf_threshold: minimum 'TF-IDF' value to accept the word as significant
        """

        # Input check
        if not isinstance(target_words, list):
            raise TypeError('target_words must be a list of strings')
        if len(target_words) == 0:
            raise TypeError('target_words must contain at least one word')
        if len(target_words) > 2:
            raise TypeError('target_words must contain at most two words')
        if not isinstance(window_center_size, int):
            raise TypeError('window_center_size must be an int')
        if not isinstance(window_offset_size, int):
            raise TypeError('window_offset_size must be an int')

        self.info = info

        if self.info is True:
            logger.info('SLIDING_WINDOWS: Start windowing of target words: {target}'.format(target=target_words))  # LOG INFO
        self.target_words = target_words

        # Get the corpus data
        self.corpus_index_path = corpus_index_path
        if self.info is True:
            logger.info('SLIDING_WINDOWS: Loading corpus data')  # LOG INFO
        self.corpus_dictionary = np.load(corpus_dictionary_path, allow_pickle=True).item()

        # Processing the corpus data
        if self.info is True:
            logger.info('SLIDING_WINDOWS: Processing corpus data')  # LOG INFO
            logger.info('SLIDING_WINDOWS: Total unique corpus words: {n}'.format(n=len(self.corpus_dictionary)))  # LOG INFO
        self.corpus_total_occurrences = self.total_occurrences(self.corpus_dictionary)
        if self.info is True:
            logger.info('SLIDING_WINDOWS: Total corpus occurrences: {n}'.format(n=self.corpus_total_occurrences))  # LOG INFO
        if self.info is True:
            logger.info('SLIDING_WINDOWS: Building corpus inverse frequency dictionary')  # LOG INFO
        self.corpus_inverse_frequency_dictionary = self.inverse_frequency_dictionary(
            self.corpus_dictionary,
            self.corpus_total_occurrences
        )

        # Define the windows' parameters
        self.window_center_size = window_center_size
        self.window_offset_size = window_offset_size

        # Create the windows
        if self.info is True:
            logger.info('SLIDING_WINDOWS: Creating windows')  # LOG INFO
        (self.windows_list, self.windows_dictionary) = self.create_windows()

        # Processing windows data
        if self.info is True:
            logger.info('SLIDING_WINDOWS: Processing windows data')  # LOG INFO
            logger.info('SLIDING_WINDOWS: Number of windows: {n}'.format(n=len(self.windows_list)))  # LOG INFO
        if len(self.windows_list) > 0:
            if self.info is True:
                logger.info('SLIDING_WINDOWS: Removing target words from windows dictionary')  # LOG INFO
            for word in self.target_words:
                self.windows_dictionary.pop(word)
            if self.info is True:
                logger.info('SLIDING_WINDOWS: Total unique windows words: {n}'.format(
                    n=len(self.windows_dictionary)))  # LOG INFO
            self.windows_total_occurrences = self.total_occurrences(self.windows_dictionary)
            if self.info is True:
                logger.info('SLIDING_WINDOWS: Total windows occurrences: {n}'.format(
                    n=self.windows_total_occurrences))  # LOG INFO
            if self.info is True:
                logger.info('SLIDING_WINDOWS: Building windows occurrence dictionary')  # LOG INFO
            self.windows_occurrence_dict = self.windows_occurrence_dictionary(self.windows_dictionary, self.windows_list)
            if self.info is True:
                logger.info('SLIDING_WINDOWS: Building windows frequency dictionary')  # LOG INFO
            self.windows_frequency_dict = self.frequency_dictionary(self.windows_dictionary, self.windows_total_occurrences)
            if self.info is True:
                logger.info('SLIDING_WINDOWS: Building windows noise dictionary')  # LOG INFO
            self.noise_dict = self.noise_dictionary(self.windows_dictionary, self.corpus_dictionary)
            if self.info is True:
                logger.info('SLIDING_WINDOWS: Building windows TF-IDF dictionary')  # LOG INFO
            self.tf_idf_dict = self.tf_idf_dictionary(self.windows_occurrence_dict, self.windows_frequency_dict, self.corpus_inverse_frequency_dictionary, self.windows_list)
            if self.info is True:
                logger.info('SLIDING_WINDOWS: Cleaning windows TF-IDF dictionary')  # LOG INFO
            cleaned_tf_idf_dict = self.clean_results(self.windows_dictionary, self.tf_idf_dict, self.noise_dict, noise_threshold, tf_idf_threshold)
            if self.info is True:
                logger.info('SLIDING_WINDOWS: Sorting cleaned TF-IDF dictionary to obtain labels')  # LOG INFO
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
        string = 'The windows of ' + self.target_words[0]
        for e in self.target_words[1:]:
            string = string + 'and ' + e
        return string

    def create_windows(self) -> ([[str]], {str: int}):
        """
        Extracts the windows from the corpus to represent the context of the target words.
        :return: the list of the windows (list of words) containing the target words and the dictionary {word: occurrences}
          related to these windows
        """
        # Limit of parsed documents
        limit = 10000000

        # Create a regular expression to define the windows in which the target words appear closely.
        # A window is a list of words that can be described by the following patterns:
        # With one target word:
        #  [offset+center/2 words] [target word 1] [offset+center/2 words]
        # With two target words:
        #  [offset words] [target word 1|2] [center words containing the target word 2|1] [offset words]
        regex = None
        re_any_word = r'(\w+\W*\s)'

        if len(self.target_words) == 1:
            re_word_one = self.target_words[0] + r'\s'
            re_offset = re_any_word + r'{' + str(self.window_offset_size + int(self.window_center_size/2)) + r'}'

            exp = re_offset + re_word_one + re_offset

            regex = re.compile(exp)

        if len(self.target_words) == 2:
            re_word_one = self.target_words[0] + r'\s'
            re_word_two = self.target_words[1] + r'\s'
            re_offset = re_any_word + r'{' + str(self.window_offset_size) + r'}'
            re_center = re_any_word + r'{' + str(self.window_center_size) + r'}'
            re_match_if = r'?=(' + re_any_word + r'){0,' + str(self.window_center_size) + r'}'

            exp1 = re_offset + re_word_one + r'(' + re_match_if + re_word_two + r')' + re_center + re_any_word + re_offset
            exp2 = re_offset + re_word_two + r'(' + re_match_if + re_word_one + r')' + re_center + re_any_word + re_offset

            regex = re.compile(r'(' + exp1 + r')' + r'|' + r'(' + exp2 + r')')

        # Search in the indexed corpus the documents containing the target words
        query_results = None
        ix = index.open_dir(self.corpus_index_path)
        if len(self.target_words) == 1:
            query = '"' + self.target_words[0] + '"'
            raw_query_results = QueryParser("content", ix.schema).parse(u'' + query)

            query_results = []
            with ix.searcher() as searcher:
                results = searcher.search(raw_query_results, limit=limit)
                for result in results:
                    query_results.append(result['content'])

        if len(self.target_words) == 2:
            query_one = '"' + self.target_words[0] + ' ' + self.target_words[1] + '"~' + str(self.window_center_size)
            query_two = '"' + self.target_words[1] + ' ' + self.target_words[0] + '\"~' + str(self.window_center_size)
            raw_query_results_one = QueryParser("content", ix.schema).parse(u'' + query_one)
            raw_query_results_two = QueryParser("content", ix.schema).parse(u'' + query_two)

            query_results = []
            with ix.searcher() as searcher:
                results = searcher.search(raw_query_results_one, limit=limit)
                for result in results:
                    query_results.append(result['content'])
            with ix.searcher() as searcher:
                results = searcher.search(raw_query_results_two, limit=limit)
                for result in results:
                    query_results.append(result['content'])

        # Filter the content of every matched document:
        #  search in every document the existence of a window in which the two words appear closely (use the previously
        #  created regular expression)
        # Apply the regular expression to every document
        query_results = [regex.search(doc) for doc in query_results]
        # Extract the matching windows (when present) from the results of the search with the regular expression
        query_results = [doc.group(0) for doc in query_results if doc is not None]
        # QSTN: what does happen here?
        # TODO: add documentation
        windows_list = []
        windows_dict = {}
        for temp in set(query_results):
            if not temp == None:
                if len(windows_list) == 0 or temp != windows_list[-1]:
                    windows_list.append(temp.split())
                    for word in windows_list[-1]:
                        windows_dict[word] = windows_dict.get(word, 0) + 1

        windows_dict = dict(sorted(windows_dict.items(), key=operator.itemgetter(1), reverse=True))

        return windows_list, windows_dict

    def total_occurrences(self, dictionary: {str: int}) -> int:
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

    def windows_occurrence_dictionary(self, dictionary: {str: int}, windows: [[str]]) -> {str: int}:
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

    def frequency_dictionary(self, occurrence_dictionary: {str: int}, total_occurrences: int) -> {
        str: (int, float, [])}:
        """
        Evaluates the frequency of every word in the dictionary.
        The frequency of a word is obtained dividing the word occurrences by the total number of word occurrences in
         the dictionary.
        :param occurrence_dictionary: dictionary {word: occurrences}
        :param total_occurrences: total amount of occurrences in the dictionary
        :return: dictionary {word: word's frequency}
        """
        occurrence_dictionary_keys = list(occurrence_dictionary.keys())
        f_dict = {}
        for word in occurrence_dictionary_keys:
            f_dict[word] = occurrence_dictionary.get(word) / total_occurrences

        return f_dict

    def inverse_frequency_dictionary(self, occurrence_dictionary, total_occurrences):
        """
        Evaluates the inverse frequency of every word in the dictionary.
        The inverse frequency of a word is obtained from the division of the total number of word occurrences in the
         dictionary by the word's occurrences. The result of the division is then scaled using a logarithm to weigh
         down the frequent terms while scale up the rare ones.
        :param occurrence_dictionary: dictionary {word: occurrences}
        :param total_occurrences: total amount of occurrences in the dictionary
        :return: dictionary {word: word's inverse frequency}
        """
        occurrence_dictionary_keys = list(occurrence_dictionary.keys())
        if_dict = {}
        for word in occurrence_dictionary_keys:
            if_dict[word] = numpy.log10(1 + total_occurrences / occurrence_dictionary.get(word))
        return if_dict

    def noise_dictionary(self, windows_dictionary, corpus_dictionary):
        """
        Evaluates the importance of a word in the defined context, or rather how much that word is characteristic for
         the defined windows. The importance is evaluated through the 'word window frequency / word corpus frequency'
         ratio, so that words appearing mainly in the windows are high rated, while words appearing often in all the
         corpus are low rated. A logarithm function is used to scale the ratio.
        :param windows_dictionary: dictionary {word: occurrences of word in windows}
        :param corpus_dictionary: dictionary {word: occurrences of word in corpus}
        :return: the noise dictionary {word: log(1 + word window frequency / word corpus frequency)}
        """
        windows_dictionary_keys = list(windows_dictionary.keys())
        total_corpus_occurrences = self.corpus_total_occurrences
        total_windows_occurrences = self.windows_total_occurrences

        noise_dict = {}
        for word in windows_dictionary_keys:
            # QSTN: all the words in the windows dictionary surely are in the corpus dictionary, 'cause windows have
            #  been extracted from the corpus. We are scrolling through the windows, so the if statement will always be
            #  False, right?
            if word not in corpus_dictionary:
                # QSTN: probably unreachable
                noise = 0
            else:
                window_frequency = windows_dictionary.get(word) / total_windows_occurrences
                corpus_frequency = corpus_dictionary.get(word) / total_corpus_occurrences
                noise = numpy.log10(1 + window_frequency / corpus_frequency)

            noise_dict[word] = noise

        return noise_dict

    def tf_idf_dictionary(self, windows_occurrence_dictionary, windows_frequency_dictionary, corpus_inverse_frequency_dictionary, windows_list):
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
            #  been extracted from the corpus. We are scrolling through the windows, so none of condition in the if
            #  statement will always be True, right?
            if word not in windows_frequency_dictionary or word not in windows_occurrence_dictionary or word not in corpus_inverse_frequency_dictionary:
                # QSTN: probably unreachable
                tf_idf_dict[word] = 0
            else:
                tf_idf_dict[word] = (windows_occurrence_dictionary.get(word) / len(windows_list)) * corpus_inverse_frequency_dictionary.get(word)

        return tf_idf_dict

    def clean_results(self, windows_dictionary, tf_idf_dictionary, noise_dictionary, noise_threshold, tf_idf_threshold):
        """
        Clean the results' dictionary removing the not so significant words.
        :param windows_dictionary: dictionary {word in window: occurrences}
        :param tf_idf_dictionary: dictionary {word in window: TF-IDF}
        :param noise_dictionary: dictionary {word: log(1 + word window frequency / word corpus frequency)}
        :param noise_threshold: minimum 'noise' value to accept the word as significant
        :param tf_idf_threshold: minimum 'TF-IDF' value to accept the word as significant
        :return: the cleaned TF-IDF dictionary {significant word in window: TF-IDF}
        """

        clean_dictionary = dict(tf_idf_dictionary)

        # Remove the words whose occurrences are higher than the total amount of unique words
        if self.info is True:
            count = 0
            logger.info('SLIDING_WINDOWS: Removing words whose occurrences are higher than the total amount of unique words')  # LOG INFO
        windows_dictionary_keys = list(windows_dictionary.keys())
        for key in windows_dictionary_keys:
            if windows_dictionary.get(key) >= len(clean_dictionary):
                if self.info is True:
                    count += 1
                    logger.info('SLIDING_WINDOWS: Removing \'{w}\': {occurences} < {unique}'.format(w=key, occurences=windows_dictionary.get(key), unique=len(clean_dictionary)))  # LOG INFO
                clean_dictionary.pop(key)
        if self.info is True:
            logger.info('SLIDING_WINDOWS: Removed {n} words'.format(n=count))  # LOG INFO

        # Remove words shorter than 3 characters
        if self.info is True:
            count = 0
            logger.info('SLIDING_WINDOWS: Removing words shorter than 3 characters')  # LOG INFO
        clean_dictionary_keys = list(clean_dictionary.keys())
        for key in clean_dictionary_keys:
            if len(key) < 3:
                if self.info is True:
                    count += 1
                    logger.info('SLIDING_WINDOWS: Removing \'{w}\': len={len} < 3'.format(w=key, len=len(key)))  # LOG INFO
                clean_dictionary.pop(key)
        if self.info is True:
            logger.info('SLIDING_WINDOWS: Removed {n} words'.format(n=count))  # LOG INFO

        # Remove words with a noise or TF-IDF lower than the specified threshold
        if self.info is True:
            count = 0
            logger.info('SLIDING_WINDOWS: Removing words with a noise or TF-IDF lower than the specified threshold')  # LOG INFO
        clean_dictionary_keys = list(clean_dictionary.keys())
        for key in clean_dictionary_keys:
            if noise_dictionary.get(key) < noise_threshold:
                if self.info is True:
                    count += 1
                    logger.info('SLIDING_WINDOWS: Removing \'{w}\': noise={noise} < {threshold}'.format(w=key, noise=noise_dictionary.get(key), threshold=noise_threshold))  # LOG INFO
                clean_dictionary.pop(key)
            elif tf_idf_dictionary.get(key) < tf_idf_threshold:
                if self.info is True:
                    count += 1
                    logger.info('SLIDING_WINDOWS: Removing \'{w}\': TF-IDF={tf_idf} < {threshold}'.format(w=key, tf_idf=tf_idf_dictionary.get(key), threshold=tf_idf_threshold))  # LOG INFO
                clean_dictionary.pop(key)
        if self.info is True:
            logger.info('SLIDING_WINDOWS: Removed {n} words'.format(n=count))  # LOG INFO

        # Remove digits
        if self.info is True:
            count = 0
            logger.info('SLIDING_WINDOWS: Removing digits')  # LOG INFO
        clean_dictionary_keys = list(clean_dictionary.keys())
        for key in clean_dictionary_keys:
            if str.isnumeric(key):
                if self.info is True:
                    count += 1
                    logger.info('SLIDING_WINDOWS: Removing \'{w}\''.format(w=key))  # LOG INFO
                clean_dictionary.pop(key)
        if self.info is True:
            logger.info('SLIDING_WINDOWS: Removed {n} words'.format(n=count))  # LOG INFO

        if self.info is True:
            logger.info('SLIDING_WINDOWS: Cleaned dictionary counts {n1} words, againts the {n2} words of the starting dictionary'.format(n1=len(clean_dictionary), n2=len(tf_idf_dictionary)))  # LOG INFO

        # Returns the cleaned dictionary
        return clean_dictionary

    def get_results(self):
        return self.results_dict

    def get_windows_dictionary(self):
        return self.windows_dictionary

    def get_windows_occurrences(self):
        return self.windows_total_occurrences
