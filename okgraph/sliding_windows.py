import operator
import re
import tqdm
from tqdm import tqdm
import scipy
from whoosh import index
from whoosh.qparser import QueryParser
import numpy as np


class SlidingWindows:
    """
    A class used to inspect the relation in a pair of words.
    The relation between the two words is described through a set of labels that are found in the corpus between the
     words which occurs in the same context of the specified words.
    The context is represented through a text window of the corpus with a specified dimension in which the two words of
     interest appear closely.
    Attributes:
        words: list of the two words whose relation has to be inspected
        corpus_index_path: path of the stored and indexed corpus
        window_center_size: max distance between the two words of interest to be considered close (sharing the context)
        window_offset_size: offset from the two words surrounding the window center
        window_total_size: total size of a context window (center + 2 * offset)
        windows_list: list of context windows
        windows_dictionary: dictionary of the windows {word in window: occurrences in windows}
        windows_total_occurrences: total number of occurrences in all of the windows
        corpus_dictionary: dictionary of the corpus {word in corpus: occurrences in corpus}
        corpus_total_occurrences: total number of occurrences in the corpus
        corpus_inverse_frequency_dictionary: dictionary {word in corpus: inverse frequency]
        results: dictionary of labels {word: TF-IDF}
        verbose: to print or not messages in the log
    """

    def __init__(self,
                 words: [str],
                 corpus_index_path: str = 'indexdir',
                 corpus_dictionary_path: str = None,
                 window_total_size: int = 14,
                 window_center_size: int = 6,
                 verbose: bool = True
                 ):
        """
        Creates a SlidingWindows object.
        Load the data and find the labels related to the two words of interest.
        :param words: list of the two words of interest
        :param corpus_index_path: path of the stored and indexed corpus
        :param corpus_dictionary_path: path of the stored dictionary of the corpus {word: occurrences}
        :param window_total_size: total size of a context window
        :param window_center_size: max distance between the two words of interest to be considered close
        """

        # Input check
        if not isinstance(words, list) and isinstance(words[0], str) and isinstance(words[1], str):
            raise TypeError('error type in words')
        if len(words) != 2:
            raise ValueError('words must be a list of 2 strings')
        if not isinstance(window_total_size, int):
            raise TypeError('error type in l')
        if not isinstance(window_center_size, int):
            raise TypeError('error type in d')

        self.verbose = verbose

        self.words = words
        # Get the corpus data
        self.corpus_index_path = corpus_index_path
        self.corpus_dictionary = np.load(corpus_dictionary_path, allow_pickle=True).item()
        self.corpus_total_occurrences = self.total_occurrences(self.corpus_dictionary)
        self.corpus_inverse_frequency_dictionary = self.inverse_frequency_dictionary(
            self.corpus_dictionary,
            self.corpus_total_occurrences
        )
        # Define the windows' parameters
        self.window_total_size = window_total_size
        self.window_center_size = window_center_size
        self.window_offset_size = int((window_total_size - (window_center_size + 2)) / 2)
        # Create the windows
        (self.windows_list, self.windows_dictionary) = self.create_windows(
            words[0],
            words[1],
            window_center_size=self.window_center_size + 1,  # QSTN: why does it do +1? Why not +2?
            window_offset_size=self.window_offset_size,
            corpus_index_path=self.corpus_index_path
        )
        self.windows_total_occurrences = self.total_occurrences(self.windows_dictionary)
        # Extract the labels
        self.results = self._label_extraction()

    def __str__(self) -> str:
        """
        Returns the object as a string.
        """
        return 'The windows of ' + self.words[0] + ' and ' + self.words[2]

    def create_windows(self,
                       word_one: str,
                       word_two: str,
                       window_center_size: int = 7,
                       window_offset_size: int = 3,
                       corpus_index_path: str = 'indexdir'
                       ) -> ([[str]], {str : int}):
        """
        Extracts the windows from the corpus, to represent the context of the two words of interest.
        A window is a list of words which respect what follows:
         - The first and the last 'window_offset_size' words are the margins of the window, with generic words. One of
            the two words of interest marks the limit between the left margin and the center of the windows, while a
            generic word marks the limit between the center and the right margin.
         - The center of the window is filled with generic words, but has to contain the other word of interest in at
            least one of the positions. The center of the window has a fixed size 'window_center_size'.
        :param word_one: first word of interest
        :param word_two: second word of interest
        :param window_center_size: max distance between the two words of interest to be considered close
        :param window_offset_size: offset from the two words surrounding the window center
        :param corpus_index_path: path of the stored and indexed corpus
        :return: the list of the windows (list of words) containing the two words and the dictionary {word: occurrences}
          related to these windows
        """
        # Limit of parsed documents (CHECK THIS)
        limit = 10000000

        # Create a regular expression to define the windows of interest in which the two words appear closely.
        re_word_one = word_one + r'\s'
        re_word_two = word_two + r'\s'
        re_any_word = r'(\w+\W*\s)'
        re_margin = re_any_word + r'{' + str(window_offset_size) + r'}'
        re_center = re_any_word + r'{' + str(window_center_size) + r'}'
        re_match_precedent_if = r'?=(' + re_any_word + r'){0,' + str(window_center_size) + r'}'

        #  Expression with word one as the limit of the left margin:
        exp1 = re_margin + re_word_one + r'(' + re_match_precedent_if + word_two + r')' + re_center + re_margin
        #  Expression two:
        exp2 = re_margin + re_word_two + r'(' + re_match_precedent_if + word_one + r')' + re_center + re_margin
        #  Final expression: exp1 OR exp2
        regex = re.compile(r'(' + exp1 + r')' + r'|' + r'(' + exp2 + r')')

        # Search in the indexed corpus the documents containing the two words
        # QSTN: the 'parse' function cares about the set of words, not their order. The two queries 'word_one word_two'
        #  and 'word_two word_one' should produce the same results. Do '~d' change something on the query? What is its
        #  function?
        ix = index.open_dir(corpus_index_path)
        query_one = '\"' + word_one + ' ' + word_two + '\"~' + str(window_center_size)
        query_two = '\"' + word_two + ' ' + word_one + '\"~' + str(window_center_size)
        query_one_results = QueryParser("content", ix.schema).parse(u'' + query_one)
        query_two_results = QueryParser("content", ix.schema).parse(u'' + query_two)

        query_results = []

        with ix.searcher() as searcher:
            results = searcher.search(query_one_results, limit=limit)
            for result in results:
                query_results.append(result['content'])
        with ix.searcher() as searcher:
            results = searcher.search(query_two_results, limit=limit)
            for result in results:
                query_results.append(result['content'])

        # Filter the content of every matched document:
        #  search in every document the existence of a window in which the two words appear closely (use the previously
        #  created regular expression)
        # Apply the regular expression to every document
        query_results = [regex.search(doc) for doc in query_results]
        # Extract the matching windows (when present) from the results of the search with the regular expression
        query_results = [doc.group(0) if doc is not None else '' for doc in query_results]
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
        if self.verbose is True:
            dictionary_keys = tqdm(list(dictionary.keys()))
        else:
            dictionary_keys = list(dictionary.keys())

        windows_occurrence_dict = {}
        for word in dictionary_keys:
            for window in windows:
                if word in window:
                    windows_occurrence_dict[word] = windows_occurrence_dict.get(word, 0) + 1

        return windows_occurrence_dict

    def frequency_dictionary(self, occurrence_dictionary: {str: int}, total_occurrences: int) -> {str: (int, float, [])}:
        """
        Evaluates the frequency of every word in the dictionary.
        The frequency of a word is obtained dividing the word occurrences by the total number of word occurrences in
         the dictionary.
        :param occurrence_dictionary: dictionary {word: occurrences}
        :param total_occurrences: total amount of occurrences in the dictionary
        :return: dictionary {word: word's frequency}
        """
        if self.verbose is True:
            occurrence_dictionary_keys = tqdm(list(occurrence_dictionary.keys()))
        else:
            occurrence_dictionary_keys = list(occurrence_dictionary.keys())

        f_dict = {}
        for word in occurrence_dictionary_keys:
            # QSTN: the third term of the tuple is never referenced, is it useless?
            #  The first term is referenced somewhere, but it could be replaced, leaving this dictionary a pure
            #  frequency dictionary, would be better, right?
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
        if self.verbose is True:
            occurrence_dictionary_keys = tqdm(list(occurrence_dictionary.keys()))
        else:
            occurrence_dictionary_keys = list(occurrence_dictionary.keys())

        if_dict = {}
        for word in occurrence_dictionary_keys:
            # TODO: change scipy.log10 (deprecated) with numpy.log10
            if_dict[word] = scipy.log10(total_occurrences / occurrence_dictionary.get(word))
        return if_dict

    def ratio_dictionary(self, windows_dictionary, corpus_dictionary):
        """
        Evaluates the importance of a word in the defined context, or rather how much that word is characteristic for
         the defined windows. The importance is evaluated through the 'word window frequency / word corpus frequency'
         ratio, so that words appearing mainly in the windows are high rated, while words appearing often in all the
         corpus are low rated. A logarithm function is used to scale the ratio.
        :param windows_dictionary: dictionary {word: occurrences of word in windows}
        :param corpus_dictionary: dictionary {word: occurrences of word in corpus}
        :return: the ratio dictionary {word: log(word window frequency / word corpus frequency)}
        """
        if self.verbose is True:
            windows_dictionary_keys = tqdm(list(windows_dictionary.keys()))
        else:
            windows_dictionary_keys = list(windows_dictionary.keys())

        total_corpus_occurrences = self.corpus_total_occurrences
        total_windows_occurrences = self.windows_total_occurrences

        ratio_dict = {}
        for word in windows_dictionary_keys:
            # QSTN: all the words in the windows dictionary surely are in the corpus dictionary, 'cause windows have
            #  been extracted from the corpus. We are scrolling through the windows, so the if statement will always be
            #  False, right?
            if word not in corpus_dictionary:
                # QSTN: probably unreachable
                ratio = 0
            else:
                # TODO: change scipy.log10 (deprecated) with numpy.log10
                window_frequency = windows_dictionary.get(word) / total_windows_occurrences
                corpus_frequency = corpus_dictionary.get(word) / total_corpus_occurrences
                ratio = scipy.log10(window_frequency / corpus_frequency)

            ratio_dict[word] = ratio

        return ratio_dict

    def tf_idf_dictionary(self, windows_occurrence_dictionary, windows_frequency_dictionary, corpus_inverse_frequency_dictionary, windows_list):
        """
        Evaluates the TF-IDF statistic for every word in the windows.
        :param windows_occurrence_dictionary: dictionary {word in windows: occurrences}
        :param windows_frequency_dictionary: dictionary {word in windows: frequency}
        :param corpus_inverse_frequency_dictionary: dictionary {word in corpus: inverse frequency}
        :param windows_list: list of windows
        :return: the TF-IDF dictionary {word in window: TF-IDF}
        """
        if self.verbose is True:
            windows_frequency_dictionary_keys = tqdm(list(windows_frequency_dictionary.keys()))
        else:
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

    def clean_results(self, windows_dictionary, tf_idf_dictionary, ratio_dictionary, threshold=0.50):
        """
        Clean the results' dictionary removing the non valid words.
        :param windows_dictionary: dictionary {word in window: occurrences}
        :param tf_idf_dictionary: dictionary {word in window: TF-IDF}
        :param ratio_dictionary: dictionary {word: log(word window frequency / word corpus frequency)}
        :param threshold: minimum 'ratio' value to accept the word as a valid one
        :return:
        """

        clean_dictionary = tf_idf_dictionary

        # Remove the words whose occurrences are higher than the total amount of unique words
        if self.verbose is True:
            windows_dictionary_keys = tqdm(list(windows_dictionary.keys()))
        else:
            windows_dictionary_keys = list(windows_dictionary.keys())
        for key in windows_dictionary_keys:
            if windows_dictionary.get(key) >= len(clean_dictionary):
                clean_dictionary.pop(key)

        # Remove words shorter than 3 characters
        if self.verbose is True:
            clean_dictionary_keys = tqdm(list(clean_dictionary.keys()))
        else:
            clean_dictionary_keys = list(clean_dictionary.keys())
        for key in clean_dictionary_keys:
            if len(key) < 3:
                clean_dictionary.pop(key)

        # Remove words with a ratio that is lower than the specified threshold
        if self.verbose is True:
            clean_dictionary_keys = tqdm(list(clean_dictionary.keys()))
        else:
            clean_dictionary_keys = list(clean_dictionary.keys())
        for key in clean_dictionary_keys:
            if ratio_dictionary.get(key) < threshold:
                clean_dictionary.pop(key)

        # Remove numbers (their string representation)
        if self.verbose is True:
            clean_dictionary_keys = tqdm(list(clean_dictionary.keys()))
        else:
            clean_dictionary_keys = list(clean_dictionary.keys())
        for key in clean_dictionary_keys:
            if str.isnumeric(key):
                clean_dictionary.pop(key)

        # Returns the cleaned dictionary
        return clean_dictionary

    def _label_extraction(self, threshold=0.50):
        """
        Extracts the labels which describe the relation between the two specified words.
        The label of interest are the words which appear in the same context of the two words, or rather in the same
         windows.
        :param threshold: minimum 'ratio' value to accept the word as a valid one
        :return: a dictionary {word (label): TF-IDF}
        """

        if len(self.windows_list) == 0:
            raise ValueError('Empty list')

        # Remove the two words, 'cause they can't be one of the label
        self.windows_dictionary.pop(self.words[0])
        self.windows_dictionary.pop(self.words[1])

        # Find the labels
        windows_occurrence_dict = self.windows_occurrence_dictionary(self.windows_dictionary, self.windows_list)
        windows_frequency_dict = self.frequency_dictionary(self.windows_dictionary, self.windows_total_occurrences)
        ratio_dict = self.ratio_dictionary(self.windows_dictionary, self.corpus_dictionary)
        dirty_result_dict = self.tf_idf_dictionary(windows_occurrence_dict, windows_frequency_dict, self.corpus_inverse_frequency_dictionary, self.windows_list)
        results_dict = self.clean_results(self.windows_dictionary, dirty_result_dict, ratio_dict, threshold)
        results_dict = dict(sorted(results_dict.items(), key=operator.itemgetter(1), reverse=True))

        return results_dict

    def get_results(self):
        return self.results

    def get_windows_dictionary(self):
        return self.windows_dictionary

    def get_windows_occurrences(self):
        return self.windows_total_occurrences
