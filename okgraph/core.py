from pymagnitude import Magnitude
from nltk.probability import FreqDist
from okgraph.file_converter import FileConverter
from os import path
import logging
from logging.config import fileConfig
from okgraph.indexing import Indexing
from okgraph.utils import create_dictionary

# Specify the task path
algorithms_package = "okgraph.task"

# Create a logger using the specified configuration
LOG_CONFIG_FILE = path.dirname(path.realpath(__file__)) + '/logging.ini'
if not path.isfile(LOG_CONFIG_FILE):
    LOG_CONFIG_FILE = path.dirname(path.dirname(LOG_CONFIG_FILE)) + '/logging.ini'
fileConfig(LOG_CONFIG_FILE)
logger = logging.getLogger()
# Print a first message in the logger
logger.info(f'Log config file is: {LOG_CONFIG_FILE}')


class OKgraph:
    """
    OKgraph currently focuses on the following tasks:
    - "set expansion": given one or a short set of words, continues this set with a list of other "same-type" words
                       (co-hyponyms)
    - "relation expansion": given one or a short set of word pairs, continues this set with a list of pairs having the
                            same implicit relation of the given pairs
    - "set labeling": given one or a short set of words, returns a list of short strings (labels) describing the given
                      set (its type or hyperonym)
    - "relation labeling": given one or a short set of word pairs, returns a list of short strings (labels) describing
                           the relation in the given set

    Attributes:
        embeddings: words vector model
        f_distribution: ???
        corpus: path of the corpus file
        index_path: path of the indexed corpus files
        dictionary_path: path of the corpus dictionary
        occurrence_corpus: ???
    """

    def __init__(self,
                 corpus: str = None,
                 embeddings: str = None,
                 k: int = 5,
                 stream: bool = False,
                 lazy_loading: int = 0,
                 index_path: str = 'indexdir/',
                 dictionary_path: str = None,
                 occurrence_corpus: str = None,
                 create_index: bool = False
                 ):
        """
        Creates an OKgraph object.

        It loads a text corpus, and optionally a word-embedding model
        - via a local resource,
        - or streaming a file on a server.

        The supported embeddings files are: ".magnitude".
        In the future will support .bin, .txt, .vec, .hdf5 by automatically converting them to a
        .magnitude file (by using the magnitude: https://github.com/plasticityai/magnitude#file-format-and-converter).

        HOW TO:

            >>> import okgraph
            >>>
            >>> okg = okgraph.OKgraph('enwik9.txt', 'model_file')
            >>>
            >>> # or equivalently:
            >>> okg = okgraph.OKgraph(corpus = 'file/path/enwik9.txt',
            >>>                       embeddings = 'file/path/model_file.magnitude')
            >>>
            >>> # another alternative
            >>> okg = okgraph.OKgraph(corpus = 'file/path/enwik9.txt',
            >>>                       embeddings = 'https://server/file/path/model_file.magnitude')

        This example usage will set enwik9.txt as corpus file and model_file.magnitude as vector
        model file in magnitude format (extension magnitude is appended automatically if not specified).

        Parameters:
        :param corpus: path of the corpus file
        :param embeddings: the file path or URL to the magnitude file
        :param k: When the embeddings is set to None and Magnitude is being used to solely featurize keys directly into
                  vectors, k should be set to the approximate upper-bound of the number of keys that will be looked up
                  with query(). If you don't know the exact number, be conservative and pick a large number, while
                  keeping in mind the bigger k is, the more memory it will consume.
        :param stream: stream the URL instead of downloading it
        :param lazy_loading: -1 = pre-load into memory,
                              0 = lazy loads with unbounded in-memory cache,
                             >0 = lazy loads with an LRU cache of that size
        :param index_path: path of the indexed corpus files
        :param dictionary_path: path of the corpus dictionary
        :param occurrence_corpus: ???
        :param create_index: forces the indexation of the corpus

        QSTN: OKgraph should performs unsupervised natural-language understanding, but it uses Magnitude, that could
               be not unsupervised
        TODO: check the accepted extension for text corpus. Magnitude can take .gz, .bz2, .txt and no extension..
        """

        if embeddings is None:
            default_magnitude_file = corpus + '.magnitude'
            if path.exists(default_magnitude_file):
                embeddings = default_magnitude_file
                logger.info(f'File model for {corpus} not specified but using model file named {embeddings} in the same directory')
            else:
                logger.info(f'File model for {corpus} not found. Generating model with default options')
                embeddings = FileConverter.corpus_to_magnitude_model(corpus_fname=corpus)
                logger.info(f'Model {embeddings} generated.')

        if not path.exists(index_path) or create_index is True:
            if dictionary_path is None:
                dictionary_path = 'dictTotal.npy'
            logger.info(f'Folder indexing for {corpus} not found. Generating index with default options')
            tmp = Indexing(corpus_path=corpus)
            tmp.indexing(index_path=index_path)
            logger.info(f'Model index generated.')
            logger.info(f'File dictTotal.npy for {corpus} not found. Generating dictTotal.npy')
            create_dictionary(corpus, dictionary_path=dictionary_path, dictionary_save=True)
            logger.info(f'dictTotal generated.')
            del tmp

        self.embeddings = Magnitude(embeddings, _number_of_values=k, stream=stream, lazy_loading=lazy_loading)
        self.f_distribution = FreqDist()
        self.corpus = corpus
        self.index = index_path
        self.dictionary = dictionary_path
        self.occurrence_corpus = occurrence_corpus

    def relation_expansion(self,
                           seed: [(str, str)],
                           k: int = 15,
                           algo: str = None,
                           options: dict = None
                           ):
        """
        Finds pairs similar to the pairs in the seed.
        :param seed: list of words pairs that has to be expanded
        :param k: limit to the number of results
        :param algo: name of the chosen algorithm
        :param options: task options: check the chosen algorithm interface to check its parameters
        """
        # Default algorithm and options
        if algo is None or options is None:
            algo = 'intersection'
            options = {'relation_labeling_algo': 'intersection',
                       'relation_labeling_options': {'dictionary': self.dictionary, 'index': self.index},
                       'relation_labeling_k': 15,
                       'set_expansion_algo': 'centroid',
                       'set_expansion_options': {},
                       'set_expansion_k': 15
                       }

        # Import the algorithm
        package = algorithms_package + '.relation_expansion.' + algo
        algorithm = getattr(__import__(package, fromlist=[algo]), algo)

        # Launch the algorithm and return the result
        return algorithm.task(seed, k, options)

    def relation_labeling(self,
                          seed: [(str, str)],
                          k: int = 15,
                          algo: str = None,
                          options: dict = None
                          ):
        """
        Finds labels describing the relation between the pairs of words in the seed.
        :param seed: list of words pairs that has to be expanded
        :param k: limit to the number of results
        :param algo: name of the chosen algorithm
        :param options: task options: check the chosen algorithm interface to check its parameters
        """
        # Default algorithm and options
        if algo is None or options is None:
            algo = 'intersection'
            options = {'dictionary': self.dictionary, 'index': self.index}

        # Import the algorithm
        package = algorithms_package + '.relation_labeling.' + algo
        algorithm = getattr(__import__(package, fromlist=[algo]), algo)

        # Launch the algorithm and return the result
        return algorithm.task(seed, k, options)

    def set_expansion(self,
                      seed: [str],
                      k: int = 15,
                      algo: str = None,
                      options: dict = None
                      ):
        """
        Finds words similar to the words in the seed (co-hyponyms).
        :param seed: list of words pairs that has to be expanded
        :param k: limit to the number of results
        :param algo: name of the chosen algorithm
        :param options: task options: check the chosen algorithm interface to check its parameters
        """
        # Default algorithm and options
        if algo is None or options is None:
            algo = 'centroid'
            options = {'embeddings': self.embeddings}

        # Import the algorithm
        # QSTN: Use itertools to convert the results to a finite list (see https://stackoverflow.com/a/5234170)
        package = algorithms_package + '.set_expansion.' + algo
        algorithm = getattr(__import__(package, fromlist=[algo]), algo)

        # Launch the algorithm and return the result
        return algorithm.task(seed, k, options)

    def set_labeling(self,
                     seed: [str],
                     k: int = 15,
                     algo: str = None,
                     options: dict = None
                     ):
        """
        Finds labels describing the relation between the words in the seed (hyperonym).
        :param seed: list of words pairs that has to be expanded
        :param k: limit to the number of results
        :param algo: name of the chosen algorithm
        :param options: task options: check the chosen algorithm interface to check its parameters
        """
        # Default algorithm and options
        if algo is None or options is None:
            algo = 'intersection'
            options = {'dictionary': self.dictionary, 'index': self.index}

        # Import the algorithm
        package = algorithms_package + '.set_labeling.' + algo
        algorithm = getattr(__import__(package, fromlist=[algo]), algo)

        # Launch the algorithm and return the result
        return algorithm.task(seed, k, options)
