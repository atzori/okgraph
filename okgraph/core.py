from pymagnitude import Magnitude
from nltk.probability import FreqDist
from okgraph.file_converter import FileConverter
from os import path
import logging
from logging.config import fileConfig
from okgraph.sliding_windows import SlidingWindows
from okgraph.indexing import Indexing
from okgraph.utils import create_dictionary

# Specify the task's path
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
    """OKgraph currently focuses on the following tasks:

    - set expansion given one or a short set of words, continues this set with a list of other "same-type" words (co-hyponyms)
    - relation expansion given one or a short set of word pairs, continues this set with a list of pairs having the same implicit relation of the given pairs
    - set labeling given one or a short set of words, returns a list of short strings (labels) describing the given set (its type or hyperonym)
    - relation labeling given one or a short set of word pairs, returns a list of short strings (labels) describing the relation in the given set

    Attributes:
        embeddings: word's vector model
        f_distribution: ???
        corpus: path of the corpus' file
        index_path: path of the indexed corpus' files
        dictionary_path: path of the corpus' dictionary
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

        The supported embeddings files are: '.magnitude'.
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
        :param corpus: path of the corpus' file
        :param embeddings: the file path or URL to the magnitude file
        :param k: When the embeddings is set to None and Magnitude is being used to solely featurize keys directly into
                  vectors, k should be set to the approximate upper-bound of the number of keys that will be looked up
                  with query(). If you don't know the exact number, be conservative and pick a large number, while
                  keeping in mind the bigger k is, the more memory it will consume.
        :param stream: stream the URL instead of downloading it
        :param lazy_loading: -1 = pre-load into memory,
                              0 = lazy loads with unbounded in-memory cache,
                             >0 = lazy loads with an LRU cache of that size
        :param index_path: path of the indexed corpus' files
        :param dictionary_path: path of the corpus' dictionary
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
            tmp = tmp.indexing(index_path=index_path)
            logger.info(f'Model index generated.')
            logger.info(f'File dictTotal.npy for {corpus} not found. Generating dictTotal.npy')
            tmp = create_dictionary(corpus, dictionary_path=dictionary_path, dictionary_save=True)
            logger.info(f'dictTotal generated.')
            del tmp

        self.embeddings = Magnitude(embeddings, _number_of_values=k, stream=stream, lazy_loading=lazy_loading)
        self.f_distribution = FreqDist()
        self.corpus = corpus
        self.index = index_path
        self.dictionary = dictionary_path
        self.occurrence_corpus = occurrence_corpus

    def set_expansion(self, seed: [str] = None, algo: str = "centroid", options: dict = {}, k: int = 5):
        """
        Returns a **generator** with results not containing the given seed
        Use itertools to convert to a finite list (see https://stackoverflow.com/a/5234170)
        e.g.: 'Spain','Portugal','Belgium', ...

        """

        # automatically get the file of the algorithm task
        package = algorithms_package + ".set_expansion." + algo
        algorithm = getattr(__import__(package, fromlist=[algo]), algo)

        options["seed"] = seed
        options["k"] = k

        return algorithm.task(self.embeddings, options=options)

    def w(self, l=12, d=6, words:[str] = None):

        return SlidingWindows(words, corpus_dictionary_path=self.dictionary, window_total_size=l, window_center_size=d, corpus_index_path=self.index)

    def relation_labeling(self, windows: [SlidingWindows] = None, algo: str = "intersection", ):

        # automatically get the file of the algorithm task
        package = algorithms_package + ".relation_labeling." + algo
        algorithm = getattr(__import__(package, fromlist=[algo]), algo)

        return algorithm.task(windows)
