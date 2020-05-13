from pymagnitude import Magnitude
from nltk.probability import FreqDist
from okgraph.file_converter import FileConverter
from os import path
from okgraph.indexing import Indexing
from okgraph.utils import create_dictionary
from okgraph.logger import logger

module_path = str.upper(__name__).replace('OKGRAPH.', '')
algorithms_package = 'okgraph.task'


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
        index: path of the indexed corpus files
        dictionary: path of the corpus dictionary
        occurrence_corpus: ???
    """

    def __init__(self,
                 corpus: str = None,
                 embeddings: str = None,
                 k: int = 5,
                 stream: bool = False,
                 lazy_loading: int = 0,
                 index: str = None,
                 dictionary: str = None,
                 occurrence_corpus: str = None,
                 ):
        """
        Creates an OKgraph object.

        It loads a text corpus, and optionally a word-embedding model
        - via a local resource,
        - or streaming a file on a server

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
        :param index: path of the indexed corpus files
        :param dictionary: path of the corpus dictionary
        :param occurrence_corpus: ???

        QSTN: OKgraph should performs unsupervised natural-language understanding, but it uses Magnitude, that could
               be not unsupervised
        TODO: check the accepted extension for text corpus. Magnitude can take .gz, .bz2, .txt and no extension..
        """

        if embeddings is None or not path.exists(embeddings):
            if embeddings is None:
                logger.info(f'{module_path}: Embedding file for {corpus} not specified. Using default values')
            else:
                logger.warning(f'{module_path}: Specified embedding file {embeddings} for {corpus} not found. Using default values')

            embeddings = corpus + '.magnitude'
            if path.exists(embeddings):
                logger.info(f'{module_path}: Embedding file with default name found. Using embedding file {embeddings}')
            else:
                logger.info(f'{module_path}: No embedding file with default name found. Generating embedding file with default settings')
                embeddings = FileConverter.corpus_to_magnitude_model(corpus_fname=corpus, save_fname=embeddings)
                logger.info(f'{module_path}: Embedding file {embeddings} generated')
        else:
            logger.info(f'{module_path}: Specified embedding file {embeddings} for {corpus} found')

        # QSTN: should we check for index_path to not be empty?
        if index is None or not path.exists(index):
            if index is None:
                logger.info(f'{module_path}: Indexing directory for {corpus} not specified. Using default values')
            else:
                logger.warning(f'{module_path}: Specified indexing directory {index} for {corpus} not found. Using default values')

            index = '/indexdir'
            if path.exists(index):
                logger.info(f'{module_path}: Default indexing directory found. Using index in directory {index}')
            else:
                logger.info(f'{module_path}: No default indexing directory found. Generating index with default settings')
                ix = Indexing(corpus_path=corpus)
                ix.indexing(index_path=index)
                del ix
                logger.info(f'{module_path}: Index {index} generated')
        else:
            logger.info(f'{module_path}: Specified indexing directory {index} for {corpus} found')

        if dictionary is None or not path.exists(dictionary):
            if dictionary is None:
                logger.info(f'{module_path}: Dictionary file for {corpus} not specified. Using default values')
            else:
                logger.warning(f'{module_path}: Specified dictionary file {dictionary} for {corpus} not found. Using default values')

            dictionary = 'dictTotal.npy'
            if path.exists(dictionary):
                logger.info(f'{module_path}: Dictionary file with default name found. Using dictionary file {dictionary}')
            else:
                logger.info(f'{module_path}: No dictionary file with default value found. Generating dictionary with default settings')
                create_dictionary(corpus, dictionary_name=dictionary, dictionary_save=True)
                logger.info(f'{module_path}: Dictionary file {dictionary} generated')
        else:
            logger.info(f'{module_path}: Specified dictionary {dictionary} for {corpus} found')

        self.embeddings = Magnitude(embeddings, _number_of_values=k, stream=stream, lazy_loading=lazy_loading)
        self.f_distribution = FreqDist()
        self.corpus = corpus
        self.index = index
        self.dictionary = dictionary
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
