"""
TRYDOC: The core module contains the library main functionalities.
"""
from pymagnitude import Magnitude
from okgraph.file_converter import WordEmbedding, MagnitudeWordEmbedding, FileConverter
import os
from os import path
import shutil
from okgraph.indexing import Indexing
from okgraph.utils import create_dictionary
from okgraph.utils import logger

algorithms_package = "okgraph.task"


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
        corpus: path of the corpus file
        embedding: words vector model
        index: path of the indexed corpus files
        dictionary: path of the corpus dictionary
    """
    corpus: str
    embeddings: WordEmbedding
    index: str
    dictionary: str

    def __init__(self,
                 corpus: str,
                 embedding: str = None,
                 k: int = 5,
                 stream: bool = False,
                 lazy_loading: int = 0,
                 index: str = None,
                 dictionary: str = None,
                 force_init: bool = False,
                 ):
        """
        Creates an OKgraph object.

        It loads a text corpus, and optionally a word-embedding model
        - via a local resource,
        - or streaming a file on a server

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
        :param corpus: path of the corpus file
        :param embedding: the file path or URL to the vector model (.magnitude file). If not specified, the default
                           path points a '.magnitude' file with the same name and parent directory of the corpus.
                           If the file exists it is used as it is; if the path points to a non existent file it is then
                           created
        :param k: When the embeddings is set to None and Magnitude is being used to solely featurize keys directly into
                  vectors, k should be set to the approximate upper-bound of the number of keys that will be looked up
                  with query(). If you don't know the exact number, be conservative and pick a large number, while
                  keeping in mind the bigger k is, the more memory it will consume
        :param stream: stream the URL instead of downloading it
        :param lazy_loading: -1 = pre-load into memory,
                              0 = lazy loads with unbounded in-memory cache,
                             >0 = lazy loads with an LRU cache of that size
        :param index: path of the directory for the indexed corpus files. If not specified, the default path points a
                      directory named 'indexdir' with the same parent directory of the corpus.
                      If the directory exists it is used as it is; if the path points to a non existent directory it is
                      then created
        :param dictionary: path of the corpus dictionary. If not specified, the default path points a file named
                           'dictTotal.npy' with the same parent directory of the corpus.
                           If the file exists it is used as it is; if the path points to a non existent file it is then
                           created
        :param force_init: when set to True, the embeddings, index and dictionary are always created from scratch. If
                           they already exists they are then overwrited

        TODO: when the specified name of the embeddings, index or dictionary is in a non existent directory, creates it
        """

        # Split the corpus name from its extension
        (corpus_name, corpus_extension) = os.path.splitext(corpus)

        # Creates the embeddings
        if embedding is None:
            embedding = corpus_name + ".magnitude"
            logger.info(f"Embedding file for corpus {corpus} not specified."
                        f" Referencing to default value {embedding}")
        if force_init is True and os.path.exists(embedding):
            logger.info(f"Removing existing embedding file {embedding}")
            os.remove(embedding)
        if not path.exists(embedding):
            if force_init is False:
                logger.info(f"Specified embedding file {embedding} for corpus {corpus} doesn't exist:"
                            f" generating a new one")
            logger.info(f"Starting embedding generation")
            embedding = FileConverter.corpus_to_magnitude_model(corpus, embedding)
            logger.info(f"Embedding file {embedding} generated")
        else:
            logger.info(f"Specified embedding file {embedding} for corpus {corpus} found")

        # Creates the index
        # QSTN: should we check for index_path to not be empty?
        if index is None:
            index = path.join(path.dirname(corpus), "indexdir")
            logger.info(f"Indexing directoy for corpus {corpus} not specified."
                        f" Referencing to default value {index}")
        if force_init is True and os.path.exists(index):
            logger.info(f"Removing existing indexing directory {index}")
            shutil.rmtree(index)
        if not path.exists(index):
            if force_init is False:
                logger.info(f"Specified indexing directory {index} for corpus {corpus} doesn't exist:"
                            f" generating a new one")
            logger.info(f"Starting index generation")
            ix = Indexing(corpus_path=corpus)
            ix.indexing(index_path=index)
            del ix
            logger.info(f"Index {index} generated")
        else:
            logger.info(f"Specified index directory {index} for corpus {corpus} found")

        # Creates the dictionary
        if dictionary is None:
            dictionary = path.join(path.dirname(corpus), "dictTotal.npy")
            logger.info(f"Dictionary file for corpus {corpus} not specified."
                        f" Referencing to default value {dictionary}")
        if force_init is True and os.path.exists(dictionary):
            logger.info(f"Removing existing dictionary file {dictionary}")
            os.remove(dictionary)
        if not path.exists(dictionary):
            if force_init is False:
                logger.info(f"Specified dictionary file {dictionary} for corpus {corpus} doesn't exist:"
                            f" generating a new one")
            logger.info(f"Starting dictionary generation")
            create_dictionary(corpus, dictionary=dictionary, save_dictionary=True)
            logger.info(f"Dictionary {dictionary} generated")
        else:
            logger.info(f"Specified dictionary file {dictionary} for corpus {corpus} found")

        # Initialize the attributes
        self.embeddings = MagnitudeWordEmbedding(embedding, k, stream, lazy_loading)
        self.corpus = corpus
        self.index = index
        self.dictionary = dictionary

    def relation_expansion(self,
                           seed: [(str,)],
                           k: int = 15,
                           algo: str = None,
                           options: dict = None
                           ) -> [(str,)]:
        """
        Finds pairs similar to the pairs in the seed.
        :param seed: list of words pairs that has to be expanded
        :param k: limit to the number of results
        :param algo: name of the chosen algorithm
        :param options: task options: check the chosen algorithm interface to check its parameters
        """
        # Default algorithm and options
        if algo is None or options is None:
            algo = "intersection"
            options = {"relation_labeling_algo": "intersection",
                       "relation_labeling_options": {"dictionary": self.dictionary, "index": self.index},
                       "relation_labeling_k": 15,
                       "set_expansion_algo": "centroid",
                       "set_expansion_options": {},
                       "set_expansion_k": 15
                       }

        # Import the algorithm
        package = algorithms_package + ".relation_expansion." + algo
        algorithm = getattr(__import__(package, fromlist=[algo]), algo)

        # Launch the algorithm and return the result
        return algorithm.task(seed, k, options)

    def relation_labeling(self,
                          seed: [(str, str)],
                          k: int = 15,
                          algo: str = None,
                          options: dict = None
                          ) -> [str]:
        """
        Finds labels describing the relation between the pairs of words in the seed.
        :param seed: list of words pairs that has to be expanded
        :param k: limit to the number of results
        :param algo: name of the chosen algorithm
        :param options: task options: check the chosen algorithm interface to check its parameters
        """
        # Default algorithm and options
        if algo is None or options is None:
            algo = "intersection"
            options = {"dictionary": self.dictionary, "index": self.index}

        # Import the algorithm
        package = algorithms_package + ".relation_labeling." + algo
        algorithm = getattr(__import__(package, fromlist=[algo]), algo)

        # Launch the algorithm and return the result
        return algorithm.task(seed, k, options)

    def set_expansion(self,
                      seed: [str],
                      k: int = 15,
                      algo: str = None,
                      options: dict = None
                      ) -> [str]:
        """
        Finds words similar to the words in the seed (co-hyponyms).
        :param seed: list of words pairs that has to be expanded
        :param k: limit to the number of results
        :param algo: name of the chosen algorithm
        :param options: task options: check the chosen algorithm interface to check its parameters
        """
        # Default algorithm and options
        if algo is None or options is None:
            algo = "centroid"
            options = {"embeddings": self.embeddings}

        # Import the algorithm
        # QSTN: Use itertools to convert the results to a finite list (see https://stackoverflow.com/a/5234170)
        package = algorithms_package + ".set_expansion." + algo
        algorithm = getattr(__import__(package, fromlist=[algo]), algo)

        # Launch the algorithm and return the result
        return algorithm.task(seed, k, options)

    def set_labeling(self,
                     seed: [str],
                     k: int = 15,
                     algo: str = None,
                     options: dict = None
                     ) -> [str]:
        """
        Finds labels describing the relation between the words in the seed (hyperonym).
        :param seed: list of words pairs that has to be expanded
        :param k: limit to the number of results
        :param algo: name of the chosen algorithm
        :param options: task options: check the chosen algorithm interface to check its parameters
        """
        # Default algorithm and options
        if algo is None or options is None:
            algo = "intersection"
            options = {"dictionary": self.dictionary, "index": self.index}

        # Import the algorithm
        package = algorithms_package + ".set_labeling." + algo
        algorithm = getattr(__import__(package, fromlist=[algo]), algo)

        # Launch the algorithm and return the result
        return algorithm.task(seed, k, options)
