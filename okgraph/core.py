"""The core module contains the library main functionalities to performs
unsupervised natural-language understanding.
"""
from okgraph.embeddings import FileConverter, MagnitudeWordEmbeddings
from okgraph.indexing import DEFAULT_INDEX_FOLDER, Indexing
from okgraph.utils import check_extension, generate_dictionary, logger
from os import path, remove
from shutil import rmtree as remove_dir
from typing import Dict, List, Tuple

ALGORITHMS_PACKAGE = "okgraph.task"
"""str: package containing the task implementations."""

DEFAULT_DICTIONARY_NAME: str = "dictTotal.npy"
"""str: default name for the corpus dictionary file."""


class OKgraph:
    """A class used to extract knowledge from unstructured text corpus.

    This class currently focuses on the following tasks:
        - **set expansion**: given one or a short set of words, continues this
          set with a list of other 'same-type' words (`co-hyponyms
          <https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy#Co-hyponyms>`_);
        - **relation expansion**: given one or a short set of word pairs,
          continues this set with a list of pairs having the same implicit
          relation of the given pairs;
        - **set labeling**: given one or a short set of words, returns a list
          of short strings (labels) describing the given set (its type or
          `hyperonym <https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy>`_);
        - **relation labeling**: given one or a short set of word pairs,
          returns a list of short strings (labels) describing the relation in
          the given set;
    All the tasks work with plain text corpus (untagged).

    Attributes:
        corpus (str): path of the corpus file.
        embeddings (MagnitudeWordEmbeddings): words embeddings (vector
            model).
        index (str): path of the indexed corpus files.
        dictionary (str): path of the corpus dictionary.

    """

    corpus: str
    embeddings: MagnitudeWordEmbeddings
    index: str
    dictionary: str

    def __init__(self,
                 corpus_file: str,
                 embeddings_file: str = None,
                 k: int = 5,
                 stream: bool = False,
                 lazy_loading: int = 0,
                 index_dir: str = None,
                 dictionary_file: str = None,
                 force_init: bool = False,
                 ):
        """The constructor creates a OKgraph object.

        Loads or generates the word embeddings, corpus index and corpus
        dictionary, whether or not they already exist.

        Args:
            corpus_file (str): path of the corpus file. The corpus file should
                be a cleaned free text file (untagged).
            embeddings_file (MagnitudeWordEmbeddings): path or URL of the
                embeddings file.
                The supported embeddings are *.magnitude*, *.bin*, *.txt*,
                *.vec* and *.hdf5*. The *.bin*, *.txt*, *.vec* and *.hdf5*
                embeddings are supported by automatically converting them to a
                *.magnitude* file. No extension will be treated as *.magnitude*.
                If the embeddings file is not specified, the default path
                points to a *.magnitude* file with the same basename of the
                corpus file.
                If the specified file is found in the path, the embeddings is
                directly loaded.
                If the specified *.magnitude* file is not found, an embeddings
                Magnitude model is created from the text corpus.
                If the specified *.bin*, *.txt*, *.vec* or *.hdf5* file is not
                found a ValueError is raised, because these models can be
                loaded but not created.
            k (int): embeddings can be queried to know the words whose vector
                representation is similar to a known vector. The OKgraph tasks
                could be using this functionality to extract their results:
                usually, the more results you are expecting from the task, the
                more words you are expecting from the embeddings.
                *k* should be set to the approximate upper-bound of the number
                of results that will be expected from the embeddings. If the
                exact number is unknown, be conservative picking a large number,
                while keeping in mind the bigger *k* is, the more memory it
                will consume.
            stream (bool): stream the URL instead of downloading it.
            lazy_loading (int): allows faster cold start when loading:

                 +----+-------------------------------------------+
                 | -1 | preload into memory                       |
                 +----+-------------------------------------------+
                 |  0 | lazy loads with unbounded in-memory cache |
                 +----+-------------------------------------------+
                 | >0 | lazy loads with an LRU cache of that size |
                 +----+-------------------------------------------+
            index_dir (str): path of the indexed corpus files directory.
                If the index directory is not specified, the default path
                points to a directory named *indexdir* with the same parent
                directory of the corpus.
                If the directory exists it is used as it is, otherwise it is
                created.
            dictionary_file (str): path of the corpus dictionary file.
                The supported files are *.npy*. No extension will be treated as
                *.npy*.
                If the dictionary file is not specified, the default path
                points to a file named *dictTotal.npy* with the same parent
                directory of the corpus.
                If the file exists it is used as it is, otherwise it is created.
            force_init (bool): forces the initialization of the embeddings,
                index and dictionary from scratch, overwriting them if already
                existing.

        Example:
            - Instantiating an OKgraph object specifing a corpus file:
                >>> from okgraph.core import OKgraph
                >>> corpus = "/OKgraph/Example/enwik9.txt"
                >>> okg = OKgraph(corpus)
              or even
                >>> okg = OKgraph(corpus_file=corpus)
              This code will create an OKgraph object using the *enwiki9.txt*
              corpus.
              No path have been specified for the embeddings, index directory
              and dictionary, so the default paths (same of the corpus)
              '/OKgraph/Example/enwik9.magnitude',
              '/OKgraph/Example/indexdir/', '/OKgraph/Example/dictTotal.npy'
              are implicitly used.
              All the resources will be searched in the default paths: if
              found they will be used, otherwise the missing one will be
              created (using the corpus).


            - Instantiating an OKgraph object specifing the resources paths:
                >>> from okgraph.core import OKgraph
                >>> corpus = "/OKgraph/Example/enwik9.txt"
                >>> embeddings = "/OKgraph/Example/NewDir/enwiki9_embeddings"
                >>> index = "/OKgraph/Example/NewDir/enwiki9_index"
                >>> dictionary = "/OKgraph/Example/NewDir/enwiki9_dictionary"
                >>> okg = OKgraph(corpus_file=corpus,
                >>>               embeddings_file=embeddings,
                >>>               index_dir=index,
                >>>               dictionary_file=dictionary)
              This code will create an OKgraph object using the *enwiki9.txt*
              corpus.
              No extension has been specified for the embeddings and the
              dictionary, so the *.magnitude* and *.npy* exntensions will be
              automatically appended.
              All the resources will be searched in the specified paths: if
              found they will be used, otherwise the missing one will be
              created (using the corpus). In the above example, the resources
              are missing and they will be created in the new directory
              *NewDir*. *NewDir* will be automatically created.


        """
        corpus_file = path.normpath(corpus_file)
        if embeddings_file is not None:
            embeddings_file = path.normpath(embeddings_file)
        if index_dir is not None:
            index_dir = path.normpath(index_dir)
        if dictionary_file is not None:
            dictionary_file = path.normpath(dictionary_file)

        embeddings_file = self._get_embeddings(
            corpus_file, embeddings_file, force_init)
        index_dir = self._get_index(
            corpus_file, index_dir, force_init)
        dictionary_file = self._get_dictionary(
            corpus_file, dictionary_file, force_init)

        self.embeddings = MagnitudeWordEmbeddings(
            embeddings_file, k, stream, lazy_loading)
        self.corpus = corpus_file
        self.index = index_dir
        self.dictionary = dictionary_file

    @staticmethod
    def _get_embeddings(corpus_file: str,
                        embeddings_file: str,
                        force_init: bool) -> str:
        """Loads or generates the embeddings whether or not it is already
        existing.

        Args:
            corpus_file (str): path of the corpus file.
            embeddings_file (str): path of the embeddings file.
            force_init (bool): if True forces the creation of the embeddings.

        Returns:
            str: the path of the loaded/generated Magnitude model.

        """
        # If no name has been given, assign a default name
        if embeddings_file is None:
            (corpus_basename, _) = path.splitext(corpus_file)
            embeddings_file = corpus_basename + ".magnitude"
            logger.info(
                f"Embeddings file for corpus {corpus_file} not specified."
                f" Referencing to default value {embeddings_file}")
        # Otherwise, check if it is a valid name file
        else:
            embeddings_file = check_extension(
                file_name=embeddings_file,
                default_extension=".magnitude",
                allowed_extensions=[".magnitude", ".txt",
                                    ".bin", ".vec", ".hdf5"]
            )

        # Split the model name in basename and extension
        (embeddings_basename, embeddings_extension) = \
            path.splitext(embeddings_file)

        magnitude_file = embeddings_basename + ".magnitude"

        # If the model exists but force_init is True, remove it
        if path.exists(embeddings_file) and force_init is True:
            logger.info(
                f"Removing existing embeddings file {embeddings_file}"
                f" to generate it again")
            remove(embeddings_file)

        # If the embeddings name is of a Magnitude model
        if embeddings_extension == ".magnitude":
            # If the model already exists, use it
            if path.exists(embeddings_file):
                logger.info(
                    f"Specified embeddings file {embeddings_file} for corpus"
                    f" {corpus_file} found: using it")
            # If no model exists, generate a new one
            else:
                if force_init is False:
                    logger.info(
                        f"Specified embeddings file {embeddings_file} for corpus"
                        f" {corpus_file} doesn't exist: generating a new one")
                embeddings_file = FileConverter.corpus_to_magnitude_model(
                    corpus_file, embeddings_file)
        # If the embeddings name is of another possible kind of embeddings
        elif embeddings_extension in [".txt", ".bin", ".vec", ".hdf5"]:
            # If the Magnitude file exists use it
            if path.exists(magnitude_file):
                logger.info(
                    f"The Magnitude model {magnitude_file} related to the"
                    f" specified {embeddings_file} embeddings already exists:"
                    f" using it")
                embeddings_file = magnitude_file
            # If the embeddings file exists, try to convert it
            elif path.exists(embeddings_file):
                logger.info(
                    f"Generating Magnitude model from specified"
                    f" {embeddings_file} embeddings")
                FileConverter.generic_model_to_magnitude_model(
                    embeddings_file, magnitude_file)
                embeddings_file = magnitude_file
            # Otherwise, an error occurred
            else:
                raise ValueError(
                    f"Specified embeddings file {embeddings_file} does not"
                    f" exists: it cannot be loaded and it cannot be generated"
                    f" (its not a Magnitude model)")

        # Return the path of the embeddings file
        return embeddings_file

    @staticmethod
    def _get_index(corpus_file: str,
                   index_dir: str,
                   force_init: bool) -> str:
        """Loads or generates the index whether or not it is already existing.

        Args:
            corpus_file: path of the corpus file.
            index_dir: path of the index directory.
            force_init: if True forces the creation of the index.

        Returns:
            str: the path of the loaded/generated index directory.
        """
        # If no name has been given, assign a default name
        if index_dir is None:
            index_dir = path.join(path.dirname(corpus_file),
                                  DEFAULT_INDEX_FOLDER)
            logger.info(
                f"Indexing directoy for corpus {corpus_file} not specified."
                f" Referencing to default value {index_dir}")

        # If the index exists but force_init is True, remove it
        if path.exists(index_dir) and force_init is True:
            logger.info(
                f"Removing existing indexing directory {index_dir}"
                f" to generate it again")
            remove_dir(index_dir)
        # If the index already exists, use it
        if path.exists(index_dir):
            logger.info(
                f"Specified index directory {index_dir} for corpus"
                f" {corpus_file} found: using it")
        # If no index exists, generate a new one
        else:
            if force_init is False:
                logger.info(
                    f"Specified indexing directory {index_dir} for corpus"
                    f" {corpus_file} doesn't exist: generating a new one")
            ix = Indexing(corpus_path=corpus_file)
            ix.indexing(index_path=index_dir)
            del ix

        # Return the path of the index directory
        return index_dir

    @staticmethod
    def _get_dictionary(corpus_file: str,
                        dictionary_file: str,
                        force_init: bool) -> str:
        """Loads or generates the dictionary whether or not it is already
        existing.

        Args:
            corpus_file: path of the corpus file.
            dictionary_file: path of the dictionary file.
            force_init: if True forces the creation of the index.

        Returns:
            str: the path of the loaded/generated dictionary file.

        """
        # If no name has been given, assign a default name
        if dictionary_file is None:
            dictionary_file = path.join(path.dirname(corpus_file),
                                        DEFAULT_DICTIONARY_NAME)
            logger.info(
                f"Dictionary file for corpus {corpus_file} not specified."
                f" Referencing to default value {dictionary_file}")
        # Otherwise, check if it is a valid name file
        else:
            dictionary_file = check_extension(
                file_name=dictionary_file,
                default_extension=".npy",
                allowed_extensions=[".npy"]
            )

        # If the dictionary exists but force_init is True, remove it
        if path.exists(dictionary_file) and force_init is True:
            logger.info(
                f"Removing existing dictionary file {dictionary_file}"
                f" to generate it again")
            remove(dictionary_file)
        # If the dictionary already exists, use it
        if path.exists(dictionary_file):
            logger.info(
                f"Specified dictionary file {dictionary_file} for corpus"
                f" {corpus_file} found: using it")
        # If no dictionary exists, generate a new one
        else:
            if force_init is False:
                logger.info(
                    f"Specified dictionary file {dictionary_file} for corpus"
                    f" {corpus_file} doesn't exist: generating a new one")
            generate_dictionary(corpus_file, dictionary=dictionary_file,
                                save_dictionary=True)

        # Return the path of the dictionary file
        return dictionary_file

    def relation_expansion(self,
                           seed: List[Tuple[str, ...]],
                           k: int = 15,
                           algo: str = None,
                           options: Dict = None
                           ) -> List[Tuple[str, ...]]:
        """Finds tuples with the same implicit relation of the seed tuples.

        Args:
            seed (List[Tuple[str, ...]]): list of tuples that has to be
                expanded.
            k (int): limit to the number of result tuples.
            algo (str): name of the chosen algorithm. The algorithm should be
                found in *okgraph.task.relation_expansion*.
            options (Dict): dictionary containing the keyword arguments for the
                chosen algorithm.

        Returns:
            List[Tuple[str, ...]]: the list of tuples related to the seed
                tuples.

        """
        # Default algorithm and options
        if algo is None or options is None:
            algo = "intersection"
            options = {
                "relation_labeling_algo": "intersection",
                "relation_labeling_options": {"dictionary": self.dictionary,
                                              "index": self.index},
                "relation_labeling_k": 15,
                "set_expansion_algo": "centroid",
                "set_expansion_options": {"embeddings": self.embeddings},
                "set_expansion_k": 15
            }

        # Import the algorithm
        package = ALGORITHMS_PACKAGE + ".relation_expansion." + algo
        algo = getattr(__import__(package, fromlist=[algo]), algo)

        # Launch the algorithm and return the result
        return algo.task(seed, k, **options)

    def relation_labeling(self,
                          seed: List[Tuple[str, ...]],
                          k: int = 15,
                          algo: str = None,
                          options: Dict = None
                          ) -> List[str]:
        """Finds labels describing the implicit relation between the seed
        tuples.

        Args:
            seed (List[Tuple[str, ...]]): list of tuples that has to be labeled.
            k (int): limit to the number of result labels.
            algo (str): name of the chosen algorithm. The algorithm should be
                found in *okgraph.task.relation_labeling*.
            options (Dict): dictionary containing the keyword arguments for the
                chosen algorithm.

        Returns:
            List[str]: the list of labels related to the seed tuples.
            
        """
        # Default algorithm and options
        if algo is None or options is None:
            algo = "intersection"
            options = {
                "dictionary": self.dictionary,
                "index": self.index
            }

        # Import the algorithm
        package = ALGORITHMS_PACKAGE + ".relation_labeling." + algo
        algorithm = getattr(__import__(package, fromlist=[algo]), algo)

        # Launch the algorithm and return the result
        return algorithm.task(seed, k, **options)

    def set_expansion(self,
                      seed: List[str],
                      k: int = 15,
                      algo: str = None,
                      options: Dict = None
                      ) -> List[str]:
        """Finds words with the same implicit relation of the seed words 
        (co-hyponyms).

        Args:
            seed (List[str]): list of words that has to be expanded.
            k (int): limit to the number of result words.
            algo (str): name of the chosen algorithm. The algorithm should be
                found in *okgraph.task.set_expansion*.
            options (Dict): dictionary containing the keyword arguments for the
                chosen algorithm.

        Returns:
            List[str]: the list of words related to the seed words.

        """
        # Default algorithm and options
        if algo is None or options is None:
            algo = "centroid"
            options = {
                "embeddings": self.embeddings
            }

        # Import the algorithm
        package = ALGORITHMS_PACKAGE + ".set_expansion." + algo
        algorithm = getattr(__import__(package, fromlist=[algo]), algo)

        # Launch the algorithm and return the result
        return algorithm.task(seed, k, **options)

    def set_labeling(self,
                     seed: List[str],
                     k: int = 15,
                     algo: str = None,
                     options: dict = None
                     ) -> List[str]:
        """Finds labels describing the implicit relation between the seed
        words (hyperonym).

        Args:
            seed (List[str]): list of words that has to be labeled.
            k (int): limit to the number of result labels.
            algo (str): name of the chosen algorithm. The algorithm should be
                found in *okgraph.task.set_labeling*.
            options (Dict): dictionary containing the keyword arguments for the
                chosen algorithm.

        Returns:
            List[str]: the list of labels related to the seed words.
        
        """
        # Default algorithm and options
        if algo is None or options is None:
            algo = "intersection"
            options = {
                "dictionary": self.dictionary,
                "index": self.index
            }

        # Import the algorithm
        package = ALGORITHMS_PACKAGE + ".set_labeling." + algo
        algorithm = getattr(__import__(package, fromlist=[algo]), algo)

        # Launch the algorithm and return the result
        return algorithm.task(seed, k, **options)
