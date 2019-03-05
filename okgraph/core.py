from pymagnitude import Magnitude
from nltk.probability import FreqDist
from okgraph.file_converter import FileConverter
import os
import logging
from logging.config import fileConfig

cwd = os.getcwd()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
fileConfig(ROOT_DIR + '/logging.ini')
logger = logging.getLogger()

algorithms_package = "okgraph.task"


class OKgraph:
    """OKgraph currently focuses on the following tasks:

    - set expansion given one or a short set of words, continues this set with a list of other "same-type" words (co-hyponyms)
    - relation expansion given one or a short set of word pairs, continues this set with a list of pairs having the same implicit relation of the given pairs
    - set labeling given one or a short set of words, returns a list of short strings (labels) describing the given set (its type or hyperonym)
    - relation labeling given one or a short set of word pairs, returns a list of short strings (labels) describing the relation in the given set

    """

    def __init__(self, corpus: str = None,
                 embeddings: str = None,
                 k: int = 5,
                 stream: bool = False,
                 lazy_loading: int = 0):
        """Create an OKgraph.

        It loads a text corpus, and optionally a word-embedding model
        - via a local resource,
        - or streaming a file on a server.

        The supported embeddings files are: '.magnitude'.
        In the future will support .bin, .txt, .vec, .hdf5 by automatically converting them to a
        .magnitude file (by using the magnitude: https://github.com/plasticityai/magnitude#file-format-and-converter).

        HOW TO:

            import okgraph

            okg = okgraph.OKgraph('enwik9.txt', 'model_file')

            # or equivalently:
            okg = okgraph.OKgraph(corpus = 'file/path/enwik9.txt',
                                  embeddings = 'file/path/model_file.magnitude')

            # another alternative
            okg = okgraph.OKgraph(corpus = 'file/path/enwik9.txt',
                                  embeddings = 'https://server/file/path/model_file.magnitude')

        This example usage will set enwik9.txt as corpus file and model_file.magnitude as vector
        model file in magnitude format (extension magnitude is appended automatically if not specified).
        """
        if embeddings is None:
            logger.info(f'File model for {corpus} not found. Generating model with default options')
            embeddings = FileConverter.corpus_to_magnitude_model(corpus_fname=corpus)
            logger.info(f'Model {embeddings} generated.')

        self.v = Magnitude(embeddings, _number_of_values=k, stream=stream, lazy_loading=lazy_loading)
        self.f = FreqDist()
        self.corpus = corpus

    def set_expansion(self, seed: [str] = None, algo: str = "centroid", options: dict = {}, k: int = 5):
        """Returns a **generator** with results not containing the given seed
        Use itertools to convert to a finite list (see https://stackoverflow.com/a/5234170)
        e.g.: 'Spain','Portugal','Belgium', ...

        """

        # automatically get the file of the algorithm task
        package = algorithms_package + ".set_expansion." + algo
        algorithm = getattr(__import__(package, fromlist=[algo]), algo)

        options["seed"] = seed
        options["k"] = k

        return algorithm.task(self.v, options=options)




