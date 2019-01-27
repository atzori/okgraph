import time, logging, numpy
from enum import Enum
from pymagnitude import Magnitude


class ALGORITHM(Enum):
    TOP5MEAN = 0


class OKgraph:
    """OKgraph currently focuses on the following tasks:

    - set expansion given one or a short set of words, continues this set with a list of other "same-type" words (co-hyponyms)
    - relation expansion given one or a short set of word pairs, continues this set with a list of pairs having the same implicit relation of the given pairs
    - set labeling given one or a short set of words, returns a list of short strings (labels) describing the given set (its type or hyperonym)
    - relation labeling given one or a short set of word pairs, returns a list of short strings (labels) describing the relation in the given set

    """

    def __init__(self, corpus: str = None, embeddings: str = None):
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

        # vecs = Magnitude('http://magnitude.plasticity.ai/word2vec/heavy/GoogleNews-vectors-negative300.magnitude',
        # stream=True)

        self.magnitude = Magnitude(embeddings)

    def set_expansion(self, seed: [str] = None, algo: str = ALGORITHM.TOP5MEAN, options: dict = None, k: int = 5):
        """Returns a **generator** with results not containing the given seed
        Use itertools to convert to a finite list (see https://stackoverflow.com/a/5234170)
        e.g.: 'Spain','Portugal','Belgium', ...

        """
        keys = []
        for key, vector in self.magnitude:
            if vector in self.magnitude.query(seed):
                keys.append(key)
                print(key)
        return keys
        # return list(filter(lambda key, vector: vector in self.magnitude.query(seed), self.magnitude))




