"""
TRYDOC: Creates embeddings
"""
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phraser, Phrases
from pymagnitude import converter
from pymagnitude import Magnitude
from okgraph.utils import logger
import numpy

# Default parameters from pymagnitude converter
DEFAULT_PRECISION = 7
DEFAULT_NGRAM_BEG = 3
DEFAULT_NGRAM_END = 6


class WordEmbedding():
    """Abstract class representing a word embedding and their basic operations """

    def w2v(self, w):
        """Given a word, returns its vector"""
        raise NotImplementedError()

    def v2w(self, v, n=1):
        """Given a vector, returns the most closest word (or a list of words if n>1)"""
        raise NotImplementedError()

    def w2w(self, w, n=1):
        """Given a word, returns the list of n closest words"""
        return self.v2w(self.w2v(w), n)

    def v2v(self, v, n=1):
        """Given a vector, returns the a list of n closest vectors"""
        return list(map(lambda x: self.w2v(x), self.v2w(v, n)))

    def exists(self, w):
        """Return True if there is a vector for the given word, False otherwise"""
        try:
            self.w2v(w)
            return True
        except:
            return False

    def get4thv(self, v1, v2, v3, n=1):
        """Compute the fourth element of analogies such as man : king = women : ?4th?
        Input vectors must be provided in the correct order, and returns a word or list of words if n>1"""
        v = v2 - v1 + v3
        return self.v2w(v, n=n)

    def get4th(self, w1, w2, w3, n=1):
        """Compute the fourth element of analogies such as man : king = women : ?4th?
        Input words must be provided in the correct order, and returns a word or list of words if n>1"""
        return self.get4thv(self.w2v(w1), self.w2v(w2), self.w2v(w3), n=n)

    def centroid(self, ws):
        def mean(lst):
            return sum(lst) / float(len(lst))

        vectors = list(map(self.w2v, ws))
        return mean(vectors)


class MagnitudeWordEmbedding(WordEmbedding):
    model: Magnitude

    def __init__(self,
                 model_name: str,
                 k: int = 5,
                 stream: bool = False,
                 lazy_loading: int = 0,
                 ):
        self.model = Magnitude(model_name,
                               _number_of_values=k,
                               stream=stream,
                               lazy_loading=lazy_loading)

    def w2v(self, w):
        """Given a word, returns its vector"""
        return self.model.query(w)

    def v2w(self, v, n=1):
        """Given a vector, returns the most closest word (or a list of words if n>1)"""
        return list(map(lambda r: r[0], self.model.most_similar(v, topn=n)))


class FileConverter:

    @staticmethod
    def corpus_to_gensim_model(corpus_fname: str, save_fname: str = None) -> str:
        """
        Reads text file and creates a gensim model file.
        :param corpus_fname: name (path) of the corpus
        :param save_fname: name (path) of the model
        :return: the model name
        """

        model = Word2Vec()

        logger.info(f"Computing phrases")
        phrases = Phrases(LineSentence(corpus_fname))

        logger.info(f"Generating bigram")
        bigram = Phraser(phrases)

        logger.info(f"Building vocabulary")
        model.build_vocab(bigram[LineSentence(corpus_fname)])

        logger.info(f"Training model with total_examples={model.corpus_count} and epochs={model.epochs}")
        model.train(bigram[LineSentence(corpus_fname)], total_examples=model.corpus_count, epochs=model.epochs)

        if save_fname is None:
            save_fname = corpus_fname + ".bin"
            logger.info(f"Save file name not specified. Using default save file named {save_fname}")

        logger.info(f"Saving... {save_fname}")
        model.wv.save_word2vec_format(save_fname, binary=True)
        logger.info(f"Saved {save_fname}")

        return save_fname

    @staticmethod
    def corpus_to_magnitude_model(corpus_fname: str, save_fname: str = None) -> str:
        """
        Reads a text file and creates a Magnitude model file.
        :param corpus_fname: name (path) of the corpus
        :param save_fname: name (path) of the model
        :return: the model name
        """

        logger.info(f"Computing file {corpus_fname}")

        if save_fname is None:
            save_fname = corpus_fname + ".magnitude"

        gensim_model_fname = corpus_fname + ".bin"

        FileConverter.corpus_to_gensim_model(corpus_fname, gensim_model_fname)

        converter.convert(gensim_model_fname,
                          output_file_path=save_fname,
                          precision=DEFAULT_PRECISION,
                          subword=False,
                          subword_start=DEFAULT_NGRAM_BEG,
                          subword_end=DEFAULT_NGRAM_END,
                          approx=False,
                          approx_trees=None,
                          vocab_path=None)

        logger.info(f"Saved {save_fname}")

        return save_fname