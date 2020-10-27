"""The 'embeddings' module contains the utilities to work with word embeddings.
"""
from abc import ABC, abstractmethod
from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.models.phrases import Phraser, Phrases
import numpy as np
from numpy import ndarray
from okgraph.utils import logger
from os import makedirs, path
from pymagnitude import converter, Magnitude
from typing import List


class WordEmbeddings(ABC):
    """An abstract class representing `word embeddings
    <https://en.wikipedia.org/wiki/Word_embedding>`_ and their usual operations.
    """

    @abstractmethod
    def w2v(self, w: str) -> ndarray:
        """Given a word, finds its vector representation.

        Args:
            w (str): word in the embeddings model.

        Returns:
            ndarray: the vector representation of the word.

        Raises:
            NotExistingWordException: if a word doesn't exist in the
                embeddings and no vector can be related to it.

        """
        pass

    @abstractmethod
    def v2w(self, v: ndarray, n: int = 1) -> List[str]:
        """Given a vector, finds the closest word/words.

        Args:
            v (ndarray): a potential vector representation of a word.
            n (int): maximum number of results.

        Returns:
            List[str]: a list of word/words whose vector representation is close
                to the given vector.

        """
        pass

    def w2w(self, w: str, n: int = 1) -> List[str]:
        """Given a word, finds the closest word/words.

        Args:
            w (str): word in the embeddings model.
            n (int): maximum number of results.

        Returns:
            List[str]: a list of word/words whose vector representation is close
                to the vector representation of the given word.

        Raises:
            NotExistingWordException: if a word doesn't exist in the
                embeddings and no vector can be related to it.

        """
        return self.v2w(self.w2v(w), n)

    def v2v(self, v: ndarray, n: int = 1) -> List[ndarray]:
        """Given a vector, finds the closest vector/vectors.

        Args:
            v (ndarray): a potential vector representation of a word.
            n (int): maximum number of results.

        Returns:
            List[ndarray]: a list of vector/vectors that are close to the given
                vector.

        """
        return list(map(self.w2v, self.v2w(v, n)))

    @abstractmethod
    def exists(self, w: str) -> bool:
        """Checks if a word exists in the embeddings model.

        Args:
            w (str): a potential word in the embeddings model.

        Returns:
            bool: True if a vector exists for the given word, False otherwise.

        """
        pass

    def get4thv(self, v1: ndarray, v2: ndarray,
                v3: ndarray, n: int = 1) -> List[str]:
        """Computes the fourth element that completes the analogy 'v1 : v2 =
        v3 : ?4th?'. Input vectors must be provided in the correct order to
        compute the analogy correctly.

        Args:
            v1 (ndarray): first vector.
            v2 (ndarray): second vector.
            v3 (ndarray): third vector.
            n (int): maximum number of results.

        Returns:
            List[str]: a list of word/words whose vector representation is close
                to the vector that completes the analogy.

        Example:
            An example of analogy that the method completes could be:
                >>> e : WordEmbeddings
                >>> ...
                >>> v1 = e.w2v("man")
                >>> v2 = e.w2v("king")
                >>> v3 = e.w2v("woman")
                >>> w4 = e.get4thv(v1, v2, v3)
                ['queen']
            expecting *w4* to be something such *['queen']* having *n=1*. With
            *n>1* other possible solutions would be added to the list of
            results.

        """
        v = v2 - v1 + v3
        return self.v2w(v, n=n)

    def get4th(self, w1: str, w2: str,
               w3: str, n: int = 1) -> List[str]:
        """Computes the fourth element that completes the analogy 'w1 : w2 =
        w3 : ?4th?'. Input words must be provided in the correct order to
        compute the analogy correctly.

        Args:
            w1 (str): first word.
            w2 (str): second word.
            w3 (str): third word.
            n (int): maximum number of results.

        Returns:
            List[str]: a list of word/words that complete the analogy.

        Raises:
            NotExistingWordException: if a word doesn't exist in the
                embeddings and no vector can be related to it.

        Example:
            An example of analogy that the method completes could be:
                >>> e : WordEmbeddings
                >>> ...
                >>> w1 = "man"
                >>> w2 = "king"
                >>> w3 = "woman"
                >>> w4 = e.get4th(w1, w2, w3)
                ['queen']
            expecting *w4* to be something such *['queen']* having *n=1*. With
            *n>1* other possible solutions would be added to the list of
            results.

        """
        return self.get4thv(self.w2v(w1), self.w2v(w2), self.w2v(w3), n=n)

    @staticmethod
    def centroidv(vs: List[ndarray]) -> ndarray:
        """Computes the average vector from the given vectors.

        Args:
            vs (List[ndarray]): list of numerical vectors.

        Returns:
            ndarray: the average vector, or centroid.

        """
        return sum(vs) / len(vs)

    def centroid(self, ws: List[str]) -> ndarray:
        """Computes the average vector from the vector representation of the
            given words.

        Args:
            ws (List[str]): list of words.

        Returns:
            ndarray: the average vector, or centroid.

        Raises:
            NotExistingWordException: if a word doesn't exist in the
                embeddings and no vector can be related to it.

        """
        return self.centroidv(list(map(self.w2v, ws)))

    @staticmethod
    def cosv(v1: ndarray, v2: ndarray) -> float:
        """Computes the cosine of the angle between two vectors.

        Args:
            v1 (ndarray): first vector.
            v2 (ndarray): second vector.

        Returns:
            float: the cosine of the angle between the two input vectors.

        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        dot_product = np.dot(v1, v2)
        return round(float(dot_product / (norm1 * norm2)), 4)

    def cos(self, w1: str, w2: str) -> float:
        """Computes the cosine of the angle between the vector representations
        of two words.

        Args:
            w1 (str): first word.
            w2 (str): second word.

        Returns:
            float: the cosine of the angle between the vector representations
                of the two input words.

        Raises:
            NotExistingWordException: if a word doesn't exist in the
                embeddings and no vector can be related to it.

        """
        return self.cosv(self.w2v(w1), self.w2v(w2))


class MagnitudeWordEmbeddings(WordEmbeddings):
    """A class used to represent word embeddings through the `Magnitude
    <https://github.com/plasticityai/magnitude/blob/master/README.md>`_ model.
    """
    model: Magnitude

    def __init__(self,
                 model_file: str,
                 k: int = 5,
                 stream: bool = False,
                 lazy_loading: int = 0,
                 ):
        self.model = Magnitude(model_file,
                               _number_of_values=k,
                               stream=stream,
                               lazy_loading=lazy_loading)

    def w2v(self, w: str) -> ndarray:
        """Given a word, finds its vector representation.

        Args:
            w (str): word in the embeddings model.

        Returns:
            ndarray: the vector representation of the word.

        Raises:
            NotExistingWordException: if a word doesn't exist in the
                embeddings and no vector can be related to it.

        """
        if self.model.__contains__(w):
            return self.model.query(w)
        else:
            raise NotExistingWordException(w)

    def v2w(self, v: ndarray, n: int = 1) -> List[str]:
        """Given a vector, finds the closest word/words.

        Args:
            v (ndarray): a potential vector representation of a word.
            n (int): maximum number of results.

        Returns:
            List[str]: a list of word/words whose vector representation is close
                to the given vector.

        """
        return list(map(lambda r: r[0], self.model.most_similar(v, topn=n)))

    def exists(self, w: str) -> bool:
        """Checks if a word exists in the embeddings model.

        Args:
            w (str): a potential word in the embeddings model.

        Returns:
            bool: True if a vector exists for the given word, False otherwise.

        """
        try:
            self.w2v(w)
            return True
        except NotExistingWordException:
            return False


class FileConverter:
    """A class used to convert text corpus and embeddings to Magnitude models.
    Text corpus should be plain text without any kind of formatting.
    """

    @staticmethod
    def _corpus_to_gensim_model(corpus_file: str,
                                model_file: str) -> str:
        """Analyzes the corpus and convert it to Gensim embeddings using
        the Word2Vec implementation.

        Args:
            corpus_file (str): path of the corpus text file.
            model_file (str): save path for the Gensim model.

        Returns:
            str: the save path of the Gensim model.
        """
        logger.info(f"Gensim: generating model {model_file}"
                    f" from {corpus_file}")

        model = Word2Vec()

        logger.info(f"Gensim: computing corpus phrases")
        phrases = Phrases(LineSentence(corpus_file))

        logger.info(f"Gensim: generating bigram")
        bigram = Phraser(phrases)

        logger.info(f"Gensim: building vocabulary")
        model.build_vocab(bigram[LineSentence(corpus_file)])

        logger.info(
            f"Gensim: training model with"
            f" total_examples={model.corpus_count} and epochs={model.epochs}")
        model.train(bigram[LineSentence(corpus_file)],
                    total_examples=model.corpus_count,
                    epochs=model.epochs)

        logger.info(f"Gensim: saving... {model_file}")
        model.wv.save_word2vec_format(model_file, binary=True)
        logger.info(f"Gensim: saved {model_file}")

        logger.info(f"Gensim: model generated")
        return model_file

    @staticmethod
    def corpus_to_magnitude_model(corpus_file: str,
                                  model_file: str) -> str:
        """Analyzes the corpus and convert it to Magnitude embeddings using
        a base Word2Vec model.
        
        Args:
            corpus_file (str): path of the corpus text file.
            model_file (str): save path for the Magnitude model.

        Returns:
            str: the save path of the Magnitude model.

        """

        logger.info(f"Magnitude: generating model {model_file}"
                    f" from {corpus_file}")

        (model_basename, _) = path.splitext(model_file)
        gensim_model_file = model_basename + ".bin"

        parent_dir = path.dirname(model_file)
        if parent_dir:
            makedirs(parent_dir, exist_ok=True)
        
        FileConverter._corpus_to_gensim_model(corpus_file, gensim_model_file)
        
        logger.info(f"Magnitude: converting Gensim model {gensim_model_file}"
                    f" to Magnitude model {model_file}")
        FileConverter.generic_model_to_magnitude_model(gensim_model_file,
                                                       model_file)

        logger.info(f"Magnitude: model generated")
        return model_file

    @staticmethod
    def generic_model_to_magnitude_model(input_model: str,
                                         output_model: str) -> None:
        """Converts the embeddings in the .txt, .bin, .vec, or .hdf5 formats
        from GloVe, Gensim or ELMo models into a Magnitude model.

        Args:
            input_model: path of the input model.
            output_model: path of the output Magnitude model.

        Returns:
            None

        """
        converter.convert(input_model,
                          output_file_path=output_model)


class NotExistingWordException(Exception):
    """An exception used to represent the error that occur when a word is
    searched in the embeddings but it's not existing.

    """

    def __init__(self, w: str):
        self.word = w

    def __str__(self) -> str:
        return f"Word {self.word} does not exist in the embeddings"
