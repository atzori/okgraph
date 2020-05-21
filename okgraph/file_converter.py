from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phraser, Phrases
from pymagnitude import converter
from okgraph.logger import logger

module_path = str.upper(__name__).replace("OKGRAPH.", "")

# Default parameters from pymagnitude converter
DEFAULT_PRECISION = 7
DEFAULT_NGRAM_BEG = 3
DEFAULT_NGRAM_END = 6


class FileConverter:

    def corpus_to_gensim_model(corpus_fname: str, save_fname: str = None) -> str:
        """
        Reads text file and creates a gensim model file.
        :param corpus_fname: name (path) of the corpus
        :param save_fname: name (path) of the model
        :return: the model name
        """

        model = Word2Vec()

        logger.info(f"{module_path}: Computing phrases")
        phrases = Phrases(LineSentence(corpus_fname))

        logger.info(f"{module_path}: Generating bigram")
        bigram = Phraser(phrases)

        logger.info(f"{module_path}: Building vocabulary")
        model.build_vocab(bigram[LineSentence(corpus_fname)])

        logger.info(f"{module_path}: Training model with total_examples={model.corpus_count} and epochs={model.epochs}")
        model.train(bigram[LineSentence(corpus_fname)], total_examples=model.corpus_count, epochs=model.epochs)

        if save_fname is None:
            save_fname = corpus_fname + ".bin"
            logger.info(f"{module_path}: Save file name not specified. Using default save file named {save_fname}")

        logger.info(f"{module_path}: Saving... {save_fname}")
        model.wv.save_word2vec_format(save_fname, binary=True)
        logger.info(f"{module_path}: Saved {save_fname}")

        return save_fname

    def corpus_to_magnitude_model(corpus_fname: str, save_fname: str = None) -> str:
        """
        Reads a text file and creates a Magnitude model file.
        :param corpus_fname: name (path) of the corpus
        :param save_fname: name (path) of the model
        :return: the model name
        """

        logger.info(f"{module_path}: Computing file {corpus_fname}")

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

        logger.info(f"{module_path}: Saved {save_fname}")

        return save_fname