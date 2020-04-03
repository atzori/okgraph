from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phraser, Phrases
from pymagnitude import converter
import logging
from logging.config import fileConfig
from os import path

LOG_CONFIG_FILE = path.dirname(path.realpath(__file__)) + '/logging.ini'
if not path.isfile(LOG_CONFIG_FILE):
    LOG_CONFIG_FILE = path.dirname(path.dirname(LOG_CONFIG_FILE)) + '/logging.ini'

fileConfig(LOG_CONFIG_FILE)
logger = logging.getLogger()
logger.info(f'Log config file is: {LOG_CONFIG_FILE}')

# from pymagnitude converter
DEFAULT_PRECISION = 7
DEFAULT_NGRAM_BEG = 3
DEFAULT_NGRAM_END = 6


class FileConverter:

    def corpus_to_gensim_model(corpus_fname: str, save_fname: str = None) -> str:
        """
        Reads text file and creates a gensim model file.
        :param save_fname: name (path) of the model
        :return: the name of the model
        """

        model = Word2Vec()

        logger.info('Computing phrases')
        phrases = Phrases(LineSentence(corpus_fname))

        logger.info('Generating bigram')
        bigram = Phraser(phrases)

        logger.info('Building vocabulary')
        model.build_vocab(bigram[LineSentence(corpus_fname)])

        logger.info('Training model with total_examples=%d and epochs=%d' % (model.corpus_count, model.epochs))
        model.train(bigram[LineSentence(corpus_fname)], total_examples=model.corpus_count, epochs=model.epochs)

        if save_fname is None:
            save_fname = corpus_fname + '.bin'

        logger.info('Saving... [' + save_fname + ']')
        model.wv.save_word2vec_format(save_fname, binary=True)
        logger.info('Saved [' + save_fname + '].')

        return save_fname

    def corpus_to_magnitude_model(corpus_fname: str, save_fname: str = None) -> str:
        """
        Reads a text file and creates a Magnitude model file
        :param save_fname: name (path) of the model
        :return: the name of the model
        """

        logger.info(f'Computing file {corpus_fname}')

        if save_fname is None:
            save_fname = corpus_fname + '.magnitude'

        gensim_model_fname = corpus_fname + '.bin'

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

        logger.info('Saved [' + save_fname + '].')

        return save_fname