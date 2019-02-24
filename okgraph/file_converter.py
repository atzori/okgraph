from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phraser, Phrases
from pymagnitude import converter


# from pymagnitude converter
DEFAULT_PRECISION = 7
DEFAULT_NGRAM_BEG = 3
DEFAULT_NGRAM_END = 6


def log(msg):
    print('OKgraph: ' + msg)


class FileConverter:

    def corpus_to_gensim_model(corpus_fname: str, save_fname: str = None) -> str:
        """Read text file and create a gensim model file."""

        model = Word2Vec()

        log('Computing phrases')
        phrases = Phrases(LineSentence(corpus_fname))

        log('Generating bigram')
        bigram = Phraser(phrases)

        log('Building vocabulary')
        model.build_vocab(bigram[LineSentence(corpus_fname)])

        log('Training model with total_examples=%d and epochs=%d' % (model.corpus_count, model.epochs))
        model.train(bigram[LineSentence(corpus_fname)], total_examples=model.corpus_count, epochs=model.epochs)

        if save_fname is None:
            save_fname = corpus_fname + '.bin'

        log('Saving... [' + save_fname + ']')
        model.wv.save_word2vec_format(save_fname, binary=True)
        log('Saved [' + save_fname + '].')

        return save_fname

    def corpus_to_magnitude_model(corpus_fname: str, save_fname: str = None) -> str:
        """Read a text file and create a Magnitude model file."""

        log(f'Computing file {corpus_fname}')

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

        log('Saved [' + save_fname + '].')

        return save_fname
