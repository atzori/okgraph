import os
import argparse

from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phraser, Phrases


def log(msg):
    print('OKGRAPH: ' + msg)


def convert(read_fname: str, save_fname: str):
    """Read text file and create a model file."""

    model = Word2Vec()

    log('Computing phrases')
    phrases = Phrases(LineSentence(read_fname))
    log('Generating phraser')
    bigram = Phraser(phrases)

    log('Building vocabulary')
    model.build_vocab(bigram[LineSentence(read_fname)])
    log('Training model with total_examples=%d and epochs=%d' % (model.corpus_count, model.iter))
    model.train(bigram[LineSentence(read_fname)], total_examples=model.corpus_count, epochs=model.iter)

    if not save_fname:
        save_fname = read_fname + '.bin'

    log('Saving... [' + save_fname + ']')
    model.wv.save_word2vec_format(save_fname, binary=True)
    log('Saved [' + save_fname + '].')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        help="input file - corpus. A simple plain .txt or any compressed Gensim supported.",
        type=str,
        required=True)

    parser.add_argument(
        "-o",
        "--output",
        help="output file. If none, the input will be used appending .bin at the end.",
        type=str)

    args = parser.parse_args()
    input_file_path = os.path.expanduser(args.input)

    if not args.output:
        output_file_path = None
    else:
        output_file_path = os.path.expanduser(args.output)

    convert(input_file_path, output_file_path)
