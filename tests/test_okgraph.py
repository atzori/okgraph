import unittest
import okgraph
from okgraph.sliding_windows import SlidingWindows
import os
from pymagnitude import Magnitude

cwd = os.getcwd()
print( cwd )
corpus_file_path = cwd + '/tests/data/text8'
embeddings_file_path = cwd + '/tests/data/text8.magnitude'
indexing_folder = cwd + '/tests/data/indexdir/text8'
dict_total= cwd + '/tests/data/dictTotal.npy'
#embeddings_file_path = cwd + '/tests/data/GoogleNews-vectors-negative300.magnitude'


class OKGraphTest(unittest.TestCase):

    def test_01_init_with_textgz(self):
        """ create a magnitude file if not exists and not passed as argument"""
        magnitude_file = corpus_file_path + '.magnitude'
        if os.path.exists(magnitude_file):
            os.remove(magnitude_file)
        self.assertFalse(os.path.exists(magnitude_file))
        okg = okgraph.OKgraph(corpus=corpus_file_path, dictionary_path=dict_total, index_path=indexing_folder, create_index=True)
        self.assertTrue(os.path.exists(magnitude_file))
        self.assertIsInstance(okg.embeddings, Magnitude)
        self.assertIsInstance(okg.corpus, str)
        self.assertGreater(len(okg.embeddings), 0)

    def test_02_init_model_exists_and_implicitly_passed(self):
        """ uses an existing magnitude file if it is named <corpus>.magnitude even if not passed as argument"""
        corpus_file = 'tests/data/text8'
        magnitude_file = corpus_file + '.magnitude'
        self.assertTrue(os.path.exists(magnitude_file))
        modificationTime = os.path.getmtime(magnitude_file)
        okg = okgraph.OKgraph(corpus=corpus_file_path, dictionary_path=dict_total, index_path=indexing_folder, create_index=True)
        self.assertTrue(os.path.exists(magnitude_file))
        self.assertEqual(modificationTime, os.path.getmtime(magnitude_file))
        self.assertIsInstance(okg.embeddings, Magnitude)
        self.assertIsInstance(okg.corpus, str)
        self.assertGreater(len(okg.embeddings), 0)


    def test_03_init_with_text_and_model(self):
        """ check passing magnitude file as argument """

        okg = okgraph.OKgraph(corpus=corpus_file_path, embeddings=embeddings_file_path, dictionary_path=dict_total, index_path=indexing_folder)

        self.assertIsInstance(okg.embeddings, Magnitude)
        self.assertIsInstance(okg.corpus, str)

        with self.assertRaises(RuntimeError):
            # test not existing file/url
            okgraph.OKgraph(corpus='none', embeddings='fake/path')

        with self.assertRaises(RuntimeError):
            okgraph.OKgraph(corpus='anotherNone', embeddings='http://not.available.org/path')


    def test_04_set_expansion(self):
        """ Test if set expansion result contains some of the expected values.
        Test if set expansion does not contains any unexpected value.
        Test if algorithm behaviour is respected.
        Test if the number of output values is lower or equal to k.
        """
        okg = okgraph.OKgraph(corpus=corpus_file_path, embeddings=embeddings_file_path)

        """
        Test set expansion algorithm "centroid"
        """

        result_1_k = 15
        result_1 = okg.set_expansion(seed=['milan', 'rome', 'turin'],
                                     algo='centroid',
                                     k=result_1_k)

        result_2_k = 14
        result_2 = okg.set_expansion(seed=['home', 'house', 'apartment'],
                                     algo='centroid',
                                     k=result_2_k)

        intersection = set(result_1) & set(result_2)

        print(result_1)
        print(result_2)

        self.assertTrue(len(intersection) == 0,
                        msg="intersection of different seeds must be empty")

        self.assertTrue(len(result_1) <= result_1_k,
                        msg="k value must be <= of the list length")

    def test_05_relation_expansion(self):
        okg = okgraph.OKgraph(embeddings=embeddings_file_path, dictionary_path=dict_total, index_path=indexing_folder)
        seed = [('rome', 'italy'), ('berlin', 'germany')]

        expansion = okg.relation_expansion(seed)

        print('Expansion of {seed} is {expansion}'.format(seed=seed, expansion=expansion))
        pass

    def test_06_relation_labeling(self):
        okg = okgraph.OKgraph(embeddings=embeddings_file_path, dictionary_path=dict_total, index_path=indexing_folder)
        pairs = [('rome', 'italy'), ('berlin', 'germany')]
        sliding_windows = []

        sliding_windows.append(SlidingWindows(words=pairs[0], corpus_dictionary_path=okg.dictionary, corpus_index_path=okg.index, info=False))
        print('Pair {pair} produced result {result}'.format(pair=pairs[0], result=sliding_windows[0].results))
        self.assertTrue(len(sliding_windows[0].results) > 0,
                        msg='windows one between rome and italy is not empty')

        sliding_windows.append(SlidingWindows(words=pairs[1], corpus_dictionary_path=okg.dictionary, corpus_index_path=okg.index, info=False))
        print('Pair {pair} produced result {result}'.format(pair=pairs[1], result=sliding_windows[1].results))
        self.assertTrue(len(sliding_windows[1].results) > 0,
                        msg='windows two between berlin and germany is not empty')

        intersection = okg.relation_labeling(pairs)

        print('Labels describing the pairs of words: {labels}'.format(labels=intersection))

        self.assertTrue(len(intersection) != 0)

    def test_07_set_labeling(self):
        pass


if __name__ == '__main__':
    unittest.main()
