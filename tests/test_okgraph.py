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

    def test_04_relation_expansion(self):
        okg = okgraph.OKgraph(corpus=corpus_file_path, embeddings=embeddings_file_path, dictionary_path=dict_total, index_path=indexing_folder)
        seed = [('rome', 'italy'), ('berlin', 'germany')]
        k = 15
        options = {'relation_labeling_algo': 'intersection',
                   'relation_labeling_options': {'dictionary': okg.dictionary, 'index': okg.index},
                   'relation_labeling_k': 15,
                   'set_expansion_algo': 'centroid',
                   'set_expansion_options': {'embeddings': okg.embeddings},
                   'set_expansion_k': 15
                   }

        expansion = okg.relation_expansion(seed, k, 'intersection', options)

        print('Expansion of {seed} is {expansion}'.format(seed=seed, expansion=expansion))

    def test_05_relation_labeling(self):
        okg = okgraph.OKgraph(corpus=corpus_file_path, embeddings=embeddings_file_path, dictionary_path=dict_total, index_path=indexing_folder)
        seed = [('rome', 'italy'), ('berlin', 'germany')]
        k = 15
        options = {'dictionary': okg.dictionary,
                   'index': okg.index}

        sliding_windows = []
        for pair in seed:
            sliding_windows.append(SlidingWindows(target_words=[pair[0], pair[1]], corpus_dictionary_path=okg.dictionary, corpus_index_path=okg.index, info=False))
            result = list(sliding_windows[-1].results_dict.keys())[:k]
            print('Pair {pair} produced result {result}'.format(pair=pair, result=result))
            self.assertTrue(len(sliding_windows[0].results_dict) > 0,
                            msg='windows one between {f} and {s} is not empty'.format(f=pair[0], s=pair[1]))

        intersection = okg.relation_labeling(seed, k, 'intersection', options)

        print('Labels describing the pairs of words: {labels}'.format(labels=intersection))

        self.assertTrue(len(intersection) != 0)

    def test_06_set_expansion(self):
        okg = okgraph.OKgraph(corpus=corpus_file_path, embeddings=embeddings_file_path, dictionary_path=dict_total, index_path=indexing_folder)

        """
        Test set expansion algorithm "centroid"
        """

        seed_1 = ['milan', 'rome', 'turin']
        seed_2 = ['home', 'house', 'apartment']
        k = 15
        options = {'embeddings': okg.embeddings}

        result_1 = okg.set_expansion(seed_1, k, 'centroid', options)

        result_2 = okg.set_expansion(seed_2, k, 'centroid', options)

        intersection = set(result_1) & set(result_2)

        print(result_1)
        print(result_2)

        self.assertTrue(len(intersection) == 0,
                        msg="intersection of different seeds must be empty")

        self.assertTrue(len(result_1) <= k,
                        msg="the expansion can't contain more than 'k' elements")

    def test_07_set_labeling(self):
        okg = okgraph.OKgraph(corpus=corpus_file_path, embeddings=embeddings_file_path, dictionary_path=dict_total, index_path=indexing_folder)
        seed = ['milan', 'rome', 'turin']
        k = 15
        options = {'dictionary': okg.dictionary,
                   'index': okg.index}
        sliding_windows = []

        for word in seed:
            sliding_windows.append(SlidingWindows(target_words=[word], corpus_dictionary_path=okg.dictionary, corpus_index_path=okg.index, info=False))
            result = list(sliding_windows[-1].results_dict.keys())[:k]
            print('Word {word} produced result {result}'.format(word=word, result=result))

        intersection = okg.set_labeling(seed, k, 'intersection', options)

        print('Labels describing the words: {labels}'.format(labels=intersection))

        self.assertTrue(len(intersection) != 0)

    def test_08_sliding_windows(self):
        okg = okgraph.OKgraph(corpus=corpus_file_path, embeddings=embeddings_file_path, dictionary_path=dict_total, index_path=indexing_folder)
        seed = [('rome', 'italy'), ('berlin', 'germany')]
        sliding_windows = []

        for pair in seed:
            sliding_windows.append(SlidingWindows(target_words=[pair[0], pair[1]], corpus_dictionary_path=okg.dictionary, corpus_index_path=okg.index, info=False))
            result = list(sliding_windows[-1].results_dict.keys())
            print('Pair {pair} produced result {result}'.format(pair=pair, result=result))
            self.assertTrue(len(sliding_windows[0].results_dict) > 0,
                            msg='windows one between {f} and {s} is not empty'.format(f=pair[0], s=pair[1]))

        for sliding_window in sliding_windows:
            print('Sliding window of target {target}'.format(target=sliding_window.target_words))
            for key in sliding_window.results_dict:
                print('{w:20s} TF-IDF={tf_idf:2.8f}, noise={noise:2.8f}'.format(w=key, tf_idf=sliding_window.tf_idf_dict.get(key), noise=sliding_window.noise_dict.get(key)))


if __name__ == '__main__':
    unittest.main()
