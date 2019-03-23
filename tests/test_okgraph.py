import unittest
import okgraph
import os
from pymagnitude import Magnitude

cwd = os.getcwd()
print( cwd )
corpus_file_path = cwd + '/tests/data/text7.gz'
embeddings_file_path = cwd + '/tests/data/text7.magnitude'
indexing_folder = cwd + '/tests/data/indexdir/'
dict_total= cwd + '/tests/data/dictTotal.npy'
#embeddings_file_path = cwd + '/tests/data/GoogleNews-vectors-negative300.magnitude'


class OKGraphTest(unittest.TestCase):

    def test_init_with_textgz(self):
        """ create a magnitude file if not exists and not passed as argument"""
        magnitude_file = corpus_file_path + '.magnitude'
        if os.path.exists(magnitude_file):
            os.remove(magnitude_file)
        self.assertFalse(os.path.exists(magnitude_file))
        okg = okgraph.OKgraph(corpus_file_path)
        self.assertTrue(os.path.exists(magnitude_file))
        self.assertIsInstance(okg.v, Magnitude)
        self.assertIsInstance(okg.corpus, str)
        self.assertGreater(len(okg.v), 0)

    def test_init_model_exists_and_implicitly_passed(self):
        """ uses an existing magnitude file if it is named <corpus>.magnitude even if not passed as argument"""
        corpus_file = 'tests/data/text7'
        magnitude_file = corpus_file + '.magnitude'
        self.assertTrue(os.path.exists(magnitude_file))
        modificationTime = os.path.getmtime(magnitude_file)
        okg = okgraph.OKgraph(corpus_file)
        self.assertTrue(os.path.exists(magnitude_file))
        self.assertEqual(modificationTime, os.path.getmtime(magnitude_file))
        self.assertIsInstance(okg.v, Magnitude)
        self.assertIsInstance(okg.corpus, str)
        self.assertGreater(len(okg.v), 0)


    def test_init_with_text_and_model(self):
        """ check passing magnitude file as argument """

        okg = okgraph.OKgraph(corpus_file_path, embeddings_file_path)

        self.assertIsInstance(okg.v, Magnitude)
        self.assertIsInstance(okg.corpus, str)

        with self.assertRaises(RuntimeError):
            # test not existing file/url
            okgraph.OKgraph(corpus='none', embeddings='fake/path')

        with self.assertRaises(RuntimeError):
            okgraph.OKgraph(corpus='anotherNone', embeddings='http://not.available.org/path')


    def test_set_expansion(self):
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

    def test_relation_expansion(self):
        pass

    def test_relation_labeling(self):
        okg = okgraph.OKgraph(embeddings=embeddings_file_path, index=indexing_folder, dict_total=dict_total)
        w1 = okg.w(words=['rome', 'italy'])
        w2 = okg.w(words=['berlin', 'germany'])

        self.assertTrue(len(w1.results)>1,
                        msg='windows one between rome and italy is not empty')

        self.assertTrue(len(w2.results) > 2,
                        msg='windows two between berlin and germany is not empty')

        print(w1.results)
        print(w2.results)

        intersection = okg.relation_labeling([w1, w2])

        self.assertTrue(len(intersection) == 0)

    def test_set_labeling(self):
        pass


if __name__ == '__main__':
    unittest.main()
