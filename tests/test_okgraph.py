import unittest
import okgraph

corpus_file_path = 'tests/text7.head.gz'
embeddings_file_path = 'tests/text7.head.magnitude'
embeddings_file_path = 'tests/GoogleNews-vectors-negative300.magnitude'

okg = okgraph.OKgraph(corpus=corpus_file_path, embeddings=embeddings_file_path)

ALGORITHM = okgraph.ALGORITHM


class OKGraphTest(unittest.TestCase):

    def test_init_with_text(self):
        okgraph.OKgraph(corpus_file_path, embeddings_file_path)
        okgraph.OKgraph(corpus=corpus_file_path, embeddings=embeddings_file_path)

        with self.assertRaises(RuntimeError):
            # test not existing file/url
            okgraph.OKgraph(corpus='none', embeddings='fake/path')
            okgraph.OKgraph(corpus='anotherNone', embeddings='http://not.available.org/path')

    def test_set_expansion(self):
        """ Test if set expansion result contains some of the expected values.
        Test if set expansion does not contains any unexpected value.
        Test if algorithm behaviour is respected.
        Test if the number of output values is lower or éequal to k.
        """
        result_1_k = 5
        result_1 = okg.set_expansion(seed=['milan', 'rome', 'turin'],
                                     algo=ALGORITHM.CENTROID.value,
                                     options={'n': 5, 'width': 10},
                                     k=result_1_k)

        result_2_k = 4
        result_2 = okg.set_expansion(seed=['home', 'house', 'apartment'],
                                     algo=ALGORITHM.CENTROID.value,
                                     options={'n': 5, 'width': 10},
                                     k=result_2_k)

        intersection = set(result_1) & set(result_2)

        print(result_1)
        print(result_2)

        self.assertTrue("cagliari" in [r[0] for r in result_1] and "florence" in [r[0] for r in result_2],
                        msg="at least some correct values must be contained")

        self.assertTrue(len(intersection) == 0,
                        msg="intersection of different seeds must be empty")

        self.assertEqual(len(result_1) <= result_1_k,
                         msg="k value must be <= of the list length")

    def test_relation_expansion(self):
        pass

    def test_relation_labeling(self):
        pass

    def test_set_labeling(self):
        pass


if __name__ == '__main__':
    unittest.main()
