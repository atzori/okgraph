import unittest
from okgraph import MathVectors


class MathVectorsTest(unittest.TestCase):

    def test_centroid(self):

        list_of_n = [1.0, 2.0, 3.0]
        centroid_list_result = MathVectors.mean(vector=list_of_n)
        self.assertTrue(centroid_list_result == 2)

        pass

    def test_euclidean_distance(self):
        pass

    def test_norm(self):
        pass

    def test_normalized(self):
        pass


if __name__ == '__main__':
    unittest.main()
