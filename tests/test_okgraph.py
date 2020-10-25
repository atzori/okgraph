from numpy import floating, ndarray
from okgraph.core import OKgraph, NotExistingCorpusException, \
    DEFAULT_DICTIONARY_NAME
from okgraph.embeddings import WordEmbeddings
from okgraph.indexing import DEFAULT_INDEX_FOLDER
from okgraph.utils import logger
from os import path
import shutil
import tests.get_test_corpus_and_resources
from tests.get_test_corpus_and_resources import TEST_DATA_FOLDER, \
    TEST_SMALL_CORPUS, TEST_MEDIUM_CORPUS, TEST_BIG_CORPUS
import unittest

_corpus_default_data = \
    {TEST_SMALL_CORPUS: {"file": "", "embeddings": "",
                         "index": "", "dictionary": ""},
     TEST_MEDIUM_CORPUS: {"file": "", "embeddings": "",
                          "index": "", "dictionary": ""},
     TEST_BIG_CORPUS: {"file": "", "embeddings": "",
                       "index": "", "dictionary": ""}}
"""Dict[str, Dict[str, str]]: dictionary containing the default values for
the corpus file, embeddings, index and dictionary paths.
"""


class OKGraphTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Perform setup operations at class level. This method is executed
        one time before starting a test session from this class.

        Checks, before running the tests, if all the needed data is existing,
        otherwise retrieve it.

        """
        print(f"Setting up test session")
        # Check the existence of the test data
        tests.get_test_corpus_and_resources.main()
        # Initialize the corpus default data
        for corpus in _corpus_default_data:
            corpus_name, _ = path.splitext(corpus)
            corpus_path = path.normpath(path.join(
                path.dirname(__file__), TEST_DATA_FOLDER, corpus_name))
            _corpus_default_data[corpus]["file"] = \
                path.normpath(path.join(corpus_path, corpus))
            _corpus_default_data[corpus]["embeddings"] = \
                path.normpath(path.join(corpus_path, corpus_name + ".magnitude"))
            _corpus_default_data[corpus]["index"] = \
                path.normpath(path.join(corpus_path, DEFAULT_INDEX_FOLDER))
            _corpus_default_data[corpus]["dictionary"] = \
                path.normpath(path.join(corpus_path, DEFAULT_DICTIONARY_NAME))

    @classmethod
    def tearDownClass(cls) -> None:
        """Perform teardown operations at class level. This method is executed
        one time before closing a test session from this class.

        """
        print(f"Tearing down test session")
        pass

    def test_core_init_from_scratch_with_default(self):
        """Tests the initialization of an OKgraph object from a given corpus
        using default parameters. Tests the corpus processing routines setting
        force_init to True.
        """
        test_corpus = TEST_SMALL_CORPUS
        corpus_file = _corpus_default_data[test_corpus]["file"]
        embeddings = _corpus_default_data[test_corpus]["embeddings"]
        index = _corpus_default_data[test_corpus]["index"]
        dictionary = _corpus_default_data[test_corpus]["dictionary"]

        # Check the existence of the corpus
        self.assertTrue(path.exists(corpus_file),
                        msg=f"The corpus file {corpus_file} should exists")

        # Force the data processing
        force_init = True

        # Creates the OKgraph object along with the embeddings, index and
        # dictionary
        okg = OKgraph(corpus_file=corpus_file,
                      force_init=force_init)

        # Check the existence of the files
        self.assertTrue(
            path.exists(embeddings),
            msg=f"The embeddings file {embeddings}"
                f" should exists")
        self.assertTrue(
            path.exists(index),
            msg=f"The indexing folder {index}"
                f" should exists")
        self.assertTrue(
            path.exists(dictionary),
            msg=f"The dictionary file {dictionary}"
                f" should exists")
        # Check the types of the OKgraph object attributes
        self.assertIsInstance(
            okg.corpus, str,
            msg=f"The corpus should be a string indicating the name of the"
                f" corpus file")
        self.assertIsInstance(
            okg.embeddings, WordEmbeddings,
            msg=f"The embeddings should be a WordEmbedding object")
        self.assertIsInstance(
            okg.index, str,
            msg=f"The index should be a string indicating the name of the"
                f" index directory")
        self.assertIsInstance(
            okg.dictionary, str,
            msg=f"The dictionary should be a string indicating the name of the"
                f" dictionary file")

    def test_core_init_from_scratch_with_parameters(self):
        """Tests the initialization of an OKgraph object from a given corpus
        using specific values for the embeddings, index and dictionary.
        Tests the corpus processing routines setting force_init to True.
        Tests the possible errors with non existing resources.

        """
        test_corpus = TEST_SMALL_CORPUS
        corpus_file = _corpus_default_data[test_corpus]["file"]
        (corpus_name, _) = path.splitext(test_corpus)
        folder = path.normpath(path.join(
            path.dirname(__file__), TEST_DATA_FOLDER, corpus_name, "new_dir"))
        embeddings = path.normpath(path.join(folder, corpus_name + "_vmodel.magnitude"))
        index = path.normpath(path.join(folder, corpus_name + "_indexdir"))
        dictionary = path.normpath(path.join(folder, corpus_name + "_dict.npy"))

        # Check the existence of the corpus
        self.assertTrue(path.exists(corpus_file),
                        msg=f"The corpus file {corpus_file} should exists")

        # Force the data processing
        force_init = True

        # Test non existing file
        with self.assertRaises(NotExistingCorpusException):
            OKgraph(corpus_file="not_existing_file")
        # TODO: test the stream option for the embeddings with non existing URL

        # Creates the OKgraph object along with the embeddings, indexes and
        # dictionary
        okg = OKgraph(corpus_file=corpus_file,
                      embeddings_file=embeddings,
                      index_dir=index,
                      dictionary_file=dictionary,
                      force_init=force_init)

        # Check the existence of the files
        self.assertTrue(
            path.exists(embeddings),
            msg=f"The embeddings file {embeddings} should exists")
        self.assertTrue(
            path.exists(index),
            msg=f"The indexing folder {index} should exists")
        self.assertTrue(
            path.exists(dictionary),
            msg=f"The dictionary file {dictionary} should exists")
        # Check the types of the OKgraph object attributes
        self.assertIsInstance(
            okg.corpus, str,
            msg=f"The corpus should be a string indicating the name of the"
                f" corpus file")
        self.assertIsInstance(
            okg.embeddings, WordEmbeddings,
            msg=f"The embeddings should be a WordEmbedding object")
        self.assertIsInstance(
            okg.index, str,
            msg=f"The index should be a string indicating the name of the"
                f" index directory")
        self.assertIsInstance(
            okg.dictionary, str,
            msg=f"The dictionary should be a string indicating the name of the"
                f" dictionary file")
        # Remove the created directory
        del okg
        # shutil.rmtree(folder)
        # FIXME: cannot remove the folder because the embeddings in the deleted
        #  okgraph instance are still locking the vector model file

    def test_core_init_with_existent_processed_data(self):
        """Tests the initialization of an OKgraph object from a given corpus
        using default parameters for the embeddings, index and dictionary
        indicating existing processed data.

        """
        test_corpus = TEST_SMALL_CORPUS
        corpus_file = _corpus_default_data[test_corpus]["file"]
        embeddings = _corpus_default_data[test_corpus]["embeddings"]
        index = _corpus_default_data[test_corpus]["index"]
        dictionary = _corpus_default_data[test_corpus]["dictionary"]

        # Check the existence of the corpus
        self.assertTrue(
            path.exists(corpus_file),
            msg=f"The corpus file {corpus_file} should exists")

        # Check the existence of the processed data
        self.assertTrue(
            path.exists(embeddings),
            msg=f"The embeddings file {embeddings}"
                f" should exists")
        self.assertTrue(
            path.exists(index),
            msg=f"The indexing folder {index}"
                f" should exists")
        self.assertTrue(
            path.exists(dictionary),
            msg=f"The dictionary file {dictionary}"
                f" should exists")

        # Get the modification time of the data
        embeddings_modification_time = \
            path.getmtime(embeddings)
        indexing_modification_time = \
            path.getmtime(index)
        dictionary_modification_time = \
            path.getmtime(dictionary)

        # Creates the OKgraph object along with the embeddings, indexes and
        # dictionary
        okg = OKgraph(corpus_file=corpus_file)

        # Check the modification time of the data: it should not be changed
        self.assertEqual(
            embeddings_modification_time,
            path.getmtime(embeddings),
            msg=f"The embeddings has been modified")
        self.assertEqual(
            indexing_modification_time,
            path.getmtime(index),
            msg=f"The indexing folder has been modified")
        self.assertEqual(
            dictionary_modification_time,
            path.getmtime(dictionary),
            msg=f"The dictionary file has been modified")

        # Check the types of the OKgraph object attributes
        self.assertIsInstance(
            okg.corpus, str,
            msg=f"The corpus should be a string indicating the name of the"
                f" corpus file")
        self.assertIsInstance(
            okg.embeddings, WordEmbeddings,
            msg=f"The embeddings should be a WordEmbedding object")
        self.assertIsInstance(
            okg.index, str,
            msg=f"The index should be a string indicating the name of the"
                f" index directory")
        self.assertIsInstance(
            okg.dictionary, str,
            msg=f"The dictionary should be a string indicating the name of the"
                f" dictionary file")

    def test_task_relation_expansion_intersection(self):
        """Tests the relation expansion task using the intersection algorithm.
        Uses an OKgraph object with default values, using pre-existent data.

        """
        test_corpus = TEST_BIG_CORPUS
        corpus_file = _corpus_default_data[test_corpus]["file"]

        okg = OKgraph(corpus_file=corpus_file)
        seed = [("rome", "italy"), ("berlin", "germany")]
        k = 15
        options = {"relation_labeling_algo": "intersection",
                   "relation_labeling_options":
                       {"dictionary": okg.dictionary,
                        "index": okg.index},
                   "relation_labeling_k": 15,
                   "set_expansion_algo": "centroid",
                   "set_expansion_options":
                       {"embeddings": okg.embeddings},
                   "set_expansion_k": 15
                   }
        results = okg.relation_expansion(seed, k, "intersection", options)

        self.assertIsInstance(
            results, list,
            msg=f"The return value of the task should be a list")
        self.assertGreater(
            len(results), 0,
            msg=f"No results obtained from the algorithm")
        self.assertLessEqual(
            len(results), k,
            msg=f"The limit of {k} results has been passed")
        for r_tuple in results:
            self.assertIsInstance(
                r_tuple, tuple,
                msg=f"The return value of the task should be a list of tuples")
            for r_elements in r_tuple:
                self.assertIsInstance(
                    r_elements, str,
                    msg=f"The return value of the task should be a list of"
                        f" tuples of strings")

        logger.info(f"Expansion of {seed} is {results}")

    def test_task_relation_expansion_centroid(self):
        """Tests the relation expansion task using the centroid algorithm.
        Uses an OKgraph object with default values, using pre-existent data.

        """
        test_corpus = TEST_BIG_CORPUS
        corpus_file = _corpus_default_data[test_corpus]["file"]

        okg = OKgraph(corpus_file=corpus_file)
        seed = [("rome", "italy"), ("berlin", "germany")]
        k = 15
        options = {"embeddings": okg.embeddings,
                   "set_expansion_algo": "centroid",
                   "set_expansion_options":
                       {"embeddings": okg.embeddings},
                   "set_expansion_k": 15
                   }
        results = okg.relation_expansion(seed, k, "centroid", options)

        self.assertIsInstance(
            results, list,
            msg=f"The return value of the task should be a list")
        self.assertGreater(
            len(results), 0,
            msg=f"No results obtained from the algorithm")
        self.assertLessEqual(
            len(results), k,
            msg=f"The limit of {k} results has been passed")
        for r_tuple in results:
            self.assertIsInstance(
                r_tuple, tuple,
                msg=f"The return value of the task should be a list of tuples")
            for r_elements in r_tuple:
                self.assertIsInstance(
                    r_elements, str,
                    msg=f"The return value of the task should be a list of"
                        f" tuples of strings")

        logger.info(f"Expansion of {seed} is {results}")

    def test_task_relation_labeling_intersection(self):
        """Tests the relation labeling task using the intersection algorithm.
        Uses an OKgraph object with default values, using pre-existent data.

        """
        test_corpus = TEST_BIG_CORPUS
        corpus_file = _corpus_default_data[test_corpus]["file"]

        okg = OKgraph(corpus_file=corpus_file)
        seed = [("rome", "italy"), ("berlin", "germany")]
        k = 15
        options = {"dictionary": okg.dictionary,
                   "index": okg.index}
        results = okg.relation_labeling(seed, k, "intersection", options)

        self.assertIsInstance(
            results, list,
            msg=f"The return value of the task should be a list")
        self.assertGreater(
            len(results), 0,
            msg=f"No results obtained from the algorithm")
        self.assertLessEqual(
            len(results), k,
            msg=f"The limit of {k} results has been passed")
        for r_label in results:
            self.assertIsInstance(
                r_label, str,
                msg=f"The return value of the task should be a list strings")

        logger.info(f"Labels of {seed} are {results}")

    def test_task_set_expansion_centroid(self):
        """Tests the set expansion task using the centroid algorithm.
        Uses an OKgraph object with default values, using pre-existent data.

        """
        test_corpus = TEST_BIG_CORPUS
        corpus_file = _corpus_default_data[test_corpus]["file"]

        okg = OKgraph(corpus_file=corpus_file)
        seed_1 = ["milan", "rome", "venice"]
        seed_2 = ["home", "house", "apartment"]
        k = 15
        options = {"embeddings": okg.embeddings}

        results_1 = okg.set_expansion(seed_1, k, "centroid", options)
        results_2 = okg.set_expansion(seed_2, k, "centroid", options)

        self.assertIsInstance(
            results_1, list,
            msg=f"The return value of the task should be a list")
        self.assertIsInstance(
            results_2, list,
            msg=f"The return value of the task should be a list")
        self.assertGreater(
            len(results_1), 0,
            msg=f"No results obtained from the algorithm for the seed {seed_1}")
        self.assertGreater(
            len(results_2), 0,
            msg=f"No results obtained from the algorithm for the seed {seed_2}")
        self.assertLessEqual(
            len(results_1), k,
            msg=f"The limit of {k} results has been passed for the seed"
                f" {seed_1}")
        self.assertLessEqual(
            len(results_2), k,
            msg=f"The limit of {k} results has been passed for the seed"
                f" {seed_2}")
        for r_word in results_1+results_2:
            self.assertIsInstance(
                r_word, str,
                msg=f"The return value of the task should be a list of strings")

        logger.info(f"Expansion of {seed_1} is {results_1}")
        logger.info(f"Expansion of {seed_2} is {results_2}")

    def test_task_set_expansion_centroid_boost(self):
        """Tests the set expansion task using the centroid algorithm.
        Uses an OKgraph object with default values, using pre-existent data.

        """
        test_corpus = TEST_BIG_CORPUS
        corpus_file = _corpus_default_data[test_corpus]["file"]

        okg = OKgraph(corpus_file=corpus_file)
        seed_1 = ["milan", "rome", "venice"]
        seed_2 = ["home", "house", "apartment"]
        k = 15
        options = {"embeddings": okg.embeddings,
                   "step": 2,
                   "fast": False}

        results_1 = okg.set_expansion(seed_1, k, "centroid_boost", options)
        results_2 = okg.set_expansion(seed_2, k, "centroid_boost", options)

        self.assertIsInstance(
            results_1, list,
            msg=f"The return value of the task should be a list")
        self.assertIsInstance(
            results_2, list,
            msg=f"The return value of the task should be a list")
        self.assertGreater(
            len(results_1), 0,
            msg=f"No results obtained from the algorithm for the seed {seed_1}")
        self.assertGreater(
            len(results_2), 0,
            msg=f"No results obtained from the algorithm for the seed {seed_2}")
        self.assertLessEqual(
            len(results_1), k,
            msg=f"The limit of {k} results has been passed for the seed"
                f" {seed_1}")
        self.assertLessEqual(
            len(results_2), k,
            msg=f"The limit of {k} results has been passed for the seed"
                f" {seed_2}")
        for r_word in results_1+results_2:
            self.assertIsInstance(
                r_word, str,
                msg=f"The return value of the task should be a list of strings")

        logger.info(f"Expansion of {seed_1} is {results_1}")
        logger.info(f"Expansion of {seed_2} is {results_2}")

    def test_task_set_expansion_depth(self):
        """Tests the set expansion task using the centroid algorithm.
        Uses an OKgraph object with default values, using pre-existent data.

        """
        test_corpus = TEST_BIG_CORPUS
        corpus_file = _corpus_default_data[test_corpus]["file"]

        okg = OKgraph(corpus_file=corpus_file)
        seed_1 = ["milan", "rome", "venice"]
        seed_2 = ["home", "house", "apartment"]
        k = 15
        options = {"embeddings": okg.embeddings,
                   "width": 10,
                   "depth": 2}

        results_1 = okg.set_expansion(seed_1, k, "depth", options)
        results_2 = okg.set_expansion(seed_2, k, "depth", options)

        self.assertIsInstance(
            results_1, list,
            msg=f"The return value of the task should be a list")
        self.assertIsInstance(
            results_2, list,
            msg=f"The return value of the task should be a list")
        self.assertGreater(
            len(results_1), 0,
            msg=f"No results obtained from the algorithm for the seed {seed_1}")
        self.assertGreater(
            len(results_2), 0,
            msg=f"No results obtained from the algorithm for the seed {seed_2}")
        self.assertLessEqual(
            len(results_1), k,
            msg=f"The limit of {k} results has been passed for the seed"
                f" {seed_1}")
        self.assertLessEqual(
            len(results_2), k,
            msg=f"The limit of {k} results has been passed for the seed"
                f" {seed_2}")
        for r_word in results_1+results_2:
            self.assertIsInstance(
                r_word, str,
                msg=f"The return value of the task should be a list of strings")

        logger.info(f"Expansion of {seed_1} is {results_1}")
        logger.info(f"Expansion of {seed_2} is {results_2}")

    def test_task_set_labeling_intersection(self):
        """Tests the set labeling task using the intersection algorithm.
        Uses an OKgraph object with default values, using pre-existent data.

        """
        test_corpus = TEST_BIG_CORPUS
        corpus_file = _corpus_default_data[test_corpus]["file"]

        okg = OKgraph(corpus_file=corpus_file)
        seed = ["milan", "rome", "venice"]
        k = 15
        options = {"dictionary": okg.dictionary,
                   "index": okg.index}
        results = okg.set_labeling(seed, k, "intersection", options)

        self.assertIsInstance(
            results, list,
            msg=f"The return value of the task should be a list")
        self.assertGreater(
            len(results), 0,
            msg=f"No results obtained from the algorithm")
        self.assertLessEqual(
            len(results), k,
            msg=f"The limit of {k} results has been passed")
        for r_label in results:
            self.assertIsInstance(
                r_label, str,
                msg=f"The return value of the task should be a list strings")

        logger.info(f"Labels of {seed} are {results}")

    def test_embeddings(self):
        """

        """
        test_corpus = TEST_BIG_CORPUS
        corpus_file = _corpus_default_data[test_corpus]["file"]

        okg = OKgraph(corpus_file=corpus_file)
        e = okg.embeddings
        n = 15

        w1 = "town"
        w2 = "boy"
        v1 = e.w2v(w1)

        r_w2v = e.w2v(w1)
        self.assertIsInstance(
            r_w2v, ndarray,
            msg=f"The w2v function must return a vector (numpy.array)")
        self.assertIsInstance(
            r_w2v[0], floating,
            msg=f"The w2v function must return a vector of floats")
        r_v2w = e.v2w(v1)
        self.assertIsInstance(
            r_v2w, list,
            msg=f"The v2w function must return a list")
        self.assertIsInstance(
            r_v2w[0], str,
            msg=f"The v2w function must return a list of words (strings)")
        r_w2w = e.w2w(w1)
        self.assertIsInstance(
            r_w2w, list,
            msg=f"The w2w function must return a list")
        self.assertIsInstance(
            r_w2w[0], str,
            msg=f"The w2w function must return a list of words (strings)")
        r_v2v = e.v2v(v1)
        self.assertIsInstance(
            r_v2v, list,
            msg=f"The v2v function must return a list")
        self.assertIsInstance(
            r_v2v[0], ndarray,
            msg=f"The v2v function must return a list of vectors (numpy.array)")
        self.assertIsInstance(
            r_v2v[0][0], floating,
            msg=f"The v2v function must return a list of vectors of floats")

        vs1 = e.v2v(v1, n)
        vs2 = list(map(lambda x: e.w2v(x), e.w2w(e.v2w(v1, 1)[0], n)))
        self.assertEqual(
            vs1, vs2,
            msg=f"The vector expansions of the same vector must be equal")

        ws1 = e.w2w(w1, n)
        ws2 = list(map(lambda x: e.v2w(x, 1)[0], e.v2v(e.w2v(w1), n)))
        self.assertEqual(
            ws1, ws2,
            msg=f"The word expansions of the same word must be equal")

        r_w4th = e.get4th("man", "king", "woman")
        self.assertIsInstance(
            r_w4th, list,
            msg=f"The get4th function must return a list")
        self.assertIsInstance(
            r_w4th[0], str,
            msg=f"The get4th function must return a list of words (strings)")

        r_v4th = e.get4thv(e.w2v("man"), e.w2v("king"), e.w2v("woman"))
        self.assertIsInstance(
            r_v4th, list,
            msg=f"The get4th function must return a list")
        self.assertIsInstance(
            r_v4th[0], str,
            msg=f"The get4th function must return a list of words (strings)")

        centroid = e.centroid(["milan", "rome", "venice"])
        self.assertIsInstance(
            centroid, ndarray,
            msg=f"The centroid function must return a vector (numpy.array)")

        not_existing_word = "iononsonounaparolachepuòesisterenelmodello"
        existence = e.exists(not_existing_word)
        self.assertFalse(
            existence,
            msg=f"{not_existing_word} cannot be in the model")

        cosine = e.cos(w1, w2)
        self.assertIsInstance(
            cosine, float,
            msg=f"The cosine must be a float value")
        self.assertTrue(
            0 < cosine < 1,
            msg=f"Cosine between {w1} and {w2} must be between 0 and 1")
        cosine = e.cos(w1, w1)
        self.assertTrue(
            cosine == 1,
            msg=f"Cosine between {w1} and {w1} must 1")


if __name__ == "__main__":
    unittest.main()
