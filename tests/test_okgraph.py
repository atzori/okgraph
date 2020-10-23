from numpy import floating, ndarray
from okgraph.core import OKgraph, DEFAULT_DICTIONARY_NAME
from okgraph.embeddings import WordEmbeddings
from okgraph.indexing import DEFAULT_INDEX_FOLDER
from okgraph.utils import logger
from os import path
import pytest
import unittest

TEST_DATA_FOLDER = path.normpath("data")
"""str: location in which store the various test corpus and resources.
"""

# Test parameters (text files with no extension)
test_small_corpus_name = "text7"
test_medium_corpus_name = "text8"
test_big_corpus_name = "text9"

# Corpus file info
(corpus_name, corpus_extension) = path.splitext(test_small_corpus_name)
data_path = path.join(path.dirname(__file__), "data", corpus_name)
test_corpus_file = path.join(data_path, test_small_corpus_name)

# Embeddings file info
default_embeddings_name = corpus_name + ".magnitude"
test_embeddings_file = path.join(data_path, "new_dir", corpus_name + "_vmodel.magnitude")
test_default_embeddings_file = path.join(data_path, default_embeddings_name)

# Corpus index folder info
default_index_folder = DEFAULT_INDEX_FOLDER
test_index_folder = path.join(data_path, "new_dir", corpus_name + "_indexdir")
test_default_index_folder = path.join(data_path, default_index_folder)

# Corpus dictionary info
default_dictionary_name = DEFAULT_DICTIONARY_NAME
test_dictionary_file = path.join(data_path, "new_dir", corpus_name + "_dict.npy")
test_default_dictionary_file = path.join(data_path, default_dictionary_name)


@pytest.fixture(scope="module", autouse=True)
def test_setup_and_teardown():
    """Perform setup and teardown operations at module level.

    Checks, before running the tests, if all the needed data is existing,
    otherwise retrieve it.
    """
    # SETUP START
    pass
    # SETUP END
    yield
    # TEARDOWN START
    pass


class OKGraphTest(unittest.TestCase):

    def test_core_init_from_scratch_with_default(self):
        """Tests the initialization of an OKgraph object from a given corpus
        using default parameters. Tests the corpus processing routines setting
        force_init to True.
        """
        corpus = test_small_corpus_name

        # Check the existence of the corpus
        self.assertTrue(path.exists(test_corpus_file),
                        msg=f"The corpus file {test_corpus_file} should exists")

        # Force the data processing
        if path.exists(test_default_embeddings_file)\
                or path.exists(test_default_index_folder)\
                or path.exists(test_default_dictionary_file):
            force_init = True
        else:
            force_init = False

        # Creates the OKgraph object along with the embeddings, index and
        # dictionary
        okg = OKgraph(corpus_file=test_corpus_file,
                      force_init=force_init)

        # Check the existence of the files
        self.assertTrue(
            path.exists(test_default_embeddings_file),
            msg=f"The embeddings file {test_default_embeddings_file}"
                f" should exists")
        self.assertTrue(
            path.exists(test_default_index_folder),
            msg=f"The indexing folder {test_default_index_folder}"
                f" should exists")
        self.assertTrue(
            path.exists(test_default_dictionary_file),
            msg=f"The dictionary file {test_default_dictionary_file}"
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
        """
        # Check the existence of the corpus
        self.assertTrue(path.exists(test_corpus_file),
                        msg=f"The corpus file {test_corpus_file} should exists")

        # Force the data processing
        if path.exists(test_embeddings_file)\
                or path.exists(test_index_folder)\
                or path.exists(test_dictionary_file):
            force_init = True
        else:
            force_init = False

        # Creates the OKgraph object along with the embeddings, indexes and
        # dictionary
        okg = OKgraph(corpus_file=test_corpus_file,
                      embeddings_file=test_embeddings_file,
                      index_dir=test_index_folder,
                      dictionary_file=test_dictionary_file,
                      force_init=force_init)

        # Check the existence of the files
        self.assertTrue(
            path.exists(test_embeddings_file),
            msg=f"The embeddings file {test_embeddings_file} should exists")
        self.assertTrue(
            path.exists(test_index_folder),
            msg=f"The indexing folder {test_index_folder} should exists")
        self.assertTrue(
            path.exists(test_dictionary_file),
            msg=f"The dictionary file {test_dictionary_file} should exists")
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

    def test_core_init_with_existent_processed_data(self):
        """Tests the initialization of an OKgraph object from a given corpus
        using default parameters for the embeddings, index and dictionary
        indicating existing processed data. Tests the possible errors with non
        existing corpus.
        """
        # Check the existence of the corpus
        self.assertTrue(
            path.exists(test_corpus_file),
            msg=f"The corpus file {test_corpus_file} should exists")

        # Check the existence of the processed data
        self.assertTrue(
            path.exists(test_default_embeddings_file),
            msg=f"The embeddings file {test_default_embeddings_file}"
                f" should exists")
        self.assertTrue(
            path.exists(test_default_index_folder),
            msg=f"The indexing folder {test_default_index_folder}"
                f" should exists")
        self.assertTrue(
            path.exists(test_default_dictionary_file),
            msg=f"The dictionary file {test_default_dictionary_file}"
                f" should exists")

        # Test non existing file/url
        with self.assertRaises(RuntimeError):
            OKgraph(corpus_file="none",
                    embeddings_file="not/available/path")
        with self.assertRaises(RuntimeError):
            OKgraph(corpus_file="another_none",
                    embeddings_file="http://not.available.org/path")

        # Get the modification time of the data
        embeddings_modification_time = \
            path.getmtime(test_default_embeddings_file)
        indexing_modification_time = \
            path.getmtime(test_default_index_folder)
        dictionary_modification_time = \
            path.getmtime(test_default_dictionary_file)

        # Creates the OKgraph object along with the embeddings, indexes and
        # dictionary
        okg = OKgraph(corpus_file=test_corpus_file)

        # Check the modification time of the data: it should not be changed
        self.assertEqual(
            embeddings_modification_time,
            path.getmtime(test_default_embeddings_file),
            msg=f"The embeddings has been modified")
        self.assertEqual(
            indexing_modification_time,
            path.getmtime(test_default_index_folder),
            msg=f"The indexing folder has been modified")
        self.assertEqual(
            dictionary_modification_time,
            path.getmtime(test_default_dictionary_file),
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
        """
        Tests the relation expansion task using the intersection algorithm.
        Uses an OKgraph object with default values, using pre-existent data.
        """
        okg = OKgraph(corpus_file=test_corpus_file)
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
        """
        self.assertGreater(
            len(results), 0,
            msg=f"No results obtained from the algorithm")
        """
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
        """
        Tests the relation expansion task using the centroid algorithm.
        Uses an OKgraph object with default values, using pre-existent data.
        """
        okg = OKgraph(corpus_file=test_corpus_file)
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
        """
        self.assertGreater(
            len(results), 0,
            msg=f"No results obtained from the algorithm")
        """
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
        """
        Tests the relation labeling task using the intersection algorithm.
        Uses an OKgraph object with default values, using pre-existent data.
        """
        okg = OKgraph(corpus_file=test_corpus_file)
        seed = [("rome", "italy"), ("berlin", "germany")]
        k = 15
        options = {"dictionary": okg.dictionary,
                   "index": okg.index}
        results = okg.relation_labeling(seed, k, "intersection", options)

        self.assertIsInstance(
            results, list,
            msg=f"The return value of the task should be a list")
        """
        self.assertGreater(
            len(results), 0,
            msg=f"No results obtained from the algorithm")
        """
        self.assertLessEqual(
            len(results), k,
            msg=f"The limit of {k} results has been passed")
        for r_label in results:
            self.assertIsInstance(
                r_label, str,
                msg=f"The return value of the task should be a list strings")

        logger.info(f"Labels of {seed} are {results}")

    def test_task_set_expansion_centroid(self):
        """
        Tests the set expansion task using the centroid algorithm.
        Uses an OKgraph object with default values, using pre-existent data.
        """
        okg = OKgraph(corpus_file=test_corpus_file)
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
        """
        self.assertGreater(
            len(results_1), 0,
            msg=f"No results obtained from the algorithm for the seed {seed_1}")
        self.assertGreater(
            len(results_2), 0,
            msg=f"No results obtained from the algorithm for the seed {seed_2}")
        """
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
        """
        Tests the set expansion task using the centroid algorithm.
        Uses an OKgraph object with default values, using pre-existent data.
        """
        okg = OKgraph(corpus_file=test_corpus_file)
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
        """
        self.assertGreater(
            len(results_1), 0,
            msg=f"No results obtained from the algorithm for the seed {seed_1}")
        self.assertGreater(
            len(results_2), 0,
            msg=f"No results obtained from the algorithm for the seed {seed_2}")
        """
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
        """
        Tests the set expansion task using the centroid algorithm.
        Uses an OKgraph object with default values, using pre-existent data.
        """
        okg = OKgraph(corpus_file=test_corpus_file)
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
        """
        self.assertGreater(
            len(results_1), 0,
            msg=f"No results obtained from the algorithm for the seed {seed_1}")
        self.assertGreater(
            len(results_2), 0,
            msg=f"No results obtained from the algorithm for the seed {seed_2}")
        """
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
        """
        Tests the set labeling task using the intersection algorithm.
        Uses an OKgraph object with default values, using pre-existent data.
        """
        okg = OKgraph(corpus_file=test_corpus_file)
        seed = ["berlin", "paris", "moscow"]
        k = 15
        options = {"dictionary": okg.dictionary,
                   "index": okg.index}
        results = okg.set_labeling(seed, k, "intersection", options)

        self.assertIsInstance(
            results, list,
            msg=f"The return value of the task should be a list")
        """
        self.assertGreater(
            len(results), 0,
            msg=f"No results obtained from the algorithm")
        """
        self.assertLessEqual(
            len(results), k,
            msg=f"The limit of {k} results has been passed")
        for r_label in results:
            self.assertIsInstance(
                r_label, str,
                msg=f"The return value of the task should be a list strings")

        logger.info(f"Labels of {seed} are {results}")

    def test_embeddings(self):
        okg = OKgraph(corpus_file=test_corpus_file)
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
        print(type(cosine))
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
