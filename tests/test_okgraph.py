from numpy import floating, ndarray
from okgraph.core import OKgraph
from okgraph.embeddings import WordEmbeddings
from okgraph.utils import logger
import os
from os import path
import unittest


cwd = path.normpath(os.getcwd())

logger.info(f"Current working directory is {cwd}")

corpus_file = "text9.txt"
(corpus_name, corpus_extension) = path.splitext(corpus_file)
data_path = path.join(cwd, "data", corpus_name)  # "tests",
test_corpus_file = path.join(data_path, corpus_file)
test_embeddings_file = path.join(data_path, "new_dir", corpus_name + "_vmodel")
test_default_embeddings_file = path.join(data_path, corpus_name + ".magnitude")
test_indexing_folder = path.join(data_path, "new_dir", corpus_name + "_indexdir")
test_default_indexing_folder = path.join(data_path, "indexdir")
test_dictionary_file = path.join(data_path, "new_dir", corpus_name + "_dict")
test_default_dictionary_file = path.join(data_path, "dictTotal.npy")


class OKGraphTest(unittest.TestCase):

    def test_core_init_from_scratch_with_default(self):
        """
        Tests the initialization of an OKgraph object from a given corpus using default parameters. Tests the corpus
         processing routines setting force_init to True if any of the embeddings, index or dictionary already exists.
        """
        # Check the existence of the corpus
        self.assertTrue(path.exists(test_corpus_file),
                        msg=f"The corpus file {test_corpus_file} should exists")

        # Force the data processing
        if path.exists(test_default_embeddings_file)\
                or path.exists(test_default_indexing_folder)\
                or path.exists(test_default_dictionary_file):
            force_init = True
        else:
            force_init = False

        # Creates the OKgraph object along with the embeddings, indexes and dictionary
        okg = OKgraph(corpus_file=test_corpus_file,
                      force_init=force_init)

        # Check the existence of the files
        self.assertTrue(path.exists(test_default_embeddings_file),
                        msg=f"The embeddings file {test_default_embeddings_file} should exists")
        self.assertTrue(path.exists(test_default_indexing_folder),
                        msg=f"The indexing folder {test_default_indexing_folder} should exists")
        self.assertTrue(path.exists(test_default_dictionary_file),
                        msg=f"The dictionary file {test_default_dictionary_file} should exists")
        # Check the types of the OKgraph object attributes
        self.assertIsInstance(okg.corpus, str,
                              msg=f"The corpus should be a string indicating the name of the corpus file")
        self.assertIsInstance(okg.embeddings, WordEmbeddings,
                              msg=f"The embeddings should be a WordEmbedding object")
        self.assertIsInstance(okg.index, str,
                              msg=f"The index should be a string indicating the name of the index directory")
        self.assertIsInstance(okg.dictionary, str,
                              msg=f"The dictionary should be a string indicating the name of the dictionary file")

    def test_core_init_from_scratch_with_parameters(self):
        """
        Tests the initialization of an OKgraph object from a given corpus using specific parameters for the embeddings,
         index and dictionary. Tests the corpus processing routines setting force_init to True if any of the embeddings,
         index or dictionary already exists.
        """
        # Check the existence of the corpus
        self.assertTrue(path.exists(test_corpus_file),
                        msg=f"The corpus file {test_corpus_file} should exists")

        # Force the data processing
        if path.exists(test_embeddings_file)\
                or path.exists(test_indexing_folder)\
                or path.exists(test_dictionary_file):
            force_init = True
        else:
            force_init = False

        # Creates the OKgraph object along with the embeddings, indexes and dictionary
        okg = OKgraph(corpus_file=test_corpus_file,
                      embeddings_file=test_embeddings_file,
                      index_dir=test_indexing_folder,
                      dictionary_file=test_dictionary_file,
                      force_init=force_init)

        # Check the existence of the files
        self.assertTrue(path.exists(test_embeddings_file),
                        msg=f"The embeddings file {test_embeddings_file} should exists")
        self.assertTrue(path.exists(test_indexing_folder),
                        msg=f"The indexing folder {test_indexing_folder} should exists")
        self.assertTrue(path.exists(test_dictionary_file),
                        msg=f"The dictionary file {test_dictionary_file} should exists")
        # Check the types of the OKgraph object attributes
        self.assertIsInstance(okg.corpus, str,
                              msg=f"The corpus should be a string indicating the name of the corpus file")
        self.assertIsInstance(okg.embeddings, WordEmbeddings,
                              msg=f"The embeddings should be a WordEmbedding object")
        self.assertIsInstance(okg.index, str,
                              msg=f"The index should be a string indicating the name of the index directory")
        self.assertIsInstance(okg.dictionary, str,
                              msg=f"The dictionary should be a string indicating the name of the dictionary file")

    def test_core_init_with_existent_processed_data(self):
        """
        Tests the initialization of an OKgraph object from a given corpus using specific parameters for the embeddings,
         index and dictionary indicating existing processed data. Tests the possible errors with non existing corpus.
        """
        # Check the existence of the corpus
        self.assertTrue(path.exists(test_corpus_file),
                        msg=f"The corpus file {test_corpus_file} should exists")

        # Check the existence of the processed data
        self.assertTrue(path.exists(test_embeddings_file),
                        msg=f"The embeddings file {test_embeddings_file} should exists")
        self.assertTrue(path.exists(test_indexing_folder),
                        msg=f"The indexing folder {test_indexing_folder} should exists")
        self.assertTrue(path.exists(test_dictionary_file),
                        msg=f"The dictionary file {test_dictionary_file} should exists")

        # Test non existing file/url
        with self.assertRaises(RuntimeError):
            OKgraph(corpus_file="none", embeddings_file="fake/path")
        with self.assertRaises(RuntimeError):
            OKgraph(corpus_file="another_none", embeddings_file="http://not.available.org/path")

        # Get the modification time of the data
        embeddings_modification_time = path.getmtime(test_embeddings_file)
        indexing_modification_time = path.getmtime(test_indexing_folder)
        dictionary_modification_time = path.getmtime(test_dictionary_file)

        # Creates the OKgraph object along with the embeddings, indexes and dictionary
        okg = OKgraph(corpus_file=test_corpus_file,
                      embeddings_file=test_embeddings_file,
                      index_dir=test_indexing_folder,
                      dictionary_file=test_dictionary_file)

        # Check the modification time of the data: it should not be changed
        self.assertEqual(embeddings_modification_time, path.getmtime(test_embeddings_file),
                         msg=f"The embeddings has been modified")
        self.assertEqual(indexing_modification_time, path.getmtime(test_indexing_folder),
                         msg=f"The indexing folder has been modified")
        self.assertEqual(dictionary_modification_time, path.getmtime(test_dictionary_file),
                         msg=f"The dictionary file has been modified")
        # Check the types of the OKgraph object attributes
        self.assertIsInstance(okg.corpus, str,
                              msg=f"The corpus should be a string indicating the name of the corpus file")
        self.assertIsInstance(okg.embeddings, WordEmbeddings,
                              msg=f"The embeddings should be a WordEmbedding object")
        self.assertIsInstance(okg.index, str,
                              msg=f"The index should be a string indicating the name of the index directory")
        self.assertIsInstance(okg.dictionary, str,
                              msg=f"The dictionary should be a string indicating the name of the dictionary file")

    def test_task_relation_expansion_intersection(self):
        """
        Tests the relation expansion task using the intersection algorithm.
        Uses an OKgraph object with default values.
        """
        okg = OKgraph(corpus_file=test_corpus_file)
        seed = [("rome", "italy"), ("berlin", "germany")]
        k = 15
        options = {"relation_labeling_algo": "intersection",
                   "relation_labeling_options": {"dictionary": okg.dictionary, "index": okg.index},
                   "relation_labeling_k": 15,
                   "set_expansion_algo": "centroid",
                   "set_expansion_options": {"embeddings": okg.embeddings},
                   "set_expansion_k": 15
                   }
        results = okg.relation_expansion(seed, k, "intersection", options)

        self.assertIsInstance(results, list,
                              msg=f"The return value of the task should be a list")
        self.assertGreater(len(results), 0,
                           msg=f"No results obtained from the algorithm")
        self.assertLessEqual(len(results), k,
                           msg=f"The limit of {k} results has been passed")
        for r_tuple in results:
            self.assertIsInstance(r_tuple, tuple,
                                  msg=f"The return value of the task should be a list of tuples")
            for r_elements in r_tuple:
                self.assertIsInstance(r_elements, str,
                                      msg=f"The return value of the task should be a list of tuples of strings")

        logger.info(f"Expansion of {seed} is {results}")

    def test_task_relation_expansion_centroid(self):
        """
        Tests the relation expansion task using the centroid algorithm.
        Uses an OKgraph object with default values.
        """
        okg = OKgraph(corpus_file=test_corpus_file)
        seed = [("rome", "italy"), ("berlin", "germany")]
        k = 15
        options = {"embeddings": okg.embeddings,
                   "set_expansion_algo": "centroid",
                   "set_expansion_options": {"embeddings": okg.embeddings},
                   "set_expansion_k": 15
                   }
        results = okg.relation_expansion(seed, k, "centroid", options)

        self.assertIsInstance(results, list,
                              msg=f"The return value of the task should be a list")
        self.assertGreater(len(results), 0,
                           msg=f"No results obtained from the algorithm")
        self.assertLessEqual(len(results), k,
                           msg=f"The limit of {k} results has been passed")
        for r_tuple in results:
            self.assertIsInstance(r_tuple, tuple,
                                  msg=f"The return value of the task should be a list of tuples")
            for r_elements in r_tuple:
                self.assertIsInstance(r_elements, str,
                                      msg=f"The return value of the task should be a list of tuples of strings")

        logger.info(f"Expansion of {seed} is {results}")

    def test_task_relation_labeling_intersection(self):
        """
        Tests the relation labeling task using the intersection algorithm.
        Uses an OKgraph object with default values.
        """
        okg = OKgraph(corpus_file=test_corpus_file)
        b_seed = [("rome", "italy"), ("berlin", "germany")]
        t_seed = [("rome", "berlin", "tokyo")]
        k = 15
        options = {"dictionary": okg.dictionary,
                   "index": okg.index}
        b_results = okg.relation_labeling(b_seed, k, "intersection", options)
        t_results = okg.relation_labeling(t_seed, k, "intersection", options)

        self.assertIsInstance(b_results, list,
                              msg=f"The return value of the task should be a list")
        self.assertGreater(len(b_results), 0,
                           msg=f"No results obtained from the algorithm")
        self.assertLessEqual(len(b_results), k,
                           msg=f"The limit of {k} results has been passed")
        for r_label in b_results:
            self.assertIsInstance(r_label, str,
                                  msg=f"The return value of the task should be a list strings")

        logger.info(f"Labels of {b_seed} are {b_results}")
        logger.info(f"Labels of {t_seed} are {t_results}")

    def test_task_set_expansion_centroid(self):
        """
        Tests the set expansion task using the centroid algorithm.
        Uses an OKgraph object with default values.
        """
        okg = OKgraph(corpus_file=test_corpus_file)
        seed_1 = ["milan", "rome", "venice"]
        seed_2 = ["home", "house", "apartment"]
        k = 15
        options = {"embeddings": okg.embeddings}

        results_1 = okg.set_expansion(seed_1, k, "centroid", options)
        results_2 = okg.set_expansion(seed_2, k, "centroid", options)

        self.assertIsInstance(results_1, list,
                              msg=f"The return value of the task should be a list")
        self.assertIsInstance(results_2, list,
                              msg=f"The return value of the task should be a list")
        self.assertGreater(len(results_1), 0,
                           msg=f"No results obtained from the algorithm for the seed {seed_1}")
        self.assertGreater(len(results_2), 0,
                           msg=f"No results obtained from the algorithm for the seed {seed_2}")
        self.assertLessEqual(len(results_1), k,
                           msg=f"The limit of {k} results has been passed for the seed {seed_1}")
        self.assertLessEqual(len(results_2), k,
                           msg=f"The limit of {k} results has been passed for the seed {seed_2}")
        for r_word in results_1+results_2:
            self.assertIsInstance(r_word, str,
                                  msg=f"The return value of the task should be a list of strings")

        logger.info(f"Expansion of {seed_1} is {results_1}")
        logger.info(f"Expansion of {seed_2} is {results_2}")

    def test_task_set_expansion_centroid_boost(self):
        """
        Tests the set expansion task using the centroid algorithm.
        Uses an OKgraph object with default values.
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

        self.assertIsInstance(results_1, list,
                              msg=f"The return value of the task should be a list")
        self.assertIsInstance(results_2, list,
                              msg=f"The return value of the task should be a list")
        self.assertGreater(len(results_1), 0,
                           msg=f"No results obtained from the algorithm for the seed {seed_1}")
        self.assertGreater(len(results_2), 0,
                           msg=f"No results obtained from the algorithm for the seed {seed_2}")
        self.assertLessEqual(len(results_1), k,
                           msg=f"The limit of {k} results has been passed for the seed {seed_1}")
        self.assertLessEqual(len(results_2), k,
                           msg=f"The limit of {k} results has been passed for the seed {seed_2}")
        for r_word in results_1+results_2:
            self.assertIsInstance(r_word, str,
                                  msg=f"The return value of the task should be a list of strings")

        logger.info(f"Expansion of {seed_1} is {results_1}")
        logger.info(f"Expansion of {seed_2} is {results_2}")

    def test_task_set_expansion_depth(self):
        """
        Tests the set expansion task using the centroid algorithm.
        Uses an OKgraph object with default values.
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

        self.assertIsInstance(results_1, list,
                              msg=f"The return value of the task should be a list")
        self.assertIsInstance(results_2, list,
                              msg=f"The return value of the task should be a list")
        self.assertGreater(len(results_1), 0,
                           msg=f"No results obtained from the algorithm for the seed {seed_1}")
        self.assertGreater(len(results_2), 0,
                           msg=f"No results obtained from the algorithm for the seed {seed_2}")
        self.assertLessEqual(len(results_1), k,
                           msg=f"The limit of {k} results has been passed for the seed {seed_1}")
        self.assertLessEqual(len(results_2), k,
                           msg=f"The limit of {k} results has been passed for the seed {seed_2}")
        for r_word in results_1+results_2:
            self.assertIsInstance(r_word, str,
                                  msg=f"The return value of the task should be a list of strings")

        logger.info(f"Expansion of {seed_1} is {results_1}")
        logger.info(f"Expansion of {seed_2} is {results_2}")

    def test_task_set_labeling_intersection(self):
        """
        Tests the set labeling task using the intersection algorithm.
        Uses an OKgraph object with default values.
        """
        okg = OKgraph(corpus_file=test_corpus_file)
        seed = ["milan", "rome", "venice"]
        k = 15
        options = {"dictionary": okg.dictionary,
                   "index": okg.index}
        results = okg.set_labeling(seed, k, "intersection", options)

        self.assertIsInstance(results, list,
                              msg=f"The return value of the task should be a list")
        self.assertGreater(len(results), 0,
                           msg=f"No results obtained from the algorithm")
        self.assertLessEqual(len(results), k,
                           msg=f"The limit of {k} results has been passed")
        for r_label in results:
            self.assertIsInstance(r_label, str,
                                  msg=f"The return value of the task should be a list strings")

        logger.info(f"Labels of {seed} are {results}")

    def test_embeddings(self):
        okg = OKgraph(corpus_file=test_corpus_file)
        e = okg.embeddings
        n = 15

        w = 'town'
        v = e.w2v(w)

        r_w2v = e.w2v(w)
        self.assertIsInstance(r_w2v, ndarray,
                              msg=f"The w2v function must return a vector (numpy.array)")
        self.assertIsInstance(r_w2v[0], floating,
                              msg=f"The w2v function must return a vector of floats")
        r_v2w = e.v2w(v)
        self.assertIsInstance(r_v2w, list,
                              msg=f"The v2w function must return a list")
        self.assertIsInstance(r_v2w[0], str,
                              msg=f"The v2w function must return a list of words (strings)")
        r_w2w = e.w2w(w)
        self.assertIsInstance(r_w2w, list,
                              msg=f"The w2w function must return a list")
        self.assertIsInstance(r_w2w[0], str,
                              msg=f"The w2w function must return a list of words (strings)")
        r_v2v = e.v2v(v)
        self.assertIsInstance(r_v2v, list,
                              msg=f"The v2v function must return a list")
        self.assertIsInstance(r_v2v[0], ndarray,
                              msg=f"The v2v function must return a list of vectors (numpy.array)")
        self.assertIsInstance(r_v2v[0][0], floating,
                              msg=f"The v2v function must return a list of vectors of floats")

        vs1 = e.v2v(v, n)
        vs2 = list(map(lambda x: e.w2v(x), e.w2w(e.v2w(v, 1)[0], n)))
        self.assertEqual(vs1, vs2,
                         msg=f"The vector expansions of the same vector must be equal")

        ws1 = e.w2w(w, n)
        ws2 = list(map(lambda x: e.v2w(x, 1)[0], e.v2v(e.w2v(w), n)))
        self.assertEqual(ws1, ws2,
                         msg=f"The word expansions of the same word must be equal")

        r_w4th = e.get4th("man", "king", "woman")
        self.assertIsInstance(r_w4th, list,
                              msg=f"The get4th function must return a list")
        self.assertIsInstance(r_w4th[0], str,
                              msg=f"The get4th function must return a list of words (strings)")

        r_v4th = e.get4thv(e.w2v("man"), e.w2v("king"), e.w2v("woman"))
        self.assertIsInstance(r_v4th, list,
                              msg=f"The get4th function must return a list")
        self.assertIsInstance(r_v4th[0], str,
                              msg=f"The get4th function must return a list of words (strings)")

        centroid = e.centroid(["milan", "rome", "venice"])
        self.assertIsInstance(centroid, ndarray,
                              msg=f"The centroid function must return a vector (numpy.array)")

        not_existing_word = "iononsonounaparolachepuòesisterenelmodello"
        existence = e.exists(not_existing_word)
        self.assertFalse(existence,
                         msg=f"{not_existing_word} cannot be in the model")


if __name__ == "__main__":
    unittest.main()
