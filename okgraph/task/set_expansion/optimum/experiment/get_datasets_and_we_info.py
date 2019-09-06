import okgraph
import random
import numpy
import numpy as np
import datetime
import scipy
import threading
import time
import copy

from dataset_helper import load
from optimization_core import Metric

print("STARTING..")

okgraph_path = 'models/'

def get_we_model_content_from_filename(filename):
    models = {
        'models/GoogleNews-vectors-negative300.magnitude' : 'Google News 100B (W2V)',
        'models/glove.6B.300d.magnitude' : 'Wikipedia 2014 + Gigaword 5 6B (GloVe)',
        'models/glove-lemmatized.6B.300d.magnitude' : 'Wikipedia 2014 + Gigaword 5 6B lemmatized (GloVe)',
        'models/glove.840B.300d.magnitude' : 'Common Crawl 840B (GloVe)',
        'models/glove.twitter.27B.200d.magnitude' : 'Twitter 27B (GloVe)',
        'models/wiki-news-300d-1M.magnitude' : 'English Wikipedia 2017 16B (fastText)',
        'models/wiki-news-300d-1M-subword.magnitude' : 'English Wikipedia 2017 + subword 16B (fastText)',
        'models/crawl-300d-2M.magnitude' : 'Common Crawl 600B (fastText)',
    }
    return models.get(filename, filename)


args = {
    "ground_truths": [
        'usa_states', 
        'universe_solar_planets', 
        'periodic_table_of_elements',
        # 'kings_of_rome',
    ],
    "models": [
        okgraph_path + 'GoogleNews-vectors-negative300.magnitude', # Google News 100B
        okgraph_path + 'glove.6B.300d.magnitude', # Wikipedia 2014 + Gigaword 5 6B
        okgraph_path + 'glove-lemmatized.6B.300d.magnitude', # Wikipedia 2014 + Gigaword 5 6B lemmatized (GloVe)
        okgraph_path + 'glove.840B.300d.magnitude', # Common Crawl 840B
        okgraph_path + 'glove.twitter.27B.200d.magnitude', # Twitter 27B (GloVe)
        okgraph_path + 'wiki-news-300d-1M.magnitude', # English Wikipedia 2017 16B	
        okgraph_path + 'wiki-news-300d-1M-subword.magnitude', # English Wikipedia 2017 + subword 16B	
        okgraph_path + 'crawl-300d-2M.magnitude', # Common Crawl 600B	
    ],
    "verbose": False,
    "lazy_loading": 0   #  You can pass in an optional lazy_loading argument to the constructor with the value
                        #   -1 to disable lazy-loading and pre-load all vectors into memory (a la Gensim), 
                        #   0 (default) to enable lazy-loading with an unbounded in-memory LRU cache, or 
                        #   an integer greater than zero X to enable lazy-loading with an LRU cache that holds the X most recently used vectors in memory.
}

def generate_csv(args):
    print("STARTED")

    ground_truths = args["ground_truths"]
    models = args["models"]
    verbose = args["verbose"]
    lazy_loading = args["lazy_loading"]

    csv_general_info_rows = []
    for embeddings_magnitude_model in models:
        okg = okgraph.OKgraph(corpus='', embeddings=embeddings_magnitude_model, lazy_loading=lazy_loading)
        for ground_truth_name in ground_truths:
            ground_truth = load(ground_truth_name) # this will read the text file
            ground_truth = [w.replace(" ", "_") for w in ground_truth]
            ground_truth_lowecase = [w.lower().replace(" ", "_") for w in ground_truth]
            ground_truth_without_not_exists = [e for e in ground_truth if e in okg.v]
            ground_truth_missing_on_we = [e for e in ground_truth if e not in okg.v]
            ground_truth_missing_on_we_len = len(ground_truth_missing_on_we)
            ground_truth_lowecase_missing_on_we = [e for e in ground_truth_lowecase if e not in okg.v]
            ground_truth_lowecase_missing_on_we_len = len(ground_truth_lowecase_missing_on_we)
                
            dataset_info = {
                "we_model": embeddings_magnitude_model,
                "we_model_content": get_we_model_content_from_filename(embeddings_magnitude_model),
                "ground_truth_name": ground_truth_name,
                "ground_truth": ground_truth,
                "ground_truth_length": len(ground_truth),
                "ground_truth_without_not_exists": ground_truth_without_not_exists,
                "ground_truth_missing_on_we": ground_truth_missing_on_we,
                "ground_truth_missing_on_we_len": ground_truth_missing_on_we_len,
                "ground_truth_lowecase_missing_on_we": ground_truth_lowecase_missing_on_we,
                "ground_truth_lowecase_missing_on_we_len": ground_truth_lowecase_missing_on_we_len,
            }

            csv_general_info_onerow = []
            csv_general_info_row_titles = [
                "we_model", 
                "we_model_content", 
                "ground_truth_name", 
                "ground_truth_length", 
                "ground_truth_missing_on_we_len", 
                "ground_truth_lowecase_missing_on_we_len",
                "ground_truth_missing_on_we",
                "ground_truth_lowecase_missing_on_we",
            ]
            for csv_general_info_row_title in csv_general_info_row_titles:
                csv_general_info_onerow.append(str(dataset_info[csv_general_info_row_title]))
            csv_general_info_rows.append(csv_general_info_onerow)

    now = datetime.datetime.now()
    filename = f'results/dataset_info_{now.strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3]}.csv'
    Metric.save(filename, row_titles=csv_general_info_row_titles, rows=csv_general_info_rows, verbose=verbose)
    print(f"SAVED {filename}")

generate_csv(args)