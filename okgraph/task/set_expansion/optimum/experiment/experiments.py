import okgraph
import random
import numpy
import numpy as np
import datetime
import scipy
import threading
import time
import copy

from dataset_helper import *
from optimization_core import *

print(f'Started')
numpy.seterr(divide='ignore', invalid='ignore')


def mean(vector: [numpy.ndarray]):
    """
    Calculate the centroid vector.
    :param vector:
    :return:
    """
    return sum(vector) / float(len(vector))


def choose_x0(vectors: [numpy.ndarray]):
    return mean(vectors)


def get_random_from(ground_truth: list, seed_size: int) -> list:
    """
    Get a list of random values from the ground truth.
    :param ground_truth:
    :param seed_size:
    :return:
    """
    out_list = set([])
    if seed_size > len(ground_truth):
        print(f'WARNING: seed size [{seed_size}] is more than the ground_truth possibilities [{len(ground_truth)}].')
    while len(out_list) < seed_size:
        out_list.add(random.choice(ground_truth))

    out_list = list(out_list)
    out_list.sort()
    return out_list


# seed_sizes = [(1, 10), (2, 10), (3, 10), (5, 10), (10, 10), (20, 10), (30, 10), (40, 10), (50, 10)]
def get_random_lists(ground_truth: list, seed_sizes: [int], verbose = False) -> [list]:
    """
    Get a list of random lists from the ground truth.
    :param ground_truth:
    :param seed_sizes: a list of tuples es. [(1, 10), (2, 10), (seed_len, seed_repetitions),...
    :return:
    """
    out_list = []
    out_list_strings = set([])
    seed_repetitions = 0

    if verbose:
        print(f'seed_sizes {seed_sizes} ...')

    for seed_size in seed_sizes:
        seed_len = seed_size[0]
        seed_repetitions += seed_size[1]
        if verbose:
            print(f'\r get random list of seed_size {seed_size} ...', end='')
        
        tmp_try = 500
        tmp_prev = []
        while tmp_try>0 and len(out_list) < seed_repetitions:
            to_add = get_random_from(ground_truth, seed_len)
            if verbose:
                print(f'len out_list {len(out_list)} ...')

            if "".join(to_add) not in out_list_strings:
                out_list += [to_add]
                out_list_strings.add("".join(to_add))

            if tmp_prev == "".join(to_add):
                tmp_try = tmp_try - 1
                if tmp_try == 0:
                    print('Too many tries on get_random_lists')
            else:
                tmp_try = 500

            tmp_prev = "".join(to_add)

    if verbose:
        print('seed_repetitions done.')
    return [l for l in out_list]


# Start


okgraph_path = 'models/'
corpus_file_path = okgraph_path + 'text7.head.gz'
embeddings_magnitude_modelGN = okgraph_path + 'GoogleNews-vectors-negative300.magnitude', # Google News 100B
embeddings_magnitude_modelWikiGigawordGlove6B = okgraph_path + 'glove.6B.300d.magnitude', # Wikipedia 2014 + Gigaword 5 6B
embeddings_magnitude_modelWikiGigawordGloveLemmatized6B = okgraph_path + 'glove-lemmatized.6B.300d.magnitude', # Wikipedia 2014 + Gigaword 5 6B lemmatized (GloVe)
embeddings_magnitude_modelCommonCrawlGlove840B = okgraph_path + 'glove.840B.300d.magnitude', # Common Crawl 840B
embeddings_magnitude_modelTwitter27B = okgraph_path + 'glove.twitter.27B.200d.magnitude', # Twitter 27B (GloVe)
embeddings_magnitude_modelEnglishWikipedia2017_16B = okgraph_path + 'wiki-news-300d-1M.magnitude', # English Wikipedia 2017 16B	
embeddings_magnitude_modelEnglishWikipedia2017Subword16B = okgraph_path + 'wiki-news-300d-1M-subword.magnitude', # English Wikipedia 2017 + subword 16B	
embeddings_magnitude_modelCommonCrawl600B = okgraph_path + 'crawl-300d-2M.magnitude', # Common Crawl 600B	

embeddings_magnitude_modelT7 = okgraph_path + 'text7.head.magnitude'
try_times = 10
# seed_sizes = [(k, try_times) for k in [1, 2, 3, 5, 10, 20, 30, 40, 50]]


def is_similar(okg: okgraph.OKgraph, doesnt_match_word: list, words: list):
    """
    It answer to the question, is doesnt_match_word list similar to the words list?
    :param okg:
    :param doesnt_match_word:
    :param words:
    :return:
    """
    simil = []
    for w in words:
        simil.append(okg.v.similarity(doesnt_match_word, w))
    simil_avg = sum(simil)/len(simil)
    print(f'matching [{doesnt_match_word}] ? {words} - '
          f'result: [{simil_avg}]'
          f' : {simil_avg >= .45}')
    return simil_avg >= .45


def remove_words_doesnt_match(okg: okgraph.OKgraph, words: list, initial_guesses: list, max: int=1, verbose=False):
    to_remove = True
    w2 = words
    tmp_max = max
    if verbose:
        print(f'words: [{words}]')
        print(f'initial_guesses: [{initial_guesses}]')
    while to_remove and tmp_max > 0:
        tmp_max -= 1
        doesnt_match_word = okg.v.doesnt_match(w2)
        if doesnt_match_word not in initial_guesses and not is_similar(okg, doesnt_match_word, initial_guesses):
            w2.remove(doesnt_match_word)
            if verbose:
                print(f'removed word: [{doesnt_match_word}]')
            to_remove = True
        else:
            to_remove = False
    return w2



def generate_seed_sizes(max: int):
    out_ss = []
    list_to_use = [1,2,3,5]
    while True:
        for i in list_to_use:
            if i<max:
                max_comb = scipy.special.comb(max, i, exact=True)
                num_comb = 10 if 10<max_comb else max
                out_ss.append((i, num_comb))
            else:
                out_ss.append((max, 1))
                return out_ss
        list_to_use = [e*10 for e in list_to_use];

    # for i in range(max-(max%10), 9, -10):
    #     max_comb = scipy.special.comb(max, i, exact=True)
    #     out_ss.append((i, 10 if 10<max_comb else max ))
    # for i in [5,3,2,1]:
    #     if i<=max:
    #         max_comb = scipy.special.comb(max, i, exact=True)
    #         num_comb = 10 if 10<max_comb else max
    #         out_ss.append((i, num_comb))
    # return out_ss

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


def run_experiments(models: list,
                    ground_truths: list,
                    optim_algos: list,
                    objective_metrics: list,
                    lazy_loading: int = 0,
                    verbose: bool = False):

    n=1

    now = datetime.datetime.now()

    for embeddings_magnitude_model in models:
    
        if verbose:
            print(f'Loading model: {embeddings_magnitude_model}')
        okg = okgraph.OKgraph(corpus=corpus_file_path, embeddings=embeddings_magnitude_model, lazy_loading=lazy_loading)
        if verbose:
            print(f'Model: {embeddings_magnitude_model} LOADED')

        for ground_truth_name in ground_truths:
            ground_truth = load(ground_truth_name) # this will read the text file
            k_topn = len(ground_truth)
            if verbose:
                print(f'Getting the ground_truth and without_not_exists')
            ground_truth = [w.replace(" ", "_") for w in ground_truth]
            ground_truth_without_not_exists = [e for e in ground_truth if e in okg.v]
            ground_truth_missing_on_we = [e for e in ground_truth if e not in okg.v]
            ground_truth_missing_on_we_len = len(ground_truth_missing_on_we)
            
            
            if verbose:
                print(f'DONE. ground_truth=[{ground_truth}] and without_not_exists=[{ground_truth_without_not_exists}]')
            enable_most_similar_approx = False
            for one_model_name in ["GoogleNews-vectors-negative300", "glove.6B.300d", "glove.840B.300d", "wiki-news-300d-1M"]:
                enable_most_similar_approx = enable_most_similar_approx or (one_model_name in embeddings_magnitude_model)
            sklearn_metric_ap_score_enabled = False  # False can reduce computation time if sklearn not used in the objective_metrics

            if verbose:
                print(f'Getting the initial_guesses_list')
            seed_sizes = generate_seed_sizes(len(ground_truth_without_not_exists))
            initial_guesses_list = get_random_lists(ground_truth_without_not_exists, seed_sizes, verbose=verbose)#[0:1]
            if verbose:
                print(f'Done. initial_guesses_list={initial_guesses_list} ')

            if verbose:
                print(f'initial_guesses_list length = {len(initial_guesses_list)}')
                print(f'ground_truth_without_not_exists length = {len(ground_truth_without_not_exists)}')
                print(f'seed_sizes = {seed_sizes}')
                print(f'k_topn = {k_topn}')
                print(f'initial_guesses_list = {initial_guesses_list}')
                print(f'optim_algos = {optim_algos}')

            for initial_guesses in initial_guesses_list:
                
                if len(initial_guesses) <= 0:
                    print(f'ERROR: Cannot calculate with an initial guesses length equal to zero [{embeddings_magnitude_model}] [{ground_truth_name}].')
                    continue # go to the next iteration
            
                for optim_algo in optim_algos:
                    filename = f'results/res_{optim_algo}_{now.strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3]}.csv'
                    for objective_metric in objective_metrics:

                        dataset_info = {
                            "we_model": embeddings_magnitude_model,
                            "we_model_content": get_we_model_content_from_filename(embeddings_magnitude_model),
                            "topn": k_topn,
                            "k": k_topn,
                            "initial_guesses": initial_guesses,
                            "initial_guesses_length": len(initial_guesses),
                            "ground_truth_name": ground_truth_name,
                            "ground_truth": ground_truth,
                            "ground_truth_length": len(ground_truth),
                            "sklearn_metric_ap_score_enabled": sklearn_metric_ap_score_enabled,
                            "enable_most_similar_approx": enable_most_similar_approx,
                            "verbose": verbose,
                            "ground_truth_missing_on_we": ground_truth_missing_on_we,
                            "ground_truth_missing_on_we_len": ground_truth_missing_on_we_len,
                        }

                        dataset_info["optim_algo"] = optim_algo
                        dataset_info["objective_metric"] = objective_metric

                        args = {
                            "okg": copy.copy(okg), 
                            "dataset_info": copy.deepcopy(dataset_info), 
                            "choose_x0_closure": copy.deepcopy(choose_x0), 
                            "filename": copy.deepcopy(filename),
                            "verbose": verbose
                        }

                        globals()['thread_list'].append(threading.Thread(name="T" + str(n), target=get_optimum, args=(args,)))
                        thread_list = globals()['thread_list']

                        if verbose:
                            print(f'\n\n')
                        # else:
                        #     print(f'ENDED ONE OF [{len(initial_guesses)}] vectors '
                        #         f'optim_algo: [{optim_algo}] '
                        #         f'by using : [{objective_metric}] >>> [{filename}]')




def experiment_t1(args_dict):
    max_at_a_time = 1 if args_dict['max_at_a_time'] is None else args_dict['max_at_a_time']
    optim_algos_list = args_dict['optim_algos_list']
    ground_truths = args_dict['ground_truths']
    models = args_dict['models']
    verbose = args_dict['verbose']
    lazy_loading = args_dict['lazy_loading']
    print(f"Starting experiments... ["+str(optim_algos_list)+"] " )
    ## (k, n) make an expertiment with a list of "k" elements for "n" times
    run_experiments(models=models,
                    ground_truths=ground_truths,
                    optim_algos=optim_algos_list,
                    objective_metrics=['AP@k'],
                    lazy_loading=lazy_loading,
                    verbose=verbose)

    thread_list = globals()['thread_list']
    MAX_RUNNING_THREADS_AT_A_TIME = max_at_a_time
    TOTAL_THREADS_TO_RUN = len(thread_list)
    thread_list_index = 0
    tmp_chr = '\\'
    print(f"Starting [{TOTAL_THREADS_TO_RUN}] threads/experiments." )
    while (thread_list_index < len(thread_list)):
        time.sleep(0.5)
        l = threading.enumerate()
        while (thread_list_index < len(thread_list)) and (len(l) < MAX_RUNNING_THREADS_AT_A_TIME):
            thread_list[thread_list_index].start()
            thread_list_index = thread_list_index + 1
            l = threading.enumerate()

        tmp_chr = '\\' if tmp_chr == '/' else '/'
        print(f'\r Threads started [{thread_list_index}] {tmp_chr} {thread_list_index}/{TOTAL_THREADS_TO_RUN} threads ({len(l)} running)', end='')




print("STARTING")
print("STARTING")

thread_list = []
n=0
args = {
    "max_at_a_time": 20,
    "verbose": False,
    "optim_algos_list": [
        'powell', 
        'nelder-mead', 
        'BFGS', 
        # 'Newton-CG', 
        # 'CG', 
        # 'TNC', 
        # 'SLSQP', 
        # 'dogleg', 
        # 'trust-ncg',
        # 'COBYLA'
    ],
    "ground_truths": [
        'usa_states', 
        'universe_solar_planets', 
        'periodic_table_of_elements',
        # 'kings_of_rome',
    ],
    "models": [
        embeddings_magnitude_modelGN,
        # embeddings_magnitude_modelWikiGigawordGlove6B,
        # embeddings_magnitude_modelWikiGigawordGloveLemmatized6B,
        # embeddings_magnitude_modelCommonCrawlGlove840B,
        # embeddings_magnitude_modelTwitter27B,
        # embeddings_magnitude_modelEnglishWikipedia2017_16B,
        # embeddings_magnitude_modelEnglishWikipedia2017Subword16B,
        # embeddings_magnitude_modelCommonCrawl600B,
    ],
    "lazy_loading": 0   #  You can pass in an optional lazy_loading argument to the constructor with the value
                        #   -1 to disable lazy-loading and pre-load all vectors into memory (a la Gensim), 
                        #   0 (default) to enable lazy-loading with an unbounded in-memory LRU cache, or 
                        #   an integer greater than zero X to enable lazy-loading with an LRU cache that holds the X most recently used vectors in memory.
}

experiment_t1(args)

