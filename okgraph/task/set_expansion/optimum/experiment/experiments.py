import okgraph
import random
import numpy

from dataset_helper import *
from optimization_core import *

print(f'Started')
np.seterr(divide='ignore', invalid='ignore')


# centroid
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
def get_random_lists(ground_truth: list, seed_sizes: [int]) -> [list]:
    """
    Get a list of random lists from the ground truth.
    :param ground_truth:
    :param seed_sizes: a list of tuples es. [(1, 10), (2, 10), (seed_len, seed_repetitions),...
    :return:
    """
    out_list = []
    out_list_strings = set([])
    seed_repetitions = 0

    for seed_size in seed_sizes:
        seed_len = seed_size[0]
        seed_repetitions += seed_size[1]
        print(f'\r seed_repetitions {seed_repetitions} ...', end='')
        while len(out_list) < seed_repetitions:
            to_add = get_random_from(ground_truth, seed_len)

            if "".join(to_add) not in out_list_strings:
                out_list += [to_add]
                out_list_strings.add("".join(to_add))

    print('seed_repetitions done.')
    return [l for l in out_list]


# Start


okgraph_path = 'models/'
corpus_file_path = okgraph_path + 'text7.head.gz'
embeddings_magnitude_modelGN = okgraph_path + 'GoogleNews-vectors-negative300.magnitude'
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


def remove_words_doesnt_match(okg: okgraph.OKgraph, words: list, initial_guesses: list, max: int=1):
    to_remove = True
    w2 = words
    tmp_max = max
    print(f'words: [{words}]')
    print(f'initial_guesses: [{initial_guesses}]')
    while to_remove and tmp_max > 0:
        tmp_max -= 1
        doesnt_match_word = okg.v.doesnt_match(w2)
        if doesnt_match_word not in initial_guesses and not is_similar(okg, doesnt_match_word, initial_guesses):
            w2.remove(doesnt_match_word)
            print(f'removed word: [{doesnt_match_word}]')
            to_remove = True
        else:
            to_remove = False
    return w2


def get_optim(embeddings_magnitude_model: str, initial_guesses: [int],
              okg: okgraph.OKgraph,
              choose_x0: callable,
              filename: str,
              ground_truth: [int],
              optim_algo: str = 'powell',
              objective_metric: str = 'AP@k',
              enable_most_similar_approx: bool = False):

    print(f'Computing x0 from [{len(initial_guesses)}] vectors '
          f'optim_algo: [{optim_algo}] '
          f'by using : [{objective_metric}]')

    while len(initial_guesses) < 50:
        i = len(initial_guesses)  # 7
        dataset_info = {
            "we_model": embeddings_magnitude_model,
            "topn": i+2,  # 9
            "k": i+2,
            "optim_algo": optim_algo,
            "initial_guesses": initial_guesses,
            "initial_guesses_length": len(initial_guesses),
            "ground_truth": initial_guesses,
            "ground_truth_length": len(initial_guesses),
            "sklearn_metric_ap_score_enabled": True,
            "objective_metric": objective_metric,
            "enable_most_similar_approx": enable_most_similar_approx
        }
        print(f'\n{i})) started')  # 7
        old_initial_guesses = initial_guesses
        print(f'old_initial_guesses: [{old_initial_guesses}]')

        initial_guesses = get_optimum(okg, dataset_info, choose_x0, filename)

        solution = initial_guesses['solution']
        the_most_similar_words = initial_guesses['the_most_similar_words']
        get_similar_and_save_results(okg, filename, solution,
                                     len(ground_truth),
                                     "ITERATION",
                                     ground_truth,
                                     dataset_info,
                                     enable_most_similar_approx=enable_most_similar_approx)
        print(f'the_most_similar_words: [{the_most_similar_words}]')

        the_most_similar_words = remove_words_doesnt_match(okg, the_most_similar_words, old_initial_guesses, max=1)  # 8

        initial_guesses = the_most_similar_words
        if initial_guesses == old_initial_guesses:
            return initial_guesses
        print(f'\n{i})) {initial_guesses}')

    return initial_guesses




# ground_truth = load('usa_states')
# ground_truth = [w.replace(" ", "_") for w in ground_truth]
# now = datetime.datetime.now()
# filename = f'results/results_{now.strftime("%Y-%m-%d_%H:%M:%S")}.csv'
# okg = okgraph.OKgraph(corpus=corpus_file_path, embeddings=embeddings_magnitude_modelGN)


#print(remove_words_doesnt_match(okg, ['Ohio', 'Nevada', 'Iowa', 'Florida', 'Nebraska', 'New_York', 'Missouri'], ['Ohio', 'Nevada']))
#is_similar(okg, 'Nevada', ['Ohio', 'Nevada', 'Iowa', 'Florida', 'Nebraska', 'New_York', 'Missouri'])


# enable_most_similar_approx = "GoogleNews-vectors-negative300" in embeddings_magnitude_modelGN
# for seed_size in [7, 6, 5, 4, 3, 2]:
#     print(f'Actual seed_size {seed_size}')
#     get_optim(embeddings_magnitude_modelGN,
#               get_random_from(ground_truth, seed_size),
#               okg=okg,
#               choose_x0=choose_x0,
#               ground_truth=ground_truth,
#               filename=filename,
#               optim_algo='powell',
#               objective_metric='AP@k',
#               enable_most_similar_approx=enable_most_similar_approx)
#     print(f'Ended {seed_size}.')


def run_experiments(models: list,
                    ground_truths: list,
                    optim_algos: list,
                    objective_metrics: list,
                    seed_sizes: list,
                    k_topn_list: list):

    for embeddings_magnitude_model in models:

        print(f'Loading model: {embeddings_magnitude_model}')
        okg = okgraph.OKgraph(corpus=corpus_file_path, embeddings=embeddings_magnitude_model)

        for ground_truth in ground_truths:
            ground_truth = [w.replace(" ", "_") for w in ground_truth]
            ground_truth_without_not_exists = [e for e in ground_truth if e in okg.v]

            enable_most_similar_approx = "GoogleNews-vectors-negative300" in embeddings_magnitude_model
            sklearn_metric_ap_score_enabled = True  # False can reduce computation time if sklearn not used in the objective_metrics

            initial_guesses_list = get_random_lists(ground_truth_without_not_exists, seed_sizes)

            now = datetime.datetime.now()
            filename = f'results/results_{now.strftime("%Y-%m-%d_%H:%M:%S")}.csv'

            print(f'initial_guesses_list length = {len(initial_guesses_list)}')
            print(f'ground_truth_without_not_exists length = {len(ground_truth_without_not_exists)}')
            print(f'seed_sizes = {seed_sizes}')
            print(f'k_topn_list = {k_topn_list}')
            print(f'initial_guesses_list = {initial_guesses_list}')
            print(f'optim_algos = {optim_algos}')
            print(f'filename = {filename}')

            total = len(optim_algos) * len(initial_guesses_list) * len(objective_metrics) * len(k_topn_list)
            tmp = total

            for k_topn in k_topn_list:
                for optim_algo in optim_algos:
                    for initial_guesses in initial_guesses_list:
                        for objective_metric in objective_metrics:

                            print(f'{tmp}) Computing x0 from [{len(initial_guesses)}] vectors '
                                  f'optim_algo: [{optim_algo}] '
                                  f'by using : [{objective_metric}]')

                            dataset_info = {
                                "we_model": embeddings_magnitude_model,
                                "topn": k_topn,
                                "k": k_topn,
                                "optim_algo": optim_algo,
                                "initial_guesses": initial_guesses,
                                "initial_guesses_length": len(initial_guesses),
                                "ground_truth": ground_truth,
                                "ground_truth_length": len(ground_truth),
                                "sklearn_metric_ap_score_enabled": sklearn_metric_ap_score_enabled,
                                "objective_metric": objective_metric,
                                "enable_most_similar_approx": enable_most_similar_approx
                            }
                            _ = get_optimum(okg, dataset_info, choose_x0, filename)
                            print(f'\n\n')
                            tmp -= 1

        print(f'Optimization ended')
        print(f'filename = {filename}')


# seed_sizes = [(k, 10) for k in [1, 2, 3, 5, 10, 20, 30, 40]]
# seed_sizes += [(48, 1)]
# print(seed_sizes)

# run_experiments(models=[embeddings_magnitude_modelGN],
#                 ground_truths=[load('usa_states')],
#                 optim_algos=['powell', 'nelder-mead', 'BFGS', 'Newton-CG', 'CG', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg'],
#                 objective_metrics=['AP@k', 'MAP', 'sklearn_metric_ap_score_weighted', 'sklearn_metric_ap_score_macro'],
#                 seed_sizes=seed_sizes,
#                 k_topn_list=[50])


def experiment_t1(sizes_list):

    seed_sizes = [(k, 10) for k in sizes_list]
    seed_sizes += [(48, 1)]
    print(seed_sizes)

    run_experiments(models=[embeddings_magnitude_modelGN],
                    ground_truths=[load('usa_states')],
                    optim_algos=['powell', 'nelder-mead', 'BFGS', 'Newton-CG', 'CG', 'TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg'],
                    objective_metrics=['AP@k', 'MAP', 'sklearn_metric_ap_score_weighted', 'sklearn_metric_ap_score_macro'],
                    seed_sizes=seed_sizes,
                    k_topn_list=[50])


import os                                                                       
from multiprocessing import Pool                                                

processes = ('process1.py', 'process2.py')                                    

def run_process(process):                                                             
    os.system('python {}'.format(process))                                       

pool = Pool(processes=3)                                                        
pool.map(run_process, processes)                                                     
