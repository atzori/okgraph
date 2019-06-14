import okgraph
import random


from okgraph.task.set_expansion.optimum.ll_dataset import *
from okgraph.task.set_expansion.optimum.ll_optimize import *

print(f'Started')
np.seterr(divide='ignore', invalid='ignore')


# centroid
def mean(vector: [float]):
    return sum(vector) / float(len(vector))


def choose_x0(vectors):
    return mean(vectors)


def get_random_from(ground_truth, seed_size):
    out_list = set([])
    while len(out_list) < seed_size:
        out_list.add(random.choice(ground_truth))

    return list(out_list)

# Start


okgraph_path = 'models/'
corpus_file_path = okgraph_path + 'text7.head.gz'
embeddings_magnitude_modelGN = okgraph_path + 'GoogleNews-vectors-negative300.magnitude'
embeddings_magnitude_modelT7 = okgraph_path + 'text7.head.magnitude'
try_times = 10
seed_sizes = [(1, try_times),
              (2, try_times),
              (3, try_times),
              (5, try_times),
              (10, try_times),
              (20, try_times),
              (30, try_times),
              (40, try_times),
              (50, try_times)]


def is_similar(okg, doesnt_match_word, words):
    simil = []
    for w in words:
        simil.append(okg.v.similarity(doesnt_match_word, w))
    simil_avg = sum(simil)/len(simil)
    print(f'matching [{doesnt_match_word}] ? {words} - '
          f'result: [{simil_avg}]'
          f' : {simil_avg >= .45}')
    return simil_avg >= .45


def remove_words_doesnt_match(okg, words, initial_guesses, max=1):
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


def get_optim(embeddings_magnitude_model, initial_guesses, k,
              okg,
              choose_x0,
              filename,
              optim_algo='powell',
              objective_metric='AP@k',
              enable_most_similar_approx=False):

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




ground_truth = load('usa_states')
ground_truth = [w.replace(" ", "_") for w in ground_truth]
now = datetime.datetime.now()
filename = f'results/results_{now.strftime("%Y-%m-%d_%H:%M:%S")}.csv'
okg = okgraph.OKgraph(corpus=corpus_file_path, embeddings=embeddings_magnitude_modelGN)



#print(remove_words_doesnt_match(okg, ['Ohio', 'Nevada', 'Iowa', 'Florida', 'Nebraska', 'New_York', 'Missouri'], ['Ohio', 'Nevada']))
#is_similar(okg, 'Nevada', ['Ohio', 'Nevada', 'Iowa', 'Florida', 'Nebraska', 'New_York', 'Missouri'])



enable_most_similar_approx = "GoogleNews-vectors-negative300" in embeddings_magnitude_modelGN
for seed_size in [7, 6, 5, 4, 3, 2]:
    print(f'Actual seed_size {seed_size}')
    get_optim(embeddings_magnitude_modelGN,
              get_random_from(ground_truth, seed_size),
              k=50,
              okg=okg,
              choose_x0=choose_x0,
              filename=filename,
              optim_algo='powell',
              objective_metric='AP@k',
              enable_most_similar_approx=enable_most_similar_approx)
    print(f'Ended {seed_size}.')




def run_experiments():
    for embeddings_magnitude_model in [embeddings_magnitude_modelGN]:

        print(f'Loading model: {embeddings_magnitude_model}')
        okg = okgraph.OKgraph(corpus=corpus_file_path, embeddings=embeddings_magnitude_model)

        ground_truth = load('usa_states')
        ground_truth = [w.replace(" ", "_") for w in ground_truth]

        k_topn_list = [50]
        enable_most_similar_approx = "GoogleNews-vectors-negative300" in embeddings_magnitude_model
        sklearn_metric_ap_score_enabled = True  # False can reduce computation time if sklearn not used in the objective_metrics

        #initial_guesses_list = [['Alaska', 'Hawaii', 'New_Mexico', 'New_York', 'South_Carolina', 'Washington'],
        #                        ['Washington', 'New_York'],
        #                        ground_truth,
        #                        ['home', 'sun', 'dog'],
        #                        ]

        initial_guesses_list = []
        for seed_size in seed_sizes:
            seed_len = seed_size[0]
            seed_repetitions = seed_size[1]
            for _ in range(0, seed_repetitions):
                initial_guesses_list += [get_random_from(ground_truth, seed_len)]


        optim_algos = ['powell']#, 'nelder-mead', 'BFGS', 'Newton-CG', 'CG']
        #optim_algos = ['TNC', 'COBYLA', 'SLSQP', 'dogleg', 'trust-ncg']
        objective_metrics = ['AP@k']
        #objective_metrics = ['MAP', 'AP@k', 'sklearn_metric_ap_score_weighted', 'sklearn_metric_ap_score_macro',

        now = datetime.datetime.now()
        filename = f'results/results_{now.strftime("%Y-%m-%d_%H:%M:%S")}.csv'

        print(f'initial_guesses_list length = {len(initial_guesses_list)}')
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
