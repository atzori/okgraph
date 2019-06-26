from metrics_helper import *
import scipy.optimize as so
import okgraph
import numpy as np
import timeit


def get_similar_and_save_results(okg: okgraph.OKgraph,
                                 filename: str,
                                 vector: [float],
                                 topn: int,
                                 save_info_title: str,
                                 ground_truth: [str],
                                 dataset_info: dict,
                                 enable_most_similar_approx=False):
    """

    :param okg:
    :param filename:
    :param vector:
    :param topn:
    :param save_info_title:
    :param ground_truth:
    :param dataset_info:
    :param enable_most_similar_approx:
    :return:
    """
    if enable_most_similar_approx is True:
        the_most_similar = okg.v.most_similar_approx(vector, topn=topn)
    else:
        the_most_similar = okg.v.most_similar(vector, topn=topn)
    the_most_similar_words = [w for w, v in the_most_similar]
    dataset_info['solution'] = vector
    dataset_info['the_most_similar_words'] = the_most_similar_words
    dataset_info['missing_words'] = [item for item in ground_truth if item not in the_most_similar_words]
    dataset_info['wrong_words'] = [item for item in the_most_similar_words if item not in ground_truth]
    p_list = [1 if word in ground_truth else 0 for word in the_most_similar_words]
    return Metric.get_print_and_save_calculus(filename, dataset_info, p_list, save_info_title=save_info_title)


def create_objective(okg, ground_truth_set, ground_truth_set_vectors, dataset_info=dict(),
                     enable_most_similar_approx=False,
                     objective_metric=None,
                     verbose=False):
    globals()['iterations'] = 0
    globals()['max_avgp'] = 0
    k = dataset_info["topn"]

    def objective(x: [np.float64]): #-> numpy.float64
        globals()['iterations'] += 1
        iterations = globals()['iterations']

        if enable_most_similar_approx is True:
            neighbours = okg.v.most_similar_approx(x, topn=k)
        else:
            neighbours = okg.v.most_similar(x, topn=k)
        the_most_similar_words = [w for w, a in neighbours]
        true_false_list = [1 if word in ground_truth_set else 0 for word in the_most_similar_words]

        if verbose:
            print(f'...', end='')
        avgp = Metric.calculate_all(dataset_info, true_false_list, verbose=False)[objective_metric]

        max_avgp = globals()['max_avgp']
        if abs(avgp) > abs(max_avgp):
            max_avgp = abs(avgp)
        globals()['max_avgp'] = max_avgp

        if verbose:
            print(f'\r\t\t\t\t\tit: [{iterations}] - {objective_metric} {max_avgp} / actual[{abs(avgp)}] \t ({len(the_most_similar_words)}):{the_most_similar_words[:3]}...', end='')
        return 1-abs(avgp)

    return objective


def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der


def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H


def get_optimum(okg: okgraph.OKgraph, dataset_info: dict, choose_x0_closure: callable, filename: str, verbose=False):
    initial_guesses = dataset_info['initial_guesses']
    ground_truth = dataset_info['ground_truth']
    ground_truth_name = dataset_info['ground_truth_name']
    enable_most_similar_approx = dataset_info['enable_most_similar_approx']
    optim_algo = dataset_info['optim_algo']
    objective_metric = dataset_info['objective_metric']
    topn = dataset_info['topn']
    start = timeit.default_timer()

    if verbose:
        print(f'\t\t\t\t\t\t\t\tStarting optimization enable_most_similar_approx:{enable_most_similar_approx} of ({len(ground_truth)}) : {ground_truth[:3]}...')
    ground_truth_vectors = okg.v.query(ground_truth)
    objective = create_objective(okg, ground_truth, ground_truth_vectors,
                                 objective_metric=objective_metric,
                                 dataset_info=dataset_info,
                                 enable_most_similar_approx=enable_most_similar_approx,
                                 verbose=verbose)
    x0 = choose_x0_closure(okg.v.query(initial_guesses))
    if verbose:
        print(f'\t\t\t\t\t\t\t\tx0 = ({len(x0)}) : {x0[:3]}... ')
    initial_res = get_similar_and_save_results(okg, filename, x0, topn, "CENTROID", ground_truth, dataset_info,
                                               enable_most_similar_approx=enable_most_similar_approx)
    if verbose:
        print(f' most similar: {initial_res["the_most_similar_words"][:3]}...\n', end='')

    if optim_algo is None:
        solution = so.minimize(objective, x0)
    elif optim_algo in ['powell', 'CG', 'TNC', 'COBYLA', 'SLSQP', 'trust-ncg']:
        solution = so.minimize(objective, x0, method=optim_algo)
    elif optim_algo == 'nelder-mead':
        solution = so.minimize(objective, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    elif optim_algo in ['BFGS', 'dogleg']:
        solution = so.minimize(objective, x0, method='BFGS', jac=rosen_der, options={'disp': True})
    elif optim_algo == 'Newton-CG':
        solution = so.minimize(objective, x0, method='Newton-CG', jac=rosen_der, hess=rosen_hess, options={'xtol': 1e-8, 'disp': True})
    else:
        print(f'Optimization algo not found: [{optim_algo}].')
        return

    if not solution.success:
        if verbose:
            print(f'\tNOT SUCCESS. Doesn\'t converge! :( - {solution.message}')
    else:
        if verbose:
            print(f'\tSUCCESS :) - {solution.message} - but wait.....')

    dataset_info["optim_message"] = solution.message
    dataset_info["nfev"] = solution.nfev

    stop = timeit.default_timer()
    dataset_info["tot_time"] = stop - start

    optimized_res = get_similar_and_save_results(okg, filename, solution.x, topn, "OPTIMIZED", ground_truth, dataset_info,
                                                 enable_most_similar_approx=enable_most_similar_approx)

    missing_words = optimized_res["missing_words"]
    wrong_words = optimized_res["wrong_words"]
    if verbose:
        print(f'\nMissing ({len(missing_words)}): {missing_words}')
        print(f'\nWrong words ({len(wrong_words)}): {wrong_words}')

    if optimized_res[objective_metric] > initial_res[objective_metric]:
        if verbose:
            print(f'\n\t\tWINNER!!!!')
            print(f'\n\n\t\tWINNER!!!!  {initial_res[objective_metric]}')
            print(f'\n\n\t\tWINNER!!!!  \tVS')
            print(f'\n\n\t\tWINNER!!!!  {optimized_res[objective_metric]}')
            print(f'\n\n\t\tWINNER!!!!  Missing ({len(missing_words)}): {missing_words}')
            print(f'\n\n\t\tWINNER!!!!  Wrong ({len(wrong_words)}): {wrong_words}')
            print(f'\n\n\t\tWINNER!!!!\n\n\n\n')

    return optimized_res
