# From:
# http://www.enseignement.polytechnique.fr/informatique/profs/Michalis.Vazirgiannis/course_slides/71_text.pdf
# https://storage.googleapis.com/kaggle-forum-message-attachments/49733/1351/rank_eval.py
# a bit edited.

from math import log
from sklearn import metrics
import datetime
import numpy as np
import csv


class Metric:

    @staticmethod
    def precision(serp: [int]):
        """

        :param serp:
        :return:
        """
        l = []
        nr_docs_retrieved = 0
        nr_relevant_docs_retrieved = 0
        for rank in serp:
            nr_docs_retrieved += 1
            nr_relevant_docs_retrieved += rank
            l.append(nr_relevant_docs_retrieved/float(nr_docs_retrieved))
        return l

    @staticmethod
    def recall(serp: [int]):
        """

        :param serp:
        :return:
        """
        l = []
        nr_relevant_docs_retrieved = 0
        nr_relevant_docs = sum([rank for rank in serp])
        if nr_relevant_docs == 0:  # avoid divide by zero
            return None
        else:
            for rank in serp:
                nr_relevant_docs_retrieved += rank
                l.append(nr_relevant_docs_retrieved/float(nr_relevant_docs))
            return l

    @staticmethod
    def interpolated_precision(serp: [int], precisions: [float] = []):
        """

        :param serp:
        :param precisions:
        :return:
        """
        l = []
        for r in range(0, len(serp)):
            l.append(max(precisions[r:]))
        return l

    @staticmethod
    def avg_precision(serp: [int], precisions: [float] = []):
        """

        :param serp:
        :param precisions:
        :return:
        """
        avg_precisions = []
        for k in range(1, len(serp)+1):
            avg_precisions.append(sum(precisions[:k])/float(k))
        return avg_precisions

    @staticmethod
    def precision_at_k(k: int, precisions: [float]):
        """

        :param k:
        :param precisions:
        :return:
        """
        try:
            return precisions[k-1] 	# K starts at 1, list index starts at 0
        except:
            return None

    @staticmethod
    def cumulative_gain(serp: [int]):
        """

        :param serp:
        :return:
        """
        cg = 0
        for rank_rel in serp:
            cg += rank_rel
        return cg

    @staticmethod
    def discounted_cumulative_gain(serp: [int]):
        """

        :param serp:
        :return:
        """
        return sum([g/log(i+2) for (i,g) in enumerate([rank for rank in serp])])

    @staticmethod
    def ideal_discounted_cumulative_gain(serp: [int]):
        """

        :param serp:
        :return:
        """
        return sum([g/log(i+2) for (i,g) in enumerate(sorted([rank for rank in serp], reverse=True))])

    @staticmethod
    def normalized_discounted_cumulative_gain(serp: [int]):
        """

        :param serp:
        :return:
        """
        if float(Metric.ideal_discounted_cumulative_gain(serp)) == 0:
            return 0
        return Metric.discounted_cumulative_gain(serp) / float(Metric.ideal_discounted_cumulative_gain(serp))

    ################################################################
    ################################################################
    ################################################################
    ################################################################

    @staticmethod
    def calculate_y_scores(length: int):
        """

        :param length:
        :return:
        """
        out_list = [x/length for x in range(length+1)][1:]
        out_list.reverse()
        return out_list

    @staticmethod
    def get_print_and_save_calculus(filename: str,
                                    dataset_info: dict,
                                    results: [int],
                                    save_info_title: str = "",
                                    verbose: bool = False,
                                    save_results: bool = True):
        """

        :param filename:
        :param dataset_info:
        :param results:
        :param save_info_title:
        :param verbose:
        :return:
        """
        with open(filename, mode='a') as my_csv_data:
            if save_results:
                writer = csv.writer(my_csv_data, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            num_lines = sum(1 for _ in open(filename))
            dataset_info['INFO'] = save_info_title
            out_calc = Metric.calculate_all(dataset_info, results, verbose)

            if not save_results:
                return out_calc

            titles = []
            results = []
            title_before = ''
            for out_calc_key in out_calc['ORDERED_KEYS']:
                if title_before == 'objective_metric':
                    titles.append('objective_metric_result')
                else:
                    titles.append(out_calc_key)
                title_before = out_calc_key

                if out_calc.get(out_calc_key) is not None:
                    results.append(out_calc.get(out_calc_key))
                else:
                    results.append('')

            if num_lines == 0 and save_results:
                writer.writerow(titles)

            if save_results:
                writer.writerow(results)

        return out_calc

    @staticmethod
    def calculate_all(dataset_info: dict,
                      results: [int] = [],
                      verbose: bool = False):
        """

        :param dataset_info:
        :param results:
        :param verbose:
        :return:
        """
        now = datetime.datetime.now()
        objective_metric = dataset_info["objective_metric"]

        out_calc = dict(DATE=now.strftime("%Y-%m-%d %H:%M:%S"))

        for key in dataset_info.keys():
            out_calc[key] = dataset_info[key]

        out_calc['TrueP'] = results.count(1)
        out_calc['TrueF'] = results.count(0)
        out_calc['PRECISION'] = Metric.precision(results)
        out_calc['RECALL'] = Metric.recall(results)
        out_calc['Interp_Prec'] = Metric.interpolated_precision(results, out_calc['PRECISION'])
        out_calc['AP'] = Metric.avg_precision(results, out_calc['PRECISION'])
        out_calc['AP@k'] = out_calc['AP'][-1]
        out_calc['MAP'] = sum(out_calc['AP'])/len(out_calc['AP'])
        for actual_k in [5, 10, 20, 50, 100]:
            out_calc[f'AP@{actual_k}'] = out_calc['AP'][actual_k-1] if actual_k-1 < len(out_calc['AP']) else None
            out_calc[f'Pa{actual_k}'] = Metric.precision_at_k(actual_k, out_calc['PRECISION'])
        out_calc['CG'] = Metric.cumulative_gain(results)
        out_calc['DCG'] = Metric.discounted_cumulative_gain(results)
        out_calc['IDCG'] = Metric.ideal_discounted_cumulative_gain(results)
        out_calc['NDCG'] = Metric.normalized_discounted_cumulative_gain(results)

        sklearn_score_averages_keys = []
        if dataset_info['sklearn_metric_ap_score_enabled'] is True:
            y_trues = np.array(results)
            y_scores = np.array(Metric.calculate_y_scores(len(y_trues)))
            sklearn_metric_ap_score_algos = [None, 'weighted', 'micro', 'macro', 'samples']
            for sklearn_score_average in sklearn_metric_ap_score_algos:
                key = f'sklearn_metric_ap_score_{sklearn_score_average}'
                sklearn_score_averages_keys.append(key)
                out_calc[key] = metrics.average_precision_score(y_trues, y_scores,
                                                                average=sklearn_score_average,
                                                                pos_label=1,
                                                                sample_weight=None)

        out_calc['ORDERED_KEYS'] = ['DATE', 'Pa50', 'topn', 'k', 'initial_guesses_length', 'ground_truth_length', 'initial_guesses',
                                    'INFO', 'optim_algo', 'objective_metric', objective_metric,
                                    'Pa5', 'Pa10', 'Pa20', 'Pa50', 'AP@k', 'MAP']
        out_calc['ORDERED_KEYS'] += sklearn_score_averages_keys
        out_calc['ORDERED_KEYS'] += ['TrueP',
                                     'optim_message', 'nfev',
                                     'CG', 'DCG', 'IDCG', 'NDCG',
                                     'AP@5', 'AP@10', 'AP@20', 'AP@50', 'AP@100',
                                     'missing_words', 'wrong_words', 'we_model',
                                     'PRECISION', 'RECALL', 'Interp_Prec', 'the_most_similar_words', 'tot_time', 'ground_truth_name', 'exp_id']

        if verbose:
        #     print(f'{out_calc}', end='')
        # else:
            print(f'{out_calc["DATE"]} [{out_calc["TrueP"]}/{out_calc["ground_truth_length"]}] : '
                  f'\t{objective_metric} {out_calc[objective_metric]} '
                  f'\t{out_calc["INFO"]} '
                  f'\t{out_calc["tot_time"]} '
                  f'\tMAP {out_calc["MAP"]} '
                  f'\tPa50 {out_calc["Pa50"]} '
                  f'\tAP@50 {out_calc["AP@50"]} '
                  f'\tPa10 {out_calc["Pa10"]} ',
                  f'\tAP@10 {out_calc["AP@10"]} ', end='')

        return out_calc
