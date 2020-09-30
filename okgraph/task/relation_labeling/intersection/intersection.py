import numpy as np
from okgraph.sliding_windows import SlidingWindows
from okgraph.utils import logger
import operator
from typing import List, Tuple


def task(seed: List[Tuple[str, ...]],
         k: int,
         dictionary: str,
         index: str,
         min_score: float = 0.125
         ) -> List[str]:
    """Finds labels describing the implicit relation between the seed tuples.

    This task is based on the distributional hypothesis. SlidingWindows objects
    are used to inspect the context of every seed tuple. A set of labels is
    obtained from every SlidingWindows object. The intersection of all the set
    of labels is the set of labels which describes the seed tuples. The common
    labels are ordered using a score evaluated as their average TF-IDF statistic
    obtained from the different windows.

    Args:
        seed (List[Tuple[str, ...]]): list of word tuples that has to be
            labeled.
        k (int): limit to the number of labels.
        dictionary (str): path of the corpus dictionary.
        index (str): path of the indexed corpus files.
        min_score (float): minimum value of the average TF-IDF statistic to
            accept a label as a valid one.

    Returns:
        List[str]: labels describing the seed.
    
    """
    logger.info(f"Starting the relation labeling of {seed}")

    # Get the SlidingWindows of every words tuple
    logger.debug(f"Start windowing of every pair in the seed")
    sliding_windows = \
        [SlidingWindows(tuple,
                        corpus_dictionary_path=dictionary,
                        corpus_index_path=index)
         for tuple in seed]
    # TODO: could be useful to add the 'sliding_windows' parameters such
    #  'window_size' and 'noise_threshold' as a parameter of this task with
    #  default values

    # Get the labels of every word tuple from the related SlidingWindows
    # objects
    logger.debug(f"Get the labels from every window")
    all_win_labels_dict = []
    for window in sliding_windows:
        all_win_labels_dict.append(window.get_results_dict())

    # Evaluates the intersection of all the sets of labels
    logger.debug(f"Evaluate the intersection of the sets of labels")
    win_labels_intersection = all_win_labels_dict[0].keys()
    for win_labels_dict in all_win_labels_dict[1:]:
        win_labels_intersection = \
            win_labels_intersection & win_labels_dict.keys()

    # Assign to the labels in the intersections the average value of the TF-IDF
    # statistic obtained from the different windows.
    labels_dict = {}
    for label in win_labels_intersection:
        avg_tf_idf = np.mean(
            [win_labels_dict[label] for win_labels_dict in all_win_labels_dict]
        )
        if avg_tf_idf >= min_score:
            labels_dict[label] = avg_tf_idf
        logger.debug(f"{label}: score={avg_tf_idf}")

    # Sort the dictionary using the TF-IDF statistic
    labels_dict = {k: v for k, v in
                   sorted(labels_dict.items(),
                          key=operator.itemgetter(1),
                          reverse=True)
                   }

    labels = list(labels_dict)
    logger.debug(f"Labels are {labels}")
    return labels[:k]
