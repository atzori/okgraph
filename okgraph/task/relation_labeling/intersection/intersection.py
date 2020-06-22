from okgraph.sliding_windows import SlidingWindows
from okgraph.utils import logger
from typing import List


def task(seed: List[str, ...],
         k: int,
         dictionary: str,
         index: str
         ) -> List[str, ...]:
    """

    Args:
        seed:
        k:
        dictionary:
        index:

    Returns:

    """
    """
    Finds labels describing the relation between the tuples in the seed.
    This task is based on the distributional hypothesis: SlidingWindows objects are used to inspect the context of every
     seed tuple. A set of labels is obtained from every SlidingWindows object. The intersection of all the set of labels
     is the set of labels which describes the seed tuples.
    :param seed: list of words tuples that has to be labeled
    :param k: limit to the number of results
    :param options: task options:
                     "dictionary" is the path of the corpus dictionary
                     "index" is the path of the indexed corpus files
    :return: the intersection of the labels describing the context of every seed pair
    """
    logger.info(f"Starting the relation labeling of {seed}")

    # Get the SlidingWindows of every words tuple
    logger.debug(f"Start windowing of every pair in the seed")
    sliding_windows = \
        [SlidingWindows(tuple, corpus_dictionary_path=dictionary, corpus_index_path=index) for tuple in seed]

    # Get the labels of every pair of words from the related SlidingWindows objects
    logger.debug(f"Get the labels from every window")
    all_labels = []
    for window in sliding_windows:
        logger.debug(f"{window._target_words} window labels: {window.get_results()}")
        all_labels.append(list(window.get_results())[:k])

    # Evaluates the intersection of all the sets of labels
    logger.debug(f"Evaluates the intersection of the sets of labels")
    labels_intersection = set(all_labels[0])
    for labels in all_labels[1:]:
        labels_intersection = set(labels_intersection) & set(labels)

    labels_intersection = list(labels_intersection)
    logger.debug(f"Labels are {labels_intersection}")
    return labels_intersection
