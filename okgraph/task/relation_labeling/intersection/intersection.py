from okgraph.sliding_windows import SlidingWindows
from okgraph.logger import logger

module_path = str.upper(__name__).replace("OKGRAPH.", "")


def task(seed: [(str,)], k: int, options: dict):
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
    # Get the task parameters
    logger.debug(f"{module_path}: Getting the parameters for the relation labeling of {seed}")
    dictionary: str = options.get("dictionary")
    index: str = options.get("index")

    # Get the SlidingWindows of every words tuple
    logger.debug(f"{module_path}: Start windowing of every pair in the seed")
    sliding_windows = \
        [SlidingWindows(tuple, corpus_dictionary_path=dictionary, corpus_index_path=index) for tuple in seed]

    # Get the labels of every pair of words from the related SlidingWindows objects
    logger.debug(f"{module_path}: Get the labels from every window")
    all_labels = []
    for window in sliding_windows:
        logger.debug(f"{module_path}: {window.target_words} window labels: {window.get_results().keys()}")
        all_labels.append({k: window.get_results()[k] for k in list(window.get_results())[:k]})

    # Evaluates the intersection of all the sets of labels
    logger.debug(f"{module_path}: Evaluates the intersection of the sets of labels")
    labels_intersection = set(all_labels[0].keys())
    for labels in all_labels:
        labels_intersection = set(labels_intersection) & set(labels.keys())

    labels_intersection = list(labels_intersection)
    logger.debug(f"{module_path}: Labels are {labels_intersection}")
    return labels_intersection
