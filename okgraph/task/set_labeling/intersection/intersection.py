from okgraph.sliding_windows import SlidingWindows
from okgraph.utils import logger

def task(seed: [str], k: int, options: dict):
    """
    Finds labels describing the relation between the words in the seed (hyperonym).
    This task is based on the distributional hypothesis: SlidingWindows objects are used to inspect the context of every
     seed word. A set of labels is obtained from every SlidingWindows object. The intersection of all the set of labels
     is the set of labels which describes the seed words.
    :param seed: list of words that has to be labeled
    :param k: limit to the number of results
    :param options: task options:
                     "dictionary" is the path of the corpus dictionary
                     "index" is the path of the indexed corpus files
    :return: the intersection of the labels describing the context of every seed word
    """
    # Get the task parameters
    logger.debug(f"Getting the parameters for the set labeling of {seed}")
    dictionary: str = options.get("dictionary")
    index: str = options.get("index")

    # Get the SlidingWindows of every word
    logger.debug(f"Start windowing of every word in the seed")
    sliding_windows = \
        [SlidingWindows((word,), corpus_dictionary_path=dictionary, corpus_index_path=index) for word in seed]

    # Get the labels of every pair of words from the related SlidingWindows objects
    logger.debug(f"Get the labels from every window")
    all_labels = []
    for window in sliding_windows:
        logger.debug(f"{window.target_words} window labels: {list(window.results_dict.keys())}")
        all_labels.append({k: window.results_dict[k] for k in list(window.results_dict)[:k]})

    # Evaluate the intersection of all the sets of labels
    logger.debug(f"Evaluate the intersection of the sets of labels")
    labels_intersection = set(all_labels[0].keys())
    for labels in all_labels:
        labels_intersection = set(labels_intersection) & set(labels.keys())

    labels_intersection = list(labels_intersection)
    logger.debug(f"Labels are {labels_intersection}")
    return labels_intersection
