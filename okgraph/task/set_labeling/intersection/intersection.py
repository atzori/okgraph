from okgraph.sliding_windows import SlidingWindows
from okgraph.utils import logger
from typing import List


def task(seed: List[str],
         k: int,
         dictionary: str,
         index: str
         ) -> List[str]:
    """Finds labels describing the implicit relation between the seed words
    (hyperonym).

    This task is based on the distributional hypothesis. SlidingWindows objects
    are used to inspect the context of every seed word. A set of labels is
    obtained from every SlidingWindows object. The intersection of all the set
    of labels is the set of labels which describes the seed.

    Args:
        seed (List[str]): list of words that has to be labeled.
        k (int): limit to the number of labels.
        dictionary (str): path of the corpus dictionary.
        index (str): path of the indexed corpus files.

    Returns:
        List[str]: labels describing the seed.

    """
    logger.info(f"Starting the set labeling of {seed}")

    # Get the SlidingWindows of every word
    logger.debug(f"Start windowing of every word in the seed")
    sliding_windows = \
        [SlidingWindows((word,),
                        corpus_dictionary_path=dictionary,
                        corpus_index_path=index)
         for word in seed]
    # TODO: could be useful to add the 'sliding_windows' parameters such
    #  'window_size' and 'noise_threshold' as a parameter of this task with
    #  default values

    # Get the labels of every pair of words from the related SlidingWindows
    # objects
    logger.debug(f"Get the labels from every window")
    all_labels = []
    for window in sliding_windows:
        logger.debug(
            f"{window.get_target_words()} window labels:"
            f" {list(window.get_results())}")
        all_labels.append(window.get_results(k))

    # Evaluate the intersection of all the sets of labels
    logger.debug(f"Evaluate the intersection of the sets of labels")
    labels_intersection = set(all_labels[0])
    for labels in all_labels[1:]:
        labels_intersection = set(labels_intersection) & set(labels)

    labels_intersection = list(labels_intersection)
    logger.debug(f"Labels are {labels_intersection}")
    return labels_intersection
