from okgraph.embeddings import WordEmbeddings, NotExistingWordException
from okgraph.sliding_windows import SlidingWindows
from okgraph.utils import logger
import operator
from typing import List

# FIXME: actually working like a set expansion algorithm and not as intended (a
#  set labeling algorithm). Needs improvements.

def task(seed: List[str],
         k: int,
         embeddings: WordEmbeddings,
         dictionary: str,
         index: str,
         min_cos: float = 0.80
         ) -> List[str]:
    """Finds labels describing the implicit relation between the seed words
    (hyperonym).

    Args:
        seed (List[str]): list of words that has to be labeled.
        k (int): limit to the number of labels.
        embeddings (WordEmbeddings): the word embeddings.
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
    all_win_labels_dict = []
    for window in sliding_windows:
        all_win_labels_dict.append(window.get_results_dict())

    # Evaluate the union of all the sets of labels
    logger.debug(f"Evaluate the union of the sets of labels")
    win_labels_union = set()
    for win_labels_dict in all_win_labels_dict:
        win_labels_union = \
            win_labels_union | win_labels_dict.keys()

    # Assign to the labels in the union the cosine similarity with the centroid
    # of the words in the seed
    labels_dict = {}
    centroid = embeddings.centroid(seed)
    for label in win_labels_union:
        try:
            cos_sim = embeddings.cosv(embeddings.w2v(label), centroid)
        except NotExistingWordException:
            cos_sim = 0
        if cos_sim >= min_cos:
            labels_dict[label] = cos_sim
        logger.debug(f"{label}: score={cos_sim}")

    # Sort the dictionary using the average cosine similarity
    labels_dict = {k: v for k, v in
                   sorted(labels_dict.items(),
                          key=operator.itemgetter(1),
                          reverse=True)
                   }

    labels = list(labels_dict)
    logger.debug(f"Labels are {labels}")
    return [label for label in labels if label not in seed][:k]
