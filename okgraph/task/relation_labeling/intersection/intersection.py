from okgraph.sliding_windows import SlidingWindows


def task(seed: [(str, str)], k: int, options: dict):
    """
    Finds labels describing the relation between the pairs of words in the seed.
    This task is based on the distributional hypothesis: SlidingWindows objects are used to inspect the context of every
     seed pair. A set of labels is obtained from every SlidingWindows object. The intersection of all the set of labels
     is the set of labels which describes the seed pairs.
    :param seed: list of words pairs that has to be labeled
    :param k: limit to the number of results
    :param options: task options:
                     "dictionary" is the path of the corpus dictionary
                     "index" is the path of the indexed corpus files
    :return: the intersection of the labels describing the context of every seed pair
    """
    # Get the task parameters
    dictionary: str = options.get('dictionary')
    index: str = options.get('index')

    # Get the SlidingWindow of every pair of words
    sliding_windows = \
        [SlidingWindows([element for element in pair], corpus_dictionary_path=dictionary, corpus_index_path=index) for pair in seed]

    # Get the labels of every pair of words from the related SlidingWindows objects
    all_labels = []
    for i in range(len(sliding_windows)):
        all_labels.append({k: sliding_windows[i].results_dict[k] for k in list(sliding_windows[i].results_dict)[:k]})

    # Evaluates the intersection of all the sets of labels
    labels_intersection = set(all_labels[0].keys())
    for labels in all_labels:
        labels_intersection = set(labels_intersection) & set(labels.keys())

    return labels_intersection
