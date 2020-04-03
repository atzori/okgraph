from okgraph.sliding_windows import SlidingWindows


def task(dictionary_path: str, index_path: str, options: dict):
    """
    Finds labels describing the relation between all the seed of words.
    :param dictionary_path: path of the corpus' dictionary
    :param index_path: path of the indexed corpus' files
    :param options: task option's:
                     'seed' is the list of words' pairs that has labeled
                     'k' is the limit to the number of results
    :return: the labels describing the relation between the pairs of words
    """
    # Get the task's parameters
    seed: [(str, str)] = options.get("seed")
    k: int = options.get("k")

    # Get the SlidingWindow of every pair of words
    sliding_windows = \
        [SlidingWindows(pair, corpus_dictionary_path=dictionary_path, corpus_index_path=index_path) for pair in seed]

    # Get the labels of every pair of words from the related SlidingWindows objects
    labels = []
    for i in range(len(sliding_windows)):
        labels.append({k: sliding_windows[i].results[k] for k in list(sliding_windows[i].results)[:k]})

    # Evaluates the intersection of all the sets of labels
    # QSTN: this part of the algorithm can work with at least two pairs of words. Should work with one too
    labels_intersection = {}
    for i in range(len(labels)):
        if i == 0:
            labels_intersection = set(labels[i].keys()) & set(labels[i + 1].keys())
        elif i + 1 < len(labels):
            labels_intersection = set(labels_intersection) & set(labels[i + 1].keys())

    return labels_intersection
