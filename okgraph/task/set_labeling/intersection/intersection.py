from okgraph.sliding_windows import SlidingWindows


def task(dictionary_path: str, index_path: str, options: dict):
    """
    Finds labels describing the relation between the words in the seed.
    :param dictionary_path: path of the corpus' dictionary
    :param index_path: path of the indexed corpus' files
    :param options: task option's:
                     'seed' is the list of words that has to be labeled
                     'k' is the limit to the number of results
    :return: the labels describing the relation between the pairs of words
    """
    # Get the task's parameters
    seed: [str] = options.get("seed")
    k: int = options.get("k")

    # Get the SlidingWindow of every pair of words
    sliding_windows = \
        [SlidingWindows([word], corpus_dictionary_path=dictionary_path, corpus_index_path=index_path) for word in seed]

    # Get the labels of every pair of words from the related SlidingWindows objects
    all_labels = []
    for i in range(len(sliding_windows)):
        all_labels.append({k: sliding_windows[i].results_dict[k] for k in list(sliding_windows[i].results_dict)[:k]})

    # Evaluates the intersection of all the sets of labels
    labels_intersection = set(all_labels[0].keys())
    for labels in all_labels:
        labels_intersection = set(labels_intersection) & set(labels.keys())

    return labels_intersection
