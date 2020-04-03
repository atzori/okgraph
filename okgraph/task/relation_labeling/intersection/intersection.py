from okgraph.sliding_windows import SlidingWindows


def task(sliding_windows: [SlidingWindows], cut_results=15):
    """
    Finds labels describing the relation between all the pairs of words.
    :param sliding_windows: list of SlidingWindows objects, containing the labels for every related pair of words
    :param cut_results: max number of labels taken from every SlidingWindows object
    :return: the intersection of the sets of labels of every SlidingWindows object
    """
    # QSTN: no options with 'seed' and 'k' as described in the README
    # QSTN: README says this should work with one pair of words, but it won't

    # Get the labels of every pair of words from the related SlidingWindows objects
    labels = []
    for i in range(len(sliding_windows)):
        labels.append({k: sliding_windows[i].results[k] for k in list(sliding_windows[i].results)[:cut_results]})

    # Evaluates the intersection of all the sets of labels
    labels_intersection = {}
    for i in range(len(labels)):
        if i == 0:
            labels_intersection = set(labels[i].keys()) & set(labels[i + 1].keys())
        elif i + 1 < len(labels):
            labels_intersection = set(labels_intersection) & set(labels[i + 1].keys())

    return labels_intersection
