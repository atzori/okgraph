from pymagnitude import Magnitude
from okgraph.core import algorithms_package


def task(seed: [(str, str)], k: int, options: dict):
    """
    Finds pairs similar to the pairs in the seed.
    This task is based on the use of the "relation labeling" and "set expansion" algorithms. Two lists are obtained from
     the first and second elements of the seed pairs. The two lists are expanded using the "set expansion" algorithm and
     the new elements from the two expansions are combined into new pairs. The "relation labeling" algorithm is used to
     obtain the labels of the seed and the labels of every new pair. The new pairs described by a set of labels
     including the seed labels are considered an expansion of the seed.
    :param seed: list of words pairs that has to be expanded
    :param k: limit to the number of results
    :param options: task options:
                     "relation_labeling_algo" is the name of the algorithm that has to be used for pair labeling
                     "relation_labeling_options" is the dictionary of options for the chosen relation labeling algorithm
                     "relation_labeling_k" is the limit to the number of results obtained from the relation labeling
                     "set_expansion_algo" is the name of the algorithm that has to be used for set expansion
                     "set_expansion_options" is the dictionary of options for the chosen set expansion algorithm
                     "set_expansion_k" is the limit to the number of results obtained from the set expansion
    :return: the new pairs of words
    """
    # Get the task parameters
    relation_labeling_algo: str = options.get('relation_labeling_algo')
    relation_labeling_options: dict = options.get('relation_labeling_options')
    relation_labeling_k: int = options.get('relation_labeling_k')
    set_expansion_algo: str = options.get('set_expansion_algo')
    set_expansion_options: dict = options.get('set_expansion_options')
    set_expansion_k: int = options.get('set_expansion_k')

    # Import the algorithms for set expansion and relation labeling
    relation_labeling_package = algorithms_package + '.relation_labeling.' + relation_labeling_algo
    relation_labeling_algorithm = getattr(__import__(relation_labeling_package, fromlist=[relation_labeling_algo]), relation_labeling_algo)
    set_expansion_package = algorithms_package + '.set_expansion.' + set_expansion_algo
    set_expansion_algorithm = getattr(__import__(set_expansion_package, fromlist=[set_expansion_algo]), set_expansion_algo)

    # Get the seed labels
    seed_labels = relation_labeling_algorithm.task(seed, relation_labeling_k, relation_labeling_options)

    # Create two separate lists:
    #  one with the first elements of every pair in the seed list, one with the second elements
    first_elements = []
    second_elements = []
    for (first,second) in seed:
        first_elements.append(first)
        second_elements.append(second)

    # Get the set expansion of the two created lists
    first_elements_expansion = [res[0] for res in set_expansion_algorithm.task(first_elements, set_expansion_k, set_expansion_options)]
    second_elements_expansion = [res[0] for res in set_expansion_algorithm.task(second_elements, set_expansion_k, set_expansion_options)]

    # Create a list of new possible pairs, combining the elements from the expansion of the first elements with the
    #  expansion of the second elements
    new_pairs = [(first, second) for first in first_elements_expansion for second in second_elements_expansion]

    # Get the labels for every created pair:
    #  if the labels of the new pair contains all the labels of the seed, save the pair as an expansion of the seed
    pairs_expansion = []
    for pair in new_pairs:
        pair_labels = relation_labeling_algorithm.task([pair], relation_labeling_k, relation_labeling_options)
        if set(seed_labels) & set(pair_labels) == set(seed_labels):
            pairs_expansion.append(pair)

    return pairs_expansion[:k]
