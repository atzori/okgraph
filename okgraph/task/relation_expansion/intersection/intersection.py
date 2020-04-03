from pymagnitude import Magnitude
from okgraph.core import algorithms_package


def task(embeddings: Magnitude, dictionary_path: str, index_path: str, options: dict):
    # Get the task's parameters
    seed: [(str, str)] = options.get('seed')
    k: int = options.get('k')

    # Import the algorithms for set expansion and relation labeling
    relation_labeling_package = algorithms_package + '.relation_labeling.intersection'
    relation_labeling_algorithm = getattr(__import__(relation_labeling_package, fromlist=['intersection']), 'intersection')
    set_expansion_package = algorithms_package + '.set_expansion.centroid'
    set_expansion_algorithm = getattr(__import__(set_expansion_package, fromlist=['centroid']), 'centroid')

    # Get the seed's labels
    seed_labels = relation_labeling_algorithm.task(dictionary_path, index_path, {'seed': seed, 'k': k})

    # Create two separate lists:
    #  one with the firsts elements of every pair in the seed list, one with the second elements
    first_elements = []
    second_elements = []
    for pair in seed:
        first_elements.append(pair[0])
        second_elements.append(pair[1])

    # Get the set expansion of the two created lists
    first_elements_expansion = [r[0] for r in set_expansion_algorithm.task(embeddings, {'seed': first_elements, 'k': k})]
    second_elements_expansion = [r[0] for r in set_expansion_algorithm.task(embeddings, {'seed': second_elements, 'k': k})]

    # Create a list of new possible pairs, combining the elements from the expansion of the first elements' set with
    #  the elements from the expansion of the second elements' set
    pairs_expansion = [(first, second) for first in first_elements_expansion for second in second_elements_expansion]

    # Get the labels for every created pair:
    #  if the labels of the new pair contains all the labels of the seed, save the label as an expansion of the seed
    labels_expansion = []
    for pair in pairs_expansion:
        pair_labels = relation_labeling_algorithm.task(dictionary_path, index_path, {'seed': [pair], 'k': k})
        if set(seed_labels) & set(pair_labels) == set(seed_labels):
            labels_expansion.append(pair)

    return labels_expansion
