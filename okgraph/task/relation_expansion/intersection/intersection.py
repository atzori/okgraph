from okgraph.core import algorithms_package
from okgraph.logger import logger

module_path = str.upper(__name__).replace('OKGRAPH.', '')


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
    logger.debug(f'{module_path}: Getting the parameters for the relation expansion of {seed}')
    relation_labeling_algo: str = options.get('relation_labeling_algo')
    relation_labeling_options: dict = options.get('relation_labeling_options')
    relation_labeling_k: int = options.get('relation_labeling_k')
    set_expansion_algo: str = options.get('set_expansion_algo')
    set_expansion_options: dict = options.get('set_expansion_options')
    set_expansion_k: int = options.get('set_expansion_k')

    # Import the algorithms for relation labeling and set expansion
    logger.debug(f'{module_path}: Importing relation labeling algorithm {relation_labeling_algo}')
    relation_labeling_package = algorithms_package + '.relation_labeling.' + relation_labeling_algo
    relation_labeling_algorithm = getattr(__import__(relation_labeling_package, fromlist=[relation_labeling_algo]), relation_labeling_algo)
    logger.debug(f'{module_path}: Importing set expansion algorithm {set_expansion_algo}')
    set_expansion_package = algorithms_package + '.set_expansion.' + set_expansion_algo
    set_expansion_algorithm = getattr(__import__(set_expansion_package, fromlist=[set_expansion_algo]), set_expansion_algo)

    # Get the seed labels
    logger.debug(f'{module_path}: Extracting labels from seed')
    seed_labels = relation_labeling_algorithm.task(seed, relation_labeling_k, relation_labeling_options)
    logger.debug(f'{module_path}: Extracted labels: {seed_labels}')

    # Create two separate lists:
    #  one with the first elements of every pair in the seed list, one with the second elements
    logger.debug(f'{module_path}: Creating list of first and second elements from the pairs in {seed}')
    first_elements = []
    second_elements = []
    for (first,second) in seed:
        first_elements.append(first)
        second_elements.append(second)
    logger.debug(f'{module_path}: Created lists {first_elements} and {second_elements}')

    # Get the set expansion of the two created lists
    logger.debug(f'{module_path}: Expanding the new lists')
    first_elements_expansion = set_expansion_algorithm.task(first_elements, set_expansion_k, set_expansion_options)
    second_elements_expansion = set_expansion_algorithm.task(second_elements, set_expansion_k, set_expansion_options)
    logger.debug(f'{module_path}: Expansion of first elements list: {first_elements_expansion}')
    logger.debug(f'{module_path}: Expansion of second elements list: {second_elements_expansion}')

    # Create a list of new possible pairs, combining the elements from the expansion of the first elements with the
    #  expansion of the second elements
    new_pairs = [(first, second) for first in first_elements_expansion for second in second_elements_expansion]
    logger.debug(f'{module_path}: Possible new pairs: {new_pairs}')

    # Get the labels for every created pair:
    #  if the labels of the new pair contains all the labels of the seed, save the pair as an expansion of the seed
    pairs_expansion = []
    for pair in new_pairs:
        logger.debug(f'{module_path}: Analyzing pair {pair}')
        pair_labels = relation_labeling_algorithm.task([pair], relation_labeling_k, relation_labeling_options)
        logger.debug(f'{module_path}: Pair {pair} labels: {pair_labels}')
        if set(seed_labels) & set(pair_labels) == set(seed_labels):
            logger.debug(f'{module_path}: Valid pair {pair}: seed labels {seed_labels} are contained in {pair_labels}')
            pairs_expansion.append(pair)
        else:
            logger.debug(f'{module_path}: Invalid pair {pair}: seed labels {seed_labels} are not contained in {pair_labels}')

    pairs_expansion = pairs_expansion[:k]
    logger.debug(f'{module_path}: Expansion is {pairs_expansion}')
    return pairs_expansion
