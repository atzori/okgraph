from okgraph.core import algorithms_package
from okgraph.utils import logger, tuple_combinations
from typing import Dict, List, Tuple


def task(seed: List[Tuple[str, ...]],
         k: int,
         relation_labeling_algo: str,
         relation_labeling_k: int,
         relation_labeling_options: Dict,
         set_expansion_algo: str,
         set_expansion_k: int,
         set_expansion_options: Dict
         ) -> List[Tuple[str, ...]]:
    """
    Finds tuples with the same implicit relation of the seed tuples.
    This task is based on the use of the "relation labeling" and "set
    expansion" algorithms. All the words in the same positions are collected in
    new lists. Every new list is expanded through a set expansion algorithm and
    new words are found to be combined into new tuples. The "relation labeling"
    algorithm is used to obtain the labels of the seed and the labels of every
    new tuple. The new tuples described by a set of labels including the seed
    labels are considered an expansion of the seed.

    Args:
        seed (List[Tuple[str, ...]]): list of word tuples that has to be
            expanded.
        k (int): limit to the number of result tuples.
        relation_labeling_algo (str): name of the chosen algorithm for the
            tuple labeling. The algorithm should be found in
            okgraph.task.relation_labeling.
        relation_labeling_k (int): limit to the number of results of the
            relation labeling algorithm.
        relation_labeling_options (Dict): dictionary containing the keyword
            arguments for the relation labeling algorithm.
        set_expansion_algo (str): name of the chosen algorithm for the
            set expansion. The algorithm should be found in
            okgraph.task.set_expansion.
        set_expansion_k (int): limit to the number of results of the
            set expansion algorithm.
        set_expansion_options (Dict): dictionary containing the keyword
            arguments for the set expansion algorithm.

    Returns:
        List[Tuple[str, ...]]: tuples similar to the tuples in the seed.

    """
    logger.info(f"Starting the relation expansion of {seed}")

    # Import the algorithms for relation labeling and set expansion
    logger.debug(
        f"Importing relation labeling algorithm {relation_labeling_algo}")
    relation_labeling_package = \
        algorithms_package + ".relation_labeling." + relation_labeling_algo
    relation_labeling_algorithm = \
        getattr(__import__(relation_labeling_package,
                           fromlist=[relation_labeling_algo]),
                relation_labeling_algo)

    logger.debug(
        f"Importing set expansion algorithm {set_expansion_algo}")
    set_expansion_package = \
        algorithms_package + ".set_expansion." + set_expansion_algo
    set_expansion_algorithm = \
        getattr(__import__(set_expansion_package,
                           fromlist=[set_expansion_algo]),
                set_expansion_algo)

    # Get the seed labels
    seed_labels = \
        relation_labeling_algorithm.task(seed,
                                         relation_labeling_k,
                                         **relation_labeling_options)
    logger.debug(f"Seed labels: {seed_labels}")

    # Create two separate lists: one with the first elements of every tuple in
    # the seed, one with the second elements
    logger.debug(
        f"Creating lists of words in the same position from the tuples in {seed}")
    relation_size = len(seed[0])
    seed_by_pos = [[] for _ in range(relation_size)]
    for t in seed:
        for i, word in zip(range(relation_size), t):
            seed_by_pos[i] += [word]

    # Expand the collection of words in the same position
    seed_by_pos_expansion = [[] for _ in range(relation_size)]
    for i, words in zip(range(relation_size), seed_by_pos):
        seed_by_pos_expansion[i] = \
            set_expansion_algorithm.task(words,
                                         set_expansion_k,
                                         **set_expansion_options)

    # Create a list of new possible tuples, by all the possible combinations of
    # elements from the seed expansions
    new_tuples = tuple_combinations(tuple(seed_by_pos_expansion))

    # Get the labels for every created tuple: if the labels of the new tuple
    # contains all the labels of the seed, save the tuple as an expansion of
    # the seed
    tuples_expansion = []
    for t in new_tuples:
        logger.debug(f"Analyzing tuple {t}")
        tuple_labels = \
            relation_labeling_algorithm.task([t],
                                             relation_labeling_k,
                                             **relation_labeling_options)
        logger.debug(f"Tuple {t} labels: {tuple_labels}")
        if set(seed_labels) & set(tuple_labels) == set(seed_labels):
            logger.debug(
                f"Valid tuple {t}:"
                f" seed labels {seed_labels} found in {tuple_labels}")
            tuples_expansion.append(t)
        else:
            logger.debug(
                f"Invalid tuple {t}:"
                f" seed labels {seed_labels} not found in {tuple_labels}")

    tuples_expansion = tuples_expansion[:k]
    logger.debug(f"Expansion is {tuples_expansion}")
    return tuples_expansion

