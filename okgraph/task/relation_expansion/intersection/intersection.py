from okgraph.core import algorithms_package
from okgraph.utils import logger
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
    expansion" algorithms. Two lists are obtained from the first and second
    elements of the seed tuples. The two lists are expanded using the "set
    expansion" algorithm and the new elements from the two expansions are
    combined into new tuples. The "relation labeling" algorithm is used to
    obtain the labels of the seed and the labels of every new tuple. The new
    tuples described by a set of labels including the seed labels are
    considered an expansion of the seed.

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
        f"Creating list of first and second elements from the tuples in {seed}")
    first_elements = []
    second_elements = []
    for (first, second) in seed:
        first_elements.append(first)
        second_elements.append(second)
    logger.debug(f"Created lists:\n"
                 f"First {first_elements}\n"
                 f"Second {second_elements}")

    # Get the set expansion of the two created lists
    first_elements_expansion = \
        set_expansion_algorithm.task(first_elements,
                                     set_expansion_k,
                                     **set_expansion_options)
    second_elements_expansion = \
        set_expansion_algorithm.task(second_elements,
                                     set_expansion_k,
                                     **set_expansion_options)
    logger.debug(f"First elements expansion: {first_elements_expansion}")
    logger.debug(f"Second elements expansion: {second_elements_expansion}")

    # Create a list of new possible tuples, combining the elements from the
    # expansion of the first elements with the expansion of the second elements
    new_tuples = [(first, second)
                 for first in first_elements_expansion
                 for second in second_elements_expansion]
    logger.debug(f"Candidate new tuples: {new_tuples}")

    # Get the labels for every created tuple: if the labels of the new tuple
    # contains all the labels of the seed, save the tuple as an expansion of
    # the seed
    tuples_expansion = []
    for tuple in new_tuples:
        logger.debug(f"Analyzing tuple {tuple}")
        tuple_labels = \
            relation_labeling_algorithm.task([tuple],
                                             relation_labeling_k,
                                             **relation_labeling_options)
        logger.debug(f"Tuple {tuple} labels: {tuple_labels}")
        if set(seed_labels) & set(tuple_labels) == set(seed_labels):
            logger.debug(
                f"Valid tuple {tuple}:"
                f" seed labels {seed_labels} found in {tuple_labels}")
            tuples_expansion.append(tuple)
        else:
            logger.debug(
                f"Invalid tuple {tuple}:"
                f" seed labels {seed_labels} not found in {tuple_labels}")

    tuples_expansion = tuples_expansion[:k]
    logger.debug(f"Expansion is {tuples_expansion}")
    return tuples_expansion

