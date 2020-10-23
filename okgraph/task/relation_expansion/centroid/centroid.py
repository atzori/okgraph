from okgraph.core import ALGORITHMS_PACKAGE
from okgraph.embeddings import WordEmbeddings
from okgraph.utils import logger
from typing import Dict, List, Tuple


def task(seed: List[Tuple[str, ...]],
         k: int,
         embeddings: WordEmbeddings,
         set_expansion_algo: str,
         set_expansion_k: int,
         set_expansion_options: Dict
         ) -> List[Tuple[str, ...]]:
    """Finds tuples with the same implicit relation of the seed tuples.

    Every seed tuple is composed by a generic number of words whose meaning is
    strictly related to the position in the tuple.
    All the words in the same positions are collected in new lists.
    Every new list is expanded through a set expansion algorithm and new words
    are found as candidates for that position in new tuples.
    The relation between two tuple words in different positions can be
    expressed by their vector difference. These vector differences are obtained
    from the seed and used to validate the new tuples obtained combining the
    new words from the set expansion into new tuples.

    Args:
        seed (List[Tuple[str, ...]]): list of word tuples that has to be
            expanded.
        k (int): limit to the number of result tuples.
        embeddings (WordEmbeddings): the word embeddings.
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
    n_closest_words = 3

    # Import the algorithm for set expansion
    logger.debug(
        f"Importing set expansion algorithm {set_expansion_algo}")
    set_expansion_package = \
        ALGORITHMS_PACKAGE + ".set_expansion." + set_expansion_algo
    set_expansion_algorithm = \
        getattr(__import__(set_expansion_package,
                           fromlist=[set_expansion_algo]),
                set_expansion_algo)

    # Create the collection of words occupying the same position in the seed
    # tuple
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

    # Define the vector differences referring to the first word in the tuples
    all_diffs = [[] for _ in range(relation_size-1)]
    for t in seed:
        for i, j in zip(range(0, relation_size - 1), range(1, relation_size)):
            all_diffs[i] += [embeddings.w2v(t[j]) - embeddings.w2v(t[0])]

    centroid_diffs = []
    for diffs in all_diffs:
        centroid_diffs += [embeddings.centroidv(diffs)]

    # Create new tuples
    new_tuples = []
    for j, word in enumerate(seed_by_pos_expansion[0]):
        str_debug = f"Count {j}: word {word}"
        tuple_list = [word]
        for i, diff in zip(range(1, relation_size), centroid_diffs):
            new_words = embeddings.v2w(embeddings.w2v(word) + diff,
                                       n_closest_words)
            for new_word in new_words:
                str_debug += f", new word {i} {new_word}"
                if new_word in seed_by_pos_expansion[i]:
                    tuple_list += [new_word]
                    break
            if len(tuple_list) == i+1:
                break
        logger.debug(str_debug)
        if len(tuple_list) == relation_size:
            new_tuples += [tuple(tuple_list)]

    return new_tuples[:k]
