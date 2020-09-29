from okgraph.embeddings import WordEmbeddings
from okgraph.utils import logger
from typing import List


def task(seed: List[str],
         k: int,
         embeddings: WordEmbeddings,
         step: int = 1,
         fast: bool = False,
         ) -> List[str]:
    """Finds words with the same implicit relation of the seed words
    (co-hyponyms).

    This task uses a 'centroid' based method. The vector representation of
    the seed words is used to calculate their average vector (centroid).
    The embeddings are then used to find the words closest to the centroid:
    some of these words are added to the seed and the centroid is updated to
    find other similar words. The iteration proceed until the seed is expanded
    with the requested number of similar words.

    Args:
        seed (List[str]): list of words that has to be expanded.
        k (int): limit to the number of result words.
        embeddings (WordEmbeddings): the word embeddings.
        step (int): specifies the number of words that are added to the seed at
            every iteration.
        fast (bool): if True, the number of words that are added to the seed at
            every iteration increases proportionally to the specified 'step' and
            the length of the expanded seed by the formula
            'len(expanded seed) * step'. If false, only #'step' words are added
            to the seed at every iteration.

    Returns:
        List[str]: words similar to the words in the seed.

    """
    logger.info(f"Starting the set expansion of {seed}")

    current_seed = list(seed)
    # Until the expansion is not greater than or equals to 'k'
    while len(current_seed) < len(seed)+k:
        # Define how many words have to be added to the seed at this iteration
        current_k = int(min(
            len(seed)+k-len(current_seed),
            step if not fast else len(current_seed)*step
        ))
        # Calculate the centroid of the current expanded seed
        v_current_seed_centroid = embeddings.centroid(current_seed)
        # Find the words closest to the centroid
        candidate_new_words = embeddings.v2w(v_current_seed_centroid,
                                             len(current_seed)+current_k)
        # Filter the closest words removing the ones that already are in the
        # expanded seed
        new_words = \
            [word for word in candidate_new_words if word not in current_seed]
        # If no new words have been found, the seed cannot be expanded anymore
        # and the algorithm stops
        if not new_words:
            break
        # If new words have been found, limit the number of new words to the
        # specified threshold for every iteration
        new_words = new_words[:current_k]
        logger.debug(f"Current k: {current_k}\n"
                     f"New words: {new_words}")
        # Add the new words to the expanded seed
        for new_word in new_words:
            current_seed.append(new_word)

    # Find the expansion of the seed and limit the number of results to the
    # specified threshold
    expansion = current_seed[len(seed):]
    co_hyponyms = [word for word in expansion if word not in seed][:k]
    logger.info(f"Expansion is {co_hyponyms}")
    return co_hyponyms
