from okgraph.utils import logger
from okgraph.embeddings import WordEmbeddings
from typing import List


def task(seed: List[str],
         k: int,
         embeddings: WordEmbeddings
         ) -> List[str]:
    """Finds words with the same implicit relation of the seed words
    (co-hyponyms).
    This task uses a 'centroid' based method. The vector representation of
    the seed words is used to calculate their average vector (centroid).
    The embeddings are then used to find the words closest to the centroid:
    these words are the seed expansion.

    Args:
        seed (List[str]): list of words that has to be expanded.
        k (int): limit to the number of result words.
        embeddings (WordEmbeddings): the word embeddings.

    Returns:
        List[str]: words similar to the words in the seed.

    """
    logger.info(f"Starting the set expansion of {seed}")

    # Calculates the centroid vector as the average vector of the seed words
    v_centroid = embeddings.centroid(seed)

    # Return the vectors that are the most similar to the centroid
    # Take the top 'k+len(seed)' results, because the seed can occur in the
    # results and has to be removed
    co_hyponyms = embeddings.v2w(v_centroid, (k + len(seed)))

    # Remove the seed from the results (if eventually present)
    co_hyponyms = [word for word in co_hyponyms if word not in seed]
    # If the list is longer than the specified 'k' length, cut the tail results
    co_hyponyms = co_hyponyms[:k]

    # Return the most similar words
    logger.info(f"Expansion is {co_hyponyms}")
    return co_hyponyms
