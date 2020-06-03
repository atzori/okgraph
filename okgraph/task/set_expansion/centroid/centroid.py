from okgraph.utils import logger
from okgraph.file_converter import WordEmbedding


def task(seed: [str], k: int, options: dict):
    """
    Finds words similar to the words in the seed (co-hyponyms).
    This task uses a 'centroid' based method. The vector representation of the seed words are used to calculate their
     average vector (centroid). The vector model is then used to find the closest words to the centroid.
    :param seed: list of words that has to be expanded
    :param k: limit to the number of results in the set expansion
    :param options: task options:
                     'embeddings' is the words vector model (Magnitude)
    :return: the words closest to the centroid vector
    """
    # Get the task parameters
    logger.info(f"Getting the parameters for the set expansion of {seed}")
    embeddings: WordEmbedding = options.get("embeddings")

    # Calculates the centroid vector as the average vector of the seed words
    v_centroid = embeddings.centroid(seed)

    # Return the vectors that are the most similar to the centroid
    # Take the top 'k+len(seed)' results, because the seed can occur in the results and has to be removed
    co_hyponyms = embeddings.v2w(v_centroid, (k + len(seed)))

    # Remove the seed from the results (if eventually present)
    co_hyponyms = [word for word in co_hyponyms if word not in seed]
    # If the list is longer than the specified 'k' length, cut the tail results
    co_hyponyms = co_hyponyms[:k]

    # Return the most similar words
    logger.info(f"Expansion is {co_hyponyms}")
    return co_hyponyms
