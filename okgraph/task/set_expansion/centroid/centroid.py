from pymagnitude import Magnitude
from okgraph.logger import logger

module_path = str.upper(__name__).replace('OKGRAPH.', '')


def task(seed: [str], k: int, options: dict):
    """
    Finds words similar to the words in the seed (co-hyponyms).
    This task uses a "centroid" based method. The vector representation of the seed words are used to calculate their
     average vector (centroid). The vector model is then used to find the closest words to the centroid.
    :param seed: list of words that has to be expanded
    :param k: limit to the number of results in the set expansion
    :param options: task options:
                     "embeddings" is the words vector model (Magnitude)
    :return: the words closest to the centroid vector
    """
    # Get the task parameters
    logger.debug(f'{module_path}: Getting the parameters for the set expansion of {seed}')
    embeddings: Magnitude = options.get('embeddings')

    # Get the vector representation of every seed-word
    logger.debug(f'{module_path}: Use the embeddings to get the vector representation of every word in seed')
    v_seed = []
    for token in seed:
        v_seed.append(embeddings.query(token))

    # Calculates the centroid vector as the average vector of the seed vectors
    logger.debug(f'{module_path}: Calculates the centroid vector as the average vector of the seed vectors')
    v_centroid = mean(v_seed)

    # Return the vectors that are the most similar to the centroid
    # Take the top "k+len(seed)" results, because the seed can occur in the results and has to be removed
    logger.debug(f'{module_path}: Search in the embeddings for the vectors that are most similar to the centroid')
    co_hyponyms = [word for (word,score) in embeddings.most_similar(v_centroid, topn=(k + len(seed)))]

    # Remove the seed from the results (if eventually present)
    logger.debug(f'{module_path}: Removing the seed words {seed} from the results {co_hyponyms}')
    co_hyponyms = [word for word in co_hyponyms if word not in seed]
    # If the list is longer than the specified "k" length, cut the tail results
    co_hyponyms = co_hyponyms[:k]

    # Return the most similar words
    logger.debug(f'{module_path}: Expansion is {co_hyponyms}')
    return co_hyponyms


def mean(vectors: [[float]]):
    """
    Calculates the average vector from an array of numerical vectors.
    :param vectors: array of vectors
    :return: the average vector
    """
    return sum(vectors) / float(len(vectors))
