from pymagnitude import Magnitude


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
    embeddings: Magnitude = options.get('embeddings')

    # Get the vector representation of every seed-word
    v_seed = []
    for token in seed:
        v_seed.append(embeddings.query(token))

    # Create the centroid vector as the average vector of the seed vectors
    v_centroid = mean(v_seed)

    # Return the vectors that are the most similar to the centroid
    # Take the top "k+len(seed)" results, because the seed can occur in the results and has to be removed
    co_hyponyms = embeddings.most_similar(v_centroid, topn=(k + len(seed)))

    # Remove the seed from the results (if eventually present)
    co_hyponyms = [result for result in co_hyponyms if result[0] not in seed]
    # If the list is longer than the specified "k" length, cut the tail results
    co_hyponyms = co_hyponyms[:k]

    # Return the most similar words
    return co_hyponyms


def mean(vectors: [[float]]):
    """
    Calculates the average vector from an array of numerical vectors.
    :param vectors: array of vectors
    :return: the average vector
    """
    return sum(vectors) / float(len(vectors))
