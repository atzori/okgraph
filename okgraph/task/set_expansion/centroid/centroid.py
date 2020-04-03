from pymagnitude import Magnitude


def mean(vector: [float]):
    """
    Calculates the average vector from an array of numerical vectors.
    :param vector: array of vectors
    :return: the average vector
    """
    return sum(vector) / float(len(vector))


def task(embeddings: Magnitude, options: dict):
    """
    Finds similar words (co-hyponyms) using a centroid vector based method.
    :param embeddings: vector model for representing words
    :param options: task option's:
                     'seed' is the list of words that has to be expanded
                     'k' is the limit to the number of results
    :return: the extension (co-hyponyms) of the seed
    """
    # Get the task's parameters
    seed: [str] = options.get("seed")
    k: int = options.get("k")

    # Get the vector representation of every seed-word
    v_seed = []
    for token in seed:
        v_seed.append(embeddings.query(token))

    # Create the centroid vector as the average vector of the seed vectors
    v_centroid = mean(v_seed)

    # Return the vectors that are the most similar to the centroid
    # Take the top 'k+len(seed)' results, because the seed could occur in the results and has to be removed
    co_hyponyms = embeddings.most_similar(v_centroid, topn=(k + len(seed)))

    # Remove the seed from the results (if eventually present)
    co_hyponyms = [result for result in co_hyponyms if result[0] not in seed]
    # If the list is longer than the specified 'k' length, cut the tail results
    co_hyponyms = co_hyponyms[:k]

    # Return the most similar words
    return co_hyponyms
