from pymagnitude import Magnitude


def mean(vector: [float]):
    """
    Calculates the average vector from an array of numerical vectors.
    :param vector: array of vectors
    :return: the average vector
    """
    return sum(vector) / float(len(vector))


def task(okgraph_model: Magnitude, options: dict = None):
    """
    Finds similar words (co-hyponyms) using a centroid vector based method.
    :param okgraph_model: vector model for representing words
    :param options: task option's: 'seed' and 'k'
    :return: the words that, according to the vector model, are the most similar to the seed-words
    """
    # Get the task's parameters
    seed: [str] = options.get("seed")
    k: int = options.get("k")

    # Get the vector representation of every seed-word: seed vector
    seed_vectors = []
    for token in seed:
        seed_vectors.append(okgraph_model.query(token))

    # Create the centroid vector as the average vector of the seed vectors
    centroid_vec = mean(seed_vectors)

    # Return the vectors that are the most similar to the centroid
    # Take the top 'k+len(seed)' results, because the seed tokens could occur in the results and they will be removed
    most_similar_vec = okgraph_model.most_similar(centroid_vec, topn=(k+len(seed)))

    # Remove the seed's tokens from the results (if eventually present)
    most_similar_vec = [result for result in most_similar_vec if result[0] not in seed]
    # If the list is longer than the specified 'k' length, cut the tail results
    most_similar_vec = most_similar_vec[:k]

    # Return the most similar vectors
    return most_similar_vec
