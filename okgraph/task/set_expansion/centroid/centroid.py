from pymagnitude import Magnitude


def mean(vector: [float]):
    return sum(vector) / float(len(vector))


def task(okgraph_model: Magnitude, options: dict = None):

    seed: [str] = options.get("seed")
    k: int = options.get("k")

    seed_vectors = []
    for token in seed:
        seed_vectors.append(okgraph_model.query(token))

    seed_as_vec = mean(seed_vectors)

    most_similar_vec = okgraph_model.most_similar(seed_as_vec, topn=(k+len(seed)))

    most_similar_vec = [result for result in most_similar_vec if result[0] not in seed]
    exceeding_results = len(most_similar_vec) - k
    if exceeding_results > 0:
        most_similar_vec = most_similar_vec[0 : len(most_similar_vec)-exceeding_results]

    return most_similar_vec