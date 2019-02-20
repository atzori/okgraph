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
    return okgraph_model.most_similar(seed_as_vec, topn=k)
