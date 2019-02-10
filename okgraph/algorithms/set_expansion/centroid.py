from okgraph import OKgraph
from okgraph import MathVectors
import tqdm


def compute(okgraph: OKgraph, seed: [str] = None, options: dict = None, k: int = 5):
    seed_vectors = []
    for token in tqdm(seed):
        seed_vectors.append(okgraph.magnitude.query(token))
    seed_as_vec = MathVectors.mean(seed_vectors)
    return okgraph.magnitude.most_similar(seed_as_vec, topn=k)
    # keys = [key for key, vector in okgraph.magnitude if vector in okgraph.magnitude.query("prova")]
    # return list(filter(lambda key, vector: vector in self.magnitude.query(seed), self.magnitude))
