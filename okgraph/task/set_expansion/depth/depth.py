from itertools import chain
from numpy.linalg import norm
from okgraph.utils import logger
from pymagnitude import Magnitude
from scipy.spatial.distance import cosine as dist_cos,\
                                   euclidean as dist_euclidean


def task(seed: [str], k: int, options: dict):
    """
    Finds words similar to the words in the seed (co-hyponyms).
    TODO: add documentation
    """
    # Get the task parameters
    logger.info(f"Getting the parameters for the set expansion of {seed}")
    embeddings: Magnitude = options.get("embeddings")
    width: int = options.get("width", 10)
    depth: int = options.get("depth", 2)
    verbose: bool = options.get("verbose", False)

    root = seed[0]

    to_expand = list(seed)
    counts = {}
    for i in range(depth):
        logger.debug(f"current depth: {i+1}, words to expand: {len(to_expand)}")
        current = []
        for c in to_expand:
            logger.debug(f"c: {c}")
            v_c = embeddings.query(c)
            similar_to_c = embeddings.most_similar(v_c, topn=width)
            current.append(similar_to_c)
        current = list_flatten(current)
        for word in current:
            counts[word] = counts.get(word, 0) + d1(embeddings, root, word)

        to_expand = current

    co_hyponyms = [(key, counts[key]) for key in sorted(counts, key=counts.get, reverse=True)][:k]
    logger.info(f"Expansion is {co_hyponyms}")
    return co_hyponyms


def normalized(v):
    return v/norm(v)


def d1(e, w1, w2):
    return 1


def d2(e, w1, w2):
    return dist_cos(e.query(w1), e.query(w2))


def d3(e, w1, w2):
    return dist_euclidean(normalized(e.query(w1)), normalized(e.query(w2)))


def list_flatten(l):
    return list(chain.from_iterable(l))
