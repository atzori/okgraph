from okgraph.utils import logger
from okgraph.file_converter import WordEmbedding


def task(seed: [str], k: int, options: dict):
    """
    Finds words similar to the words in the seed (co-hyponyms).
    TODO: add documentation
    """
    # Get the task parameters
    logger.info(f"Getting the parameters for the set expansion of {seed}")
    embeddings: WordEmbedding = options.get("embeddings")
    step: int = options.get("step", 2)
    fast: bool = options.get("fast", False)
    verbose: bool = options.get("verbose", False)

    current_seed = list(seed)
    while len(current_seed) < len(seed)+k:
        current_k = int(min(
            len(seed)+k-len(current_seed),
            step if not fast else len(current_seed)*step
        ))
        v_current_seed_centroid = embeddings.centroid(current_seed)
        candidate_new_words = embeddings.v2w(v_current_seed_centroid, len(current_seed)+current_k)
        new_words = [word for word in candidate_new_words if word not in current_seed]
        if not new_words:
            break
        new_words = new_words[:current_k]
        if verbose: logger.debug(f"current k: {current_k}, new words: {new_words}")
        for new_word in new_words:
            current_seed.append(new_word)

    expansion = current_seed[len(seed):]
    co_hyponyms = [word for word in expansion if word not in seed][:k]
    logger.info(f"Expansion is {co_hyponyms}")
    return co_hyponyms
