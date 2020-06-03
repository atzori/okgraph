from okgraph.utils import logger
from okgraph.file_converter import WordEmbedding

def task(seed: [(str, str)], k: int, options: dict):
    # Get the task parameters
    logger.debug(f"Getting the parameters for the relation expansion of {seed}")
    embeddings: WordEmbedding = options.get("embeddings")

    current_seed = list(seed)
    while len(current_seed) < len(seed)+k:
        current_k = 1
        fs = [f for f, s in current_seed]
        ss = [s for f, s in current_seed]
        centr_fs = embeddings.centroid(fs)
        centr_ss = embeddings.centroid(ss)

        new_f = [word for word in embeddings.v2w(centr_fs, len(current_seed)+1) if word not in fs][0]
        if not new_f:
            break
        new_s = embeddings.get4thv(centr_fs, centr_ss, embeddings.w2v(new_f))[0]
        if not new_s:
            break

        current_seed.append((new_f, new_s))

    expansion = current_seed[len(seed):]
    new_pairs = [tuple for tuple in expansion if tuple not in seed][:k]
    logger.info(f"Expansion is {new_pairs}")
    return new_pairs
