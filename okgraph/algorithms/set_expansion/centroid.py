

def compute(okgraph, seed: [str] = None, options: dict = None, k: int = 5):
    seed2vec = okgraph.magnitude.query(seed)
    keys = []
    for key, vector in okgraph.magnitude:
        if vector in seed2vec:
            keys.append(key)
            print(len(keys))
            if len(keys) >= k:
                return keys
    return keys[:k]
    # keys = [key for key, vector in okgraph.magnitude if vector in okgraph.magnitude.query("prova")]
    # return list(filter(lambda key, vector: vector in self.magnitude.query(seed), self.magnitude))
