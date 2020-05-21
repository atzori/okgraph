from pymagnitude import Magnitude
from okgraph.core import algorithms_package


def task(seed: [(str, str)], k: int, options: dict):
    """
    Finds pairs similar to the pairs in the seed.
    ......
    ......
    ......
    :param seed: list of words pairs that has to be expanded
    :param k: limit to the number of results
    :param options: task options:
                     "embeddings" is the words vector model (Magnitude)
                     "relation_labeling_algo" is the name of the algorithm that has to be used for pair labeling
                     "relation_labeling_options" is the dictionary of options for the chosen relation labeling algorithm
                     "relation_labeling_k" is the limit to the number of results obtained from the relation labeling
                     "set_expansion_algo" is the name of the algorithm that has to be used for set expansion
                     "set_expansion_options" is the dictionary of options for the chosen set expansion algorithm
                     "set_expansion_k" is the limit to the number of results obtained from the set expansion
    :return: the new pairs of words
    """
    # Get the task parameters
    embeddings: Magnitude = options.get("embeddings")
    relation_labeling_algo: str = options.get("relation_labeling_algo")
    relation_labeling_options: dict = options.get("relation_labeling_options")
    relation_labeling_k: int = options.get("relation_labeling_k")
    set_expansion_algo: str = options.get("set_expansion_algo")
    set_expansion_options: dict = options.get("set_expansion_options")
    set_expansion_k: int = options.get("set_expansion_k")

    # Import the algorithms for set expansion and relation labeling
    relation_labeling_package = algorithms_package + ".relation_labeling." + relation_labeling_algo
    relation_labeling_algorithm = getattr(__import__(relation_labeling_package, fromlist=[relation_labeling_algo]), relation_labeling_algo)
    set_expansion_package = algorithms_package + ".set_expansion." + set_expansion_algo
    set_expansion_algorithm = getattr(__import__(set_expansion_package, fromlist=[set_expansion_algo]), set_expansion_algo)

    # Get the seed labels
    seed_labels = relation_labeling_algorithm.task(seed, relation_labeling_k, relation_labeling_options)

    expansion = []
    i = 0
    while True:
        i += 1
        print('Iterazione', i)
        # Ottieni i primi elementi
        f_lista = [f for (f, ss) in seed+expansion]
        print('Primi elementi:', f_lista)

        # Ottieni i vettori differenza di rappresentazione delle due coppie
        v_differenza = [[y-x for (x, y) in zip(embeddings.query(pair[0]), embeddings.query(pair[1]))] for pair in seed+expansion]
        # Ottieni il centroide
        centroid = mean(v_differenza)

        # Ottieni una espansione del set di primi elementi
        f_espansione = set_expansion_algorithm.task(f_lista, set_expansion_k-len(expansion), set_expansion_options)
        print('Espansione primi elementi:', f_espansione)
        print('Espansione:', f_espansione)

        # Per ogni primo elemento, trova il possibile secondo elemento e certifica la coppia
        new_pair = None
        for f in f_espansione:
            print('f:', f)
            v_f = embeddings.query(f)

            # Ottieni il possibile secondo elemento
            v_s = [x+d for(x,d) in zip(v_f,centroid)]
            ss = [res[0] for res in embeddings.most_similar(v_s, topn=set_expansion_k)]
            print('Possibili secondi elmenti di', f, 'sono:', ss)

            for s in ss:
                if f == s:
                    continue
                try_pair = (f, s)
                print('Coppia prova:', try_pair)
                pair_labels = relation_labeling_algorithm.task([try_pair], relation_labeling_k, relation_labeling_options)
                if set(seed_labels) & set(pair_labels) == set(seed_labels):
                    print(try_pair, 'è una coppia valida')
                    new_pair = try_pair
                    expansion.append(new_pair)
                    break

            if new_pair is not None:
                break

        if new_pair is None or len(expansion) >= k:
            break

    return expansion


def mean(vectors: [[float]]):
    """
    Calculates the average vector from an array of numerical vectors.
    :param vectors: array of vectors
    :return: the average vector
    """
    return [sum([v[i] for v in vectors])/len(vectors) for i in range(0, len(vectors[0]))]
