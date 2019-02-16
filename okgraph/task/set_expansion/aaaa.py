
from pymagnitude import Magnitude
from tqdm import tqdm
import numpy as np


magnitude = Magnitude('tests/GoogleNews-vectors-negative300.magnitude', _number_of_values=5, log=True)

print('Started')

def avg_vector(vector_list):
    def avg(lst): return sum(lst) / float(len(lst))
    return avg(vector_list)


def compute(magnitude, seed: [str] = None, options: dict = None, k: int = 5):
    seed_vectors = []
    for token in tqdm(seed):
        all = magnitude.query(token)
        top_5 = np.mean(all[:5])
        all = set(top_5) & set(all[5:])
        seed_vectors.append(all)
    seed_as_vec = avg_vector(seed_vectors)
    return magnitude.most_similar(seed_as_vec, topn=k)


result_1 = compute(magnitude,
    seed=['milan', 'rome', 'turin'],
    options={'n': 5, 'width': 10},
    k=5)

print(result_1)

