from okgraph.embeddings import WordEmbeddings
from okgraph.utils import list_flatten, logger
from typing import List


def task(seed: List[str],
         k: int,
         embeddings: WordEmbeddings,
         width: int = 10,
         depth: int = 2,
         ) -> List[str]:
    """Finds words with the same implicit relation of the seed words
    (co-hyponyms).

    Conceptually, this task expands the seed in a way similar to a tree
    expansion.
    The root is connected to the seed words, so that every seed word defines a
    node in the first level of the tree.
    The tree can be expanded in depth: for every word in a level (node), its
    similar words can be found from the embeddings and used as its children,
    making the new level grow in width.
    A score is assigned to every word in the tree, according to how many nodes
    are being occupied by that word. The most scored words are the result of
    the expansion.

    Args:
        seed (List[str]): list of words that has to be expanded.
        k (int): limit to the number of result words.
        embeddings (WordEmbeddings): the word embeddings.
        width (int): max number of children for every word (node).
        depth (int): max height of the tree.

    Returns:
        List[str]: words similar to the words in the seed.

    """
    logger.info(f"Starting the set expansion of {seed}")

    words_in_level = list(seed)
    scores = {}
    for level in range(depth):
        logger.debug(
            f"Current level is {level+1},"
            f" {len(words_in_level)} words to expand")
        children = []
        for word in words_in_level:
            similar_words = embeddings.w2w(word, width)
            children.append(similar_words)
        words_in_new_level = list_flatten(children)
        for word in words_in_new_level:
            scores[word] = scores.get(word, 0) + 1
        words_in_level = words_in_new_level

    co_hyponyms = \
        [key for key in sorted(scores, key=scores.get, reverse=True)
         if key not in seed][:k]
    logger.info(f"Expansion is {co_hyponyms}")
    return co_hyponyms
