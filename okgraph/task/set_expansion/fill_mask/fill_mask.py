from transformers import pipeline
from okgraph.utils import logger
from typing import Callable, List


def task(seed: List[str],
         k: int,
         boost: bool = True,
         min_score: float = 0.001,
         model: str = "distilbert-base-uncased",
         pattern: Callable[[List], str] = lambda seed: ", ".join(
             seed) + " and [MASK]"
         ):
    """Finds words with the same implicit relation of the seed words
    (co-hyponyms).

    This task uses masked models (default is 'distilbert-base-uncased'),
     which may need to be downloaded to be used, such as:
    ```
    $ git clone https://huggingface.co/distilbert-base-uncased
    ```

    Args:
        seed (List[str]): list of words that has to be expanded.
        k (int): number of gross results expected from the model; the algorithm
         filters duplicate results and symbols.
        boost (bool): if 'True' allows to use a more precise version that uses
         hearst patterns.
        min_score (float): precision parameter, that is the minimun acceptable
         score for the result (default is '0.001' for best results).
        model (str): name of any pre-trained BERT-based model available from the
         hugginface library; only models trained for the 'fill mask' task can be
         used here.
        pattern (Callable[[List], str]): if the model isn't masked with token
         [MASK] you can adapt it changing with the token needed or if the
         language used isn't english you can change and with another conjunction
         default set expansion -> "pattern" : lambda seed: ", ".join(seed)+" and [MASK]"
         to find set category you can change it with -> "pattern" : lambda seed: "[MASK] such as " +", ".join(seed)
         TODO 'pattern' needs a better explanation

    Example:

        You have to create an 'OKgraph' object to execute this task. Even though
         it's not necessary for this model to work, you have to specify a '.txt
         courpus' and a 'dictionary file' to create the OKgraph instance.
        >>> from okgraph.core import OKgraph
        >>> okg = OKgraph("tests/data/text9/text9.txt",
        >>>               dictionary_file="corpus_dictionary.npy")
        >>> okg.set_expansion(
        >>>     seed = ['illinois', 'arizona', 'california'],
        >>>     algo = "fill_mask",
        >>>     k = 20,
        >>>     options = {}
        >>> )

    Returns:
        List[str]: words similar to the words in the seed.

    """
    logger.info(f"Starting the set expansion of {seed}")

    dis = pipeline('fill-mask', model=model, top_k=k)

    # Filter
    def dub(couple):
        """Check if the word is not in the set seed given in input and if the
         word is not a symbol."""
        alnum = False
        for i in couple["token_str"]:
            if i.isalnum():
                alnum = True
                break
        return couple["token_str"].lower() not in seed and alnum

    if boost == True:
        pattern = lambda seed: "[MASK]" + " such as " + ", ".join(seed)
        categories = dis(pattern(seed))
        category = categories[0]["token_str"]
        pattern = lambda seed: category + " such as " + ", ".join(
            seed) + " and [MASK]"

    # Transformation from set to query
    query = pattern(seed)

    result = dis(query)

    new_words = []
    # Saving model results to a list
    for i in result:
        if (i["score"] >= min_score):
            new_words.append({
                "token_str": i["token_str"],
                "score": i["score"]
            })

    # Removing unnecessary words and symbols
    new_words_filtered = list(filter(dub, [li for li in new_words]))

    return [w["token_str"] for w in new_words_filtered]
