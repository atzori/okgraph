from transformers import pipeline
import sys
from okgraph.utils import logger
from typing import List

def task(seed: List[str],
         k: int,
         boost = True,
         min_score = 0.001,
         model = "distilbert-base-uncased",
         pattern = lambda seed: ", ".join(seed)+" and [MASK]"        
         ):
    """ set expansion using masked models [default is distilbert base uncased]      

        k: number of gross results expected from distilbert, the algorithm filters duplicate results and symbols
        boost: boolean that if true allows to use a more precise version that uses hearst patterns
        min_score: precision parameter, minimun acceptable score for the result (default is 0.001 for best results)
        model: you can add any pre-trained bert-based model available from the hugginface library, only models trained for the "fill mask" task can be used here
        pattern: if the model isn't masked with token [MASK] you can adapt it changing with the token needed 
        or if the language used isn't english you can change and with another conjunction
            default set expansion -> "pattern" : lambda seed: ", ".join(seed)+" and [MASK]"
            to find set category you can change it with -> "pattern" : lambda seed: "[MASK] such as " +", ".join(seed)
        
               
        You can use it downloading your model, example:
            git clone https://huggingface.co/distilbert-base-uncased

        Then you have to import okgraph in this way: 
            from okgraph.core import OKgraph
            okg = OKgraph("tests/data/text9/text9.txt", dictionary_file="corpus_dictionary.npy")
        Even if it's not necessary for this model to work, you have to insert a courpus .txt and a dictionary file, they can be empty in this case (bug)

        Example of use:
        okg.set_expansion(
            seed = ('illinois', 'arizona', 'california'),
                algo = "fill_mask",
                k = 20,
                options = {}
            )
    """

    dis = pipeline('fill-mask', model = model, top_k = k)

    #filter
    def dub(couple):
        """check if the word is not in the set seed given in input and if the word is not a symbol"""
        alnum = False
        for i in couple["token_str"]:
            if i.isalnum():
                alnum = True
                break
        return couple["token_str"].lower() not in seed and alnum
    
    if boost == True:
        pattern = lambda seed: "[MASK]" + " such as " +", ".join(seed)
        categories = dis(pattern(seed))
        category = categories[0]["token_str"]
        pattern = lambda seed: category + " such as " +", ".join(seed)+" and [MASK]"

    """transformation from set to query"""
    query = pattern(seed)

    result = dis(query)

    new_words = []
    #saving distilbert results to a list
    for i in result:
        if(i["score"] >= min_score):
            new_words.append({
                "token_str": i["token_str"],
                "score": i["score"]
            })

    #removing unnecessary words and symbols
    new_words_filtered = list(filter(dub, [li for li in new_words]))

    return(new_words_filtered)

