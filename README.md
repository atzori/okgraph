[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

How to use the _okgraph_ library
================================

_okgraph_ is a python3 library that performs unsupervised [natural-language understanding](https://en.wikipedia.org/wiki/Natural-language_understanding) (NLU).

It currently focuses on the following tasks:

  - **set expansion** given one or a short set of words, continues this set with a list of other "same-type" words ([co-hyponyms](https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy#Co-hyponyms)) 
  - **relation expansion** given one or a short set of word pairs, continues this set with a list of pairs having the same implicit relation of the given pairs
  - **set labeling** given one or a short set of words, returns a list of short strings (labels) describing the given set (its type or [hyperonym](https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy))
  - **relation labeling** given one or a short set of word pairs, returns a list of short strings (labels) describing the relation in the given set

Being unsupervised, it only takes a free (untagged) text corpus as input, in any space-separated language. [scriptio-continua](https://en.wikipedia.org/wiki/Scriptio_continua) corpora and languages needs third-party tokenization techniques (e.g. [micter](https://github.com/tkng/micter)).

How to install
--------------
Please ensure you are using **python 3.7** (previous versions are not supported).
After cloning the repository (`git clone https://bitbucket.org/semanticweb/okgraph.git && cd okgraph`), run the followings **from the root directory of the project**:
```
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools devtools
pip install -r requirements.txt # this may take several minutes
python setup.py install
```

To download a little corpus and model for tests, run:
```
    source tests/get_corpus_and_model.sh
```

To run the tests, from the root directory, run:
```
    python -m unittest discover tests/
```

How to contribute
-----------
Tools that may be useful (not mandatory):

  - [hatch](https://github.com/ofek/hatch) ([commands reference](https://github.com/ofek/hatch/blob/master/COMMANDS.rst))
  - [git flow](https://github.com/nvie/gitflow) ([simple guide](https://jeffkreeftmeijer.com/git-flow/)) and also [git-flow-completion](https://github.com/bobthecow/git-flow-completion)

To send a contribution:

  - git checkout master
  - git flow init -d (to set the default settings)
  - git flow feature start *my-cool-feature* (use an appropriate feature name, for bugs use git flow bugfix start ...) 
  - add, commit and push your work (it will be in branch *feature/my-cool-feature*)
  - follow the link suggested after the push to create a new push request to "develop" branch and start a discussion with the maintainer
  - the maintainer will merge your work into develop (or master in case of new releases)


Loading a corpus
----------------
The first step is to create an _okgraph_ instance by loading a text corpus, and optionally a word-embedding model. 

    # DOWNLOAD CORPUS     
    # wget http://mattmahoney.net/dc/text8.zip && unzip text8.zip && mv text8 text8.txt

    # DOWNLOAD MODEL
    # wget http://magnitude.plasticity.ai/word2vec/medium/GoogleNews-vectors-negative300.magnitude
    

    import okgraph
    
    okg = okgraph.OKgraph('text8.txt', 'model_file')
    
    #or equivalently: 
    okg = okgraph.OKgraph(corpus='text8.txt', embeddings='model_file')

This example usage will set `text8.txt` as corpus file and `model_file.magnitude` as vector model file in [magnitude format](https://github.com/plasticityai/magnitude) (extension `magnitude` is appended automatically if not specified).

If the embeddings are not specified, okgraph will search a file with the same name of the corpus file with appended `.magnitude` (e.g. `text8.txt.magnitude`). If this file is not available, will be created using default options (word2vec, d=100, ...) and then loaded.


    okg = okgraph.OKgraph('text8.txt')
    > file text8.txt.magnitude not found, generating model with default options
    > model text8.txt.magnitude generated


If the second (embeddings) parameter is a url (starts with 'http://' or 'https://'), a [remote streamed version of the file](https://github.com/plasticityai/magnitude#remote-streaming-over-http) will be used.

Corpus preprocessing
--------------------
The classes and methods in `okgraph.preprocessing.*` are useful to parse and prepare a text corpus **before** creating the OKgraph instance. For instance, these methods helps, e.g., to convert MediaWiki format into cleaned free text used by okgraph. Some methods are also useful to make it lowercase, for stemming and for co-occurrence (n-gram) tokenization.

These functions are taken from the [Gensim preprocessing module](https://radimrehurek.com/gensim/parsing/preprocessing.html) and [Gensim phrases module](https://radimrehurek.com/gensim/models/phrases.html)


TBD


Tasks
-----
Every task has an associated method that take a _seed_ as first argument. It can be any iterable that produces  strings (e.g. a Set of strings, list of strings, generators, ...).
The second parameter is an integer `k`. If specified, a list of at most `k` strings is returned. Otherwise, a generator (potentially infinite) of strings is returned.


### Set Expansion ###
    okg.set_expansion(['Italy','France','Germany'])
    > returns a **generator** with results not containing the given seed 
    > user itertools to convert to a finite list (see https://stackoverflow.com/a/5234170)
    > e.g.: 'Spain','Portugal','Belgium', ... 

    okg.set_expansion(['Italy','France','Germany'], 5)
    okg.set_expansion(['Italy','France','Germany'], k=5)
    > returns a **list** of the specified size (5 in the example) with results not containing the given seed
    > e.g.: ['Spain','Portugal','Belgium','Greece','Denmark']


    okg.set_expansion(['Italy','France','Germany'], algo='depth',options={'n':5,'width':10}) # specify an algoritm


### Relation Expansion ###

    okg.rel_expansion([('Italy','Rome'),('Germany','Berlin')])
    > e.g.: ['Spain','Madrid'),('Belgium','Brussels')]

### Set Labeling ###

    okg.set_labeling(['Italy','France','Germany'])
    > e.g.: 'country','state','nation','football_team', ...

### Relation Labeling ###

    okg.rel_labeling([('Italy','Rome'),('Germany','Berlin')])
    > e.g.: 'capital','capital_of','soccer_team', ...

Evaluation
----------
Classes and methods in `okgraph.evaluation.*` evaluate the performance of algorithms in each task based on several benchmarks.

TBD


    
    
