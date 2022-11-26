[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

What is _okgraph_
======

_okgraph_ is a python3 library that performs unsupervised [natural-language understanding](
https://en.wikipedia.org/wiki/Natural-language_understanding) (NLU).

It currently focuses on the following tasks:

  - **set expansion** given one or a short set of words, continues this set with a list of other 'same-type' words
    ([co-hyponyms](https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy#Co-hyponyms));
  - **relation expansion** given one or a short set of word tuples, continues this set with a list of tuples having the 
    same implicit relation of the given tuples;
  - **set labeling** given one or a short set of words, returns a list of short strings (labels) describing the given 
    set (its type or [hyperonym](https://en.wikipedia.org/wiki/Hyponymy_and_hypernymy));
  - **relation labeling** given one or a short set of word tuples, returns a list of short strings (labels) describing 
    the implicit relation of the tuples in the given set.

Being unsupervised, it only takes a free (untagged) text corpus as input, in any space-separated language.
[Scriptio-continua](https://en.wikipedia.org/wiki/Scriptio_continua) corpora and languages needs third-party
tokenization techniques (e.g. [micter](https://github.com/tkng/micter)).

How to use the _okgraph_ library
======

How to install
------

### Creating the virtual environment ###
Please ensure **python 3.7** or newer is being used (previous versions are not supported). After cloning the repository 
(`https://github.com/atzori/okgraph.git && cd okgraph`), run the followings commands **from the root
directory of the project**.

- Linux _Bash_
    ```bash
    $ python3.7 -m venv venv
    $ source venv/bin/activate
    (venv) $ python -m pip install --upgrade pip setuptools devtools
    (venv) $ pip install -r requirements.txt  # this may take several minutes
    (venv) $ python setup.py install
    ```

- Windows 10 _cmd_ (ensure the _Windows 10 SDK_ have been installed by the _Visual Studio Installer_)
    ```bat
    > py -3.7 -m venv venv
    > .\venv\Scripts\activate
    (venv) > python -m pip install --upgrade pip setuptools devtools
    (venv) > pip install -r requirements.txt  & this may take several minutes
    (venv) > python setup.py install
    ```

### Acquiring the test data ###
To run the tests some _text corpora_ are required. The following script will provide the required _text corpora_, along
with their _word-embeddings_, _corpus indexes_ and _corpus dictionaries_:
```bash
$ python tests/get_test_corpus_and_resources.py
```
This procedure may take a while: it will download the _wiki-english-20171001_ corpus from the
[Gensim-data](https://github.com/RaRe-Technologies/gensim-data) and use it to generate three corpora: `text7.txt`,
`text8.txt` and `text9.txt` (obtained respectively from the first 10<sup>7</sup>, 10<sup>8</sup> and 10<sup>9</sup>
bytes of the _wiki-english-20171001_ corpus). Then, all the related resources (embeddings, index and dictionary) are
created. The brandly new corpora and their related resources can be found in `tests/data` from the project directory.

### Generating the docs ###
This library uses [Sphinx](https://www.sphinx-doc.org/en/master/) to automatically integrate the in-code comments
within the library documentation. To obtain the code documentation run the following:
```bash
$ python docs/make_docs.py
```
The README, modules and packages will be automatically parsed to obtain an _html_ documentation that can be found in
`docs/build/html/index.html` from the project directory.

Loading a corpus
------
The first step is to create an `OKgraph` instance by loading a _text corpus_. Any `OKgraph` instance will also require a 
_word-embedding model_, a _corpus index_ and a _corpus dictionary_, but specifying any value for them is optional.

The _word-embedding model_ can be obtained processing the specified _corpus_, or can be a pre-existent model.
The _word-embedding model_ is available through one of the extension of the `okgraph.embeddings.WordEmbeddings` abstract
class which introduces the _word-embeddings_ and the operations that can be done with them. The
`okgraph.embeddings.MagnitudeWordEmbeddings` class extends the `okgraph.embeddings.WordEmbeddings` class and provides
the currently one and only implementation available for the _word-embeddings_, using the
[Magnitude](https://github.com/plasticityai/magnitude) library.

The _corpus index_ and _corpus dictionary_ are strictly related to the _corpus_ itself and are always obtained from its
processing.

### Specifing a corpus ###
This example creates an `OKgraph` instance based on the `text8.txt` corpus:
```python
from okgraph.core import OKgraph
okg = OKgraph("text8.txt")
```
The file `text8.txt` will be set as _corpus file_ and the _word-embeddings_, _corpus index_ and _corpus dictionary_ will
be searched using their default values starting from the same directory of the _corpus_ (`text8.magnitude`, `indexdir/`
and `dictTotal.npy`). If found, they will be used as they are, otherwise they will be automatically generated processing
the _corpus_.

### Specifying a corpus and model ###
This example creates an `OKgraph` instance based on the `text8.txt` _corpus_ with a specified _word-embedding_ model:
```python
from okgraph.core import OKgraph
okg = OKgraph("text8.txt", "model_file")
```
or equivalently:
```python
import okgraph
okg = okgraph.OKgraph(corpus="text8.txt", embeddings="model_file")
```
The file `text8.txt` will be set as _corpus file_ and the _word-embeddings_ is searched, starting from the same
directory of the corpus, in a file named `model_file.magnitude` (when the extension is not provided, `.magnitude` is
automatically appended). If found, the _word-embeddings_ will be loaded, otherwise it will be generated processing the
_corpus_ and stored as `model_file.magnitude`.

If instead of a file name, the _word-embeddings_ argument is a URL (starts with 'http://' or 'https://'), a
[remote version of the file](https://github.com/plasticityai/magnitude#remote-streaming-over-http) will be used
(**to be fixed**).
```python
from okgraph.core import OKgraph
okg = OKgraph(corpus="text8.txt", embeddings="https://model_file")
```
The _stream_ argument allows to stream the _model_, instead of downloading it:
```python
from okgraph.core import OKgraph
okg = OKgraph(corpus="text8.txt", embeddings="https://model_file", stream=True)
```
### Specifying all the resources ###
This example creates an `OKgraph` instance based on the `text8.txt` _corpus_ with a specified value for the
_word-embeddings_, the _corpus index_ and _corpus dictionary_:
```python
from okgraph.core import OKgraph
okg = OKgraph(corpus="text8.txt", embeddings="model_file",
                      index_dir="corpus_index/", dictionary_file="corpus_dictionary.npy")
```
The file `text8.txt` will be set as _corpus file_ and the _word-embeddings_, _corpus index_ and _corpus dictionary_ will
be searched using their specified values starting from the same directory of the _corpus_. If found, they will be used
as they are, otherwise they will be automatically generated processing the _corpus_ and stored with the specified names.

### Forcing the resources generation ###
When an `OKgraph` instance is created loading a _corpus_, the _word-embeddings_, _corpus index_ and _corpus dictionary_
are searched starting from the corpus directory to be loaded as they are, if they exist. To avoid this resources to be
loaded as they are and force their re-generation from the _corpus_ processing, the _force_init_ argument can be set
to `True`:
```python
from okgraph.core import OKgraph
okg = OKgraph(corpus="text8.txt", force_init=True)
```
This code will force the `OKgraph` constructor to generate again the resources and overwrite them, if they exist.

Preparing a corpus
------
The classes and methods in `okgraph.preprocessing.*` are useful to parse and prepare a text corpus **before** creating 
the `OKgraph` instance. For instance, these methods helps, e.g., to convert _xml_ (MediaWiki) and _html_ format into
cleaned free text usable by _okgraph_.

Some methods are also useful to _make it lowercase_, for _stemming_, for _co-occurrence (n-gram) tokenization_,
_stop-words removal_ or _idioms identification_.

Some of these functions are taken from the [Gensim preprocessing module](https://radimrehurek.com/gensim/parsing/preprocessing.html) 
and [Gensim phrases module](https://radimrehurek.com/gensim/models/phrases.html)

(**to be done**)

Executing a task
------
Once an `OKgraph` object has been instantiated, it can be used to execute the four task using its four methods: 
`set_expansion()`, `relation_expansion()`, `set_labeling()` and `relation_labeling()`.

Every method takes four arguments:
- `seed`: is the generator used to compute the results. It will be: a list of strings in the `set_expansion` and
  `set_labeling` tasks; a list of string tuples in the `relation_expansion` and `relation_labeling` tasks;
- `k` (optional): is an integer specifying the limit to the number of results returned (setting it to `-1` will avoid
  the limit). It is set to `15` by default;
- `algo` (optional): is a string specifying the name of the algorithm chosen as implementation of the task. Every task
  has its own default algorithm with default arguments, so this argument can be optionally not specified along with
  the `options` argument;
- `options` (optional): is a dictionary of the type _{'argument': value}_ containing the values of the arguments requested by the
  chosen algorithm. Every task has its own default algorithm with default arguments, so this argument can be optionally
  not specified along with the `algo` argument;

To correctly execute a task it's important to know which implementations are available for each task, and so which
values can be assigned to the `algo` and `options` arguments. Every implementation has its own package, inside the
respective task package, containing a same-name module that implements the method `task()`. The `task()` method is
the one being called by the `OKgraph` instance with the unpacked dictionary of arguments `**options`.
The `algo` arguments has to be the name of one of the packages and the `options` arguments has to contain the packed 
arguments for the respective `task()` method.

### Executing a set expansion algorithm ###
All the _set expansion_ algorithms can be found in the `okgraph.task.set_expansion` package.

This is an example of usage with default values(using embeddings):
```python
from okgraph.core import OKgraph
okg = OKgraph(corpus="text9.txt")

okg.set_expansion(["Italy", "France", "Germany"])

> e.g.: ["Spain", "Portugal", "Belgium", ...]
```
And another example using a specific algorithm (using embeddings):
```python
from okgraph.core import OKgraph
okg = OKgraph(corpus="text9.txt")

okg.set_expansion(
    seed=["Italy", "France", "Germany"],
    k=15,
    algo='centroid_boost',
    options={"embeddings": okg.embeddings,
             "step": 2,
             "fast": False}
)

> e.g.: ["Spain", "Portugal", "Belgium", ...]
```

If you want to use a pretrained masked model you can use the `fill-mask` algorithm, for example:

```python
okg.set_expansion(
                seed = ('italy', 'france', 'germany'),
                k = 20,
                algo = "fill_mask",
                options = {}
            )
```

### Executing a relation expansion algorithm ###
All the _relation expansion_ algorithms can be found in the `okgraph.task.relation_expansion` package.

This is an example of usage with default values:
```python
from okgraph.core import OKgraph
okg = OKgraph(corpus="text9.txt")

okg.relation_expansion([("Italy", "Rome"), ("Germany", "Berlin")])

> e.g.: [("Spain", "Madrid"),("Belgium", "Brussels"), ...]
```
And another example using a specific algorithm:
```python
from okgraph.core import OKgraph
okg = OKgraph(corpus="text9.txt")

okg.relation_expansion(
    seed=[("Italy", "Rome"), ("Germany", "Berlin")],
    k=15,
    algo="centroid",
    options={"embeddings": okg.embeddings,
             "set_expansion_algo": "centroid",
             "set_expansion_options": {"embeddings": okg.embeddings},
             "set_expansion_k": 15}
)

> e.g.: [("Spain", "Madrid"),("Belgium", "Brussels"), ...]
```

### Executing a set labeling algorithm ###
All the _set labeling_ algorithms can be found in the `okgraph.task.set_labeling` package.

This is an example of usage with default values:
```python
from okgraph.core import OKgraph
okg = OKgraph(corpus="text9.txt")

okg.set_labeling(["Italy", "France", "Germany"])

> e.g.: ["country", "state", "nation", ...]
```
And another example using a specific algorithm:
```python
from okgraph.core import OKgraph
okg = OKgraph(corpus="text9.txt")

okg.set_labeling(
    seed=["Italy", "France", "Germany"],
    k=15,
    algo='intersection',
    options={"dictionary": okg.dictionary,
             "index": okg.index}
)

> e.g.: ["country", "state", "nation", ...]
```

### Executing a relation labeling algorithm ###
All the _relation labeling_ algorithms can be found in the `okgraph.task.relation_labeling` package.

This is an example of usage with default values:
```python
from okgraph.core import OKgraph
okg = OKgraph(corpus="text9.txt")

okg.relation_labeling([("Italy", "Rome"), ("Germany", "Berlin")])

> e.g.: ["capital", "capital_of", "soccer_team", ...]
```
And another example using a specific algorithm:
```python
from okgraph.core import OKgraph
okg = OKgraph(corpus="text9.txt")

okg.relation_labeling(
    seed=[("Italy", "Rome"), ("Germany", "Berlin")],
    k=15,
    algo='intersection',
    options={"dictionary": okg.dictionary,
             "index": okg.index}
)

> e.g.: ["capital", "capital_of", "soccer_team", ...]
```

Evaluation
------
The classes and methods in `okgraph.evaluation.*` evaluate the performance of algorithms in each task based on several 
benchmarks.

(**to be done**)

Embeddings operations
--------------------
The followings are examples of use of the embeddings in _okgraph_:
```python
from okgraph.core import OKgraph
okg = OKgraph(corpus="text9.txt")

# This is a WordEmbeddings class, specifically a MagnitudeWordEmbeddings
# instance
emb = okg.embeddings

# Obtain the vector representation of "town"
emb.w2v("town")

# Obtain the 5 closest word to "town"
emb.w2w("town", 5)

# Obtain the 5 words closest to the given vector representation (compatible with
# the embeddings)
v: ndarray
emb.v2w(v, 5)

# Obtain the 5 vector representations of the 5 words closest to the given vector
# representation (compatible with the embeddings)
v: ndarray
emb.v2v(v, 5)

```
More can be found in the `okgraph.embeddings` module.

How to contribute
======

Tools that may be useful (not mandatory):

  - [hatch](https://github.com/ofek/hatch) ([commands reference](https://github.com/ofek/hatch/blob/master/COMMANDS.rst))
  - [git flow](https://github.com/nvie/gitflow) ([simple guide](https://jeffkreeftmeijer.com/git-flow/)) and also 
    [git-flow-completion](https://github.com/bobthecow/git-flow-completion)

To send a contribution:

  - git checkout master
  - git flow init -d (to set the default settings)
  - git flow feature start *my-cool-feature* (use an appropriate feature name, for bugs use git flow bugfix start ...) 
  - add, commit and push your work (it will be in branch *feature/my-cool-feature*)
  - follow the link suggested after the push to create a new push request to "develop" branch and start a discussion 
    with the maintainer
  - the maintainer will merge your work into develop (or master in case of new releases)

Implementing a task
------
To implement a new task with an algorithm named `new_implementation`, the `new_implementation` package must be created
inside the task package. The `new_implementation` package will have to contain the `new_implementation.py` module
containing the `task()` method, where the `task()` method will effectively contain the algorithm.

For example, to implement a new _relation labeling_ algorithm named `my_rel_label_alg`, the following path will have
to exist: `okgraph/task/relation_labeling/my_rel_label_alg/my_rel_label_alg.py`.

Look at existing methods for practical examples, e.g.: `/tasks/set_expansion/centroid/centroid.py`

Documenting your work
------
All the project has been documented using the in-code [Google Style Python Docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html).
The comments are extracted by [Sphinx](https://www.sphinx-doc.org/en/master/) to automatically generate the library
documentation. To update the documentation to match your updates, run the following script:
```bash
$ python docs/make_docs.py
```

Testing
------
To run the tests, from the root directory, run:
```
python -m unittest discover tests/ -v
```
If the data required to run the tests has not been acquired yet, the
`tests/get_test_corpus_and_resources.py` script will be executed before testing.