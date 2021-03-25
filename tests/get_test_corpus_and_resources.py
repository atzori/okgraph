"""This module contains a script to provide the corpus, embeddings, indexes and
dictionaries that are used in testing.

The data are provided by the 'gensim' library data. Specifically, the
'wiki-english-20171001' corpus is downloaded and used to create 3 other corpus:
'text9', 'text8' and 'text7', obtained extracting respectively the first 10^9,
10^8 and 10^7 bytes from 'wiki-english-20171001'. The corpus are available in
the specified folder inside the 'test' package. It generates automatically the
needed embeddings, indexes and dictionaries, with default values (as used in the
testing tasks).

Note: Other corpus and models can be found through the 'gensim' library, that
provides other useful data, as shown `here
<https://github.com/RaRe-Technologies/gensim-data>`_.
"""
import gensim.downloader as gensimapi
from gensim.downloader import BASE_DIR
from okgraph.core import OKgraph
import os
from os import makedirs, path
import re
import shutil

GENSIM_CORPUS: str = "wiki-english-20171001"
"""str: name of the 'gensim' data to download.
"""

KEEP_GENSIM_CORPUS: bool = False
"""bool: if True, store the 'gensim' data as a corpus, otherwise use it just
to generate the other smaller corpus. When set to True a ~17GB corpus will be
created, requiring the processing of ~5 million articles and a lot of time.
"""

GENSIM_PATH: str = path.normpath(BASE_DIR)
"""str: path in which the 'gensim' data downloader stores the data.
"""

TEST_DATA_FOLDER = path.normpath("data")
"""str: location in which store the various test corpus and resources.
"""

TEST_SMALL_CORPUS = "text7.txt"
TEST_MEDIUM_CORPUS = "text8.txt"
TEST_BIG_CORPUS = "text9.txt"


def main():
    # Save the current working directory
    _cwd_backup = path.normpath(os.getcwd())
    # Set the current working directory to be the one containing this script
    cwd = path.dirname(path.normpath(__file__))
    os.chdir(cwd)

    print(f"\nRunning script '{path.normpath(__file__)}' from '{cwd}'.")

    bar_char = "\u2588"
    bar_length = 50

    print(f"Checking the existence of the test data.")

    # If the directory that will contain the test corpus doesn't exist,
    # create it
    base = path.normpath(TEST_DATA_FOLDER)
    if not path.exists(base):
        print(f"'{base}' folder not existing. Creating '{base}' folder.")
        makedirs(base)
    else:
        print(f"'{base}' folder already existing.")

    # Corpus data
    corpus_data = {TEST_SMALL_CORPUS:  {"size": 10 ** 7},
                   TEST_MEDIUM_CORPUS: {"size": 10 ** 8},
                   TEST_BIG_CORPUS:    {"size": 10 ** 9}}
    biggest_corpus = \
        max(map(lambda x: (corpus_data[x]["size"], x),
                list(corpus_data.keys()))
            )[1]

    # Check which corpus are already existing and which ones have to be
    # generated
    for corpus in corpus_data.keys():
        corpus_name, _ = path.splitext(corpus)
        corpus_folder = path.normpath(path.join(base, corpus_name))
        corpus_file = path.normpath(path.join(corpus_folder, corpus))
        corpus_data[corpus]["file"] = corpus_file
        if not path.exists(corpus_file):
            corpus_data[corpus]["to_gen"] = True
            # Create the folder to contain corpus and resources if not
            # existing
            makedirs(corpus_folder, exist_ok=True)
        else:
            print(f"'{corpus}' file already existing.")
            corpus_data[corpus]["to_gen"] = False

    # Check if the 'gensim' data it's needed, because explicitly expressed or
    # the biggest corpus is missing
    if not KEEP_GENSIM_CORPUS:
        need_gensim_data = corpus_data[biggest_corpus]["to_gen"]
    else:
        need_gensim_data = True

    # Download the 'wiki-english-20171001' if it's not existing and it's
    # needed
    wiki_corpus = f"{GENSIM_CORPUS}.txt"
    wiki_corpus_name, _ = path.splitext(wiki_corpus)
    wiki_corpus_folder = path.normpath(path.join(base, wiki_corpus_name))
    wiki_corpus_file = path.normpath(path.join(wiki_corpus_folder, wiki_corpus_name))
    if not path.exists(wiki_corpus_file) and need_gensim_data:
        print(f"'{wiki_corpus}' file not existing.")
        # Create the folder, if not existing, to contain the wiki_corpus
        makedirs(wiki_corpus_folder, exist_ok=True)

        print(f"Downloading {wiki_corpus_name} from 'gensim' source data.")
        wiki_corpus_f = gensimapi.load(wiki_corpus_name)
        print(f"Download completed.")

        with open(wiki_corpus_file, "wb") as fb_wiki:
            written_bytes = 0
            N_ARTICLES = 4924893  # Obtained iterating through the dataset
            for i, wiki_article in enumerate(wiki_corpus_f):
                # Join the article sections in a unified string
                k_txt = "section_texts"
                k_ttl = "section_titles"
                excluded_titles = ["See also", "References", "Further reading",
                                   "External links"]
                sections_text = \
                    [text for text, title
                     in zip(wiki_article[k_txt], wiki_article[k_ttl])
                     if title not in excluded_titles]
                sections_text = " ".join(sections_text)

                # Make the string plain
                cleaned_text = sections_text + " "
                cleaned_text = re.sub(r"[\n\t']", " ", cleaned_text)
                cleaned_text = re.sub(r"[^\w\s]", "", cleaned_text)
                cleaned_text = re.sub(r"[\s]+", " ", cleaned_text)
                cleaned_text = re.sub(r"^\s", "", cleaned_text)
                cleaned_text = cleaned_text.lower()

                # Convert to bytes
                text_bytes = str.encode(cleaned_text, encoding="utf-8")
                written_bytes += len(text_bytes)

                # Write the bytes in the wiki corpus
                fb_wiki.write(text_bytes)

                # Show the progresses for every article
                if KEEP_GENSIM_CORPUS:
                    done_percentage = i / N_ARTICLES
                    done_bar = int(bar_length * done_percentage)
                    print(f"\r{done_percentage * 100:4.1f}% "
                          f"|{bar_char * done_bar}{' ' * (bar_length - done_bar)}|"
                          f" = {i}/{N_ARTICLES} articles",
                          end='')
                # Go on until the minimum number of bytes needed is reached
                else:
                    limit_reached = \
                        written_bytes >= corpus_data[biggest_corpus]["size"]
                    if i % 1000 == 0 or limit_reached:
                        print(f"\rRead articles: {i+1}.", end="")
                    if limit_reached:
                        break
            print()
        print(f"Removing the 'gensim' source data folder {GENSIM_PATH}.")
        shutil.rmtree(f"{GENSIM_PATH}")
    elif need_gensim_data:
        print(f"'{wiki_corpus}' file already existing.")

    # Check if the biggest corpus has to be generated
    if corpus_data[biggest_corpus]["to_gen"]:
        print(f"'{corpus_data[biggest_corpus]['file']}' corpus not existing.")
        # Open the 'wiki-english-20171001' corpus
        with open(wiki_corpus_file, "rb") as fb_wiki:
            # Open the corpus and write the needed bytes
            corpus = biggest_corpus
            with open(corpus_data[corpus]["file"], "wb") as fb_corpus:
                print(f"Writing the {corpus_data[corpus]['file']} corpus"
                      f" from {wiki_corpus_file}.")
                fb_corpus.write(fb_wiki.read(corpus_data[corpus]["size"]))
    else:
        print(f"'{corpus_data[biggest_corpus]['file']}' corpus already existing.")

    # Delete the wiki corpus if not needed
    if not KEEP_GENSIM_CORPUS and path.exists(wiki_corpus_folder):
        print(f"Removing the 'gensim' corpus folder {wiki_corpus_folder}.")
        shutil.rmtree(f"{wiki_corpus_folder}")

    # Open the corpus files and write the needed bytes
    for corpus in corpus_data:
        if corpus_data[corpus]["to_gen"] and corpus != biggest_corpus:
            print(f"'{corpus_data[corpus]['file']}' corpus not existing.")
            with open(corpus_data[corpus]["file"], "wb") as fb_corpus, \
                 open(corpus_data[biggest_corpus]["file"], "rb") as fb_big_corpus:
                print(f"Writing the {corpus_data[corpus]['file']} corpus"
                      f" from {corpus_data[biggest_corpus]['file']}.")
                fb_corpus.write(fb_big_corpus.read(corpus_data[corpus]["size"]))
        elif corpus != biggest_corpus:
            print(f"'{corpus_data[corpus]['file']}' corpus already existing.")

    # Process every corpus to obtain the related resources:
    # embeddings, corpus index and dictionary
    for corpus in corpus_data:
        print(f"Checking for the '{corpus}' resources.")
        corpus_file = corpus_data[corpus]["file"]
        OKgraph._get_embeddings(corpus_file, None, False)
        OKgraph._get_index(corpus_file, None, False)
        OKgraph._get_dictionary(corpus_file, None, False)

    print(f"Test data check completed.")

    print(f"Ended script '{path.normpath(__file__)}'.\n")


if __name__ == "__main__":
    main()
