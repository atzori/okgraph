"""This module contains a script to provide usable corpus, models, and other
resources (corpus index and dictionary) that can be used in testing. The corpus
are provided by the 'gensim' library data. Specifically, the 'text8'
corpus is downloaded, while a second even smaller 'text7' corpus is obtained
from 'text8' extracting its first 10^7 bytes. The embeddings ('Magnitude') and
the other useful resources are built from the corpus. The corpus and models are
available in a specific folder inside the 'test' package. Other corpus and
models can be found through the 'gensim' library, that provides other useful
data, as shown `here <https://github.com/RaRe-Technologies/gensim-data>`_.
"""
from gensim import downloader
import gzip
from okgraph.core import OKgraph
import os
from os import path
import shutil

TEST_CORPUS_FOLDER = "data"
"""str: location in which store the various test corpus and resources.
"""
GENSIM_CORPUS = ["text8"]
"""List[str]: names of the 'gensim' data corpus to download.
"""
GENSIM_PATH = f"{os.path.expanduser('~')}/gensim-data"
"""str: path in which the 'gensim' data downloader stores the corpus.
"""


if __name__ == "__main__":
    # Save the current working directory
    cwd_backup = path.normpath(os.getcwd())
    # Set the current working directory to be the one containing this script
    cwd = path.dirname(path.normpath(__file__))
    os.chdir(cwd)

    all_corpus = GENSIM_CORPUS + ["text7"]

    # If the directory that will contain the test corpus doesn't exist, 
    # create it
    base = TEST_CORPUS_FOLDER
    if not path.exists(base):
        print(f"\n'{base}' folder not existing. Creating '{base}' folder.")
        os.mkdir(base)
    else:
        print(f"\n'{base}' folder already existing.")

    # Download the corpus that are non existing
    for corpus in GENSIM_CORPUS:
        if not path.exists(f"{base}/{corpus}/{corpus}"):
            print(f"\n'{corpus}' file not existing.")
            # Create the folder, if not existing, to contain corpus and
            # resources
            if not path.exists(f"{base}/{corpus}/"):
                os.mkdir(f"{base}/{corpus}")
            # Download the corpus
            print(f"Downloading {corpus} from 'gensim' data.")
            downloader._download(corpus)
            # Move the downloaded file to the test directory
            shutil.move(f"{GENSIM_PATH}/{corpus}/{corpus}.gz",
                        f"{base}/{corpus}/{corpus}.gz")
            shutil.rmtree(f"{GENSIM_PATH}")
            # Extract the corpus
            with gzip.open(f"{base}/{corpus}/{corpus}.gz", "rb") as f_in:
                with open(f"{base}/{corpus}/{corpus}", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(f"{base}/{corpus}/{corpus}.gz")
        else:
            print(f"\n'{corpus}' file already existing.")

    # Generates the 'text7' corpus if it's not existing
    if not path.exists(f"{base}/text7/text7"):
        print(f"\n'text7' file not existing.")
        # Create the folder to contain corpus and model if not existing
        if not path.exists(f"{base}/text7/"):
            os.mkdir(f"{base}/text7")
        # Open 'text8' and save its first 10^7 bites as 'text7'
        print(f"Creating 'text7' from 'text8'.")
        with open(f"{base}/text8/text8", "rb") as text8:
            with(open(f"{base}/text7/text7", "wb")) as text7:
                text7.write(text8.read(10000000))
    else:
        print(f"\n'text7' file already existing.")

    # Process every corpus to obtain the related resources:
    # embeddings, corpus index and dictionary
    for corpus in all_corpus:
        # Instantiate an 'OKgraph' object to create all the resources
        print(f"\nInstantiating an OKgraph object to create all the"
              f" '{corpus}' resources.")
        OKgraph(corpus_file=f"{base}/{corpus}/{corpus}")

    # Go back to the previous working directory
    # NOTE: possibly useless
    os.chdir(cwd_backup)
