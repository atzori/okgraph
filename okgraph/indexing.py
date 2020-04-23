import logging
import os
from os import path
from logging.config import fileConfig
from okgraph.utils import get_words
from whoosh import index
from whoosh.fields import Schema, TEXT

# Create a logger using the specified configuration
LOG_CONFIG_FILE = path.dirname(path.realpath(__file__)) + '/logging.ini'
if not path.isfile(LOG_CONFIG_FILE):
    LOG_CONFIG_FILE = path.dirname(path.dirname(LOG_CONFIG_FILE)) + '/logging.ini'
fileConfig(LOG_CONFIG_FILE)
logger = logging.getLogger()


class Indexing:
    """
    A class used to organize a corpus in sub-documents and allow faster searches of word's occurrences into it.
    Every sub-document (or document) is a text window extracted from the corpus. All of the documents have the same
     specified dimension and adjoining documents are partially overlaid. Every document has an unique title (ID) and a
     content (the text contained in the window). All of the documents are stored and indexed.
    Attributes:
        corpus_path: path of the file (text corpus)
        schema: schema (whoosh Schema) used to represent a document: (title (text ID): content (text))
        info: to print or not messages in the log
    """

    # Schema's fields
    FIELD_TITLE = 'title'
    FIELD_CONTENT = 'content'

    def __init__(self, corpus_path: str, info: bool = True):
        """
        Define an 'Indexing' object with a reference to the corpus and the document's storing schema.
        :param corpus_path: path (with name) of the text corpus
        QSTN: index_path could be a valid attribute of the object, not a parameter of indexing
        """
        self.corpus_path = corpus_path
        self.schema = Schema(
            title=TEXT(stored=True),   # TEXT field for corpus title (indexed and stored)
            content=TEXT(stored=True)  # TEXT field for corpus content (indexed and stored)
        )
        self.info = info

    def __str__(self) -> str:
        """
        Returns the object as a string.
        """
        return self.corpus_path.__str__()

    def indexing(self, index_path: str = 'indexdir') -> None:
        """
        Starts the indexing process.
        :param index_path: path in which the documents will be stored
        """
        if self.info is True:
            logger.info('INDEXING: Start documents\' indexing in corpus')  # LOG INFO

        # Indexing parameters
        document_overlay = 20  # Number of words shared between two documents
        document_center = 40  # Number of words at the center of the document, not shared
        document_size = document_center + 2 * document_overlay  # Total size of a document

        document_list = []  # List of words that defines a document
        document_list_count = document_overlay  # Number of words in the document's constructor list
        #  (first document has no left overlay: let the counter start like it had and it has been already processed)
        document_index = 0  # ID of the document being indexed and saved
        document_count = 0  # Counter for found documents
        document_count_limit = 500000  # Max number of indexable and savable documents

        log_count = 0  # Log counter for the found documents
        log_frequency = 10000  # Frequency of log messages in term of found documents

        # Create a new schema
        # QSTN: A new Schema is created but never assigned to self.schema? Why doesn't it use the Schema in the object?
        schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))

        # Index the corpus if there is no trace of an index in the specified path
        if not path.exists(index_path):
            # Create the path and the index for the specified schema
            os.makedirs(index_path, exist_ok=True)
            ix = index.create_in(index_path, schema)
            writer = ix.writer()

            # Scroll through the corpus word by word
            #  Divide the corpus in partially overlaid documents
            #  Index and save every document using the specified schema
            for word in get_words(self.corpus_path):
                # Add the word to list of document's word
                document_list.append(word)
                document_list_count += 1

                # If a document has been completed
                if document_list_count == document_size:
                    # Count the new document
                    document_index += 1
                    document_count += 1
                    log_count += 1

                    if self.info is True:
                        if log_count == 1:
                            logger.info('INDEXING: Indexing document number: {0}'.format(document_count))  # LOG INFO
                        if log_count == log_frequency:
                            log_count = 0

                    # Convert the temporary list of words, representing the document, into plain text
                    document_content = ' '.join(map(str, document_list))
                    # Index the document's content using the document's index as its title
                    writer.add_document(title=str(hex(document_index)), content=document_content)

                    # Get rid of the words that do not overlay with the next document
                    del document_list[:-document_overlay]
                    document_list_count = document_overlay

                    # If the limit has been reached, commit the changes and reset the index
                    #  (start overwriting the old documents)
                    if document_index == document_count_limit:
                        if self.info is True:
                            logger.info('INDEXING: Limit of {0} document reached: '
                                        'committing changes'.format(document_count_limit))  # LOG INFO
                        writer.commit()
                        ix = index.open_dir(index_path)
                        writer = ix.writer()
                        document_index = 0

            if document_count != 0:
                if self.info is True:
                    logger.info('INDEXING: Indexed last document with number: {0}'.format(document_count))  # LOG INFO
                    logger.info('INDEXING: Committing')  # LOG INFO
                writer.commit()

            if self.info is True:
                logger.info('INDEXING: Ended documents\' indexing in corpus')  # LOG INFO
