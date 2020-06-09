"""The 'indexing' module contains the utilities used to organize a corpus in
sub-documents and allow faster searches of word occurrences into it.
"""
from okgraph.utils import get_words, logger
from os import makedirs, path
from whoosh import index
from whoosh.fields import Schema, TEXT

DEFAULT_INDEX_DIR: str = "indexdir"
"""str: Default path for the index directory."""
FIELD_ID: str = "id"
"""str: Field that stores the ID of a document in the Schema"""
FIELD_CONTENT: str = "content"
"""str: Field that stores the content of a document in the Schema"""


class Indexing:
    """A class used to create an index of the corpus.

    The corpus is divided in sub-documents with a unique ID to identify their
    content. All the documents are then stored and indexed.
    Every sub-document (or document) is a text window extracted from the corpus.
    All the documents have the same specified dimension and adjoining documents
    are partially overlaid.

    Attributes:
        corpus_path (str): path of the text corpus.
        schema (Schema): the index structure composed by the following fields:
            "id": str
            "content": str

    """

    def __init__(self, corpus_path: str):
        """The constructor creates an Indexing object.

        Saves the reference to the corpus file and defines the structure for
        the index (schema).

        Args:
            corpus_path (str): path of the text corpus.

        """
        self.corpus_path = corpus_path
        self.schema = Schema(
            title=TEXT(stored=True),
            content=TEXT(stored=True)
        )

    def __str__(self) -> str:
        return self.corpus_path.__str__()

    def indexing(self,
                 index_path: str = DEFAULT_INDEX_DIR,
                 document_overlay: int = 20,
                 document_center: int = 40,
                 num_processes: int = 1,
                 memory_limit: int = 128
                 ) -> None:
        """Starts the indexing process.

        Args:
            index_path (str): path in which the index will be stored.
            document_overlay (int): number of words shared between two
                documents. This value should be greater than the expected
                maximum size of the windows created trough the SlidingWindows
                objects.
            document_center (int): number of words at the center of the
                document, not shared.
            num_processes (int): number of processes used by the index writer.
            memory_limit (int): memory (MB) used by every single writer process.

        Returns:
            None

        """
        if not document_overlay > 0:
            raise ValueError(f"document_overlay can't be negative or zero")
        if not document_center > 0:
            raise ValueError(f"document_center can't be negative or zero")
        if not num_processes > 0:
            raise ValueError(f"num_processes can't be negative or zero")
        if not memory_limit > 0:
            raise ValueError(f"memory_limit can't be negative or zero")

        logger.info(f"Start documents indexing in corpus")

        # Indexing parameters
        document_size = document_center + 2 * document_overlay

        # List of words that defines a document
        document_list = []
        # Number of words in the documents constructor list.
        # The first document has no left overlay: let the counter start like it
        # had and it has been already processed
        document_list_count = document_overlay

        # ID of the document being indexed and saved
        document_index = 0
        # Counter for found documents
        document_count = 0
        # Max number of indexable documents without saving and committing
        document_count_limit = 500000

        # Log counter for the found documents
        log_count = 0
        # Frequency of log messages in term of found documents
        log_frequency = 10000

        # Define the index schema
        schema = self.schema

        # Index the corpus if there is no trace of an index in the specified
        # path
        if not path.exists(index_path):
            # Creates the path and the index for the specified schema
            makedirs(index_path, exist_ok=True)
            ix = index.create_in(index_path, schema)
            writer = ix.writer(procs=num_processes,
                               multisegment=True,
                               limitmb=memory_limit)

            # Scrolls through the corpus word by word:
            # divide the corpus in partially overlaid documents;
            # index and save every document using the specified schema
            for word in get_words(self.corpus_path):
                # Add the word to the list of words in the document
                document_list.append(word)
                document_list_count += 1

                # If a document has been completed
                if document_list_count == document_size:
                    # Count the new document
                    document_index += 1
                    document_count += 1
                    log_count += 1

                    if log_count == 1:
                        logger.info(
                            f"Indexing document number: {document_count}")
                    if log_count == log_frequency:
                        log_count = 0

                    # Convert the temporary list of words, representing the
                    # document, into plain text
                    document_content = " ".join(map(str, document_list))
                    # Index the document content using the document index as
                    # its ID
                    writer.add_document(
                        title=str(hex(document_index)),
                        content=document_content)

                    # Get rid of the words that do not overlay with the next
                    # document
                    del document_list[:-document_overlay]
                    document_list_count = document_overlay

                    # If the limit has been reached, commit the changes and
                    # start saving the next documents into a new file
                    if document_index == document_count_limit:
                        logger.info(
                            f"Limit of {document_count_limit} document reached:"
                            f" committing changes")
                        writer.commit()
                        ix = index.open_dir(index_path)
                        writer = ix.writer()
                        document_index = 0

            if document_count != 0:
                logger.info(
                    f"Indexed last document with number: {document_count}")
                logger.info(
                    f"Committing")
                writer.commit()

            logger.info(f"Ended documents indexing in corpus")
