import logging
import os
from os import path
from logging.config import fileConfig
from okgraph.utils import get_word
from whoosh import index
from whoosh.fields import Schema, TEXT

LOG_CONFIG_FILE = path.dirname(path.realpath(__file__)) + '/logging.ini'
if not path.isfile(LOG_CONFIG_FILE):
    LOG_CONFIG_FILE = path.dirname(path.dirname(LOG_CONFIG_FILE)) + '/logging.ini'

fileConfig(LOG_CONFIG_FILE)
logger = logging.getLogger()

class Indexing:

    def __init__(self, path_corpus: str):
        self.schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))
        self.path_corpus = path_corpus

    def __str__(self):
        return print(self.path_corpus)

    def indexing(self, name_path='indexdir'):
        logger.info('Start indexing windows in corpus')
        count = 0
        c = 0
        m = 0
        log = 0
        flag = False
        l = []

        schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))

        if not os.path.exists(name_path):
            os.mkdir(name_path)
            ix = index.create_in(name_path, schema)

            ix = index.open_dir(name_path+'/')
            writer = ix.writer()

            for word in get_word(self.path_corpus):
                l.append(word)
                if c <= 60:
                    c += 1

                if flag == False and c == 60 and len(l) == 60:
                    count += 1
                    m += 1
                    logger.info('indexing document number: %d' % (m,))
                    writer.add_document(title=str(hex(count)), content=' '.join(map(str, l)))
                    writer.commit()
                    ix = index.open_dir(name_path+'/')
                    writer = ix.writer()
                    flag = True
                    del l[:40]

                if len(l) == 80:
                    count += 1
                    m += 1
                    log += 1
                    if (log >= 10000):
                        logger.info('indexing document number: %d' % (m,))
                        log = 0
                    writer.add_document(title=str(hex(count)), content=' '.join(map(str, l)))
                    if count == 500000:
                        writer.commit()
                        ix = index.open_dir(name_path+'/')
                        writer = ix.writer()
                        count = 0
                    del l[:60]

            if count != 0:
                logger.info('indexing document number: %d' % (m,))
                writer.commit()
            logger.info('End indexing windows in corpus')
