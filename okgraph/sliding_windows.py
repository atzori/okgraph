import operator
import re
import tqdm
from tqdm import tqdm
import scipy
from whoosh import index
from whoosh.qparser import QueryParser
import numpy as np


class SlidingWindows:

    def __init__(self, words, dict_total=None, l=14, d=6, occurrence_corpus=None, path="indexdir/"):

        if not isinstance(words, list) and isinstance(words[0], str) and isinstance(words[1], str):
            raise TypeError('error type in words')
        if len(words) != 2:
            raise ValueError('words must be a list of 2 strings')
        if not isinstance(l, int):
            raise TypeError('error type in l')
        if not isinstance(d, int):
            raise TypeError('error type in d')

        self.dict_total = np.load(dict_total).item()
        self.words = words
        self.l = l
        self.d = d
        self.offset = int((l - (d + 2)) / 2)
        self.path = path
        self.listWindows, self.dictionary = self.windowsDict(words[0], words[1], d=self.d + 1, offset=self.offset, path=self.path)
        self.occurrences = self.getNumberOccurrences(self.dictionary)
        self.occurrence_corpus = self.getNumberOccurrences(self.dict_total)
        self.dict_total_freq = self.freqTotalDict(self.dict_total, self.occurrence_corpus)
        self.results = self._labelExtraction()


    def __str__(self):
        return 'The window of ' + self.words[0] + ' and ' + self.words[2]

    def occurrence(self, dictionary, list_windows):
        occurence_dict = {}
        for word in tqdm(list(dictionary.keys())):
            for l in list_windows:
                if word in l:
                    occurence_dict[word] = occurence_dict.get(word, 0) + 1
        return occurence_dict

    def cleanResults(self, dictionary, dirty_dictionary, ratio_dictionary, threshold=0.50):
        new_dictionary = dirty_dictionary
        for key in tqdm(list(dictionary.keys())):
            if dictionary.get(key) >= len(new_dictionary):
                new_dictionary.pop(key)
        for key in tqdm(list(new_dictionary.keys())):
            if len(key) < 3:
                new_dictionary.pop(key)
        for key in tqdm(list(new_dictionary.keys())):
            if ratio_dictionary.get(key) < threshold:
                new_dictionary.pop(key)
        for key in tqdm(list(new_dictionary.keys())):
            if str.isnumeric(key):
                new_dictionary.pop(key)
        return new_dictionary

    def dirtyResults(self, presence_dictionary, occurrence, tot_occurrence, list_word):
        new_dictionary = {}
        for word in tqdm(list(occurrence.keys())):
            if word not in occurrence or word not in presence_dictionary or word not in tot_occurrence:
                new_dictionary[word] = occurrence.get(word)[0], occurrence.get(word)[1], 0
            else:
                new_dictionary[word] = occurrence.get(word)[0], occurrence.get(word)[1], (
                        (presence_dictionary.get(word) / len(list_word)) * tot_occurrence.get(word))
        return new_dictionary

    def freqTotalDict(self, tot_dictionary, number_total_dictionary_word):
        dizionario_frequenza_tot = {}
        for word in tqdm(list(tot_dictionary.keys())):
            dizionario_frequenza_tot[word] = scipy.log10(number_total_dictionary_word / tot_dictionary.get(word))
        return dizionario_frequenza_tot

    def frequency(self, dictionary, number_total_dictionary_word):
        occurrence = {}
        for word in tqdm(list(dictionary.keys())):
            occurrence[word] = (dictionary.get(word), dictionary.get(word) / number_total_dictionary_word, [])
        return occurrence

    def getNumberOccurrences(self, dizionario):
        """Restituisce il conteggio di tutte le parole contenute nel dizionario"""
        countWordTot = 0
        for num in dizionario.values():
            countWordTot += num

        return countWordTot

    def noise(self, dictionary, total_dictionary):
        """
        Restituisce il rapporto normalizzato tra le frequenze di tutte le parole
        tutte le parole delle finestre"""
        ratioDict = {}

        n_words_tot = self.occurrence_corpus
        n_words_windows = self.occurrences

        # per ogni parola nelle finestre
        for w in tqdm(list(dictionary.keys())):
            # calcola il rapporto
            if w not in total_dictionary:
                print(w)
                ratio = 0
            else:
                ratio = scipy.log10(dictionary.get(w) / n_words_windows) + scipy.log10(
                    n_words_tot / total_dictionary.get(w))

            ratioDict[w] = ratio

        return ratioDict

    def windowsDict(self, word_one, word_two, d=7, offset=3, path=None):
        dict_windows = {}
        ix = index.open_dir(path)

        exp1 = '((\w+\W*\s){' + str(offset) + '}' + word_one + '\s(?=(\w+\W*\s){0,' + str(
            d) + '}' + word_two + ')(\w+\W*\s){' + str(d) + '}(\w+\W*\s){' + str(offset) + '}'
        exp2 = '(\w+\W*\s){' + str(offset) + '}' + word_two + '\s(?=(\w+\W*\s){0,' + str(
            d) + '}' + word_one + ')(\w+\W*\s){' + str(d) + '}(\w+\W*\s){' + str(offset) + '})'
        exp = exp1 + '|' + exp2
        regex = re.compile(r'' + exp)
        query_one = '\"' + word_one + ' ' + word_two + '\"~' + str(d)
        query_two = '\"' + word_two + ' ' + word_one + '\"~' + str(d)
        query_one = QueryParser("content", ix.schema).parse(u'' + query_one)
        query_two = QueryParser("content", ix.schema).parse(u'' + query_two)
        temporary = []
        windows = []

        with ix.searcher() as searcher:
            results = searcher.search(query_one, limit=10000000)
            for result in results:
                temporary.append(result['content'])
        with ix.searcher() as searcher:
            results = searcher.search(query_two, limit=10000000)
            for result in results:
                temporary.append(result['content'])
        lista = temporary
        temporary = [regex.search(doc) for doc in temporary]
        temporary = [doc.group(0) if doc is not None else '' for doc in temporary]
        for temp in set(temporary):
            if not temp == None:
                if len(windows) == 0 or temp != windows[-1]:
                    windows.append(temp.split())
                    for word in windows[-1]:
                        dict_windows[word] = dict_windows.get(word, 0) + 1

        dict_windows = dict(sorted(dict_windows.items(), key=operator.itemgetter(1), reverse=True))

        return windows, dict_windows

    def _labelExtraction(self, threshold=0.50):

        """Algoritmo di estrazione"""

        if len(self.listWindows) == 0:
            raise ValueError('Empty List')

        self.dictionary.pop(self.words[0])
        self.dictionary.pop(self.words[1])

        occurrence_dict = self.occurrence(self.dictionary, self.listWindows)
        dict_frequency = self.frequency(self.dictionary, self.occurrences)
        dict_ratio = self.noise(self.dictionary, self.dict_total)
        dict_dirty = self.dirtyResults(occurrence_dict, dict_frequency, self.dict_total_freq, self.listWindows)
        new_dict = self.cleanResults(self.dictionary, dict_dirty, dict_ratio, threshold)

        new = {}
        for w in list(new_dict.keys()):
            tmp = new_dict.get(w)[2]
            new[w] = tmp

        new_dict = dict(sorted(new.items(), key=operator.itemgetter(1), reverse=True))

        return new_dict

    def getResults(self):
        return self.results

    def getDictW(self):
        return self.dictionary

    def getNumberW(self):
        return self.occurrences
