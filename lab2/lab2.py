from collections import Counter, defaultdict
import sys
import re
import math
import itertools
import functools
from functools import partial

def lines(filename):
    """Returns a list of all lines from a file"""
    with open(filename,'r') as file:
        return list(iter(file.readline,''))

def tokenize_scentence(scentence):
    """Split a scentece into tokens"""
    #remove punctuation and leading/trailing spaces left by said removal
    punct_removed = re.sub("[.?,]","",scentence).strip()
    #split on whitespace, and before apostrophe follwed by letters
    split = re.split("\s+|(?='\w+)",punct_removed)
    #TODO: find a more efficient and pythonic way to do this wrapping in start and end tokens
    return ['<s>'] + [s.lower() for s in split] + ['</s>']

def load_training_data(corpus_file):
    return [tokenize_scentence(line) for line in lines(corpus_file)]

def load_questions(questions_file):
    """load all questions"""
    def load_line(line):
        """load one question"""
        #scentence and candidate words are separated by :
        scentence,candidate_words = line.split(":")
        #individual candidate words are separated by :
        candidate_words = [word.strip() for word in candidate_words.split("/")]
        #spit the scentence around where we should insert our candidate words
        scentence = scentence.strip().split("____")
        return scentence,candidate_words
    return [load_line(line) for line in lines(questions_file)]

def extract_ngrams(data,ngram_size):
    for i in range(len(data)-ngram_size):
        yield tuple(data[i:i+ngram_size])

def dict_add(to,addend):
    """accumulate a vector represented as a dict (addend) into a defaultdict (to)"""
    for k in addend:
        to[k] += addend[k]

class DictionaryModel:
    def __init__(self,ngram_size,kappa):
        self.ngram_size = ngram_size
        self.kappa = kappa

    def train(self,train_data):
        self.counts = []
        self.probs = []
        self.len_v = sum([len(scentence) for scentence in train_data])
        for i in range(self.ngram_size):
            self.counts.append(defaultdict(int))
            for scentence in train_data:
                dict_add(self.counts[i],Counter(extract_ngrams(scentence,i+1)))
            prev_count = sum(self.counts[i].values())
            self.probs.append({})
            for ngram in self.counts[i]:
                self.probs[i][ngram] = (self.counts[i][ngram] + self.kappa)/(prev_count + (self.kappa*self.len_v))
        self.probs

    def __p_single(self,ngram):
        if ngram in self.probs[len(ngram)-1]:
            return self.probs[len(ngram)-1][ngram]
        elif len(ngram) == 1:
            return self.counts[0][ngram]/self.len_v
        else:
            prev_count = sum(self.counts[len(ngram)-1].values())
            #count is 0, since the ngram is not in probs, so was never seen
            return (0 + self.kappa)/(prev_count + (self.kappa*self.len_v))

    def p(self,ngrams):
        p = 1
        for ngram in ngrams:
            p *= self.__p_single(ngram)
        return p

def rank_scentences(training_data,questions,ngram_size,kappa):
    """yields arrays of probabilities and candidate scentences in descending order of probability"""
    #print("training model")
    model = DictionaryModel(ngram_size,kappa=kappa)
    model.train(training_data)

    #print("extracting best candidates")
    for scentence,words in questions:
        candidates = [scentence[0] + word + scentence[1] for word in words]
        probs = [(model.p(extract_ngrams(tokenize_scentence(candidate),ngram_size)),candidate) for candidate in candidates]
        yield sorted(probs,key=lambda x: x[0],reverse=True)

def print_latex_table(iter):
    print(r"selected & prob & vs \\")
    print(r"\hline")
    for i in iter:
        print(fr"{i[0][1]}&\num{{{i[0][0]:.3e}}}&\num{{{i[1][0]:.3e}}}\\")

if __name__ == "__main__":
    print("loading data")
    training_data = load_training_data(sys.argv[1])
    questions = load_questions(sys.argv[2])

    #unigram
    unigram = rank_scentences(training_data,questions,ngram_size=1,kappa=0)
    #bigram
    bigram = rank_scentences(training_data,questions,ngram_size=2,kappa=0)
    #bigram + add 1 smoothing
    bigram_smoothed = rank_scentences(training_data,questions,ngram_size=2,kappa=1)
    print_latex_table(unigram)
    print_latex_table(bigram)
    print_latex_table(bigram_smoothed)
