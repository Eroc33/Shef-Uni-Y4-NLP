import os
import sys
import re
import numpy as np
import io
import random
from collections import Counter,defaultdict
from matplotlib import pyplot as plt
from copy import copy

#TODO: possibly filter the set of words we use and use numpy arrays instead of this
def dict_dot(a,b):
    acc = 0
    for key in b.keys():
        acc += a[key]*b[key]
    return acc

def dict_add(a,b):
    for key in b.keys():
        a[key] = a[key]+b[key]

def dict_sub(a,b):
    for key in b.keys():
        a[key] = a[key]-b[key]

def sign(n):
    if n >= 0:
        return 1
    else:
        return -1

def train(w,d_train):
    """Perceptron training"""
    for (vec,cls) in d_train:
        prediction = sign(dict_dot(w,vec))
        if prediction != cls:
            if cls == 1:
                dict_add(w,vec)
            else:
                dict_sub(w,vec)

    return w

#TODO: use precision/recall if they can be use for unbiased data
def test(w,d_test):
    total = len(d_test)
    correct = 0
    for (vec,cls) in d_test:
        prediction = sign(dict_dot(w,vec))
        correct += 1 if prediction==cls else 0
    return correct/total

def bag_of_words(text):
    """bag of words feature extractor"""
    return re.sub("[^\w']"," ",text).split()

def ngrams(n):
    """ngram feature extractor"""
    def text_to_feature(text):
        split = re.sub("[^\w']"," ",text).split()
        for i in range(len(split)-(n-1)):
            yield " ".join(split[i:i+(n)])
    return text_to_feature

def read_class(class_folder,cls,extract_features):
    """Reads the corpus for a class, and converts each example to a tuple of (weight_vector,class)"""
    #find all files for this class
    files = []
    with os.scandir(class_folder) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                files.append(entry.path)
    #read the words for each file
    data = []
    for file in files:
        with io.open(file,'r') as file:
            data.append((Counter(extract_features(file.read())),cls))
    return data

def main(extract_features,save_fig=None,shuffle=True):
    max_iter = 50
    #set the seed for reproducibility of randomness
    random.seed(345978439)

    data_path = sys.argv[1]

    if not data_path:
        raise "You must pass a folder with training data"

    print("loading data")

    #load data
    pos = read_class(data_path+"/txt_sentoken/pos",1,extract_features)
    neg = read_class(data_path+"/txt_sentoken/neg",-1,extract_features)
    #split training and testing data
    pos_train = pos[:800]
    pos_test = pos[800:]
    neg_train = neg[:800]
    neg_test = neg[800:]

    #recombine neg and pos sets
    d_train = pos_train+neg_train
    d_test = pos_test+neg_test

    print("training")
    w = [defaultdict(int)]
    p = []
    #setup the axis limits
    gca = plt.gca()
    gca.set_ylim([0,1])
    gca.set_xlim([-1,max_iter])

    #createa a new plot
    train_plt, = gca.plot([],[],'b-')
    avg_plt, = gca.plot([],[],'r--')
    #show it without blocking, so we can update it later
    plt.show(block=False)

    for i in range(max_iter):
        #randomize training and testing data
        if shuffle:
            random.shuffle(d_train)
            random.shuffle(d_test)
        #do a round of training
        w.append(train(copy(w[i]),d_train))
        #store the test result
        p.append(test(w[i],d_test))
        #update plot (in realtime!)
        train_plt.set_data(range(i+1),p)
        plt.pause(0.001)

    w_avg = defaultdict(int)

    for key in w[max_iter].keys():
        for w_i in w:
            w_avg[key] += w_i[key]
        w_avg[key] /= len(w)

    random.shuffle(d_test)
    p_avg = test(w_avg,d_test)
    #update plot (in realtime!)
    avg_plt.set_data([0,max_iter],[p_avg,p_avg])
    plt.pause(0.001)
    if save_fig:
        plt.savefig(save_fig)

    #pause on the plot
    plt.show()
    sorted_weights = sorted(w[max_iter].items(),key=lambda x: x[1])
    print("top negative:",sorted_weights[:10])
    print("top positive:",sorted_weights[-10:])

if __name__ == "__main__":
    #The commented lines below generated the various other graphs used in my report
    #main(ngrams(1),save_fig="unshuffled-unigrams.png",shuffle=False)
    main(ngrams(1),save_fig="unigrams.png")
    #main(ngrams(2),save_fig="bigrams.png")
    #main(ngrams(3),save_fig="trigrams.png")
