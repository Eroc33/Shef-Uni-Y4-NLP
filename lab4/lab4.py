import sys
import itertools
from collections import Counter, defaultdict
from sklearn.metrics import f1_score
import random
from random import shuffle
from copy import copy
import argparse
import time

LABELS = ['O', 'ORG','MISC','PER','LOC']

def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets=[]
    inputs=[]
    zip_inps=[]
    with open(file_path) as f:
        for line in f:
            sent, tags=line.split('\t')
            words=[token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags=[target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)

def mk_cw_cl_counts(corpus):
    counter = Counter()
    for sentence in corpus:
        counter.update(sentence)
    return {k:v for (k,v) in counter.items() if v >= 3}

def phi_1(x, y, cw_cl_counts):
    return Counter((cw_cl for cw_cl in zip(x,y) if cw_cl in cw_cl_counts))

def argmax(fn,over):
    """Argmax of fn over the iterator over"""
    return max([(arg,fn(arg)) for arg in over],key=lambda v: v[1])[0]

def max_over(fn,over):
    """Max of fn over the iterator over"""
    return max([fn(arg) for arg in over])

def dot(a,b):
    """Dot product of two vectors,represented as dicts"""
    acc = 0
    for k in b.keys():
        acc += a[k] * b[k]
    return acc

def predict(w,x,phi):
    return argmax(lambda y: dot(w,phi(x,y)), over=gen(len(x)))

def predict_viterbi(w,x,phi):
    v = Counter()
    b = {}
    for n in range(len(x)):
        for y in LABELS:
            # instead of passing the whole of the the word and tag sequences to phi I only
            # pass the current word, and potential current label, as this is easier to reason about.
            # This unfortunately does not generalise to models such as phi2, but that was not
            # a requirement for this lab
            cw = [x[n]]
            v[(y,n)] = max_over(lambda y_prime: v[(y_prime,n-1)]+dot(w,phi(cw,[y])), over=LABELS)
            b[(y,n)] = argmax(lambda y_prime: v[(y_prime,n-1)]+dot(w,phi(cw,[y])), over=LABELS)
    #convert back pointers into a sequence of labels
    seq = []
    prev_label = None
    for n in range(len(x)):
        if not prev_label:
            prev_label = argmax(lambda y_prime: v[(y_prime,len(x)-1)], over=LABELS)
        else:
            prev_label = b[(prev_label,len(x)-n)]
        seq.append(prev_label)
    return list(reversed(seq))

def predict_beam(w,x,phi,k=3):
    def top(beam,k):
        #sort descending with key score
        ordered = sorted(beam,key=lambda b: b[1],reverse=True)
        #take k
        return ordered[:k]
    beam = [([],0)]
    for n in range(len(x)):
        beam_prime = []
        for b in beam:
            for y in LABELS:
                y_prime = b[0]+[y]
                score = dot(w,phi(x,y_prime))
                beam_prime.append((y_prime,score))
        beam = top(beam_prime,k)
    #get top sequence
    return top(beam,1)[0][0]
        

def gen(length):
    """Generate all sequences of the labels for a given length"""
    return itertools.product(LABELS,repeat=length)

def train(train_data,phi,predict,passes=5):
    w_avg = Counter()
    last_w = Counter()
    #do several training passes
    for i in range(passes):
        #shuffle the training data for each training pass
        shuffle(train_data)
        #starting weights for this pass are the last iteration's weights, or an empty vector
        w = copy(last_w)
        for s in train_data:
            x,y = zip(*s)
            prediction = predict(last_w,x,phi)
            if list(prediction) != list(y):
                w += phi(x,y) - phi(x,prediction)
        last_w = copy(w)
        #running average
        w_avg += Counter({k:v/passes for k,v in w.items()})
    return w_avg

def test(w,test_data,phi,predict,name,verbose):
    pred_all = []
    correct_all = []
    #for every scentence make a prediction, and store that and the correct labels to be passed to f1_score
    for scentence in test_data:
        x,y = zip(*scentence)
        prediction = predict(w,x,phi)
        if list(prediction) != list(y):
            for (t1,t2,word) in zip(prediction,y,x):
                if t1 != t2:
                    #if we have a weight for the mismatched tags
                    keys = [(k,v) for k,v in w if k == word]
                    if verbose:
                        if len(keys) > 1:
                            print(f"Guessed `{word}` as `{t1}` it could have been: {keys}")
                        else:
                            print(f"Guessed `{word}` as `{t1}` due to no weight")
        pred_all.extend(prediction)
        correct_all.extend(y)
    f1_micro = f1_score(correct_all,pred_all,average='micro',labels=['ORG','MISC','PER','LOC'])
    print(f"{name} f1_micro:", f1_micro)
    print(w[('ST','LOC')])
    print(w[('ST','ORG')])

def main(training_file, test_file, predict, latex_output, seed,verbose):

    random.seed(seed)

    train_data = load_dataset_sents(training_file)
    test_data = load_dataset_sents(test_file)

    cw_cl_counts = mk_cw_cl_counts(train_data)
    #pre apply phi functions to their counts arguments so they conform to phi(x,y)
    phi1 = lambda x,y: phi_1(x,y,cw_cl_counts)
    #train and test phi funtion
    start = time.perf_counter()
    weights = train(train_data,predict=predict,phi=phi1)
    elapsed = time.perf_counter() - start
    print(f"Training complete in {elapsed} seconds")

    test(weights,test_data,predict=predict,phi=phi1,name='phi1',verbose=verbose)
    for label in ['ORG','MISC','PER','LOC','O']:
        label_features = [((w,l),c) for ((w,l),c) in weights.items() if l == label]
        top_label_features = sorted(label_features,key=lambda i: i[1], reverse=True)[:10]
        #format the output for LaTeX if the flag is set
        if latex_output:
            joined_features = " \\\\ \n & ".join((f"{f} = {c}" for (f,c) in top_label_features))
            print(fr"\multirow{{{len(top_label_features)}}}{{*}}{{{label}}} & {joined_features} \\")
        #otherwise just print it normally
        else:
            joined_features = "\n\t".join((f"{f} = {c}" for (f,c) in top_label_features))
            print(f"Top 10 features for {label} for phi1: {joined_features}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entity labling with structured perceptron.')
    parser.add_argument('-v', dest='predict', action='store_const',
                        const=predict_viterbi, default=predict,
                        help='use viterbi preictor')
    parser.add_argument('-b', dest='predict', action='store_const',
                        const=predict_beam, default=predict,
                        help='use beam search predictor')
    parser.add_argument('--latex', dest='latex_output', action='store_const',
                        const=True, default=False,
                        help='whether to use latex table output')
    parser.add_argument('--seed', type=int, default=77324,
                        help='the seed to use')
    parser.add_argument('--verbose', action='store_const',
                        const=True, default=False,
                        help='print extra info')
    parser.add_argument('training_file')
    parser.add_argument('test_file')

    args = parser.parse_args()

    main(**vars(args))
