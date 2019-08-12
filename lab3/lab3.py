import sys
import itertools
from collections import Counter, defaultdict
from sklearn.metrics import f1_score
from random import shuffle
from copy import copy

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

def mk_pl_cl_counts(corpus):
    counter = Counter()
    for s in corpus:
        x,y = zip(*s)
        counter.update(zip(y[:-1],y[1:]))
    return {k:v for (k,v) in counter.items() if v >= 3}

def phi_2(x,y,pl_cl_counts):
    return Counter((pl_cl for pl_cl in zip(y[:-1],y[1:]) if pl_cl in pl_cl_counts))

def argmax(fn,over):
    """Argmax of fn over the iterator over"""
    return max([(arg,fn(arg)) for arg in over],key=lambda v: v[1])[0]

def dot(a,b):
    """Dot product of two vectors,represented as dicts"""
    acc = 0
    for k in b.keys():
        acc += a[k] * b[k]
    return acc

def predict(w,x,phi):
    return argmax(lambda y: dot(w,phi(x,y)), over=gen(len(x)))

def gen(length):
    """Generate all sequences of the labels for a given length"""
    return itertools.product(['O', 'ORG','MISC','PER','LOC'],repeat=length)

def train(train_data,phi,passes=5):
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
            if prediction != y:
                w += phi(x,y) - phi(x,prediction)
        last_w = copy(w)
        #running average
        w_avg += Counter({k:v/passes for k,v in w.items()})
    return w_avg

def test(w,test_data,phi,name):
    pred_all = []
    correct_all = []
    #for every scentence make a prediction, and store that and the correct labels to be passed to f1_score
    for scentence in test_data:
        x,y = zip(*scentence)
        prediction = predict(w,x,phi)
        pred_all.extend(prediction)
        correct_all.extend(y)
    f1_micro = f1_score(correct_all,pred_all,average='micro',labels=['ORG','MISC','PER','LOC'])
    print(f"{name} f1_micro:", f1_micro)

def main(training_file, test_file):
    train_data = load_dataset_sents(training_file)
    test_data = load_dataset_sents(test_file)

    cw_cl_counts = mk_cw_cl_counts(train_data)
    pl_cl_counts = mk_pl_cl_counts(train_data)
    #pre apply phi functions to their counts arguments so they conform to phi(x,y)
    phi1 = lambda x,y: phi_1(x,y,cw_cl_counts)
    phi2 = lambda x,y: phi_2(x,y,pl_cl_counts)
    #phi1+2 from merging the outputs of ph1 and phi2
    phi12 = lambda x,y: phi1(x,y)+phi2(x,y)
    #train and test each phi funtion
    for name,phi in {'phi1':phi1,'phi2':phi2,'phi1+2':phi12}.items():
        weights = train(train_data,phi=phi)
        test(weights,test_data,phi=phi,name=name)
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
                print(f"Top 10 features for {label} for {name}: {joined_features}")

#whether to print output as latex table
latex_output = False

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])
