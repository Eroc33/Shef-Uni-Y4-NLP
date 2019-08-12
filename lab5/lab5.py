# -*- coding: utf-8 -*-
# Adapted from (provided) code by: Robert Guthrie

import torch
import torch.cuda
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import itertools

def mk_trigrams(sent):
        return [([sent[i], sent[i + 1]], sent[i + 2])
                    for i in range(len(sent) - 2)]

def sanity_check_score(predict):
    #sanity check
    sanity_correct = 0
    sanity_total = 0
    for context,target in mk_trigrams("START The mathematician ran to the store . END".split()):
        prediction = predict(context)
        if prediction == target:
            sanity_correct += 1
        sanity_total += 1
    return (sanity_correct/sanity_total)

def main(config,reporter,tuning=True):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    CONTEXT_SIZE = 2
    test_sentences = [
        "START The mathematician ran . END",
        "START The mathematician ran to the store . END",
        "START The physicist ran to the store . END",
        "START The philosopher thought about it . END",
        "START The mathematician solved the open problem . END"
    ]
    sents_trigrams = []
    vocab = set()
    for test_sentence in test_sentences:
        test_sentence = test_sentence.split()
        # we should tokenize the input, but we will ignore that for now
        # build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
        sents_trigrams.append(mk_trigrams(test_sentence))
        vocab.update(test_sentence)

    word_to_ix = {word: i for i, word in enumerate(vocab)}


    class NGramLanguageModeler(nn.Module):

        def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
            super(NGramLanguageModeler, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, vocab_size)

        def forward(self, inputs):
            embeds = self.embeddings(inputs).view((1, -1))
            out = F.relu(self.linear1(embeds))
            out = self.linear2(out)
            log_probs = F.log_softmax(out, dim=1)
            return log_probs



    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocab), config['embedding_dim'], CONTEXT_SIZE,config['hidden_dim'])
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

    def probs_for_context(context):
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        return model(context_var)

    #for prediction we want to find the word with the lowest loss
    def predict(context,vocab=vocab):
        def loss_for_word(word):
            log_probs = probs_for_context(context)
            loss = loss_function(log_probs, autograd.Variable(
                    torch.LongTensor([word_to_ix[word]])))
            loss.backward()
            return loss.item()
        return min(vocab,key = loss_for_word)

    def gap_fill_score(display=False): 
        def loss_for_sent(sent):
            total_loss = torch.Tensor([0])
            for context,target in mk_trigrams(sent.split()):
                log_probs = probs_for_context(context)
                loss = loss_function(log_probs, autograd.Variable(
                        torch.LongTensor([word_to_ix[target]])))
                loss.backward()
                total_loss += loss.data
            return total_loss.item()
        candidate_sents = ["START The "+candidate+" solved the open problem . END" for candidate in ["physicist","philosopher"]]
        losses = list(map(lambda s: (s,loss_for_sent(s)), candidate_sents))
        sorted_candidates = sorted(losses,key=lambda t: t[1])
        if display:
            print("Best candidate sentence is :",sorted_candidates[0],"followed by", sorted_candidates[1])
            print("Losses were: ", losses)
        gap_fill_score = losses[1][1] - losses[0][1]
        if display:
            print("gap_fill_score:",gap_fill_score)
        return gap_fill_score

    for epoch in range(config['passes']):
        total_loss = torch.Tensor([0])
        for trigrams in sents_trigrams:
            for context, target in trigrams:
                log_probs = probs_for_context(context)

                # Step 4. Compute your loss function. (Again, Torch wants the target
                # word wrapped in a variable)
                loss = loss_function(log_probs, autograd.Variable(
                    torch.LongTensor([word_to_ix[target]])))

                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()

                total_loss += loss.data
            losses.append(total_loss)
        mean_loss = total_loss.double() / epoch
        gap_score=gap_fill_score()
        sanity=sanity_check_score(predict)
        total_score=gap_score*sanity
        reporter(neg_mean_loss=-mean_loss.item(),sanity=sanity,gap_fill_score=gap_score,total_score=total_score)
    if not tuning:
        print(losses)
        gap_fill_score(display=True)
        embeddings = {}
        for word in ["physicist","mathematician","philosopher"]:
            context_idxs = [word_to_ix[word]]
            context_var = autograd.Variable(torch.LongTensor(context_idxs))
            model.zero_grad()
            embeddings[word] = model.embeddings(context_var).view((1,-1))
            print(word, "embeddings:", embeddings[word])
        cos = nn.CosineSimilarity()
        for a,b in itertools.combinations(embeddings.keys(),r=2):
            print(a, "similarity to", b, ":", cos(embeddings[a],embeddings[b]))
        

EMBEDDING_DIM = 50
HIDDEN_DIM = 64

if __name__ == "__main__":
    def fake_reporter(**kwargs):
        print(kwargs)
    #config and fake reporter are passed like this to make the function compatible with ray's "tune" hyper parameter tuning framework
    main({'embedding_dim':EMBEDDING_DIM,'hidden_dim':HIDDEN_DIM,'learning_rate':0.009,'passes': 50},fake_reporter,tuning=False)