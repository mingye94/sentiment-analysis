#!/usr/bin/python

import random
import collections
import math
import sys
from util import readExamples, evaluatePredictor
from model import extractFeatures, learnPredictor, plot_loss
import unicodedata
import re
from sklearn.model_selection import train_test_split

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z\u4e00-\u9fa5.!?，。？]+", r" ", s)
    return s

def TestModel(numIters, eta, reg, mode, train_data, test_data, train_label, test_label):
    #train_set = readExamples('data/data_rt.train')
    #train_new = []
    #for example in train_set:
        #text = normalize_string(example[0])
        #label = example[1]
        #train_new.append((text, label))

    
    #train_corpus = [example[0] for example in train_new]
    
    #train_label = [example[1] for example in train_new]
   
    #train_embed = extractFeatures(train_corpus, mode)
    
    #train_data, val_data, train_l, val_l = train_test_split(train_embed, train_label, test_size = 0.1, random_state = 31)
    weight, bias, training_loss, test_error_list = learnPredictor(train_data, test_data, train_label, test_label, numIters=numIters, eta=eta, reg = reg)
    
    plot_loss(training_loss, 'train', mode, eta)
    plot_loss(test_error_list, 'test', mode, eta)

    trainError = evaluatePredictor(train_data, train_label, weight, bias)
    testError = evaluatePredictor(test_data, test_label, weight, bias)
    
    print ("training error = %s, test error = %s" % (trainError, testError))
    return weight, bias, testError

if __name__ == "__main__":
    train_set = readExamples('data/data_rt.train')
    train_new = []
    for example in train_set:
        text = normalize_string(example[0])
        label = example[1]
        train_new.append((text, label))

    train_corpus = [example[0] for example in train_new]
    train_label = [example[1] for example in train_new]
    
    test_set = readExamples('data/data_rt.test')
    test_new = []
    for example in test_set:
        text = normalize_string(example[0])
        label = example[1]
        test_new.append((text, label))
    
    test_corpus = [example[0] for example in test_new]
    test_label = [example[1] for example in test_new]
    
    train_num = len(train_corpus)
    mode_list = ['BOW', 'Bigram', 'Trigram', 'Combo', 'Word2Vec', 'Glove']
    combo_embed = extractFeatures(train_corpus + test_corpus, mode_list[1])
    train_embed = combo_embed[:train_num]
    test_embed = combo_embed[train_num:]
    #train_data, val_data, train_l, val_l = train_test_split(train_embed, train_label, test_size = 0.1, random_state = 31)

    min_error = 1.0
    iters_list = [200000]
    learning_rate = [0.02, 0.1]
    reg_list = [0.0]
    best_w = None
    best_b = None
    best_combo = None
    for iters in iters_list:
        for lr in learning_rate:
            for reg in reg_list:
                for mode in mode_list:
                    combo_embed = extractFeatures(train_corpus + test_corpus, mode)
                    train_embed = combo_embed[:train_num]
                    test_embed = combo_embed[train_num:]
                    w, b, testError = TestModel(iters, lr, reg, mode, train_embed, test_embed, train_label, test_label)
                    print('Test error of {} is: {}'.format(mode, testError))
                    
                    #if valError < min_error:
                        #best_w = w
                        #best_b = b
                        #best_combo = [iters, lr, reg]
    #print('Best hyper combo is: ')
    #print(best_combo)

    #testError = evaluatePredictor(test_embed, test_label, best_w, best_b)
    #print('Test error is: {}'.format(testError))
