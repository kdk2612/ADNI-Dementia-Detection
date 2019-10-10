# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 09:24:58 2017

@author: kkothari
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from tflearn.data_utils import VocabularyProcessor
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.estimator import regression

import csv
import re,pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn import decomposition
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from matplotlib import pyplot as plt
from sklearn import metrics
import scipy.sparse as sp
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn import naive_bayes
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.metrics import matthews_corrcoef

from gensim import corpora,matutils,similarities
from gensim.corpora import Dictionary as dictbuild
from nltk.corpus import stopwords
from gensim.models import ldamulticore
from gensim.models import LdaModel
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from gensim.models import HdpModel
from pprint import pprint   # pretty-printer
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from pprint import pprint   # pretty-printer
from nltk.stem import PorterStemmer, WordNetLemmatizer
import io
from sklearn.metrics import roc_auc_score, auc
from collections import Counter
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim import utils
from random import shuffle
import os
import pandas as pd
from tensorflow.contrib.learn.python.learn.preprocessing import CategoricalVocabulary

#raw_data = r'raw_data'
#voca_file=r'final_voc.csv'
#voc_file= pd.read_csv(os.path.join(raw_data,voca_file))
#voc= list(set(voc_file['word']))
#print(voc[1:5])
#dnn_vocab= CategoricalVocabulary()
#for x in voc:
#    dnn_vocab.add(x)
#dnn_vocab.freeze()
#print(len(dnn_vocab))


def convert_docs(documents,no_class=2,MAX_DOCUMENT_LENGTH=200,vocab=None):
    '''Takes list of docs and associated clas list as input.
    Prepares it for the tflearn library. documents should be a list of strings and 
    clas should be a numbered list of classes encoded into 0,1,2 etc.
    no_classes is the number of classes that are going to be used in the model
    this is defaulted to 2'''
    
    if MAX_DOCUMENT_LENGTH is None:
        list_docs = []
        for x in documents:
            list_docs.append(x.split())
    
        MAX_DOCUMENT_LENGTH = max(len(l) for l in list_docs) 
        print(MAX_DOCUMENT_LENGTH)
    else:
        MAX_DOCUMENT_LENGTH=MAX_DOCUMENT_LENGTH
    
    if vocab is None:
        vocab=None
        vocab_processor = VocabularyProcessor(MAX_DOCUMENT_LENGTH,min_frequency=10)
        data = np.array(list(vocab_processor.fit_transform(documents)))
        n_words = len(vocab_processor.vocabulary_)
    elif vocab is not None:
        vocab_processor=vocab
        #vocab_processor = VocabularyProcessor(MAX_DOCUMENT_LENGTH,vocabulary=vocab)
        data = np.array(list(vocab_processor.transform(documents)))
        n_words = len(vocab_processor.vocabulary_)
    
    return data,vocab_processor, n_words, MAX_DOCUMENT_LENGTH



    
def classify_DNN(data,clas,model):
    from sklearn.cross_validation import StratifiedKFold
    folds = 10 #number of folds for the cv 
    skf = StratifiedKFold(n_folds=folds,y=clas)
    fold = 1
    cms = np.array([[0,0],[0,0]])
    accs = []
    aucs=[]
    mccs=[]
    for train_index, test_index in skf:
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = clas[train_index], clas[test_index]
        trainy= to_categorical(y_train, nb_classes=2)
        model.fit(X_train, trainy, n_epoch = 10, shuffle=True)
        prediction = model.predict(X_test)
        pred=np.argmax(prediction,axis=1)
        acc = accuracy_score(y_test,pred)
        mcc=matthews_corrcoef(y_test,pred)
        cm = confusion_matrix(y_test,pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
        print('Test Accuracy for fold {} : {}'.format(fold,round((acc*100),2)))
        au = metrics.auc(fpr, tpr)
        #au=roc_auc_score(testY, pred)
        print('AUC for fold {} : {}'.format(fold,round((au*100),2)))
        fold +=1
        cms += cm
        accs.append(acc)
        aucs.append(au)
        mccs.append(mcc)

    #print('CV test accuracy: {}\n{}'.format(round((np.mean(accs)*100),2),cms))
    #print('\nCV AUC: {}'.format(round(np.mean(aucs)*100),2))
    fp,tn,fn,tp=cms[0][1],cms[0][0],cms[1][0],cms[1][1]
    fpr=fp/(fp+tn)
    tpr=tp/(tp+fn)
    mcc = (tp*tn - fp*fn) / math.sqrt( float(tp + fp)*float(tp + fn)*float(tn + fp)*float(tn + fn) )
    print('\nCV accuracy: %.3f +/- %.3f' % (round((np.mean(accs)*100),2),round((np.std(accs)*100),2)))
    print('\nCV ROC AUC: %.3f +/- %.3f' % (round((np.mean(aucs)*100),2),round((np.std(aucs)*100),2)))
    print('\nCV False Positive Rate: %.3f '% (round(fpr*100,2)))
    print('\nCV True Positive Rate: %.3f ' % (round(tpr*100,2)))
    print(mcc)
    print('mean',np.mean(mccs))
    return model, round(np.mean(accs)*100,2), round(np.mean(aucs)*100,2),fpr,tpr
    
    
def create_CNN(MAX_DOCUMENT_LENGTH,n_words,data,clas,model_path):
    graph1 = tf.Graph()
    with graph1.as_default():
        network = input_data(shape=[None, MAX_DOCUMENT_LENGTH])
        network = tflearn.embedding(network, input_dim=n_words, output_dim=128)
        branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
        branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
        branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
        network = merge([branch1, branch2, branch3], mode='concat', axis=1)
        network = tf.expand_dims(network, 2)
        network = global_max_pool(network)
        network = dropout(network, 0.5)
        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy', name='target')
        model = tflearn.DNN(network, tensorboard_verbose=0)
        clf, acc, roc_auc,fpr,tpr =classify_DNN(data,clas,model)
        clf.save(model_path)
    return acc,roc_auc,fpr,tpr

def load_CNN(MAX_DOCUMENT_LENGTH,n_words,model_path):
    MODEL = None
    with tf.Graph().as_default():
    # Building deep neural network
        network = input_data(shape=[None, MAX_DOCUMENT_LENGTH])
        network = tflearn.embedding(network, input_dim=n_words, output_dim=128)
        branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
        branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
        branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
        network = merge([branch1, branch2, branch3], mode='concat', axis=1)
        network = tf.expand_dims(network, 2)
        network = global_max_pool(network)
        network = dropout(network, 0.5)
        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy', name='target')
        new_model = tflearn.DNN(network, tensorboard_verbose=3)
        new_model.load(model_path)
        MODEL = new_model
        return MODEL  
    
def create_RNN(MAX_DOCUMENT_LENGTH,n_words,data,clas,model_path):
    graph1 = tf.Graph()
    with graph1.as_default():
        # Building deep neural network
        net = input_data(shape=[None, MAX_DOCUMENT_LENGTH])
        net = embedding(net, input_dim=n_words, output_dim=200)
        net = bidirectional_rnn(net, BasicLSTMCell(200), BasicLSTMCell(200))
        net = dropout(net, 0.5)
        net = fully_connected(net, 2, activation='softmax')
        net = regression(net, optimizer='adam', loss='categorical_crossentropy')
        model = tflearn.DNN(net, tensorboard_verbose=0)
        clf, acc, roc_auc,fpr,tpr =classify_DNN(data,clas,model)
        clf.save(model_path)
    return acc,roc_auc,fpr,tpr

def load_RNN(MAX_DOCUMENT_LENGTH,n_words,model_path):
    MODEL = None
    with tf.Graph().as_default():
    # Building deep neural network
        net = input_data(shape=[None, MAX_DOCUMENT_LENGTH])
        net = embedding(net, input_dim=n_words, output_dim=200)
        net = bidirectional_rnn(net, BasicLSTMCell(200), BasicLSTMCell(200))
        net = dropout(net, 0.5)
        net = fully_connected(net, 2, activation='softmax')
        net = regression(net, optimizer='adam', loss='categorical_crossentropy')
        # Training
        # load the trained model
        new_model = tflearn.DNN(net, tensorboard_verbose=3)
        new_model.load(model_path)
        MODEL = new_model
        return MODEL
    
    

    
def pred_user_dnn(user_transformed, clf, y=None):
    '''
    Used for predicting the class of the user string given the transformed user input and the pretrained classifier
    Arguments:
        user_transformed= the transformed doc using the one used on the training data.. Must have same dimension as the training data
        clf= classifier pre trained on the training data of the one returned from cros_val()
        y= the training labels
    returns:
        string- Yes if the predicted label is 0
        No is the predicted label is 1
    '''

    usr_p = clf.predict(user_transformed)
    prob=np.max(usr_p)*100
    usr_p= np.argmax(usr_p,1)
    print('\nUser class'+str(usr_p))
    for x in usr_p:
        if x==0:
            print("Case recovery eligibility is: Yes")
            return 'Yes',prob
        elif x==1:
            print("Case recovery eligibility is: No")
            return 'No',prob
#            
            
            
            
            
            