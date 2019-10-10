#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:17:34 2017

@author: karankothari
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Imputer
import csv
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn import metrics
import scipy.sparse as sp
 
from sklearn.decomposition import LatentDirichletAllocation as LDA
from gensim import corpora,matutils,similarities
from gensim.corpora import Dictionary as dictbuild
from nltk.corpus import stopwords
from gensim.models import ldamulticore
from gensim.models import LdaModel
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import naive_bayes
from random import shuffle
from gensim.models import word2vec
from sklearn.neural_network import MLPClassifier

#from deep_learning import convert_docs,create_CNN,load_CNN,create_RNN,load_RNN,classify_DNN,pred_user_dnn
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from pprint import pprint   # pretty-printer
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics import roc_auc_score, auc
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from sklearn.cross_validation import cross_val_score
from final_cleaner import normalize_user
import datetime
from classification2 import MajorityVoteClf
from classification2 import cross_ens
import eli5
from final_deploy import readcsv, read_compare_set
from final_deploy import tfidf, bow, save_html,user_transform_lda,lsa_test_transform,getAvgFeatureVecs
from final_deploy import makeFeatureVec,LabeledLineSentence
import pandas as pd
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
from gensim.corpora import Dictionary as dictbuild
import os
import numpy as np
from final_deploy import voted
from final_deploy import get_filename,get_most_recent, get_most_recent_classifiers
from flask import Markup
from flask import flash


# Preparing the Classifier
raw_data = r'raw_data'
features_path= r'features'
trained = r'trained_classifiers'
model_path = r'model'
model_date = get_most_recent_classifiers(trained)
print('Selecting the classifier created on:', model_date)
transformer = 'lsa'
model_name='mv_lsa'
path_model = get_filename(name=transformer,ext='.model',date=model_date,folder_path = model_path)
lsa_model = pickle.load(open(os.path.join(model_path,path_model),'rb'))

transformer='lda'
model_name='mv_lda'
path_model = get_filename(name=transformer,ext='.model',date=model_date,folder_path = model_path)
lda_model = pickle.load(open(os.path.join(model_path,path_model),'rb'))
finalDict = dictbuild.load('vocab/lda_dict.dict')
path_clf= get_filename(name=model_name,ext='.clf',date=model_date,folder_path=trained)

transformer='doc2vec'
model_name='mv_doc2vec'
path_model = get_filename(name=transformer,ext='.model',date=model_date,folder_path = model_path)
doc_model = pickle.load(open(os.path.join(model_path,path_model),'rb'))

transformer='word2vec'
model_name='mv_word2vec'
path_model = get_filename(name=transformer,ext='.model',date=model_date,folder_path = model_path)
word_model = pickle.load(open(os.path.join(model_path,path_model),'rb'))
            
            
transformer='ensemble'
model_name='mv_ensemble'
path_clf= get_filename(name=model_name,ext='.clf',date=model_date,folder_path=trained)
ensemble_clf = pickle.load(open(os.path.join(trained,path_clf),'rb'))

def classify(user_text):
    import classification
    import final_deploy
    from classification import cross_val
    from final_deploy import getAvgFeatureVecs
    
    clf2  = RandomForestClassifier(n_estimators=1000)#,class_weight={0:.95,1:.05})
    clf1 = linear_model.LogisticRegression(penalty='l1')#,class_weight={0:.95,1:.05})
    
    import time
    from classification import pred_user
    import pandas as pd
    from classification2 import pred_user_ens
    from final_cleaner import normalize_user
    most_recent= get_most_recent(raw_data)
    current_cleaned = get_filename(name='preprocessed',ext='.csv',date=most_recent,folder_path=raw_data)
    path = os.path.join(raw_data,current_cleaned)

    cleaned_doc = normalize_user(user_text)
    clean_user_token=cleaned_doc.split()
    #the reason for the list is so that the code of the model does not need to be changed
    clean_user_list=[' '.join(clean_user_token)]        
    #the user string is read for the transformation of getting features
    clas, documents= readcsv(path)
    #voc=get_voc(documents)
    model_date, confidence, _= read_compare_set()
    label=[]
    probability=[]
    global finalDict
    global lda_model
    global lsa_model
    global ensemble_clf
    #global rnn_model
    #global rnnvocab_processor
    #global cnnvocab_processor
    #global cnn_model
    global doc_model
    global word_model
    
    docs_tfidf, user_tfidf,tfi = tfidf(user_string=clean_user_list,docs=documents)#,vocab=voc)
    user_lsa = lsa_test_transform(lsa_model,user_tfidf)
    print('lsa done')
    
    user_lda = user_transform_lda(clean_user_list, finalDict,lda_model)
    print('lda done')
       
    clean_user_l= [x.split() for x in clean_user_list]
    user_doc = getAvgFeatureVecs(clean_user_l,doc_model)
    user_doc = Imputer().fit_transform(user_doc)
    print('doc2vec done')
    
    user=[]
    for x in clean_user_list:
        user.append(x.split())
        user_word = getAvgFeatureVecs(user,word_model)
        user_word = Imputer().fit_transform(user_word)
    print('word2vec done')
    
    transformer = 'bow'
    model_name='mv_bow'
    docs_bow, user_bow,tf = bow(user_string=clean_user_list,docs=documents)#,vocab=voc)
    print('bow done')
    clf2.fit(docs_bow,clas)
    #abcx=sorted(zip(map(lambda x: round(x, 4), clf2.feature_importances_), voc), reverse=True)
    #top_terms = abcx[:15]
    #top_terms = [y for x,y in top_terms]
    explain = save_html(clf2,cleaned_doc,tf)
    
    user_string = [user_bow.toarray(),user_tfidf.toarray(),user_lda,user_lsa,user_doc,user_word]
    
    
    result,confidence=  pred_user_ens(user_string, ensemble_clf)     
    label.append(result)
    probability.append(confidence)
    print(label)
    return result ,confidence,explain
    
app = Flask(__name__)
app.secret_key = 'random string'

class ReviewForm(Form):
	moviereview = TextAreaField('',
			[validators.DataRequired(), validators.length(min=100)])

@app.route('/css/styles.css') # the route name, <file> is like a request.args
def css():
  return render_template('styles.css') 
 
@app.route('/')
def index():
	form = ReviewForm(request.form)
	return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba,explain = classify(review)
        explain = Markup(explain)
        flash(explain)
        return render_template('results.html',
                               content=review,
                               prediction=y,
                               probability=proba,
                               explaination=explain)
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])

def feedback():
	feedback = request.form['feedback_button']
	review = request.form['review']
	prediction = request.form['prediction']
	inv_label = {'Early Dementia': 0, 'Normal': 1}
	y = inv_label[prediction]
	if feedback == 'Incorrect':
		y = int(not(y))
	train(review, y)
	sqlite_entry(db, review, y)
	return render_template('thanks.html')

if __name__ == '__main__':
	app.run(debug=True,port=5000)

