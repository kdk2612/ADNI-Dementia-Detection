# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:59:45 2016
 
@author: kkothari
"""
from __future__ import division, print_function, absolute_import
 

from classification import MajorityVoteClassifier
 
import os
import pickle
import time
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Imputer
import csv
from sklearn import linear_model
from matplotlib import pyplot as plt

from gensim.corpora import Dictionary as dictbuild
from nltk.corpus import stopwords
from gensim.models import LdaModel
from sklearn.svm import SVC
from random import shuffle
from gensim.models import word2vec

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from final_cleaner import normalize_user
import datetime
from classification2 import MajorityVoteClf
from classification2 import cross_ens
import eli5

 
    
def get_filename(name,ext,date,folder_path):
    '''Read the file for the given parameters
    Arguments:
        name= substring of the file name that is unique to that file
        ext= extension of the file on disk
        date = date in string format that you wish to read the file for
        folder_path = path of the folder that you wish to read the file from
    Return:
        read_bydate = the file that is read by the date specified
    '''
    for file in os.listdir(folder_path):
        if file in  name+ '_'+ str(date)+ext:
            read_bydate = file
            return read_bydate
 
 
def save_file(model,name, ext, date,folder_path):
    '''save the files to the disk with the specified parameters in the function
    Arguments:
        model: the model or trained clf that needs to be saved
        name= name to be given to the file.'''
        
    final_name = name.lower()+'_'+date+ext
    
    pickle.dump(model,open(os.path.join(folder_path,final_name),'wb'))
 
 
 
def get_most_recent(path):
    
    if path == None:
        print('ValueError:Please enter a valid path')
    else:
        match = [re.search(r'\d{4}-\d{2}-\d{2}', file) for file in  os.listdir(path)]
        if match.count(None)==len(match):
            print('Folder has no file with date in the File_name')
        else:
            dates= [datetime.datetime.strptime(m.group(), '%Y-%m-%d').date() for m in match if m != None]
            return str(max(dates))
    
    
def get_most_recent_classifiers(path):
    if path == None:
        print('ValueError:Please enter a valid path')
    else:
        match = [re.search(r'ensemble_\d{4}-\d{2}-\d{2}', file) for file in  os.listdir(path)]
        if match.count(None)==len(match):
            print('Folder has no Classifiers with date in the File_name')
        else:
            file = max([m.group() for m in match if m!=None])
            date_match = re.search(r'\d{4}-\d{2}-\d{2}', file).group()
            return date_match
    
    
    
#path='/Users/karankothari/Desktop/capstone_karan/' 
 
#setting the dates to be used to read the files
now= str(datetime.datetime.now().date())
 
#now= '2016-12-08'
start_delta = datetime.timedelta(weeks=1)
last_week = str(datetime.datetime.now().date()- start_delta)
 
 
#setting the path to the current directory that is required
current_dir = os.getcwd()
check_dir = os.path.basename(os.path.normpath(current_dir))
if check_dir == 'ADNI-Dementia-Detection':
    path = os.getcwd()
else:
    os.chdir(r'ADNI-Dementia-Detection')
    path = os.getcwd()
 
#paths of the folders that has different things in it
raw_data = r'raw_data'
features_path= r'features'
trained = r'trained_classifiers'
model_path = r'model'
 
most_recent= get_most_recent(raw_data)
current_cleaned = get_filename(name='preprocessed',ext='.csv',date=most_recent,folder_path=raw_data)
print('most recent file:',most_recent )
print('Current claned file:', current_cleaned)
data_path = os.path.join(raw_data,current_cleaned)
print('final data path', data_path)
m=587
 
 
def load_all_files(directory,ext):
    '''Not using this any more.. applied a different logic '''
    global features
    global path
    global trained
    global now 
    global last_week
    
    
    current_files={}
    previous_files= {}
    
    current_list = [file for file in os.listdir(directory) if '_'+ str(now)+ext in  file]
    previous_list = [file for file in os.listdir(directory) if '_'+str(last_week)+ext in file]
    
    
    for file in current_list:
        current_files[file] = pickle.load(open(os.path.join(directory,file), "rb"))
    
    for file in previous_list:
        previous_files[file]= pickle.load(open(os.path.join(directory,file),'rb'))
 
    return current_files, previous_files
        
def readcsv(path=data_path,label_column=0,doc_column=1):
    '''
    This function is used to read the read the column with the document and the label from the csv
    Takes in 3 arguments:
        path= the path to the .csv file
        label_column =(value of the column should be 'Y ' or 'N ') integer number of the label column  (numbering starts with 0 so be sure to provide the right column)
        doc_column= integer number of the column with documents (numbering starts with 0 so be sure to provide the right column)
    Returns:
        class = numpy array of the class with 0:Y and 1:N
        documents= list of strings-> each string represents 1 document
        
    '''
    csvReader = csv.reader(open(data_path,'r',encoding='utf-8'))
    clas=[]
    for row in csvReader:
        clas.append((row[label_column]))
    c=[]
    for x in clas:
        if x=='NL':
            c.append(1)
        elif x=='EMCI':
            c.append(0)
    clas2=np.asarray(c)
    csvReader = csv.reader(open(data_path,'r',encoding='utf-8'))
    documents=[]
    next(csvReader)
    for row in csvReader:
        documents.append(row[doc_column])
    print('Documents read.')
    return clas2, documents 
    
 
clas, documents= readcsv(path)
 
def bow(docs,*positional_parameters,**keyword_parameters):
    '''
    This function returns a BOW model for the docs and the user_string
    BOW- words and thier frequency.
    Can be modified to get the vocabulary as well.
    Arguments:
        user_string= the doc that the use inputs for clssification
        docs= the list of docs collected from read_csv
        vocab= Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are indices in the feature matrix, or an iterable over terms. 
        If not given, a vocabulary is determined from the input documents. 
        Indices in the mapping should not be repeated and should not have any gap between 0 and the largest index.
    Returns: 
        documents= the transformed docs (matrix)
        user_transformed= the matrix for the user string
    '''
    if 'vocab' in keyword_parameters:
        for key,value in keyword_parameters.items():
            if key =='vocab':
                vocab = value
    else:
        vocab=None
    
    if 'user_string' in keyword_parameters:
        for key,value in keyword_parameters.items():
            if key =='user_string':
                user_str = value
        tf = CountVectorizer(vocabulary=vocab)
        doc_bow = tf.fit_transform(docs)
        user_transformed = tf.transform(user_str) 
        return doc_bow,user_transformed,tf
    else:
        #pickle.load(open("bow_feature.pkl", "rb"))
        tf = CountVectorizer(vocabulary=vocab) 
        doc_bow = tf.fit_transform(docs)
        return doc_bow,tf
    print("Bag of words done")
 
    
def tfidf(docs,*positional_parameters,**keyword_parameters):
    ''' This function returns a tfidf model for the docs and the user_string
    tfidf- words and the term freq inverse the doc frequency.
    Can be modified to get the vocabulary as well.
    Arguments:
        user_string= the doc that the use inputs for clssification
        docs= the list of docs collected from read_csv
        vocab= Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are indices in the feature matrix, or an iterable over terms. 
        If not given, a vocabulary is determined from the input documents and the same will be used for the user string. 
        Indices in the mapping should not be repeated and should not have any gap between 0 and the largest index.
    Returns:
        documents= the transformed docs (matrix)
        user_transformed= the matrix for the user string'''
    
    
    if 'vocab' in keyword_parameters:
        for key,value in keyword_parameters.items():
            if key =='vocab':
                vocab = value
    else:
        vocab=None
        
    if 'user_string' in keyword_parameters:
        for key,value in keyword_parameters.items():
            if key =='user_string':
                user_str = value
        vectorizer = TfidfVectorizer(min_df=2, stop_words='english',use_idf=True,vocabulary=vocab)
        docs_tfidf = vectorizer.fit_transform(docs)
        vocab = vectorizer.vocabulary_
        user_tfidf = vectorizer.transform(user_str) 
        return docs_tfidf, user_tfidf,vectorizer
    else:
        ##pickle.load(open("tfidf_feature.pkl", "rb"))
        vectorizer = TfidfVectorizer(min_df=2, stop_words='english',use_idf=True,vocabulary=vocab) 
        docs_tfidf = vectorizer.fit_transform(docs)
        return docs_tfidf,vectorizer
 
    
def load_vocab(feature_type=None):
    if feature_type =='BOW':
        save_vocab = pickle.load(open("bow_feature.pkl", "rb"))
    elif feature_type == 'TFIDF':
        save_vocab = pickle.load(open("tfidf_feature.pkl", "rb"))
    else:
        print('Wrong selection, Please choose BOW or TFIDF')
    return save_vocab
 
    
    
class LabeledLineSentence(object):
    
    def __init__(self, documents):
        self.sources = documents
    
    def __iter__(self):
        prefix='doc'
        for item_no, line in enumerate(documents):
            print(item_no)
            yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        prefix='doc'
        self.sentences = []
        for item_no, line in enumerate(documents):
            self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences
     
    
    
    
def initiate_doc2vec(sentences,n=10, existing_model=None):
    '''
    initiates a Doc2Vec model for the documnets that have been inputed
    Arguments:
        docs= the docs converted using the convert_to()
        array_for_vocab = the array obtained from to_array()
        n= number of training iterations (default =10)
        existing_model= path to the pretrained model that was used previously (default None)
    Returns:
        returns the trained model for Doc2Vec that can be used in different ways
    '''
    if existing_model == None:
        print('Initializing the Doc2vec model: Might take time')
        model = Doc2Vec(min_count=1,dm=1, window=10, size=200, sample=1e-4, negative=10, workers=7,alpha=0.5)
        model.build_vocab(sentences.to_array())
        for epoch in range(10):
            model.train(sentences.sentences_perm(),total_examples=None,total_words=model.corpus_count,epochs=model.iter)
       
    elif existing_model!=None:
        model = Doc2Vec.load(existing_model)
    return model
 
 
def makeFeatureVec(words, model, num_features):
    """
    given a list of docs, define the feature vector by averaging the feature vectors
    of all words that exist in the model vocabulary in the doc
    :param docs:
    :param model:
    :param num_features:
    :return:
    """
 
    featureVec = np.zeros(num_features, dtype=np.float32)
    nwords = 0
    # index2word is the list of the names of the words in the model's vocabulary.
    # convert it to set for speed
    vocabulary_set = set(model.wv.index2word)
    
    # loop over each word in the review and add its feature vector to the total
    # if the word is in the model's vocabulary
    for word in words:
        if word in vocabulary_set:
            nwords = nwords + 1
            # add arguments element-wise
            # if x1.shape != x2.shape, they must be able to be casted
            # to a common shape
            featureVec = np.add(featureVec, model[word])
    
    featureVec = np.true_divide(featureVec,nwords)
    return featureVec
    
    
def getAvgFeatureVecs (docs, model):
    '''
    passes the data to makeFeatureVec and stores the data for the same in a new list
    Arguments:
        docs: the list of docs or single doc(user string)
        model: the model obtained from the initiate_doc2vec
    Returns:
        docFV= transformed feature vector of the list of docs which will be used for classification
    Usage:
        features = getAvgFeatureVecs(docs,model)
    '''
    # initialize variables
    counter = 0
    num_features = model.wv.syn0.shape[1]
    if type(docs) is list:
        docsFV = np.zeros((len(docs),num_features), dtype=np.float32)
        for line in docs:
            docsFV[counter] = makeFeatureVec(line, model, num_features)
            counter += 1
    elif type(docs) is str:
        docsFV=np.zeros((len(docs),num_features), dtype=np.float32)
        docsFV[1] = makeFeatureVec(docs, model, num_features)
        
    return docsFV  
 
 
    
    
def lsa(docs_tfidf):
    '''
    Transforms the tfidf features to LSA features.
    Arguments:
        docs_tfidf= Feature of the tfidf()
    Returns:
        lsa_features= LSA features'''
    t0 = time.time()
    # Project the tfidf vectors onto the first 150 principal components.
    # Though this is significantly fewer features than the original tfidf vector,
    # they are stronger features, and the accuracy is higher.
    svd = TruncatedSVD(300)
    lsa = make_pipeline(svd, Normalizer(copy=False))
            
    # Run SVD on the training data, then project the training data.
    lsa_features = lsa.fit_transform(docs_tfidf)
    print("  done in %.3fsec" % (time.time() - t0))
    explained_variance = svd.explained_variance_ratio_.sum()
    print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    return lsa,lsa_features
   
    
def lsa_test_transform(model,user_tfidf):
    '''
    Used for transforming the user string for LSA features
    Arguments:
        modle= obtained from the lsa()
        user_tfidf= the tfidf feature version of the user string obtained from tfidf()
    Returns:
        user_feature= lsa transformed feature of the user strings'''
    lsa=model
    user_feature = lsa.transform(user_tfidf)
    return user_feature
    
    
    
def initiate_LDA(docs=None,no_topics=100,*positional_parameters,**keyword_parameters):
    '''
    Initiates a 
    '''
    finalDict = dictbuild()
    stringSplit = re.compile('\\s')
    stop = stopwords.words('english')
    if 'vocab' in keyword_parameters:
        for key,value in keyword_parameters.items():
            if key =='vocab':
                vocab = value
                finalDict.add_documents([vocab])
    else:
        
        count =0
        for line in docs:
            lines = stringSplit.split(line.lower())
            finalDict.add_documents([lines])
            count+=1
            
    for_corpus=[]
    for line in docs:
        lines = stringSplit.split(line.lower())
        for_corpus.append(lines)
    finalDict.filter_tokens(stop)
    #once_ids = [tokenid for tokenid, docfreq in finalDict.dfs.items() if docfreq <6] #5642 
    #finalDict.filter_tokens(once_ids)
    finalDict.compactify() 
    
    
    corpus = [finalDict.doc2bow(line) for line in for_corpus]
    
    n=no_topics
    lda_model = LdaModel(corpus,num_topics=n,id2word=finalDict)
    topicsDist=[]
    for x in corpus:
        topics = lda_model[x]
        temp=[0]*n
        for t in topics:
            temp[t[0]]=t[1]
        topicsDist.append(temp)
    
    print("topics generated")    
    topicsDist = np.asarray(topicsDist)
    return topicsDist, finalDict,lda_model
 
    
def user_transform_lda(user_string, dictionary,model,no_topics=100):
    
    x=str(user_string)[2:-2]
    bow_new = dictionary.doc2bow(x.split())
    
    user_topic=model[bow_new]
    
    new_temp= [0]*no_topics
    for t in user_topic:
        new_temp[t[0]]=t[1]
    userdist = np.asarray(new_temp)
    u=list(userdist)
    userdist=np.asarray(u)
    user_features=userdist.reshape(1,-1)
    return user_features
 
    
 
def save_results(date,area_under_curve,clf_name):
    if date==None or area_under_curve==None or clf_name == None:
        print('Please enter all the variables, cannot work without date or accuracy or auc or clf_name')
    else:
        myCsvRow = [date,area_under_curve,clf_name]
        with open(r'weekly_results.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(myCsvRow)      
                
                
def save_html(classifier,documnet,transformer):
    
    import eli5
    from bs4 import BeautifulSoup, Tag
    v=transformer.get_feature_names()
    html=eli5.show_prediction(classifier,documnet,vec=transformer,top=len(v)).data
    soup = BeautifulSoup(html,"lxml")
    for tag in soup.find_all(['table','b']):
        tag.replaceWith('')
    html=str(soup)
    html=html.replace('(probability )','')
    html=html.replace('top features','')
    return html
            
 
def read_compare_set():
    '''
    Get the best model name and the date of the model with the best estimator paramenters
    Arguments:
        blank- It reas the weekly_results csv file and gets the best results
    Return:
        model_data: the date associated to the best parameter
        model_name: the date associated to the best parameter
    '''
    rows=[]
    with open(r'weekly_results.csv','r') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for r in csvreader:
            rows.append(r)
    rows = [x for x in rows if x != []]
    rows.sort(key = lambda row: (row[1]))
    model_date = rows[-1][0]
    confidence= rows[-1][1]
    model_name = rows[-1][2]
    return model_date,confidence, model_name  
 
    
def for_api_input(user_text,plan_type,plan_year):
    '''
    This function is used for the API input and out as the final model
    the function takes 3 inputs
    Arguments:
        user_text = string... the doc that needs to be classified
        plan_type= string... the type of the plan
        plan_year= string... the year of the plan
    Returns:
        Recovery_eligibility= if the user is eligible 
        confidencelevel= the confidedce of the prediction
        explaination= the explaination of the classification
        dateRecieved = When the model started execution
        date'''
    import time
    from classification import pred_user
    import pandas as pd
    voca_file=r'final_voc.csv'
    voc_file= pd.read_csv(os.path.join(raw_data,voca_file))
    voc= list(set(voc_file['word']))
    print(voc[1:5])
    dnn_vocab= CategoricalVocabulary()
    for x in voc:
        dnn_vocab.add(x)
    dnn_vocab.freeze()
    global m
    #print(type(user_text))
    #user_text= user_text.encode('utf-8')
    if not user_text or type(user_text)!= str or \
     not plan_type or not  plan_year:
        #type_is = type(user_text)
        raise ValueError('User input needs to be string')
        #print("Invalid input,\n Please check the input\n Need the input as str")
    else:
        modelstart = time.strftime("%c")
        cleaned_doc = normalize_user(user_text)
        clean_user_token=cleaned_doc.split()
        #the reason for the list is so that the code of the model does not need to be changed
        clean_user_list=[' '.join(clean_user_token)]        
        #the user string is read for the transformation of getting features
        model_date, confidence, model_name= read_compare_set()
        clas, documents= readcsv(path)
        if '_' in model_name:
            transformer = model_name.split('_')[1]
        else: transformer = model_name
        print(transformer)
        if transformer == 'doc2vec':
            path_model = get_filename(name=transformer,ext='.model',date=model_date,folder_path = model_path)
            trained_model = pickle.load(open(os.path.join(model_path,path_model),'rb'))
            clean_user_list= [x.split() for x in clean_user_list]
            user_transformed = getAvgFeatureVecs(clean_user_list,trained_model)
            print(user_transformed)
            user_transformed = Imputer().fit_transform(user_transformed)
            path_clf= get_filename(name=model_name,ext='.clf',date=model_date,folder_path=trained)
            trained_clf = pickle.load(open(os.path.join(trained,path_clf),'rb'))
            result,confidence =  pred_user(user_transformed, trained_clf)     
        
        
        elif transformer == 'lsa':
            path_model = get_filename(name=transformer,ext='.model',date=model_date,folder_path = model_path)
            trained_model = pickle.load(open(os.path.join(model_path,path_model),'rb'))
            docs_tfidf, user_tfidf = tfidf(user_string=clean_user_list,docs=documents,vocab=voc)
            user_transformed = lsa_test_transform(trained_model,user_tfidf)
            trained_clf = pickle.load(open(os.path.join(trained,path_clf),'rb'))
            result,confidence =  pred_user(user_transformed, trained_clf)
        
        elif transformer=='lda':
            path_model = get_filename(name=transformer,ext='.model',date=model_date,folder_path = model_path)
            trained_model = pickle.load(open(os.path.join(model_path,path_model),'rb'))
            finalDict = dictbuild.load('vocab/lda_dict.dict')
            path_clf= get_filename(name=model_name,ext='.clf',date=model_date,folder_path=trained)
            user_transformed = user_transform_lda(clean_user_list, finalDict,trained_model)
            trained_clf = pickle.load(open(os.path.join(trained,path_clf),'rb'))
            result,confidence =  pred_user(user_transformed, trained_clf)
            
        elif transformer == 'bow':
            docs_transformed, user_transformed = bow(user_string=clean_user_list,docs=documents,vocab=voc)
            path_clf= get_filename(name=model_name,ext='.clf',date=model_date,folder_path=trained)
            trained_clf = pickle.load(open(os.path.join(trained,path_clf),'rb'))
            result,confidence =  pred_user(user_transformed, trained_clf)
            
        elif transformer =='tfidf':
            docs_tfidf, user_transformed = tfidf(user_string=clean_user_list,docs=documents,vocab=voc)
            path_clf= get_filename(name=model_name,ext='.clf',date=model_date,folder_path=trained)
            trained_clf = pickle.load(open(os.path.join(trained,path_clf),'rb'))
            result,confidence =  pred_user(user_transformed, trained_clf) 
            
        elif transformer == 'word2vec':
            path_model = get_filename(name=transformer,ext='.model',date=model_date,folder_path = model_path)
            trained_model = pickle.load(open(os.path.join(model_path,path_model),'rb'))
            user=[]
            for x in clean_user_list:
                user.append(x.split())
            user_transformed = getAvgFeatureVecs(user,trained_model)
            user_transformed = Imputer().fit_transform(user_transformed)
            print(user_transformed)
            path_clf= get_filename(name=model_name,ext='.clf',date=model_date,folder_path=trained)
            trained_clf = pickle.load(open(os.path.join(trained,path_clf),'rb'))
            print(path_clf)
            result,confidence =  pred_user(user_transformed, trained_clf)
        
        xpe= 'There will be an explaination'
        return({'recoveryeligible': result
                ,'confidencelevel': confidence
                ,'explanation': xpe
                ,'datereceived': modelstart
                ,'transformer used': transformer
                ,'datereturned': time.strftime("%c")})
                     


 
def weekly_run_create():
    
 
    import classification
    from classification import cross_val
    from classification import MajorityVoteClassifier
    
    clas, documents = readcsv(os.path.join(path,'raw_data',current_cleaned))

    ## Initiate the things we need 
    clf2  = RandomForestClassifier(n_estimators=1000)#,class_weight={1:0.05,0:.95})
    clf1 = linear_model.LogisticRegression(penalty='l1')#,class_weight={1:0.05,0:.95})
    clf3  = SVC(C=1000000.0,probability=True)#,class_weight={1:0.05,0:.95})
    clf4  = DecisionTreeClassifier()#class_weight={1:0.05,0:.95})
    clf5  = AdaBoostClassifier(n_estimators=1000)
    clf6 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,random_state=0)
    #clf7=  MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(200,150,50,20), random_state=1)
    clf_labels = ['Logistic Regression','Random Forest','SVC','Decision Tree','Ada Boost','Gradient Boosting','MLP-DNN']
    mv_clf = MajorityVoteClassifier(classifiers=[clf1, clf2, clf3,clf4,clf5,clf6])

    
    ## First we will do the BOW model and the Classifier trained on it
    print('\nstart BOW')
    docs_bow,_ = bow(docs=documents)#,vocab=voc)
    print(docs_bow.shape)
    save_file(model=docs_bow,name='bow', ext='.data', date=now,folder_path=features_path)
    bclf,bacc,broc_auc,bfp,btp=cross_val(docs_bow.toarray(),clas,folds=5,clf=mv_clf,cv=True)
    save_file(model=bclf,name='mv_bow',ext='.clf',date=now,folder_path=trained)
    save_results(date=now,area_under_curve=broc_auc,clf_name='mv_bow')
    print('BOW done')
    
    #Tfidf
    print('\nstart tfidf')
    docs_tfidf,_ = tfidf(docs=documents)#,vocab=voc)
    print(docs_tfidf.shape)

    save_file(model=docs_tfidf,name='tfidf', ext='.data', date=now,folder_path=features_path)
    tclf,tacc,troc_auc,tfp,ttp=cross_val(docs_tfidf.toarray(),clas,folds=5,clf=mv_clf,cv=True)    
    save_file(model=tclf,name='mv_tfidf',ext='.clf',date=now,folder_path=trained)
    save_results(date=now,area_under_curve=troc_auc,clf_name='mv_tfidf')
 
    print('\ntf-idf done')
    
    #LDA
    voc=None
    print('\nStart LDA')
    if voc:
        topicsDist, finalDict,lda_model = initiate_LDA(docs=documents,no_topics=100,vocab=voc)
    elif voc is None:
        topicsDist, finalDict,lda_model = initiate_LDA(docs=documents,no_topics=100)#,vocab=voc)
    print(topicsDist.shape)
    save_file(model=topicsDist,name='lda',ext='.data',date=now,folder_path=features_path)
    save_file(model=lda_model,name='lda',ext='.model',date=now,folder_path=model_path)
    finalDict.save('vocab/lda_dict.dict')
    ldclf,ldacc,ldroc_auc,ldafp,ldatp=cross_val(topicsDist,clas,folds=5,clf=mv_clf,cv=True)
    save_file(model=ldclf,name='mv_lda',ext='.clf',date=now,folder_path=trained)
    save_results(date=now,area_under_curve=ldroc_auc,clf_name='mv_lda')
    
    print('LDA done')
    
    #voc=get_voc(documents)
    #LSA
    print('\nStart LSA')
    docs_tfidf,_ = tfidf(docs=documents)#,vocab=voc)
    model,docs_lsa = lsa(docs_tfidf)
    print(docs_lsa.shape)
    save_file(model=docs_lsa,name='lsa',ext='.data',date=now,folder_path=features_path)
    #lsa model cannot be saved as we have used pipeline fr it
    save_file(model=model,name='lsa',ext='.model',date=now,folder_path=model_path)
    lsclf,lsacc,lsroc_auc,lsafp,lsatp=cross_val(docs_lsa,clas,folds=5,clf=mv_clf,cv=True)
    save_file(model=lsclf,name='mv_lsa',ext='.clf',date=now,folder_path=trained)
    save_results(date=now,area_under_curve=lsroc_auc,clf_name='mv_lsa')
 
    print('LSA done')
    
    
    
    print('\nStart doc2vec')
    sentences = LabeledLineSentence(documents)
    model = initiate_doc2vec(sentences,n=10)
    save_file(model = model,name='doc2vec',ext='.model',date=now,folder_path=model_path)
    doc_words=[]
    for x in documents:
        doc_words.append(x.split())
    feature_doc2vec = getAvgFeatureVecs(doc_words,model)
    feature_doc2vec = Imputer().fit_transform(feature_doc2vec)
    save_file(model=feature_doc2vec,name='doc2vec',ext='.data',date=now,folder_path=features_path)
    dclf,dacc,droc_auc,dfp,dtp=cross_val(feature_doc2vec,clas,folds=5,clf=mv_clf,cv=True)
    save_file(model=dclf,name='doc2vec',ext='.clf',date=now,folder_path=trained)
    save_results(date=now,area_under_curve=droc_auc,clf_name='mv_doc2vec')
    print('doc2vec done')
 
    print('\nStarting word2vec')
    bag_sentences=[]
    for x in documents:
        bag_sentences.append(x.split())
    print("Training word2vec model...")
    model = word2vec.Word2Vec(bag_sentences,sg=0, workers=4,size=300, min_count=1,window=10, sample=1e-3)
    model.init_sims(replace=True)
    save_file(model = model,name='word2vec',ext='.model',date=now,folder_path=model_path)
    datavec = getAvgFeatureVecs( bag_sentences, model )
    word_features = Imputer().fit_transform(datavec)
    save_file(model=word_features,name='word2vec',ext='.data',date=now,folder_path=features_path)
    wclf,wacc,wroc_auc,wfp,wtp=cross_val(word_features,clas,folds=5,clf=mv_clf,cv=True)
    save_file(model=wclf,name='word2vec',ext='.clf',date=now,folder_path=trained)
    save_results(date=now,area_under_curve=wroc_auc,clf_name='mv_word2vec')
    print('word2vec done')
#    
    print("\nStaring the final Ensemble")
    new_clf= [bclf,tclf,ldclf,lsclf,dclf,wclf]
    data = [docs_bow.toarray(),docs_tfidf.toarray(),topicsDist,docs_lsa,feature_doc2vec,word_features]
    mvc = MajorityVoteClf(classifiers=new_clf,weights=[1,5,3,1,5,1])
    mclf,macc,mroc_auc,mfp,mtp=cross_ens(clas,data,folds=5,clf=mvc,cv=True)
    save_file(model=mclf,name='ensemble',ext='.clf',date=now,folder_path=trained)
    save_results(date=now,area_under_curve=mroc_auc,clf_name='mv_ensemble')
    print('done Ensemble')
    

 
    import json
    with open('dominostats.json', 'w') as f:
        f.write(json.dumps({
                            #"RNN-ROC":rroc_auc,"RNN-FPR":rfp,"RNN-TPR":rtp,
                            #"CNN-ROC":croc_auc,"CNN-FPR":cfp,"CNN-TPR":ctp,
                            "BOW-ROC":broc_auc,"BOW-FPT":bfp,"BOW-TPR":btp,
                           "Tfidf-ROC":troc_auc,"Tfidf-FPT":tfp,"Tfidf-TPR":ttp,
                            "LDA-ROC": ldroc_auc,"LDA-FPR":ldafp,"LDA-TPR":ldatp,
                            "LSA-ROC":lsroc_auc,"LSA-FPR":lsafp,"LSA-TPR":lsatp,
                            "ParagraphEmbedding-ROC":droc_auc,"ParagraphEmbedding-FPR":dfp,"ParagraphEmbedding-TPR":dtp,
                            "WordEmbedding-ROC":wroc_auc,"WordEmbedding-FPR":wfp,"WordEmbedding-ROC":wtp,
                            "Ensemble-ROC":mroc_auc,"Ensemble-FPR":mfp,"Ensemble-ROC":mtp
            
        }))
 

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


def voted(user_text,plan_year,plan_type):
    import classification
    from classification import cross_val
    clf2  = RandomForestClassifier(n_estimators=1000,class_weight={0:.95,1:.05})
    clf1 = linear_model.LogisticRegression(penalty='l1',class_weight={0:.95,1:.05})

    import time
    from classification import pred_user
    import pandas as pd
    from classification2 import pred_user_ens

    
    if not user_text or type(user_text)!= str or \
     not plan_type or not  plan_year:
    #type_is = type(user_text)
        raise ValueError('User input needs to be string')
    #print("Invalid input,\n Please check the input\n Need the input as str")

    modelstart = time.strftime("%c")
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

    #xpe= 'Classification based on the language like'+ str(top_terms)
    return({'recoveryeligible': label
            ,'confidencelevel': confidence
            ,'explanation': explain
            ,'datereceived': modelstart
            ,'transformer used': 'Ensemble'
            ,'datereturned': time.strftime("%c")})
