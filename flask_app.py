#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:17:34 2017

@author: karankothari
"""

import pandas as pd
from flask import Flask, render_template, request, jsonify
from wtforms import Form, TextAreaField, validators
import pickle

from final_deploy import voted
from final_deploy import get_filename,get_most_recent, get_most_recent_classifiers



app = Flask(__name__)

def vote():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     #query = pd.get_dummies(query_df)
         import classification
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
    voc=get_voc(documents)
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

    docs_tfidf, user_tfidf,tfi = tfidf(user_string=clean_user_list,docs=documents,vocab=voc)
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
    docs_bow, user_bow,tf = bow(user_string=clean_user_list,docs=documents,vocab=voc)
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
    

#    print('starting prediction')
#    user_transformed = np.array(list(rnnvocab_processor.transform(clean_user_list)))
#    user_transformed = pad_sequences(sequences=user_transformed,maxlen=MAX_DOCUMENT_LENGTH, value=0.)
#    result,confidence = pred_user_dnn(user_transformed, rnn_model)
#    label.append(result)
#    probability.append(confidence)
    
#    print('starting prediction')
#    user_transformed = np.array(list(cnnvocab_processor.transform(clean_user_list)))
#    user_transformed = pad_sequences(sequences=user_transformed,maxlen=MAX_DOCUMENT_LENGTH, value=0.)
#    result,confidence = pred_user_dnn(user_transformed, cnn_model)             
#    label.append(result)        
#    probability.append(confidence)
#    print(label)
#    print(probability)
    
#    proba=[]
#    mapping={'Yes':0,'No':1}
#    predicted=np.asarray([mapping[x] for x in label])
#    
#
#
#    final_pred =np.apply_along_axis(lambda x:np.argmax(np.bincount(x,weights=probability)),axis=0,arr=predicted)
#    for x,y in zip(predicted,probability):
#        if x == final_pred:
#            proba.append(y)
#    probab = sum(proba) /len(proba)
#    print(final_pred, round(probab,2)*100)
#    return final_pred, probab
#    return label,confidence


    #xpe= 'Classification based on the language like'+ str(top_terms)
    return({'recoveryeligible': label
            ,'confidencelevel': confidence
            ,'explanation': explain
            ,'datereceived': modelstart
            ,'transformer used': 'Ensemble'
            ,'datereturned': time.strftime("%c")})
    
     #prediction = clf.predict(query)
     #return jsonify({'prediction': list(prediction)})



if __name__ == '__main__':
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
    app.run(port=8080)
