# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:56:01 2017
 
@author: kkothari
"""
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn import naive_bayes
from sklearn import metrics
 
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.metrics import matthews_corrcoef
from sklearn.pipeline import _name_estimators
from sklearn.externals import six
from sklearn.base import clone
import operator
 
 
class MajorityVoteClf(BaseEstimator, ClassifierMixin):
    """ A majority vote ensemble classifier
 
    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble
 
    vote : str, {'classlabel', 'probability'} (default='label')
      If 'classlabel' the prediction is based on the argmax of
        class labels. Else if 'probability', the argmax of
        the sum of probabilities is used to predict the class label
        (recommended for calibrated classifiers).
 
    weights : array-like, shape = [n_classifiers], optional (default=None)
      If a list of `int` or `float` values are provided, the classifiers
      are weighted by importance; Uses uniform weights if `weights=None`.
 
    """
    def __init__(self, classifiers, vote='classlabel', weights=None):
 
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
 
    def fit(self,y, X=[]):
        """ Fit classifiers.
 
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.
 
        y : array-like, shape = [n_samples]
            Vector of target class labels.
 
        Returns
        -------
        self : object
 
        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'"
                             "; got (vote=%r)"
                             % self.vote)
 
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))
 
        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf,x in zip(self.classifiers,X):
            fitted_clf = clone(clf).fit(x, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
 
    def predict(self, X):
        """ Predict class labels for X.
 
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.
 
        Returns
        ----------
        maj_vote : array-like, shape = [n_samples]
            Predicted class labels.
            
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' vote
 
            #  Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(x)
                                      for clf,x in zip(self.classifiers_,X)]).T
 
            maj_vote = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote
 
    def predict_proba(self, X):
        """ Predict class probabilities for X.
 
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
 
        Returns
        ----------
        avg_proba : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
 
        """
        probas = np.asarray([clf.predict_proba(x)
                             for clf,x in zip(self.classifiers_,X)])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba
 
    
def cross_ens(y,X=[],folds=10,clf=None,cv=True):
    '''
    Used for training the classifier given the data and the class
    default folds =10
    Arguments:
        clf= classifier- pretrained on training data
        X= The features in the form of numpy array
        y= clas label in form of numpy array
        cv =False, 
    Returns:
        Prints the output and the class
        if cv = True will return the best trained classifier from the cross validation,best accuracy, and best confusion matrix
        if cv = False will return the classifier that is trained on the entier data,
        
    '''
    if clf is None:
        print('Please provide a pre-trained classifer')
    else:
        clf=clf
    #kf = KFold(len(y), n_folds=folds)
    skf = StratifiedKFold(n_splits=folds)
    fold = 1
    cms = np.array([[0,0],[0,0]])
    accs = []
    aucs=[]
    mccs=[]
    best_cms = np.array([[0,0],[0,0]])
    best_auc=0
    best_acc, best_clf = 0, None
    print(len(X), len(y))
    recall=[]
    precision=[]
    y_predicted_overall = None
    y_test_overall = None
    bow=X[0]
    tfidf=X[1]
    lda=X[2]
    lsa=X[3]
    doc2vec=X[4]
    word2vec=X[5]
    print('collected data')
    if cv:
        for train_index, test_index in skf.split(bow, y):
            bow_train, bow_test = bow[train_index], bow[test_index]
            tfidf_train, tfidf_test = tfidf[train_index], tfidf[test_index]
            lda_train, lda_test = lda[train_index],lda[test_index]
            lsa_train,lsa_test = lsa[train_index], lsa[test_index]
            doc_train,doc_test = doc2vec[train_index],doc2vec[test_index]
            word_train,word_test = word2vec[train_index],word2vec[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(y_train,[bow_train,tfidf_train,lda_train,lsa_train,doc_train,word_train])
            prediction = clf.predict([bow_test,tfidf_test,lda_test,lsa_test,doc_test,word_test])
            acc = accuracy_score( y_test,prediction)
            cm = confusion_matrix(y_test,prediction)
            #misclassified = np.where(y_test != prediction)
            #missed_indices.append(misclassified)
            pred_probas = clf.predict_proba([bow_test,tfidf_test,lda_test,lsa_test,doc_test,word_test])[:,1]
            fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_probas)
            mcc=matthews_corrcoef(y_test,prediction)
            #print('Test Accuracy for fold {}: {}\n{}'.format(fold,round((acc*100),2),cm))
            roc_auc = auc(fpr,tpr)
            #print('AUC for fold {} : {}'.format(fold,round((roc_auc*100),2)))
            fold +=1
            cms += cm
            accs.append(acc)
            mccs.append(mcc)
            aucs.append(roc_auc)
            #y_test = np.asarray(y_test)
            if y_predicted_overall is None:
                y_predicted_overall = prediction
                y_test_overall = y_test
            else: 
                y_predicted_overall = np.concatenate([y_predicted_overall, prediction])
                y_test_overall = np.concatenate([y_test_overall, y_test])
            rec=recall_score( y_test,prediction,)
            pre=precision_score(y_test,prediction)
            recall.append(rec)
            precision.append(pre)
            #misclassified = np.where(y_test != clf.predict(X_test))
            #missed_indices.extend(misclassified)
            if acc > best_acc:
                best_acc, best_clf, = acc, clf
                best_auc = roc_auc
                best_cms = cm
            
        fp,tn,fn,tp=cms[0][1],cms[0][0],cms[1][0],cms[1][1]
        fpr=fp/(fp+tn)
        tpr=tp/(tp+fn)
        fnr=fn/(fn+tp)
        tnr=tn/(fp+tn)
        mcc = (tp*tn - fp*fn) / math.sqrt( float(tp + fp)*float(tp + fn)*float(tn + fp)*float(tn + fn) )
        print('Classification Report')
        print (metrics.classification_report(y_test_overall, y_predicted_overall, digits=3))
        print('Over all recall: ',recall_score(y_test_overall, y_predicted_overall))
        print('Over all precision: ',precision_score(y_test_overall, y_predicted_overall))
        print('Recall: {}'.format(np.mean(recall)))
        print('Precision: {}'.format(np.mean(precision)))
        print('CV accuracy: %.3f +/- %.3f' % (round(np.mean(accs)*100,2),round(np.std(accs)*100,2)))
        print('CV ROC AUC: %.3f +/- %.3f' % (round(np.mean(aucs)*100,2),round(np.std(aucs)*100,2)))
        print('CV False Positive Rate(Fallout): %.3f '% (round(fpr*100,2)))
        print('CV True Positive Rate(Hit Rate): %.3f ' % (round(tpr*100,2)))
        print('CV True Negative Rate(Specificity): %.3f ' % (round(tnr*100,2)))
        print('CV False Negative Rate(Miss Rate): %.3f ' % (round(fnr*100,2)))
        print('CV the Mathews Correlation coefficient: '+ str(mcc))
        clf_return = clf.fit(y,X)
        #print('\nPeak accuracy: '+str(round((np.amax(accs)*100),2)))
        #print('\nPeak ROC AUC: '+str(round((np.amax(aucs)*100),2)))
        return clf_return,round(np.mean(accs)*100,2),round(np.mean(aucs)*100,2),fpr,tpr#,missed_indices
    else:
        X_train= X
        y_train=y
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_train)
        single_acc = accuracy_score(prediction, y_train)
        cm = confusion_matrix(y_train,prediction)
        pred_probas = clf.predict_proba(X_train)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_train, pred_probas)
        #print('Test Accuracy for fold {}: {}\n{}'.format(fold,round((acc*100),2),cm))
        single_roc_auc = auc(fpr,tpr)
        #print('AUC for fold {} : {}'.format(fold,round((roc_auc*100),2)))
 
        print('\nTraining accuracy:  '+ str(round((single_acc)*100,2)))
        print('\nTraining ROC AUC:  '+str(round((single_roc_auc)*100,2)))
        #print('\nPeak accuracy: '+str(round((np.amax(accs)*100),2)))
        #print('\nPeak ROC AUC: '+str(round((np.amax(aucs)*100),2)))
        return clf,round((single_acc)*100,2),round((single_roc_auc)*100,2)
        
def pred_user_ens(user_transformed, clf, y=None):
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
    label = {1:'Normal',0:'Early Dementia'}
    predicted=label[clf.predict(user_transformed)[0]] ,np.max(clf.predict_proba(user_transformed))*100
    return predicted[0],predicted[1]
                    