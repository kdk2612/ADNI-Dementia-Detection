# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 09:48:40 2016

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
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
import math
from sklearn.metrics import matthews_corrcoef

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators
from sklearn.externals import six
from sklearn.base import clone
import operator
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def baseline_from_labels(labels):
    """Compute the majority class baseline of a set of enumerated labels

    Args:
        labels: list of labels, should already be enumerated NOT strings
    Returns:
        ( Fraction for majority class label,
          Majority class label/number )
    """
    counts = np.bincount(labels).astype('float')
    return counts.max()/counts.sum(), counts.argmax()

    
def baseline_from_cm(conf_mat):
    """Compute the majority class baseline of a confusion matrix

    Args:
        conf_mat: scikit confusion matrix with ROWS as GROUND TRUTH
    Returns:
        ( Fraction for majority class label,
          Majority class label/number )
    """
    # sum the ROWS (ground truth)
    counts = conf_mat.sum(axis=1, dtype='float')
    return counts.max()/counts.sum(), counts.argmax()

    
    


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
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

    def fit(self, X, y):
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
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
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
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

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
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

        
        
def using_max_vote(X_train,y_train):    
    clf = linear_model.LogisticRegression(penalty='l1',class_weight={0:.05,1:.95})
    
    clf2  = RandomForestClassifier(n_estimators=100,class_weight={0:.05,1:.95})
    
    clf3  = SVC(C=1000000.0,probability=True,class_weight={0:.05,1:.95})
    
    
    clf5  = DecisionTreeClassifier(class_weight={0:.05,1:.95})
    
    clf_labels = ['Logistic Regression','Random Forest','SVC','Decision Tree']
            
    print('10-fold cross validation:\n')
    for clf, label in zip([clf, clf2, clf3,clf5], clf_labels):
        scores = cross_val_score(estimator=clf,
                                 X=X_train,
                                 y=y_train,
                                 cv=10,
                                 scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), label))
    
    mv_clf = MajorityVoteClassifier(classifiers=[clf, clf2, clf3,clf5])
    
    clf_labels += ['Majority Voting']
    all_clf = [clf, clf2, clf3,clf5, mv_clf]
    
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf,
                                 X=X_train,
                                 y=y_train,
                                 cv=10,
                                 scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), label))
        
        
    
def cross_val(X,y,folds=10,clf=None,cv=True):
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
    skf = StratifiedKFold(n_splits=folds)
    fold = 1
    cms = np.array([[0,0],[0,0]])
    accs = []
    aucs=[]
    recall=[]
    precision=[]
    mccs=[]
    missed_indices=[]
    best_cms = np.array([[0,0],[0,0]])
    best_auc=0
    best_acc, best_clf = 0, None
    y_predicted_overall = None
    y_test_overall = None
    if cv:
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            prediction = clf.predict(X_test)
            acc = accuracy_score( y_test,prediction)
            cm = confusion_matrix(y_test,prediction)
            misclassified = np.where(y_test != prediction)
            missed_indices.append(misclassified)
            pred_probas = clf.predict_proba(X_test)[:,1]
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
            y_test = np.asarray(y_test)
            misclassified = np.where(y_test != clf.predict(X_test))
            missed_indices.extend(misclassified)
            if y_predicted_overall is None:
                y_predicted_overall = prediction
                y_test_overall = y_test
            else: 
                y_predicted_overall = np.concatenate([y_predicted_overall, prediction])
                y_test_overall = np.concatenate([y_test_overall, y_test])
            rec=recall_score( y_test,prediction)
            pre=precision_score(y_test,prediction)
            recall.append(rec)
            precision.append(pre)
            if acc > best_acc:
                best_acc, best_clf, = acc, clf
                best_auc = roc_auc
                best_cms = cm
            
        tn, fp, fn, tp = cms.ravel()
        fpr=fp/(fp+tn)
        tpr=tp/(tp+fn)
        mcc = (tp*tn - fp*fn) / math.sqrt( float(tp + fp)*float(tp + fn)*float(tn + fp)*float(tn + fn) )
        print('\nClassification Report')
        print (metrics.classification_report(y_test_overall, y_predicted_overall, digits=3))
        print('Recall: {}'.format(np.mean(recall)))
        print('Precision: {}'.format(np.mean(precision)))
        print('CV accuracy: %.3f +/- %.3f' % (round(np.mean(accs)*100,2),round(np.std(accs)*100,2)))
        print('CV ROC AUC: %.3f +/- %.3f' % (round(np.mean(aucs)*100,2),round(np.std(aucs)*100,2)))
        print('CV False Positive Rate: %.3f '% (round(fpr*100,2)))
        print('CV True Positive Rate: %.3f ' % (round(tpr*100,2)))
        print('CV Matthews Correlation Coefficient:',mcc)
        print('mean',np.mean(mccs))
        clf_return = clf.fit(X,y)
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
        
def pred_user(user_transformed, clf, y=None):
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
    label = {0:'Yes',1:'No'}
    predicted=label[clf.predict(user_transformed)[0]],np.max(clf.predict_proba(user_transformed))*100
    return predicted[0],predicted[1]
                    
