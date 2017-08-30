#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#########################################################
### your code goes here ###
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf = SVC(C=10000, kernel='rbf')
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

pred_test = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred_test)

pred_test_label1 = pred_test[pred_test == 1]
print "label 1 test num:", len(pred_test_label1)

pred_test10 = clf.predict(features_test[10])
pred_test26 = clf.predict(features_test[26])
pred_test50 = clf.predict(features_test[50])

print "accuracy:", accuracy
print "pred_test10:", pred_test10
print "pred_test26:", pred_test26
print "pred_test50:", pred_test50
#########################################################


