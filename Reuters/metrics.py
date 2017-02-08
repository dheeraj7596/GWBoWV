import time;
from gensim.models import Word2Vec
import pandas as pd
from sklearn.cluster import KMeans
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
import os
from KaggleWord2VecUtility_Dheeraj import KaggleWord2VecUtility
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from numpy import float32
from sklearn.preprocessing import Imputer
import math
from sklearn.ensemble import RandomForestClassifier
import sys
from random import uniform
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn import svm
import pickle
import cPickle
from math import *
import matplotlib.pyplot as plt
from sklearn import svm, datasets, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn import grid_search
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from pandas import DataFrame
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
from sklearn.externals import joblib

def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def func(a):
	t = []
	while a < 10.:
		t.append(a)
		a+=0.2
	return t

print "Fitting One vs Rest SVM classifier to labeled cluster training data..."

start = time.time()

num_features = int(sys.argv[1])
num_clusters = int(sys.argv[2])

all = pd.read_pickle('all.pkl')

lb = MultiLabelBinarizer()
Y = lb.fit_transform(all.tags)
train_data, test_data, Y_train, Y_test = train_test_split(all["text"], Y, test_size=0.3, random_state=42)

train = DataFrame({'text': []})
test = DataFrame({'text': []})

train["text"] = train_data.reset_index(drop=True)
test["text"] = test_data.reset_index(drop=True)

gwbowv = np.load("old_GWBOWV_cluster_feature_matrix.npy")
gwbowv_test = np.load("old_TEST_GWBOWV_cluster_feature_matrix.npy")

# percentiles = (80, 82, 83, 60, 70)
print "Number of Clusters :", num_clusters

# for percentile in percentiles:
	# transformer = feature_selection.SelectPercentile(feature_selection.f_classif, percentile = percentile)
gwbowv_transformed = gwbowv
gwbowv_test_transformed = gwbowv_test
clf = OneVsRestClassifier(LogisticRegression(C = 100.0), n_jobs = 30)
clf = clf.fit(gwbowv_transformed, Y_train)
pred = clf.predict(gwbowv_test_transformed)
pred_proba = clf.predict_proba(gwbowv_test_transformed)
# print "Percentile: ", percentile
K = [1,3,5]

for k in K:
    Total_Precision = 0
    Total_DCG = 0
    norm = 0
    for i in range(k):
        norm += 1/math.log(i+2)

    loop_var = 0
    for item in pred_proba:
        classelements = sorted(range(len(item)), key=lambda i: item[i])[-k:]
        classelements.reverse()
        precision = 0
        dcg = 0
        loop_var2 = 0
        for element in classelements:
            if Y_test[loop_var][element] == 1:
                precision += 1
                dcg += 1/math.log(loop_var2+2)
            loop_var2 += 1
        
        Total_Precision += precision*1.0/k
        Total_DCG += dcg*1.0/norm
        loop_var += 1
    print "Precision@", k, ": ", Total_Precision*1.0/loop_var 
    print "NDCG@", k, ": ", Total_DCG*1.0/loop_var

print "Coverage Error: ", coverage_error(Y_test, pred_proba)
print "Label Ranking Average precision score: ", label_ranking_average_precision_score(Y_test, pred_proba) 
print "Label Ranking Loss: ", label_ranking_loss(Y_test, pred_proba)
print "Hamming Loss: ", hamming_loss(Y_test, pred)
print "Weighted F1score: ", f1_score(Y_test, pred, average = 'weighted')
print "*"*80
#joblib.dump(transformer, "anova_transformer.pkl")
	
endtime = time.time()
print "Total time taken: ", endtime-start, "seconds." 

print "********************************************************"
