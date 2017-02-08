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
from sklearn import svm, datasets, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn import grid_search

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

train = pd.read_csv( 'data/train_v2.tsv', header=0, delimiter="\t")
test = pd.read_csv( 'data/test_v2.tsv', header=0, delimiter="\t")

num_features = int(sys.argv[1])
num_clusters = int(sys.argv[2])

gwbowv_name = "GWBOWV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_orig.npy"
gwbowv = np.load(gwbowv_name)
test_gwbowv_name = "TEST_GWBOWV_" + str(num_clusters) + "cluster_" + str(num_features) + "feature_matrix_orig.npy"
gwbowv_test = np.load(test_gwbowv_name)

percentiles = [83]
print "Number of Clusters :", sys.argv[2]

for percentile in percentiles:
	transformer = feature_selection.SelectPercentile(feature_selection.f_classif, percentile = percentile)
	gwbowv_transformed = transformer.fit_transform(gwbowv, train["class"])
	gwbowv_test_transformed = transformer.transform(gwbowv_test)

	#joblib.dump(transformer, "anova_transformer.pkl")
	for i in drange(0.1, 6, 0.2):
		clf = svm.LinearSVC(C = i)
		clf.fit(gwbowv_transformed, train["class"])
		print percentile, "---------", i, "------------",clf.score(gwbowv_test_transformed,test["class"])

endtime = time.time()
print "Total time taken: ", endtime-start, "seconds." 

print "********************************************************"
