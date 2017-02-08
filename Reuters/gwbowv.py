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
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn import svm
import pickle
import cPickle
from math import *
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from pandas import DataFrame


def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

def create_cluster_vector_and_gwbowv(wordlist, word_centroid_map, dimension, word_idf_dict, featurenames):
	# The number of clusters is equal to the highest cluster index
	# in the word / centroid map
	num_centroids = max( word_centroid_map.values() ) + 1
	bag_of_centroids = np.zeros( num_centroids * dimension, dtype="float32" )
	gwbowv = np.zeros( num_centroids * (dimension + 1), dtype="float32" )
	icf = np.zeros(num_centroids, dtype="float32")
	index2word_set = set(model.index2word)
	for word in wordlist:
		if word in word_centroid_map:
			index = word_centroid_map[word]
			if word in index2word_set:
				bag_of_centroids[index*dimension:(index+1)*dimension] += model[word]
				if word in featurenames:
					icf[index] += word_idf_dict[word]   
	
	norm_cluster_vector = np.linalg.norm(bag_of_centroids)
	if norm_cluster_vector!=0:
		bag_of_centroids = np.divide(bag_of_centroids,norm_cluster_vector)
	
	norm_icf = np.linalg.norm(icf)
	if norm_icf!=0:
		icf = np.divide(icf, norm_icf)
	
	gwbowv = np.hstack((bag_of_centroids, icf))
	return bag_of_centroids, gwbowv

if __name__ == '__main__':

	start = time.time()

	num_features = int(sys.argv[1])     # Word vector dimensionality
	min_word_count = 20   # Minimum word count
	num_workers = 8       # Number of threads to run in parallel
	context = 10          # Context window size
	downsampling = 1e-3   # Downsample setting for frequent words

	model_name = str(num_features) + "features_" + str(min_word_count) + "minwords_" + str(context) + "context_len2alldata_sgns"
  	model = Word2Vec.load(model_name)
	word_vectors = model.syn0

	all = pd.read_pickle('all.pkl')

	num_clusters = int(sys.argv[2])
	# Initalize a k-means object and use it to extract centroids
	# print "Running K means"
	
	kmeans_clustering = KMeans( n_clusters = num_clusters , n_jobs = -1 )
	idx = kmeans_clustering.fit_predict( word_vectors )
	joblib.dump(idx, 'kmeans_latestclusmodel.pkl')
	print "Cluster Model Saved..."

	idx = joblib.load('kmeans_latestclusmodel.pkl')
	print "Cluster Model Loaded..."

	# Get the end time and print how long the process took
	end = time.time()
	elapsed = end - start
	print "Time taken for K Means clustering: ", elapsed, "seconds."

	# Create a Word / Index dictionary, mapping each vocabulary word to
	# a cluster number
	word_centroid_map = dict(zip( model.index2word, idx ))

	lb = MultiLabelBinarizer()
	Y = lb.fit_transform(all.tags)
	train_data, test_data, Y_train, Y_test = train_test_split(all["text"], Y, test_size=0.3, random_state=42)
	
	train = DataFrame({'text': []})
	test = DataFrame({'text': []})

	train["text"] = train_data.reset_index(drop=True)
	test["text"] = test_data.reset_index(drop=True)

	traindata = []
	
	for i in range( 0, len(train["text"])):
		traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["text"][i], True)))
	
	tfv = TfidfVectorizer(min_df=5,strip_accents='unicode',dtype=np.float32)
	tfidfmatrix_traindata = tfv.fit_transform(traindata)
	featurenames = tfv.get_feature_names()
	idf = tfv._tfidf.idf_

	#Creating a dictionary with word mapped to its idf value 
	print "Creating word-idf dictionary for Training texts..."

	word_idf_dict = {}
	for pair in zip(featurenames, idf):
		word_idf_dict[pair[0]] = pair[1]

	print "Creating Cluster Vectors and Graded Weighted Bag of Word Vectors..."

	#bowv and gwbowv are matrices which contain normalised bag of word vectors and normalised gwbowv.
	bowv = np.zeros( (train["text"].size, num_clusters*num_features), dtype="float32")
	gwbowv = np.zeros( (train["text"].size, num_clusters*(num_features+1)), dtype="float32")

	counter = 0

	for review in train["text"]:
		words = KaggleWord2VecUtility.review_to_wordlist( review, \
            remove_stopwords=True )
		bowv[counter], gwbowv[counter] = create_cluster_vector_and_gwbowv(words, word_centroid_map, num_features, word_idf_dict, featurenames)
		counter+=1
		if counter % 1000 == 0:
			print "Train Text Covered : ",counter

    #saving the bowv and gwbowv matrices
	#np.save("Bag_of_word_vector_250cluster_500feature_matrix", bowv)
	np.save("old_GWBOWV_cluster_feature_matrix", gwbowv)

	bowv_test = np.zeros( (test["text"].size, num_clusters*num_features), dtype="float32")
	gwbowv_test = np.zeros( (test["text"].size, num_clusters*(num_features+1)), dtype="float32")

	counter = 0

	for review in test["text"]:
		words = KaggleWord2VecUtility.review_to_wordlist( review, \
            remove_stopwords=True )
		bowv_test[counter], gwbowv_test[counter] = create_cluster_vector_and_gwbowv(words, word_centroid_map, num_features, word_idf_dict, featurenames)
		counter+=1
		if counter % 1000 == 0:
			print "Test Text Covered : ",counter

    #saving the bowv and gwbowv test matrices
	#np.save("TEST_Bag_of_word_vector_250cluster_500feature_matrix", bowv_test)
	np.save("old_TEST_GWBOWV_cluster_feature_matrix", gwbowv_test)

	# print "Fitting a SVM classifier to labeled training data..."

	# for i in drange(0.1,10.0,0.2):
	# 	clf=svm.LinearSVC(C=i)
	# 	clf.fit(gwbowv, train["sentiment"])
	# 	print i,"------------",clf.score(gwbowv_test,test["sentiment"])
	
	# endtime = time.time()
	# print "Total time taken: ", endtime, "seconds." 

	# print "********************************************************"