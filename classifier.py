#!/bin/python

import string
import sys
import random
import operator
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#read in raw data from file and return a list of (label, article) tuples
def get_data(filename): 
    data = [line.strip().split('\t') for line in open(filename).readlines()]
    random.shuffle(data)
    return data

def get_labels() :
    data = get_data('articles')
    labels = [int(label) for (label, _) in data]
    return labels

def get_stopwords():
    stopwords = set()
    for line in open('stopwords.txt', 'r'):
        stopwords.add(line.strip())
    return stopwords

#this is the main function you care about; pack all the cleverest features you can think of into here.
def get_features(X) : 
    features = []
    feature_set = set()
    stopwords = get_stopwords()
    rule_bigrams = ['man shot', 'person shot', 'people shot', 'child shot', 'was shot', 'got shot', 'been shot', 'were shot']
    for x in X : 
        #x is an article in string form - corresponds to a row of the matrix
        #each feature is a column of the matrix
        #putting all words in as a feature is a very strong baseline - we have to try to beat it
        f = {}
        x_lower = [word.strip(string.punctuation).lower() for word in x.split()]
        for word in x_lower:
            #if word not in f:
            #   f[word] = 0
            if word not in stopwords:
                f[word] = 1
                feature_set.add(word)

        # hardcode bigram features
        for bigram in rule_bigrams:
            if bigram in x.lower():
                f[bigram] = 1
                feature_set.add(bigram)
        """
        When creating features for all words:
        Statistical classification
        Fold 0 : 0.98298
        Fold 1 : 0.98388
        Fold 2 : 0.98256
        Fold 3 : 0.98256
        Fold 4 : 0.98256
        Test Average : 0.98291
        """
        """
        if "shooting" in x_lower :
            f['shooting'] = 1

        if "gun" in x_lower :
            f['gun'] = 1
        """
        features.append(f)
    #calculate_feature_distances(features,feature_set)
    return features

def calculate_best_features(features, feature_set) :
    labels = get_labels()
    print len(features)
    print len(features[0])
    print len(labels)
    
    distances = {}
    #rows are articles, columns are features
    for column in feature_set :
        vector = []
        for row in features :
            if column in row :
                vector.extend([1])
            else :
                vector.extend([0])

        distance_squared = sum([1 for i in xrange(len(labels)) if labels[i] != vector[i]])

        distances[column] = distance_squared
        #print column + " is " + str(distance_squared) + " away from the labels."   

    top_hun_words = sorted(distances.items(), key = lambda x: x[1])[:100]  
    for word, distance in top_hun_words:
      print word + " is " + str(distance) + " away from the labels."      
    
#vectorize feature dictionaries and return feature and label matricies
def get_matricies(data) : 
    dv = DictVectorizer(sparse=True) 
    le = LabelEncoder()
    y = [d[0] for d in data]
    texts = [d[1] for d in data]
    X = get_features(texts)
    #Here we are returning 5 things, the label vector y and feature matrix X, but also the texts from which the features were extracted and the 
    #objects that were used to encode them. These will come in handy for your analysis, but you can ignore them for the initial parts of the assignment
    return le.fit_transform(y), dv.fit_transform(X), texts, dv, le

#train and multinomial naive bayes classifier
def train_classifier(X, y):
    clf = LogisticRegression()
    clf.fit(X,y)
    return clf 

#test the classifier
def test_classifier(clf, X, y):
    return clf.score(X,y)

#cross validation    
def cross_validate(X, y, numfolds=5):
    test_accs = []
    split = 1.0 / numfolds
    for i in range(numfolds):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=i)
        clf = train_classifier(x_train, y_train)
        test_acc = test_classifier(clf, x_test, y_test)
        test_accs.append(test_acc)
        print 'Fold %d : %.05f'%(i,test_acc)
    test_average = float(sum(test_accs))/ numfolds
    print 'Test Average : %.05f'%(test_average)
    print
    return test_average

#run a rule based classifier and calculate the accuracy
def rule_based_classifier(data):
    correct = 0.0; total = 0.0
    for label, text in data : 
        prediction = '0'
        #TODO add more keywords, see how well they do alone and in combination
        #if "news" in text : prediction = '0' - no effect on accuracy
        if "shooting" in text : prediction = '1' #increases accuracy considerably
        #if "shot" in text : prediction = '1' - decreases accuracy considerably (~1.7%)
        #if "gun" in text : prediction = '1' - decreases accuracy
        #if "murder" in text : prediction = '1' - decreases accuracy
        #if "pistol" in text : prediction = '1' - decreases accuracy
        #if "rifle" in text : prediction = '1' - decreases accuracy
        if "handgun" in text : prediction = '1' #no effect on accuracy
        if "man shot" in text : prediction = '1' #increases accuracy
        if "woman shot" in text : prediction = '1' #no effect on accuracy
        if "person shot" in text : prediction = '1' #increases accuracy
        if "people shot" in text : prediction = '1' #increases accuracy
        if "child shot" in text : prediction = '1' #increases accuracy
        if "was shot" in text : prediction = '1' #increases accuracy
        if "got shot" in text : prediction = '1' #increases accuracy
        if "been shot" in text : prediction = '1' #increases accuracy
        if "were shot" in text : prediction = '1' #increases accuracy
        #if "shot at" in text : prediction = '1' - decreases accuracy
        #if "shooter" in text : prediction = '1' - decreases accuracy
        if "children shot" in text : prediction = '1' #no effect on accuracy
        #if " shoot " in text : prediction = '1' - decreases accuracy
        #if "shoot" in text : prediction = '1' - decreases accuracy
        #if "shoot him" in text : prediction = '1' - decreases accuracy
        #if "shoot her" in text : prediction = '1' - decreases accuracy
        #if "police" in text : prediction = '1' - decreases accuracy
        if prediction == label : correct += 1
        total += 1
    print 'Rule-based classifier accuracy: %.05f'%(correct / total)

#train and multinomial naive bayes classifier
def get_top_features(X, y, dv):
    clf = train_classifier(X, y)
    #the DictVectorizer object remembers which column number corresponds to which feature, and return the feature names in the correct order
    feature_names = dv.get_feature_names() 
    rule_words = ['shooting', 'handgun', 'man shot', 'person shot', 'people shot', 'child shot', 'was shot', 'got shot', 'been shot', 'were shot']
    #TODO: You will have to write code here to get the weights from the classifier, and print out the weights of the features you are interested in
    for i in xrange(len(feature_names)):
        if feature_names[i] in rule_words:
            print feature_names[i] + ': ' + str(clf.coef_[0][i])
    
def get_misclassified_examples(y, X, texts) :
    x_train, x_test, y_train, y_test, train_texts, test_texts = train_test_split(X, y, texts)
    clf = train_classifier(x_train, y_train)

    #TODO: You will have to write some code to call your classifier on each of the test examples, and check whether its prediction was right or wrong
    #for x in x_test :
    #    clf.predict(x)

if __name__ == '__main__' : 

    raw_data = get_data('articles')
    
    print '\nRule-based classification'
    rule_based_classifier(raw_data)

    print '\nStatistical classification'
    y, X, texts, dv, le = get_matricies(raw_data)
    print 'y'
    print y.shape
    print 'x'
    print X.shape
    cross_validate(X,y)

    get_top_features(X, y, dv)
#    get_misclassified_examples(y, X, texts)
