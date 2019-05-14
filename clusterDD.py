'''
    CSCS 466 Project 2 : machine learning
    Makena Kong & Whitney Larsen

    K-Means Algorithm Implementation :
    Will return a list of cluster ids
    and the number of items in each
'''

import sys
import time
import pandas as pd
import numpy as np
from random import randint
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.cluster import KMeans


''' KMEANS IMPLEMENTATION FUNCTIONS '''

def k_means(test, train, k):
    model = fit(train, k)
    predictions = predict(test,model[1])
    return predictions

# find centroids in cluster?
def get_nearest_centroid(obs, centroids):
    dists = np.sqrt(((obs - centroids) ** 2).sum(axis=1))
    return dists.idxmin()

# given a series of features
# return a list of cluster ids and number of items in each
def fit(data, k):
    # get random k centroids
    centroids = data.sample(k).reset_index()
    #print("initial centroids ...")
    #print(centroids)

    # get clusters
    clusters = data.apply(get_nearest_centroid, centroids=centroids, axis=1)

    # get new centroids
    new_centroids = data.groupby(clusters).mean()

    # do until centroids stop changing
    while not (new_centroids.equals(centroids)):
        centroids = new_centroids
        clusters = data.apply(get_nearest_centroid, centroids=centroids, axis=1)
        new_centroids = data.groupby(clusters).mean()

    #print("final centroids....")
    #print(centroids)
    # return clusters?
    return (clusters,centroids)

# find closest centroids
def predict(test,centroids):
    clusters = test.apply(get_nearest_centroid, centroids=centroids, axis=1)
    #nearest_centroids = test.groupby(clusters).mean()
    return clusters.values

# pass in single text field
# return vector of features
def getFeatures(record):

    # strips punctuation and listisizes text field
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(record.lower())

    features = {}
    lm = WordNetLemmatizer()
    for word in text:
        # checks stemmed word and associated synonyms against entries in features
        if not word in stopwords.words('english'):
            word = lm.lemmatize(word)
            entry_exists = False
            # if word or any synonyms of word appear in features, add 1 to that count
            # else create a new entry of word in features
            if word in features:
                entry_exists = True
                features[word] += 1
            if not entry_exists:
                features[word] = 1
    return features

def IGetSynsNotTragedies(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

'''SKLEARN FUNCTION '''

# kmeans algorithm from scikitlearn
def real_k_means(test,train,k):
    model = KMeans(n_clusters=k)
    model.fit(train)
    #model.predict(test)
    centroids = model.cluster_centers_
    clusters = model.labels_
    return clusters

'''ANAYLSIS/COMPARISON FUNCTIONS'''

# compare our kmeans to that of sklearn
def compare_results(our_results, sklearn_results, k):
    match = 0
    for i in range(k):
        if (our_results[i] == sklearn_results[i]):
            match += 1
    return (match / k)

# compare our kmeans to the actual results
def get_accuracy(actual, results):
    pass

def main():

    if (len(sys.argv)!= 2):
        print("Usage: python clusterDD <filename>")
        sys.exit()
    start = time.time()
    file = sys.argv[1]

    # read file into a dataframe
    data = pd.read_csv(file, sep="\t")

    # run kmeans algorithm
    # features = data.text.apply(getFeatures)
    # our_results = k_means(features)
    # actual_results = real_k_means(features)

    # testing it on a portion of the data
    test_data = data.iloc[0:7716]
    f = data.text.apply(getFeatures)
    feats = pd.DataFrame.from_dict(f)
    features = pd.DataFrame(list(feats['text'])).fillna(0)

    splitPoint = len(features.index) // 3
    test = features.iloc[:splitPoint, :].reindex()
    train = features.iloc[splitPoint:, :].reindex()

    # comparing our cluster ids
    our_model = fit(train,10)[0].values
    sklearn_model = real_k_means(test, train, 10)
    print("...OUR clusters...")
    print(our_model)
    #mid = time.time()
    #print('Time So Far:')
    #print(mid - start)
    #print("...SKLEARN clusters ...")
    #print(sklearn_model)

    # see how much ours match sklearns
    #match_percentage = compare_results(our_model, sklearn_model, 10)
    #print("\n OUR CLUSTERING IS %2.2f SIMILAR TO SKLEARN'S CLUSTERING" % match_percentage)

    # our predictions
    our_results = k_means(test, train, 10)
    print("\n...Our predictions")
    print(our_results)
    end = time.time()
    #print('Total Time:')
    #print(end - start)

    # see how accurate our clustering actually is
    # accuracy = get_accuracy()
    #print("\n OUR CLUSTERING IS %2.2f ACCURATE" % 0)
    #print("...note: this part is in the works...\n")

if __name__=="__main__":
    main()
