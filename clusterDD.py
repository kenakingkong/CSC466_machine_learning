'''
    CSCS 466 Project 2 : machine learning
    Makena Kong & Whitney Larsen

    K-Means Algorithm Implementation :
    Will return a list of cluster ids
    and the number of items in each
'''

import sys
import pandas as pd
import numpy as np
from random import randint
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as guru
from nltk.corpus import stopwords
from sklearn.cluster import KMeans

# code for the kmeans algorithm
# given a series of features
# return a list of cluster ids and number of items in each
def k_means(data, k):
    # get random k centroids
    centroids = data.sample(k).reset_index()
    print("initial centroids ...")
    print(centroids)

    # get clusters
    clusters = data.apply(get_nearest_centroid, centroids=centroids, axis=1)

    # get new centroids
    new_centroids = data.groupby(clusters).mean()

    # do until centroids stop changing
    while not (new_centroids.equals(centroids)):
        centroids = new_centroids
        clusters = data.apply(get_nearest_centroid, centroids=centroids, axis=1)
        new_centroids = data.groupby(clusters).mean()

    print("final centroids....")
    print(centroids)
    return clusters

# find centroids in cluster?
def get_nearest_centroid(obs, centroids):
    dists = np.sqrt(((obs - centroids) ** 2).sum(axis=1))
    return dists.idxmin()

# pass in single text field
# return vector of features
def getFeatures(record):

    # strips punctuation and listisizes text field
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(record.lower())

    features = {}
    ps = PorterStemmer()
    for word in text:
        # checks stemmed word and associated synonyms against entries in features
        if word not in stopwords.words('english')
            word = ps.stem(word)
            entry_exists = False
            # if word or any synonyms of word appear in features, add 1 to that count
            # else create a new entry of word in features
            for syn in getSyns(word):
                if syn in features:
                    entry_exists = True
                    break
            if not entry_exists:
                features[word] = 1
    return features

def getSyns(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

# kmeans algorithm from scikitlearn
def real_k_means(data,k):
    model = KMeans(n_clusters=k)
    model.fit(data)
    centroids = model.cluster_centers_
    clusters = model.labels_
    return clusters

def main():

    if (len(sys.argv)!= 2):
        print("Usage: python clusterDD <filename>")
        sys.exit()

    file = sys.argv[1]

    # read file into a dataframe
    data = pd.read_csv(file, sep="\t")

    # run kmeans algorithm
    # features = data.text.apply(getFeatures)
    # our_results = k_means(features)
    # actual_results = real_k_means(features)

    # testing it on a portion of the data
    test_data = data.iloc[0:100]

    f = test_data.text.apply(getFeatures)
    feats = pd.DataFrame.from_dict(f)
    features = pd.DataFrame(list(feats['text']))

    splitPoint = len(features.index) // 3
    test = features.iloc[:splitPoint, :].reindex()
    train = features.iloc[splitPoint:, :].reindex()

    our_results = k_means(train, 10)
    actual_result = real_k_means(train, 10)

    print("...OUR cluster ids...")
    print(our_results.values)

    print("...REAL cluster ids ...")
    print(actual_result)

    # test our results???


if __name__=="__main__":
    main()
