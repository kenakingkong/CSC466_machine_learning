'''
    CSCS 466 Project 2 : machine learning
    Makena Kong & Whitney Larsen

    K-Nearest-Neighbors Algorithm:
    Predicts the speaker of a text

    Implemented with SciKitLearn
'''
import sys
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.simplefilter("ignore")

def choose_k(X_train, y_train):

    print("choosing k for Count Vectorizer")

    k = 1
    vec = CountVectorizer(max_features=50)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

    k = 3
    vec = CountVectorizer(max_features=50)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

    k = 5
    vec = CountVectorizer(max_features=50)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

    k = 10
    vec = CountVectorizer(max_features=50)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

    k = 13
    vec = CountVectorizer(max_features=50)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

    k = 15
    vec = CountVectorizer(max_features=50)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

    k = 17
    vec = CountVectorizer(max_features=50)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

    k = 20
    vec = CountVectorizer(max_features=50)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

def choose_k_tfidf(X_train, y_train):

    print("choosing k for tfidf vectorizer")

    k = 1
    vec =TfidfVectorizer(max_features=15)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

    k = 3
    vec = TfidfVectorizer(max_features=15)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

    k = 5
    vec = TfidfVectorizer(max_features=15)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

    k = 10
    vec = TfidfVectorizer(max_features=15)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

    k = 13
    vec = TfidfVectorizer(max_features=15)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

    k = 15
    vec = TfidfVectorizer(max_features=15)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

    k = 17
    vec = TfidfVectorizer(max_features=15)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

    k = 20
    vec = TfidfVectorizer(max_features=15)
    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("k=%d, accuracy=%2.6f" % (k, accuracy))

def choose_scaler(X_train, y_train, vec, k):

    print("choosing a scaler")

    scaler = StandardScaler(with_mean = False)
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("Standard scaler, accuracy=%2.6f" % ( accuracy))

    scaler = Normalizer()
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("Normalizer, accuracy=%2.6f" % (accuracy))

    scaler = MaxAbsScaler()
    model = KNeighborsClassifier(n_neighbors = k)
    pipeline = Pipeline([('vectorizer', vec),('scaler',scaler), ('model',model)])
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy").mean()
    print("MaxAbsScaler, accuracy=%2.6f" % (accuracy))


def main():
    if (len(sys.argv)!= 3):
        print("Usage: python speakerDD committee_utterances.tsv <speaker's text file>")
        sys.exit()
    file = sys.argv[1]
    speaker = sys.argv[2]

    # read file into a dataframe
    data = pd.read_csv(file, sep="\t")
    speaker_data = pd.read_csv(speaker, sep="\t")

    # only care about the most spoken speakers
    n = 60
    top_speakers = data.pid.value_counts().index[0:n]
    top_speakers_df = data.loc[data.pid.isin(top_speakers)]
    top_speakers_df = shuffle(top_speakers_df)

    splitPoint = len(top_speakers_df.index) // 3
    test = top_speakers_df.iloc[:splitPoint, :].reindex()
    train = top_speakers_df.iloc[splitPoint:, :].reindex()

    # get our data and labels
    X_train = train.text
    y_train = train.pid

    # Best K = 15 with Count Vectorizer
    #choose_k(X_train, y_train)
    #choose_k_tfidf(X_train, y_train)

    # Best Scaler is Normalizer with CountVectorizer
    #vec1 = CountVectorizer(max_features=50)
    #choose_scaler(X_train, y_train, vec1, 15)
    #vec2 = TfidfVectorizer(max_features=15)
    #choose_scaler(X_train, y_train, vec2, 15)

    # vectorize the training data
    vec = CountVectorizer(max_features=50, lowercase=False, stop_words="english")
    #vec = TfidfVectorizer(max_features=15)
    vec.fit(X_train)
    x_train_vec = vec.transform(X_train)

    # normalize the training data
    #scaler = StandardScaler(with_mean = False)
    scaler = Normalizer()
    scaler.fit(x_train_vec)
    x_train_sc = scaler.transform(x_train_vec)

    # create the knn model
    model = KNeighborsClassifier(n_neighbors = 15)
    model.fit(x_train_sc,y_train)

    # Predict on Test Set
    X_test = test.text
    y_test = test.pid
    x_test_vec = vec.transform(X_test)
    x_test_sc = scaler.transform(x_test_vec)
    y_pred = model.predict(x_test_sc)

    # Model scores
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test,y_pred, average='weighted')
    f1 = f1_score(y_test,y_pred, average='weighted')
    print("Accuracy: %2.5f" % accuracy)
    print("Precision: %2.5f" % precision)
    print("Recall: %2.5f" % recall)
    print("f1_score: %2.5f" % f1)

    # PREDICT GIVEN SPEAKER DATA
    x_new = speaker_data.text
    y_new = speaker_data.pid
    x_new_vec = vec.transform(x_new)
    x_new_sc = scaler.transform(x_new_vec)
    prediction = model.predict(x_new_sc)
    print("The predicted speaker is %d" % prediction)
    print("The actual speaker is %d" % y_new)


if __name__=="__main__":
    main()
