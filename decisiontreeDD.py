'''
    CSCS 466 Project 2 : machine learning
    Makena Kong & Whitney Larsen

    Decision Tree Algorithm Implementation:
    Classify text fields/ hearing by committee

    Help from "Decision Tree. It begins here."
                    by Rishabh Jain on Medium
'''

import sys
import pandas as pd
import numpy as np
import scipy.stats as st
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

'''
Node Class
Represents a Decision Tree Node
Decision Trees have Root, Decision and Terminal Nodes
Attributes: data, entropy, children
'''
class Node:
    def __init__(self):
        self.data = ''    # index to features
        self.entropy = 0  # node entropy
        self.left = None
        self.right = None

    def __eq__(self,other):
        return (self.data == other.data and self.entropy == other.entropy
                and self.left == other.left and self.right == other.right)

    def __repr__(self):
        if (self.left and self.right):
            return ("Node: %s , Entropy: %2.2f and LChild %s and RChild %s" %
                        (self.data, self.entropy, self.left.data, self.right.data))
        elif (self.left):
            return ("Node: %s , Entropy: %2.2f and LChild %s and RChild NONE" %
                        (self.data, self.entropy, self.left.data))
        elif (self.right):
            return ("Node: %s , Entropy: %2.2f and LChild NONE and RChild %s" %
                        (self.data, self.entropy, self.right.data))
        else:
            return ("Node: %s , Entropy: %2.2f and no children :(" %
                    (self.data, self.entropy))

    def is_leaf(self):
        #return (self.left == None and self.right == None)
        return self.entropy == 0

    def print_tree(self):
        print(self)
        if self.left: self.left.print_tree()
        if self.right: self.right.print_tree()

'''
Information Gain & Entropy
IG is a measure of the change in entropy
Gain = Entropy(Target) - Entropy(Target,SubTarget)

Goal: get max decrease in entropy on an attribute split
'''

# E(S) = E(c1,c2,...) = - sum of (p(logp)) from i=1 to c
def entropy(data,feature,total_rows):

    # get total amount of each outcome
    feature_count_left = (data[feature] == 0).sum()
    feature_count_right = (data[feature] == 1).sum()

    # get left entropy
    if (feature_count_left == 0):
        entropy_left = 0
    else:
        probability_left = feature_count_left / total_rows
        entropy_left = probability_left * np.log(probability_left)

    # get right entropy
    if (feature_count_right == 0):
        entropy_right = 0
    else:
        probability_right = feature_count_right / total_rows
        entropy_right = probability_right * np.log(probability_right)

    total_entropy = np.absolute(entropy_left + entropy_right)
    return total_entropy

# Entropy(T,X) = Sum of P(c)E(c) from c to X
def split_entropy(data,feature, total_rows):

    # get the total probabilities of each outcome
    total_feature_count_left = (data[feature] == 0).sum()
    total_feature_count_right = (data[feature] == 1).sum()
    total_probability_left = total_feature_count_left / total_rows
    total_probability_right = total_feature_count_right / total_rows

    # get the entropy for each specific outcome
    entropy_left = entropy(data, feature, total_rows)
    entropy_right = entropy(data, feature, total_rows)

    # calculate the total entropy
    total_entropy = ((total_probability_left * entropy_left) +
                        (total_probability_right * entropy_right))
    return total_entropy

def decision_tree(test,train):

    tree = build_tree(train)

    print("\n**Tree**")
    tree.print_tree()

    predictions = predict(test, tree)

    print('\n**Predictions**')
    print(predictions)
    #return predictions

    pass

# begin building the decision tree
def build_tree(train):

    features = train.columns.tolist()

    # find the root node
    root = Node()
    min_entropy = 0
    total_rows = len(train)
    for feature in features:
        new_entropy = entropy(train, feature, total_rows)
        if new_entropy > min_entropy :
            min_entropy = new_entropy
            root.data = feature
            root.entropy = min_entropy

    # remove the Root's feature
    features.remove(root.data)

    # expand on the left and right
    left_train = train.loc[train[root.data]==0]
    right_train = train.loc[train[root.data]==1]

    root.left = expand_tree(left_train, features, root, total_rows)
    root.right = expand_tree(right_train, features, root, total_rows)

    return root

# recursively builds the decision tree
def expand_tree(data, features, target, total_rows):

    print("--expanding on --")
    print(target)
    #print("--with data--")
    #print(data)

    # stop if target is a leaf
    if target.is_leaf():
        return None

    # stop when no more attributes or data
    if (data.empty):
        return None

    # use info gain to decide next node
    min_entropy = target.entropy
    node = Node()
    #total_rows = len(data)
    for attribute in features:
        new_entropy = split_entropy(data,attribute,total_rows)
        #print("\t %s: %d" % (attribute,new_entropy))
        if new_entropy < min_entropy :
            print("\t %s: %d" % (attribute,new_entropy))
            min_entropy = new_entropy
            node.data = attribute
            node.entropy = min_entropy

    # remove added feature from consideration
    if (node.data == ''): return None

    features.remove(node.data)
    left_train = data.loc[data[target.data]==0.0]
    right_train = data.loc[data[target.data]==1.0]

    node.left = expand_tree(left_train,features, node, total_rows)
    node.right = expand_tree(right_train,features, node, total_rows)

    return node


# predict by parsing decision tree
def predict(test, tree):
    results = test.apply(parse_tree, tree=tree, axis=1)
    return results.values

def parse_tree(obs,tree):
    while (tree.is_leaf() == False):
        if obs[tree.data] == 0 :
            tree = tree.left
        else:
            tree = tree.right
    return tree

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
        if not word in stopwords.words('english'):
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

def main():

    if (len(sys.argv)<2 or len(sys.argv)>3):
        print("Usage: python3 decisiontreeDD [-h] <filename> ")
        sys.exit()

    if (len(sys.argv)==2):
        file = sys.argv[1]

    if (len(sys.argv)==3):
        h_flag = sys.argv[1]
        file = sys.argv[2]

    # read file into a dataframe
    data = pd.read_csv(file, sep="\t")

    # testing it on a portion of the data
    test_data = data.iloc[0:100]

    f = test_data.text.apply(getFeatures)
    feats = pd.DataFrame.from_dict(f)
    features = pd.DataFrame(list(feats['text'])).fillna(0)

    splitPoint = len(features.index) // 3
    test = features.iloc[:splitPoint, :].reindex()
    train = features.iloc[splitPoint:, :].reindex()

    decision_tree(test,train)


if __name__=="__main__":
    main()

'''
# this is if there are more than two outcomes????
# i feel like all this looping will take longer
def entropy(data, feature, total_rows):

    # get the probabilities of each outcome
    probabilities = []
    outcomes = data.features.unique().count()
    for outcome in outcomes:
        count = data.groupby(data.feature == outcome).count()
        probability = count / total_rows
        probabilities.append(probability)

    # get the entropy of each outcome
    total_entropy = 0
    length = len(outcomes)
    for l in range(length):
        entropy = probabilities[l] * np.log(probabilities[l])
        total_entropy += entropy

    return total_entropy


'''
