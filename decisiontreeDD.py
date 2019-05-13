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
from nltk.tokenize import RegexpTokenizer

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
                        self.data, self.entropy, self.left.data, self.right.data)
        elif (self.left):
            return ("Node: %s , Entropy: %2.2f and LChild %s and RChild NONE" %
                        self.data, self.entropy, self.left.data)
        elif (self.left):
            return ("Node: %s , Entropy: %2.2f and LChild NONE and RChild %s" %
                        self.data, self.entropy, self.right.data)
        else:
            return ("Node: %s , Entropy: %2.2f and no children :(" %
                    (self.data, self.entropy))

    def is_leaf(self):
        return (self.left == None and self.right == None)

'''
Information Gain & Entropy
IG is a measure of the change in entropy
Gain = Entropy(Target) - Entropy(Target,SubTarget)

Goal: get max decrease in entropy on an attribute split
'''


# E(S) = E(c1,c2,...) = - sum of (p(logp)) from i=1 to c
def entropy(data,feature,total_rows):
    # get the probabilities of each outcome
    #feature_count_left = data.groupby(data[feature] == 0).count()
    #feature_count_right = data.groupby(data[feature] == 1).count()
    feature_count_left = (data[feature] == 0).sum()
    feature_count_right = (data[feature] == 1).sum()
    probability_left = feature_count_left / total_rows
    probability_right = feature_count_right / total_rows

    # get the entropy of each outcome
    entropy_left = probability_left * np.log(probability_left)
    entropy_right = probability_right * np.log(probability_right)
    total_entropy = entropy_left + entropy_right

    return - total_entropy

# Entropy(T,X) = Sum of P(c)E(c) from c to X
def split_entropy(data, node, feature, total_rows):

    # get the total probabilities of each outcome
    total_feature_count_left = (data[feature] == 0).sum()
    total_feature_count_right = (data[feature] == 1).sum()
    total_probability_left = feature_count_left / total_rows
    total_probability_right = feature_count_right / total_rows

    # get the entropy for each specific outcome
    data_left = data.loc[data[feature]==0]
    data_right = data.loc[data[feature]==1]
    entropy_left = entropy(data_left, node.data, len(data_left))
    entropy_right = entropy(data_right, node.data, len(data_right))

    # calculate the total entropy
    total_entropy = ((total_probability_left * entropy_left) +
                        (total_probability_right * entropy_right))
    return total_entropy

def decision_tree(test,train):
    tree = build_tree(train)
    #predictions = predict(test, tree)
    #return predictions
    pass

'''
1. Root attribute /tree
2. calc entropy of tree
3. calc entropy of possible attributes
    - for each feature - split on its attributes
    - sum the entropy for each attribute
4. split on the largest gain (entropy decrease)
5. entropy of 0 is a leaf node
'''
# build the tree
def build_tree(train):

    # will be stripping dataframe at each node
    data = train
    features = data.columns

    # get the root node
    root = Node()
    min_entropy = 1
    total_rows = len(data)
    for feature in features:
        new_entropy = entropy(data, feature, total_rows)
        print(new_entropy)
        if new_entropy < min_entropy :
            min_entropy = new_entropy
            root.data = feature
            root.entropy = min_entropy

    print("**ROOT**")
    print(root)

    previous_feature = root.data
    target = root
    for attribute in features:
        total_rows =
        new_entropy = split_entropy(data,target,attribute,total_rows)

    while ()


    # create two data frames
    #left_train = train.loc[train[root.data]==0]
    #right_train = train.loc[train[root.data]==1]





def expand_tree()


# predict by parsing decision tree
def predict(test, tree):
    pass

# binary features only
def getFeatures(record):

    features = {}

    # strips punctuation and listisizes text field
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(record.lower())

    # temp stuff
    if 'go' in text:
        features["go"] = 1
    if (len(text) > 20):
        features['length > 20'] = 1
    if 'and' in text:
        features["and"] = 1
    if 'government' in text:
        features['government'] = 1

    return features

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
