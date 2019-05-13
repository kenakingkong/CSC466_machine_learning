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

'''
Node Class
Represents a Decision Tree Node
Decision Trees have Root, Decision and Terminal Nodes
Attributes: data, entropy, children
'''
class Node:
    def __init__(self, data):
        self.data = data        # index to features dataframe
        self.children = []      # list of indices to
        self.entropy = 0        # node entropy
    def __eq__(self,other):
        return self.data == other.data and self.children == other.children
    def __repr__(self):
        return "Node %d with children %s" % self.data, ", ".join(self.children)

'''
Information Gain & Entropy
IG is a measure of the change in entropy
Gain = Entropy(Target) - Entropy(Target,SubTarget)

Goal: get max decrease in entropy on an attribute split
'''

# E(S) = E(c1,c2,...) sum of (-p(logp)) from i=1 to c
def node_entropy():
    pass

# Entropy(T,X) = Sum of P(c)E(c) from c to X
def split_entropy():
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
def build_decision_tree():
    pass

def getFeatures(record):
    pass

def main():

    if (len(sys.argv)<2 or len(sys.argv)>3):
        print("Usage: python clusterDD <filename>")
        sys.exit()

    if (len(sys.argv)==2):
        file = sys.argv[1]

    if (len(sys.argv)==3):
        h_flag = sys.argv[1]
        file = sys.argv[2]

    # call decision tree functions here


if __name__=="__main__":
    main()
