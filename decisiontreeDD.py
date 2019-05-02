'''
    CSCS 466 Project 2 : machine learning 
    Makena Kong & Whitney Larsen

    Decision Tree Algorithm Implementation:

'''

import sys
import pandas as pd


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