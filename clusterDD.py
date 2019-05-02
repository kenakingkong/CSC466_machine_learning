'''
    CSCS 466 Project 2 : machine learning 
    Makena Kong & Whitney Larsen

    K-Means Algorithm Implementation :
    Will return a list of cluster ids 
    and the number of items in each
'''

import sys
import pandas as pd
            
# code for the kmeans algorithm
def k_means(data);
    pass

# pass in single text field
# return vector of features
def getFeatures(record):
    pass


def main():

    if (len(sys.argv)!= 2):
        print("Usage: python clusterDD <filename>")
        sys.exit()

    file = sys.argv[1]

    # read file into a dataframe
    data = pd.read_csv(file, sep=",")

    # run kmeans algorithm
    k_means(data.text)




if __name__=="__main__":
    main()