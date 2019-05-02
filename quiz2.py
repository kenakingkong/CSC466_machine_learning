import sys
import pymysql
import pandas as pd
import numpy as np

# find centroids in cluster?
def get_nearest_centroid(obs, centroids):
    dists = np.sqrt(((obs - centroids) ** 2).sum(axis=1))
    return dists.idxmin()

# print the Centroids
def print_centroids(centroids , i):

    midterm_1 = centroids.iloc[0]['midterm']
    final_1 = centroids.iloc[0]['final']
    midterm_2 = centroids.iloc[1]['midterm']
    final_2 = centroids.iloc[1]['final']

    print("Centroids %d: (%0.1f, %0.1f) (%0.1f, %0.1f)" % (i,midterm_1,final_1,midterm_2,final_2))


def main():

    # Connect to the database
    connection = pymysql.connect(host='localhost',
                                 user='quiz',
                                 password='quiz2F2019',
                                 db='quiz2F19',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)

    try:
        with connection.cursor() as cursor:

            # get the centroids
            get_centroids_sql = "SELECT * FROM centroids";
            cursor.execute(get_centroids_sql)
            result = cursor.fetchall()
            centroids = pd.DataFrame(result)

            # get the scores
            get_scores_sql = "SELECT * FROM scores";
            cursor.execute(get_scores_sql)
            result = cursor.fetchall()
            scores = pd.DataFrame(result)

    finally:
        connection.close

    # K MEANS HERE

    # iterations 1
    clusters = scores.apply(get_nearest_centroid, centroids=centroids, axis=1)
    print_centroids(centroids, 1)

    # iterations 2
    centroids = scores.groupby(clusters).mean()
    clusters = scores.apply(get_nearest_centroid, centroids=centroids, axis=1)
    print_centroids(centroids, 2)

    # iterations 3
    centroids = scores.groupby(clusters).mean()
    clusters = scores.apply(get_nearest_centroid, centroids=centroids, axis=1)
    print_centroids(centroids,3)


if __name__=="__main__":
    main()
