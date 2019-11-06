from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def total_distance(cluster, center):
    """
    Iterates over each row, then finds in each dimension the difference in the cluster value and the centroid value,
    sums them, and raises it to the power corresponding to the dimension of the space. It then sums all of these
    distances to find the total distance of all the points in the cluster from the centroid
    :param cluster: an iterable array of arrays that represents all the points in a cluster
    :param center: the coordinates of the cluster corresponding to what was passed in
    :return: the total distance according to the dimensionality of the data
    """
    dim = len(center)
    return sum([sum([(row[i] - center[i]) ** dim for i in range(len(row))]) ** (1/dim) for row in cluster.values])


def main():
    # Load in data
    iris = load_iris()
    X = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
    Y = pd.DataFrame(data=iris['target'], columns=['species'])

    # Create a range of models with N=1-10 clusters and keep track of the total squared distance
    model_stats = {}
    for i in range(1, 11):
        # Fit model
        model = KMeans(n_clusters=i).fit(X)
        model_data = model.predict(X)

        # Extract each cluster, its centroid and find the total squared distance
        distances = 0
        for j, center in enumerate(model.cluster_centers_):
            cluster = X.iloc[np.where(model_data == j)[0], ]
            distances += total_distance(cluster, center)
        model_stats[i] = distances

    # Plot total sum of distances
    plt.plot(list(model_stats.keys()), list(model_stats.values()))
    plt.xticks(list(model_stats.keys()))
    plt.xlabel('# of clusters')
    plt.ylabel('Sum of within-cluster distances')
    plt.show()


if __name__ == '__main__':
    main()
