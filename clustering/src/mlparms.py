import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import DBSCAN



def plot_kmeans_kvalue_elbow(X):

    wsse = []
    k_vals = list(range(1, 20))

    for k in k_vals:
        km = KMeans(n_clusters=k, max_iter=25)
        km.fit(X)
        wsse.append(km.inertia_)

    plt.figure(figsize=(6, 6))
    plt.plot(k_vals, wsse)
    plt.xlabel('Number of Clusters (k-value)')
    plt.ylabel('Within-Cluster Sum of Squared Errors');
    plt.draw()



def plot_dbscan_epsilon_elbow(X):

    knn = NearestNeighbors(n_neighbors=4)
    fit = knn.fit(X)
    distances, _ = fit.kneighbors(X)
    mean_knn = np.mean(distances, axis=1)
    mean_knn = np.sort(mean_knn, axis=0)

    plt.figure(figsize=(6,6))
    plt.plot(mean_knn)
    plt.grid()
    plt.ylabel('Epsilon Value');
    plt.draw()



def evaluate_dbscan_epsilon(X, min_samples=4, epsilon_list=[]):
    """ Returns a list of evaluation scores for a DBSCAN model over a range of
        Epsilon values. Three evaluation metrics are generated along with the
        cluster count:
          * Calinski-Harabasz Index
          * Davies-Bouldin Index
          * Silhoette Coefficient
          * Cluster Count
        The three evaluation metrics will be normalized using the MinMaxScaler
        whereas the Cluster Count will be returned in its raw form.

        Return value is a tuple of lists that is parallel to `epsilon_list`:
          ([Calniski-Harabasz], [Davies-Bouldin], [Silhoette], [Cluster Count])
    """

    calinski_harabasz = []
    davies_bouldin = []
    silhoette = []
    cluster_count = []

    # Try a DBSCAN using each possible value for epsilon
    for eps in epsilon_list:

        # Create a test model using these input parameters
        model = DBSCAN(eps=eps, min_samples=min_samples)
        predict = model.fit_predict(X)

        if len(np.unique(predict)) > 1:
            # Calculate the three evaluation scores
            silhoette.append(metrics.silhouette_score(X, predict))
            calinski_harabasz.append(metrics.calinski_harabasz_score(X, predict))
            davies_bouldin.append(metrics.davies_bouldin_score(X, predict))

            # Its also important that we know how many clusters these parameters produced
            cluster_count.append(len(np.unique(predict)))

        # If there's only one label, we consider it an error
        else:
            silhoette.append(-1)
            calinski_harabasz.append(-1)
            davies_bouldin.append(-1)
            cluster_count.append(-1)

    # Scale the three evaluation scores since they all use different ranges
    scores = list(zip(calinski_harabasz, davies_bouldin, silhoette))
    scaler = preprocessing.MinMaxScaler()
    scores = scaler.fit_transform(scores)

    # Separate the scores to make it easier on the caller
    calinski_harabasz, davies_bouldin, silhoette = zip(*scores)

    return (calinski_harabasz, davies_bouldin, silhoette, cluster_count)



def plot_dbscan_metrics(min_samples, epsilon_list, calinski_harabasz, davies_bouldin, silhoette, cluster_count):
    """ Helps calibrate a DBSCAN model by plotting four evaluation scores over
        a range of epsilon values using a static min_values parameter:
          * Calniski-Harabasz Indices,
          * Davies-Bouldin Indices,
          * Silhoette Coefficients,
          * Cluster Counts
        Silhoette and Calinski-Harabasz scores are shown in red because high
        scores are generally considered better. Davies-Bouldin is shown in blue
        because low scores are considered better. The cluster count is shown in
        green and requires domain knowledge to diagnose.
    """

    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(epsilon_list, silhoette, color='orangered', ls='solid', lw=3, label='Silhoette (max)')
    ax1.plot(epsilon_list, calinski_harabasz, color='indianred', ls='solid', lw=3, label='Cski-Hrb (max)')
    ax1.plot(epsilon_list, davies_bouldin, color='dodgerblue', ls='solid', lw=3, label='Dav-Bldn (min)')
    ax1.set_title("Cluster Evaluation for min_samples={}".format(min_samples), fontsize=16)
    ax1.set_ylabel("Scaled Evaluation Score", fontsize=14)
    ax1.set_xlabel("Value of Epsilon", fontsize=14)
    ax1.yaxis.grid()
    ax1.xaxis.grid()

    ax2 = ax1.twinx()
    ax2.plot(epsilon_list, cluster_count, color='seagreen', ls='solid', lw=3, label='Cluster Count')
    ax2.set_ylabel("Number of Clusters\n(including outliers)", fontsize=14)

    (handles1, labels1) = ax1.get_legend_handles_labels()
    (handles2, labels2) = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, facecolor='white', framealpha=1)

    plt.tight_layout()
    plt.show()
