
################################################################################
# KMeans Clustering                                                            #
#   Clusters multi-dimensional data by repeatedly grouping each data point to  #
#   a centroid and then recalculating the centroid for the new group. The      #
#   algorithm finishes when the centroids reach a stable position.             #
#                                                                              #
#   file:   kmeans.py                                                          #
#   author: prof-tallman                                                       #
################################################################################



################################################################################
#                                   IMPORTS                                    #
################################################################################

import math
import random
import statistics



################################################################################
#                                   MODULE                                     #
################################################################################

def euclid_dist(point1, point2):
    """ Calculates the Euclidean Distance between two points of arbitrary
        dimension. The two points must be parallel tuples.
    """

    sum_of_dimensions = 0
    for idx in range(len(point1)):
        sum_of_dimensions += (point1[idx]-point2[idx])**2
    distance = math.sqrt(sum_of_dimensions)

    return distance



def get_closest_centroid(point, centroid_list):
    """ Returns the index of the closest centroid using Euclidean Distance.
        Each centroid must have parallel dimension to the point.
    """

    # Create a list of distances to each of the different centroids
    distance_to_centroid = []
    for cluster_index in range(len(centroid_list)):
        distance = euclid_dist(point, centroid_list[cluster_index])
        distance_to_centroid.append(distance)

    # Which centroid is closest? Get its index
    min_distance = min(distance_to_centroid)
    closest_indx = distance_to_centroid.index(min_distance)
    min_centroid = centroid_list[closest_indx]

    return closest_indx



def create_clusters(sample_list, k, centroid_list):
    """ Assigns each sample to the closest centroid. Returns a list of clusters
        that is parallel to the original list of centroids.
    """

    # same number of clusters as there are centroids (same as `k`)
    cluster_list = [[] for i in range(k)]

    # Find closest centroid and assign to the parallel item in the cluster list
    for sample in sample_list:
        closest_indx = get_closest_centroid(sample, centroid_list)
        cluster_list[closest_indx].append(sample)

    return cluster_list



def calculate_single_centroid(cluster, dimension_count):
    """ Returns the statistical mean of all points in this cluster. Empty
        clusters raise a ValueError.
    """

    # Sanity check to avoid divide-by-zero errors
    if len(cluster) == 0:
        msg = "Cannot find center of an empty cluster"
        raise ValueError(msg)

    # Calculate the mean coordinate in each dimension, one at a time
    centroid = []
    for idx in range(dimension_count):
        sum_in_this_dimension = sum(point[idx] for point in cluster)
        avg_in_this_dimension = sum_in_this_dimension / len(cluster)
        centroid.append(avg_in_this_dimension)

    # Convert the centroid to an immutable type
    return tuple(centroid)



def calculate_centroids(cluster_list, dimension_count):
    """ Returns a list of centroids whose coordinates are the statistical
        mean of its cluster.
    """

    # Calculate each centroid one at a time
    centroid_list = []
    for cluster in cluster_list:
        centroid = calculate_single_centroid(cluster, dimension_count)
        centroid_list.append(centroid)

    return centroid_list



def _choose_random_centroids(samples, k):
    """ Randomly creates a list of `k` centroids from the given data points.
        If the data contains fewer than `k` samples, a ValueError is raised.
    """

    # Sanity check to avoid an infinite loop
    if len(samples) < k:
        msg = "Cannot choose {} centroids from only {} data points"
        msg = msg.format(k, len(samples))
        raise ValueError(msg)

    # Keep looping until we have `k` unique centroids
    centroids = [] # our k centroids will go in here
    while len(centroids) < k:
        random_centroid = random.choice(list(samples))
        if random_centroid not in centroids:
            centroids.append(random_centroid)

    return centroids



def run_kmeans_algorithm(data, k, max_passes=10):
    """ Cluster a dataset with the KMeans algorithm using random samples as
        the initial centroids. The algorithm stops when it reaches stability
        or the maximum number of passes. Returns (centroids, clusters).
    """

    prev_centroids = _choose_random_centroids(data, k)
    arbitrary_centroid = prev_centroids[0]
    dimension_count = len(arbitrary_centroid)

    pass_count = 0
    while pass_count < max_passes:

        # Run a single round of the algorithm and check for stability
        clusters = create_clusters(data, k, prev_centroids)
        centroids = calculate_centroids(clusters, dimension_count)
        if prev_centroids == centroids:
            break

        # Update variables for the next pass
        prev_centroids = centroids
        pass_count += 1

    return (centroids, clusters)



################################################################################
#         REMOVES OUTLIERS FROM CLUSTERS USING IQR OUTLIER DETECTION           #
#                      NOT PART OF STUDENT ASSIGNMENT                          #
################################################################################

# Using interquartile range (IQR) for outlier detection
# https://medium.com/analytics-vidhya/effect-of-outliers-on-k-means-algorithm-using-python-7ba85821ea23
# We don't care about the lower threshold because it would mean that the sample
# is close to the centroid (good)... we care about the upper threshold

def iqr_outlier_threshold(centroid, cluster):
    """ Returns the max distance at which a sample should be considered an
        outlier rather than a member of the cluster. Uses the Interquartile
        Range method to set the max distance at Q3 + 1.5 * IQR.
    """

    # We need to a sorted list of all of the Euclidean Distances in this cluster
    distances = [euclid_dist(sample, centroid) for sample in cluster]
    distances = sorted(distances)

    if len(distances) < 3:
        outlier_threshold = max(distances)

    else:
        # Divide up the data into 1st-half/2nd-half and then 1st-Qtr / 3rd-Qtr
        # 1st-Qtr is the median of the 1st-half data
        # 3rd-Qtr is the median of the 2nd-half data
        # Interquartile Range is the difference between Q1 and Q3
        halfway = len(distances) // 2
        H1 = distances[:halfway]
        H2 = distances[halfway:]
        Q1 = statistics.median(H1)
        Q3 = statistics.median(H2)
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR

    return outlier_threshold



def remove_outliers_from_cluster(centroid, cluster, threshold):
    """ Modifies this cluster by removing any data point whose distance from
        its centroid exceeds the threshold.
    """

    # Remova all samples whose distance to the centroid exceeds the threshold
    outliers = [c for c in cluster if euclid_dist(c, centroid) > threshold]
    for sample in outliers:
        cluster.remove(sample)

    return outliers



def remove_outliers_all_clusters(centroids, clusters):
    """ Modifies all of the clusters by removing any data point whose distance
        from its centroid exceeds the threshold.
    """

    outliers = []
    for cluster, centroid in zip(clusters, centroids):
        threshold = iqr_outlier_threshold(centroid, cluster)
        outliers += remove_outliers_from_cluster(centroid, cluster, threshold)

    return outliers



################################################################################
#                                  TESTING                                     #
################################################################################

if __name__ == "__main__":

    print("Running Unit Tests:  ", end='')
    origin2d = (0, 0)
    unit345t = (3, 4)
    origin4d = (0, 0, 0, 0)
    unit1_4d = (1, 1, 1, 1)
    assert euclid_dist(origin2d, unit345t) == 5
    assert euclid_dist(origin4d, unit1_4d) == 2

    test_centroids = [(0.5, 0.5), (9.5, 9.5)]
    assert get_closest_centroid(origin2d, test_centroids) == 0

    test_k = len(test_centroids)
    test_data = [(2, 2), (10, 10), (0, 2), (2, 0), (8, 10), (8, 8), (0, 0), (10, 8)]
    tpos_clusters = [[(2, 2), (0, 2), (2, 0), (0, 0)], [(10, 10), (8, 10), (8, 8), (10, 8)]]
    test_clusters = create_clusters(test_data, test_k, test_centroids)
    assert test_clusters == tpos_clusters

    tpos_centroids = [(1, 1), (9, 9)]
    assert calculate_single_centroid(test_clusters[0], test_k) == tpos_centroids[0]
    assert calculate_single_centroid(test_clusters[1], test_k) == tpos_centroids[1]
    assert calculate_centroids(test_clusters, test_k) == tpos_centroids

    test_random = _choose_random_centroids(test_data, test_k)
    assert len(test_random) == test_k
    print("Passed")

    print("Running System Test: Check Results")
    (final_centroids, final_clusters) = run_kmeans_algorithm(test_data, test_k)
    print("Centroids: {}".format(final_centroids))
    print("Clusters:  {}".format(final_clusters))
