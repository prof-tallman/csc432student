
################################################################################
# DBSCAN Clustering                                                            #
#   Clusters multi-dimensional data by repeatedly following densly located     #
#   samples from one to the next. Neighboring samples are found by calculating #
#   a circle around the current sample and checking whether there are some     #
#   minimum number of samples within this circle.                              #
#                                                                              #
#   file:   dbscan.py                                                          #
#   author: prof-tallman                                                       #
#                                                                              #
# Acknowledgements:                                                            #
#   Moosa Ali - https://becominghuman.ai/dbscan-clustering-algorithm-          #
#               implementation-from-scratch-python-9950af5eed97                #
#   Erik Lindernoren - https://github.com/eriklindernoren/ML-From-Scratch/     #
#                      blob/master/mlfromscratch/unsupervised_learning/        #
#                      dbscan.py                                               #
#                                                                              #
################################################################################



################################################################################
#                                   IMPORTS                                    #
################################################################################

import math



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



def find_neighbors(epsilon, sample_idx, all_samples, distance_fn):
    """ Returns the indices of all the samples that are neighbors to the
        single sample located at `index`.
    """

    neighbors = []
    sample_point = all_samples[sample_idx]
    for (i, s) in enumerate(all_samples):
        if i != sample_idx and distance_fn(sample_point, s) <= epsilon:
            neighbors.append(i)
    return neighbors



def create_neighbors_list(all_samples, epsilon, distance_fn):
    """ Returns a list of of all neighbors for each and every sample, using
        the `epsilon` radius and distance function. Every neighbor is given
        as an index into the original `all_samples` list.
    """

    dbscan_neighbors = []
    for (i, sample) in enumerate(all_samples):
        neighbors_list = find_neighbors(epsilon, i, all_samples, distance_fn)
        dbscan_neighbors.append(neighbors_list)
    return dbscan_neighbors



def run_dbscan_algorithm(X, epsilon=0.5, min_samples=5, distance_fn=euclid_dist, debug=False):
    """ Clusters an N-dimensional dataset using the DBSCAN algorithm. Output
        is a list of cluster IDs with #0 representing outliers and all other
        ID#s assigned accordingly. The output list is parallel to the input.
        The DBSCAN model uses `epsilon` as a circle radius, `min_samples` to
        as the minimum number of neighbors for a sample to be considered core,
        and `distance_fn` as the distance calculation (Euclidean by default).
        This algorithm is O(n^2).

        Setting `debug=True` will print the step-by-step DBSCAN calculations.
    """

    # Parallel OUTPUT list which identifies the cluster ID# for each sample.
    # We start at ID#1 because cluster ID#0 is reserved for outliers (noise)
    clusters = [-1] * len(X)
    cluster_id = 1

    # This processing queue is an alternative to recursion
    processing_queue = set()

    # Creates a list of neighbors for each sample - O(n^2) and then creates
    # a second list to avoid loops within each cluster
    dbscan_neighbors = create_neighbors_list(X, epsilon, distance_fn)
    unvisited = [idx for idx in range(len(dbscan_neighbors))]

    # Run the algorithm until we've run out of samples
    # Each time through this top-level loop is a new cluster
    while len(unvisited) > 0:

        # Prime the queue to start a new cluster... it's possible this is an
        # outlier, in which case we would NOT want to update the cluster ID#
        processing_queue.add(unvisited[-1])
        update_cluster_id = False

        if debug:
            print("Starting Cluster #{} - {} samples remain\n".format(cluster_id, len(unvisited)))

        # Step through each sample in the current cluster
        while len(processing_queue) > 0:

            if debug:
                print("Processing Queue: {}".format(processing_queue))

            # Get the next sample from this cluster and lookup its neighbors
            index = processing_queue.pop()
            neighbor_list = dbscan_neighbors[index]
            neighbor_count = len(neighbor_list)
            unvisited.remove(index)

            if debug:
                print("Processing sample {} which has {} neighbors".format(index, neighbor_count))

            # Noise Point: Assign all outliers to cluster ID#0
            if neighbor_count == 0:
                clusters[index] = 0
                if debug:
                    print("Noise: a lonely outlier all by iteslf\n")

            # Border Point: Add sample to the cluster but ignore its neighbors
            elif neighbor_count < min_samples:
                clusters[index] = cluster_id
                if debug:
                    print("Border: Adding this sample but not its neighbors\n")

            # Core Point: Add sample to the current cluster and then expand
            elif neighbor_count >= min_samples:
                clusters[index] = cluster_id
                to_process = [idx for idx in neighbor_list if idx in unvisited]
                processing_count = len(to_process)
                processing_queue.update(to_process)
                update_cluster_id = True
                if debug:
                    print("Core: Adding {} of {} neighbors {}\n".format(processing_count, neighbor_count, to_process))

        # If this last cluster was not a single outlier, increment the ID#
        if update_cluster_id:
            cluster_id += 1

    # When finished, return the clusters array that is parallel to input X
    return clusters



################################################################################
#                                  TESTING                                     #
################################################################################

if __name__ == "__main__":

    print("RUNNING UNIT TESTS: MANUALLY VERIFY PLOTS")

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_circles, make_moons

    # Use sklearn to create interesting datasets: add an obvious outlier to each
    (circle_samples, circle_labels) = make_circles(factor=0.4, noise=0.06, n_samples=500)
    circle_samples = np.append(circle_samples, [[1.5, 1.5]], axis=0)
    circle_labels = np.append(circle_labels, [-1], axis=0)
    (moon_samples, moon_labels) = make_moons(noise=0.06, n_samples=500)
    moon_samples = np.append(moon_samples, [[2, 1.5]], axis=0)
    moon_labels = np.append(moon_labels, [-1], axis=0)

    # Test with circle samples: must manually inspect the output graphs
    all_samples = [tuple(sample) for sample in circle_samples]
    clusters = run_dbscan_algorithm(all_samples, 0.2, 10, debug=True)
    sns.scatterplot(x=circle_samples[:,0], y=circle_samples[:,1], hue=clusters)
    plt.show()

    # Test with moon samples: must manually inspect the output graphs
    all_samples = [tuple(sample) for sample in moon_samples]
    clusters = run_dbscan_algorithm(all_samples, 0.2, 10, debug=True)
    sns.scatterplot(x=moon_samples[:,0], y=moon_samples[:,1], hue=clusters)
    plt.show()

    print("UNIT TESTS COMPLETED")
