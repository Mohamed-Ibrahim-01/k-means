import numpy as np
import scipy.spatial.distance as dist


class MyKmeans:
    def __init__(self, n_clusters=8, init='random', n_init=10,
                 max_iter=300, algorithm='lloyd'):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.algorithm = algorithm

    def fit(self, X):
        converged = False
        n_features_in_ = X.shape[1]
        n_iters_ = 0
        labels_ = np.array([])
        centroids = np.random.rand(self.n_clusters, n_features_in_)

        while(n_iters_ < 300 and not converged):
            cent_distances = self._cent_dist(X, centroids)
            labels_ = self._label_examples(cent_distances)
            new_centroids = self._recentroide(X, labels_, centroids)
            converged = self._is_converged(centroids, new_centroids)
            centroids = new_centroids
            n_iters_ += 1

        self.n_features_in_ = n_features_in_
        self.n_iters_ = n_iters_
        self.labels_ = labels_
        self.cluster_centers_ = centroids
        self.clusters_ = self._get_clusters(labels_, centroids)

    def _cent_dist(self, X, centroids):
        distances = np.zeros((X.shape[0], centroids.shape[0]))
        for i, example in enumerate(X):
            for j, centroid in enumerate(centroids):
                distances[i][j] = self._calc_distance(example, centroid)
        return distances

    def _label_examples(self, cent_distances):
        labels = np.zeros(cent_distances.shape[0])
        for i, example_dist in enumerate(cent_distances):
            labels[i] = np.argmax(example_dist).astype(int)
        return labels

    def _recentroide(self, X, labels_, centroids):
        new_centroids = np.copy(centroids)
        for label in range(centroids.shape[0]):
            labeled_points = np.array(list(zip(labels_, X)), dtype=object)
            curr_cluster = X[labeled_points[:, 0] == label]
            if(curr_cluster.shape[0]):
                mean = np.mean(curr_cluster, axis=0)
                new_centroids[label] = mean
        return new_centroids

    def _is_converged(self, centroids, new_centroids):
        return np.all(np.isclose(centroids, new_centroids, atol=1e-03))

    def _calc_distance(self, example, centroid):
        return dist.euclidean(example, centroid)

    def _get_clusters(self, labels_, centroids):
        clusters = []
        for label in range(centroids.shape[0]):
            curr_cluster = np.where(labels_ == label)[0].tolist()
            clusters.append(curr_cluster)
        return np.array(clusters, dtype=object)
