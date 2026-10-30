import sys
import math
import numpy as np
from sklearn.metrics import silhouette_score
import symnmf


class Vector:
    def __init__(self, cord):
        self.cord = cord
        self.dim = len(cord)
        self.cluster = None


def distance_between(vec, cent):
    diff_sum = 0.0
    for i in range(vec.dim):
        diff = vec.cord[i] - cent[i]
        diff_sum += diff * diff
    return math.sqrt(diff_sum)


def assign_vectors_to_centroids(vectors, centroids):
    for vec in vectors:
        cluster = 0
        min_distance = distance_between(vec, centroids[0])

        for i in range(1, len(centroids)):
            curr_distance = distance_between(vec, centroids[i])
            if curr_distance < min_distance:
                min_distance = curr_distance
                cluster = i

        vec.cluster = cluster


def recompute_centroids(vectors, centroids, k, dim):
    max_shift = 0.0

    for i in range(k):
        cluster_points = [v.cord for v in vectors if v.cluster == i]

        if not cluster_points:
            continue

        new_centroid = []
        for m in range(dim):
            coord_sum = sum(p[m] for p in cluster_points)
            new_centroid.append(coord_sum / len(cluster_points))

        shift_sq = 0.0
        for m in range(dim):
            diff = new_centroid[m] - centroids[i][m]
            shift_sq += diff * diff

        shift = math.sqrt(shift_sq)
        if shift > max_shift:
            max_shift = shift

        centroids[i] = new_centroid

    return max_shift


def run_kmeans(X, k, max_iter=300, eps=1e-4):
    vectors = [Vector(row.tolist()) for row in X]
    dim = X.shape[1]

    centroids = [vectors[i].cord[:] for i in range(k)]

    for _ in range(max_iter):
        assign_vectors_to_centroids(vectors, centroids)
        max_shift = recompute_centroids(vectors, centroids, k, dim)
        if max_shift < eps:
            break

    labels = np.array([v.cluster for v in vectors])
    return labels


def main():
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        file_name = sys.argv[2]

        X = np.loadtxt(file_name, delimiter=",")
        X = np.atleast_2d(X)

        # SymNMF
        W = symnmf.norm(X)

        np.random.seed(1234)
        n = W.shape[0]
        m = np.mean(W)
        upper_bound = 2.0 * np.sqrt(m / k)
        H_init = np.random.uniform(0.0, upper_bound, size=(n, k))

        H_final = symnmf.symnmf(W, H_init)
        nmf_labels = np.argmax(H_final, axis=1)
        nmf_score = silhouette_score(X, nmf_labels)

        # KMeans
        kmeans_labels = run_kmeans(X, k, max_iter=300, eps=1e-4)
        kmeans_score = silhouette_score(X, kmeans_labels)

        print(f"nmf: {nmf_score:.4f}")
        print(f"kmeans: {kmeans_score:.4f}")

    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()