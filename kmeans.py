import sys
import math

class vector:
    def __init__(self, cord):
        self.cord = cord
        self.dim = len(cord)
        self.cluster = None

# Calculate the distance Betweem Two given vectors
def distance_between(vec, cent):

    diff_sum = 0
    for i in range(vec.dim):
        diff = vec.cord[i] - cent[i] 
        diff_sum += diff * diff

    return diff_sum**0.5

def assign_vectors_to_centroids(vectors, centroids):

    for vector in vectors:
        
        cluster = 0
        min_distance = distance_between(vector, centroids[0])
        
        for i in range(len(centroids)):
            curr_distance = distance_between(vector, centroids[i])
            if curr_distance < min_distance:
                min_distance = curr_distance
                cluster = i
        
        vector.cluster = cluster

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

def print_centroids(centroids):
    for row in centroids:
        print(",".join(f"{x:.4f}" for x in row))
        
def main():
    vectors = []
    centroids = []
    eps = 0.001

    if (len(sys.argv) == 2):
        k = sys.argv[1]
        iter = 400

    elif(len(sys.argv) == 3):
        k = sys.argv[1]
        iter = sys.argv[2]

        if (not iter.isdigit() or int(iter) < 1 or int(iter) > 800):
            print("Incorrect maximum iteration!")
            exit(1) 
        
        iter = int(iter)
        
    else:
        print("An Error Has Occurred")
        exit(1) 

    # Initialize vectors list from input file
    vectors = []
    expected_dim = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]

        if any(p == "" for p in parts):
            print("An Error Has Occurred")
            sys.exit(1)

        try:
            cords = [float(p) for p in parts]
        except ValueError:
            print("An Error Has Occurred")
            sys.exit(1)

        if expected_dim is None:
            expected_dim = len(cords)
            if expected_dim == 0:
                print("An Error Has Occurred")
                sys.exit(1)

        elif len(cords) != expected_dim:
            print("An Error Has Occurred")
            sys.exit(1)

        vectors.append(vector(cords))

    if not vectors:
        print("An Error Has Occurred")
        sys.exit(1)

    dim = expected_dim

    if (not k.isdigit() or int(k) >= len(vectors) or int(k) <= 1):
        print("Incorrect number of clusters!")
        exit(1) 
        
    k = int(k)
    
    # Initialize centroids
    for i in range(k):
        centroids.append(vectors[i].cord)

    for i in range(iter):
        assign_vectors_to_centroids(vectors, centroids)
        max_shift = recompute_centroids(vectors, centroids, k, dim)
        if (max_shift < eps):
            break
    
    print_centroids(centroids)

if __name__ == "__main__":
    main()

