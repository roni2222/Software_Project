import sys
import numpy as np
import symnmf

def print_matrix(matrix):
    """
    Prints a matrix (list of lists or numpy array) formatted 
    with exactly 4 decimal places, comma-separated.
    """
    for row in matrix:
        print(",".join([f"{val:.4f}" for val in row]))

def main():
    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        file_name = sys.argv[3]

        X = np.loadtxt(file_name, delimiter=',')
        X = np.atleast_2d(X)

        n = X.shape[0]
        if k <= 1 or k >= n:
            print("An Error Has Occurred")
            sys.exit(1)

        if goal == "sym":
            result = symnmf.sym(X)
            print_matrix(result)

        elif goal == "ddg":
            result = symnmf.ddg(X)
            print_matrix(result)

        elif goal == "norm":
            result = symnmf.norm(X)
            print_matrix(result)

        elif goal == "symnmf":
            W = symnmf.norm(X)
            W_np = np.array(W)
            
            np.random.seed(1234)
            n = W_np.shape[0]
            m = np.mean(W_np)
            
            upper_bound = 2.0 * np.sqrt(m / k)
            H = np.random.uniform(0.0, upper_bound, size=(n, k))
            
            final_H = symnmf.symnmf(W_np, H)
            print_matrix(final_H)

        else:
            print("An Error Has Occurred")
            sys.exit(1)

    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()