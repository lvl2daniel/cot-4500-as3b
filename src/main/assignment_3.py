import numpy as np

def f(t, y):
    return t - y**2

def euler(f, t0, y0, T, n):
    h = (T - t0) / n
    t, y = t0, y0
    for _ in range(n):
        y += h * f(t, y)
        t += h
    return y

def rk(f, t0, y0, T, n):
    h = (T - t0) / n
    t, y = t0, y0
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h * k1/2)
        k3 = f(t + h/2, y + h * k2/2)
        k4 = f(t + h, y + h * k3)
        y += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h
    return y

def gaussian_elimination(A_aug):
    A = np.array(A_aug, dtype=float)
    n = A.shape[0]

    for i in range(n):
        pivot = A[i, i]
        if np.isclose(pivot, 0.0):
            found_pivot = False
            for k in range(i + 1, n):
                if not np.isclose(A[k, i], 0.0):
                    A[[i, k]] = A[[k, i]]
                    pivot = A[i, i]
                    found_pivot = True
                    break
            if not found_pivot:
                return None


        for j in range(i + 1, n):
            multiplier = A[j, i] / pivot
            A[j, i:] = A[j, i:] - multiplier * A[i, i:]

    for i in range(n):
        if np.allclose(A[i, :n], 0) and not np.isclose(A[i, n], 0):
            return None

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ax = np.dot(A[i, i+1:n], x[i+1:n])
        if np.isclose(A[i,i], 0.0):
             return None
        x[i] = (A[i, n] - sum_ax) / A[i, i]

    return x


def lu_factorization(A):
    n = A.shape[0]
    L = np.eye(n, dtype=float)
    U = np.array(A, dtype=float)

    for j in range(n):
        pivot = U[j, j]

        for i in range(j + 1, n):
            multiplier = U[i, j] / pivot
            L[i, j] = multiplier
            U[i, j:] = U[i, j:] - multiplier * U[j, j:]

    det = np.prod(np.diag(U))

    return det, L, U


def is_diagonally_dominant(A):
    A_abs = np.abs(A)
    n = A.shape[0]
    for i in range(n):
        diag_element = A_abs[i, i]
        sum_off_diag = np.sum(A_abs[i, :]) - diag_element
        if diag_element <= sum_off_diag:
            return False
    return True


def is_positive_definite(A):
    if A.shape[0] != A.shape[1]:
        return False

    if not np.allclose(A, A.T):
        return False

    try:
        eigenvalues = np.linalg.eigvalsh(A)
        return np.all(eigenvalues > 1e-10)
    except np.linalg.LinAlgError:
        return False


if __name__ == "__main__":
    print(euler(f, 0, 1, 2, 10))
    print("\n")
    print(rk(f, 0, 1, 2, 10))
    print("\n")

    A1_aug = np.array([
        [2, -1,  1,  6],
        [1,  3,  1,  0],
        [-1, 5,  4, -3]
    ])
    solution1 = gaussian_elimination(A1_aug)
    if solution1 is not None:
        print(f"{solution1}\n")


    A2 = np.array([
        [1,  1,  0,  3],
        [2,  1, -1,  1],
        [3, -1, -1,  2],
        [-1, 2,  3, -1]
    ])
    det2, L2, U2 = lu_factorization(A2)
    if det2 is not None:
        print(f"{det2}\n")
        print(f"{L2}\n")
        print(f"{U2}\n")


    A3 = np.array([
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ])
    dominant3 = is_diagonally_dominant(A3)
    print(f"{dominant3}\n")


    A4 = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ])
    positive_definite4 = is_positive_definite(A4)
    print(f"{positive_definite4}\n")
