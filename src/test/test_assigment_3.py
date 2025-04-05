import unittest
import numpy as np
from src.main.assignment_3 import euler, rk, f, gaussian_elimination, lu_factorization, is_diagonally_dominant, is_positive_definite

class TestAssignment3(unittest.TestCase):
    def test_euler(self):
        self.assertAlmostEqual(euler(f, 0, 1, 2, 10), 1.2446380979332121, places=6)

    def test_rk4(self):
        self.assertAlmostEqual(rk(f, 0, 1, 2, 10), 1.251316587879806, places=6)

    def test_gaussian_elimination(self):
        A_aug = np.array([
            [2, -1,  1,  6],
            [1,  3,  1,  0],
            [-1, 5,  4, -3]
        ])
        expected = np.array([2.0, -1.0, 1.0])
        result = gaussian_elimination(A_aug)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_lu_factorization(self):
        A = np.array([
            [1,  1,  0,  3],
            [2,  1, -1,  1],
            [3, -1, -1,  2],
            [-1, 2,  3, -1]
        ])
        det_expected = 39.0
        L_expected = np.array([
            [ 1.,  0.,  0.,  0.],
            [ 2.,  1.,  0.,  0.],
            [ 3.,  4.,  1.,  0.],
            [-1., -3.,  0.,  1.]
        ])
        U_expected = np.array([
            [  1.,   1.,   0.,   3.],
            [  0.,  -1.,  -1.,  -5.],
            [  0.,   0.,   3.,  13.],
            [  0.,   0.,   0., -13.]
        ])
        det, L, U = lu_factorization(A)
        self.assertAlmostEqual(det, det_expected, places=6)
        np.testing.assert_allclose(L, L_expected, rtol=1e-6)
        np.testing.assert_allclose(U, U_expected, rtol=1e-6)

    def test_diagonal_dominance(self):
        A = np.array([
            [9, 0, 5, 2, 1],
            [3, 9, 1, 2, 1],
            [0, 1, 7, 2, 3],
            [4, 2, 3, 12, 2],
            [3, 2, 4, 0, 8]
        ])
        self.assertFalse(is_diagonally_dominant(A))

    def test_positive_definite(self):
        A = np.array([
            [2, 2, 1],
            [2, 3, 0],
            [1, 0, 2]
        ])
        self.assertTrue(is_positive_definite(A))

if __name__ == '__main__':
    unittest.main()

