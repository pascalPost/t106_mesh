# Copyright (c) 2022 Pascal Post
# This code is licensed under AGPL license (see LICENSE.txt for details)

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def derivatives(X, x_xi, x_eta, y_xi, y_eta):
    x = X[:, :, 0]
    y = X[:, :, 1]

    # loop over internal mesh points for now
    for i in range(1, X.shape[0] - 1):
        for j in range(1, X.shape[1] - 1):
            x_xi[i, j] = 0.5 * (x[i+1, j] - x[i-1, j])
            x_eta[i, j] = 0.5 * (x[i, j+1] - x[i, j-1])
            y_xi[i, j] = 0.5 * (y[i+1, j] - y[i-1, j])
            y_eta[i, j] = 0.5 * (y[i, j+1] - y[i, j-1])


def derivatives2(X, x_xi2, x_eta2, y_xi2, y_eta2, x_xieta, y_xieta):
    x = X[:, :, 0]
    y = X[:, :, 1]

    for i in range(1, X.shape[0]-1):
        for j in range(1, X.shape[1]-1):
            x_xi2[i, j] = x[i+1, j] - 2 * x[i, j] + x[i-1, j]
            x_eta2[i, j] = x[i, j+1] - 2 * x[i, j] + x[i, j-1]
            y_xi2[i, j] = y[i+1, j] - 2 * y[i, j] + y[i-1, j]
            y_eta2[i, j] = y[i, j+1] - 2 * y[i, j] + y[i, j-1]

            x_xieta[i, j] = 0.25 * (x[i + 1][j + 1] - x[i + 1]
                                    [j - 1] - x[i - 1][j + 1] + x[i - 1][j - 1])
            y_xieta[i, j] = 0.25 * (y[i + 1][j + 1] - y[i + 1]
                                    [j - 1] - y[i - 1][j + 1] + y[i - 1][j - 1])


def matrix_index(i, j, I):
    return (i-1) + (j-1) * (I-2)


def smooth_block(X, iterations=10):
    I = X.shape[0]
    J = X.shape[1]

    x = X[:, :, 0]
    y = X[:, :, 1]

    print("Laplace smoothing Block")
    S = np.zeros((I, J))
    T = np.zeros((I, J))

    # allocations
    x_xi = np.zeros((I, J))
    x_eta = np.zeros((I, J))
    y_xi = np.zeros((I, J))
    y_eta = np.zeros((I, J))

    N = (I-2) * (J-2)

    x_last = np.zeros(N)
    y_last = np.zeros(N)

    for i in range(1, I - 1):
        for j in range(1, J - 1):
            idx = matrix_index(i, j, I)

            x_last[idx] = x[i, j]
            y_last[idx] = y[i, j]

    for it in range(iterations):
        print(f"  iteration {it}")

        derivatives(X, x_xi, x_eta, y_xi, y_eta)

        P = x_eta**2 + y_eta**2
        Q = x_xi * x_eta + y_xi * y_eta
        R = x_xi**2 + y_xi**2

        a_i_j = -2*P-2*R
        a_ip1_j = P + 0.5 * S
        a_im1_j = P - 0.5 * S
        a_i_jp1 = R + 0.5 * T
        a_i_jm1 = R - 0.5 * T
        a_ip1_jp1 = -0.5 * Q
        a_ip1_jm1 = 0.5 * Q
        a_im1_jp1 = 0.5 * Q
        a_im1_jm1 = -0.5 * Q

        b_x = np.zeros(N)
        b_y = np.zeros(N)

        A = coo_matrix((N, N), dtype=np.float64)

        A = A.todok()  # convert to dok
        for i in range(1, I - 1):
            for j in range(1, J - 1):
                idx = matrix_index(i, j, I)

                A[idx, idx] = a_i_j[i, j]

                # boundary values
                if i == 1:
                    b_x[idx] -= a_im1_j[i, j] * x[i-1, j]
                    b_y[idx] -= a_im1_j[i, j] * y[i-1, j]
                else:
                    A[idx, matrix_index(i-1, j, I)] = a_im1_j[i, j]

                if i == I - 2:
                    b_x[idx] -= a_ip1_j[i, j] * x[i+1, j]
                    b_y[idx] -= a_ip1_j[i, j] * y[i+1, j]
                else:
                    A[idx, matrix_index(i+1, j, I)] = a_ip1_j[i, j]

                if j == 1:
                    b_x[idx] -= a_i_jm1[i, j] * x[i, j-1]
                    b_y[idx] -= a_i_jm1[i, j] * y[i, j-1]
                else:
                    A[idx, matrix_index(i, j-1, I)] = a_i_jm1[i, j]

                if j == J - 2:
                    b_x[idx] -= a_i_jp1[i, j] * x[i, j+1]
                    b_y[idx] -= a_i_jp1[i, j] * y[i, j+1]
                else:
                    A[idx, matrix_index(i, j+1, I)] = a_i_jp1[i, j]

                if i == I - 2 or j == J - 2:
                    b_x[idx] -= a_ip1_jp1[i, j] * x[i+1, j+1]
                    b_y[idx] -= a_ip1_jp1[i, j] * y[i+1, j+1]
                else:
                    A[idx, matrix_index(i+1, j+1, I)] = a_ip1_jp1[i, j]

                if i == I-2 or j == 1:
                    b_x[idx] -= a_ip1_jm1[i, j] * x[i+1, j-1]
                    b_y[idx] -= a_ip1_jm1[i, j] * y[i+1, j-1]
                else:
                    A[idx, matrix_index(i+1, j-1, I)] = a_ip1_jm1[i, j]

                if i == 1 or j == J - 2:
                    b_x[idx] -= a_im1_jp1[i, j] * x[i-1, j+1]
                    b_y[idx] -= a_im1_jp1[i, j] * y[i-1, j+1]
                else:
                    A[idx, matrix_index(i-1, j+1, I)] = a_im1_jp1[i, j]

                if i == 1 or j == 1:
                    b_x[idx] -= a_im1_jm1[i, j] * x[i-1, j-1]
                    b_y[idx] -= a_im1_jm1[i, j] * y[i-1, j-1]
                else:
                    A[idx, matrix_index(i-1, j-1, I)] = a_im1_jm1[i, j]
        A = A.tocoo()  # convert back to coo

        x_new = spsolve(A, b_x)
        y_new = spsolve(A, b_y)

        res = np.linalg.norm(np.append(x_new, y_new) -
                             np.append(x_last, y_last))

        print(f"    residual {res}")

        # x_xi2 = np.zeros((I, J))
        # x_eta2 = np.zeros((I, J))
        # y_xi2 = np.zeros((I, J))
        # y_eta2 = np.zeros((I, J))
        # x_xieta = np.zeros((I, J))
        # y_xieta = np.zeros((I, J))

        # derivatives2(X, x_xi2, x_eta2, y_xi2, y_eta2, x_xieta, y_xieta)

        # res_x_0 = P * x_xi2 - 2 * Q * x_xieta + R * x_eta2 + S * x_xi + T * x_eta
        # res_y_0 = P * y_xi2 - 2 * Q * y_xieta + R * y_eta2 + S * y_xi + T * y_eta

        for i in range(1, I - 1):
            for j in range(1, J - 1):
                idx = matrix_index(i, j, I)

                x[i, j] = x_new[idx]
                y[i, j] = y_new[idx]

        # derivatives2(X, x_xi2, x_eta2, y_xi2, y_eta2, x_xieta, y_xieta)

        # res_x_1 = P * x_xi2 - 2 * Q * x_xieta + R * x_eta2 + S * x_xi + T * x_eta
        # res_y_1 = P * y_xi2 - 2 * Q * y_xieta + R * y_eta2 + S * y_xi + T * y_eta

        x_last = x_new
        y_last = y_new
