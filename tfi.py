import numpy as np


def _boundary_blended_control_function(U):
    """
    Boundary-Blended Control Functions eq. (3.21) from Thompson et al., Eds.,
    Handbook of grid generation. Boca Raton, Fla: CRC Press, 1999.

    Expected shape of U is [iMax, jMax, 2] with u = [:,:,0] & v = [:,:,1]
    """

    u = U[:, :, 0]
    v = U[:, :, 1]

    s1 = u[:, 0]
    s2 = u[:, -1]
    t1 = v[0, :]
    t2 = v[-1, :]

    # only non boundary values must be updated here
    for i in range(1, U.shape[0] - 1):
        for j in range(1, U.shape[1] - 1):
            u[i, j] = ((1 - t1[j]) * s1[i] + t1[j] * s2[i]) / \
                (1 - (s2[i] - s1[i]) * (t2[j] - t1[j]))
            v[i, j] = ((1 - s1[i]) * t1[j] + s1[i] * t2[j]) / \
                (1 - (t2[j] - t1[j]) * (s2[i] - s1[i]))


def _arclength_control_function(U, X):
    """
    arclength control function, see section 3.6.4, Thompson et
    al., Eds., Handbook of grid generation. Boca Raton, Fla: CRC Press, 1999.
    """

    def _arclength_control_function_line(X, U):
        for i in range(1, U.shape[0]):
            U[i] = U[i - 1] + np.sqrt((X[i, 0] - X[i-1, 0])
                                      ** 2 + (X[i, 1] - X[i-1, 1])**2)

        U[:] = U[:] / U[-1]

    if U.shape != X.shape:
        raise RuntimeError("Different sized coordinate fields encountered.")

    # Bottom
    _arclength_control_function_line(X[:, 0, :], U[:, 0, 0])

    # Top
    _arclength_control_function_line(X[:, -1, :], U[:, -1, 0])
    U[:, -1, 1] = 1

    # Left
    _arclength_control_function_line(X[0, :, :], U[0, :, 1])

    # Right
    _arclength_control_function_line(X[-1, :, :], U[-1, :, 1])
    U[-1, :, 0] = 1


def _intermediate_control_domain(X):
    """
    creation of the intermediate control domain, see section 3.6, Thompson et
    al., Eds., Handbook of grid generation. Boca Raton, Fla: CRC Press, 1999.
    """
    U = np.zeros_like(X)
    _arclength_control_function(U, X)
    _boundary_blended_control_function(U)

    # import matplotlib.pyplot as plt

    # plt.clf()
    # plt.close()
    # plt.plot(U[:, :, 0], U[:, :, 1], 'o')

    return U


def tfi_linear_2d(X):
    """
    linear TFI as described in chapter 3.5.1 of Thompson et al., Eds.,
    Handbook of grid generation. Boca Raton, Fla: CRC Press, 1999.

    Expected shape of X is [iMax, jMax, 2] with x = [:,:,0] & y = [:,:,1].

    The coordinates of the boundaries of the block must be set before calling
    this function. On this basis, the coordinates inside the block are computed.
    """

    U = _intermediate_control_domain(X)

    xi = U[:, :, 0]
    eta = U[:, :, 1]

    for i in range(1, X.shape[0]):
        for j in range(1, X.shape[1]):
            u_ij = (1 - xi[i, j]) * X[0, j] + xi[i, j] * X[-1, j]
            v_ij = (1 - eta[i, j]) * X[i, 0] + eta[i, j] * X[i, -1]
            uv_ij = \
                xi[i, j] * eta[i, j] * X[-1, -1] + \
                xi[i, j] * (1 - eta[i, j]) * X[-1, 0] + \
                (1 - xi[i, j]) * eta[i, j] * X[0, -1] + \
                (1 - xi[i, j]) * (1-eta[i, j]) * X[0, 0]
            X[i, j] = u_ij + v_ij - uv_ij
