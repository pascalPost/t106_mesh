# Copyright (c) 2022 Pascal Post
# This code is licensed under AGPL license (see LICENSE.txt for details)

import math
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tfi import tfi_linear_2d

pitch = 88.36 * 1e-3  # m

df_ss = pd.read_csv('T106_ss.dat', comment='#',
                    header=None, names=['x', 'y'], delim_whitespace=True)
df_ps = pd.read_csv('T106_ps.dat', comment='#',
                    header=None, names=['x', 'y'], delim_whitespace=True)

# combine profile
df_blade = pd.concat([df_ss[:], df_ps[1:]])


# create clustering around the blade

# create spline representation wich goes from
# 0 - 0.5 on pressure side
# 0.5 - 1 on suction side

# compute arclength on suction side
u = [0]
for i in range(1, len(df_ss.index)):
    x_i, y_i = df_ss.iloc[i]
    x_im1, y_im1 = df_ss.iloc[i-1]
    s = np.sqrt((x_i - x_im1)**2 + (y_i - y_im1)**2)  # euclidian distance
    u.append(s + u[i-1])  # cumsum of euclidian distance

# compute relative arc length for suction side between 0 and 0.5
u_min = u[0]
u_max = u[-1]
du = u_max - u_min
for i in range(len(u)):
    u[i] = (u[i] - u_min) / du * 0.5


# compute arclength on pressure side
u_ps = [0]
for i in range(1, len(df_ps.index)):
    x_i, y_i = df_ps.iloc[i]
    x_im1, y_im1 = df_ps.iloc[i-1]
    s = np.sqrt((x_i - x_im1)**2 + (y_i - y_im1)**2)  # euclidian distance
    u_ps.append(s + u_ps[i-1])  # cumsum of euclidian distance

# compute relative arc length for suction side between 0.5 and 1
u_min = u_ps[0]
u_max = u_ps[-1]
du = u_max - u_min
for i in range(len(u_ps)):
    u_ps[i] = 0.5 + (u_ps[i] - u_min) / du * 0.5

# remove first and last element
del u_ps[0]
#del u_ps[-1]

# combined u
u.extend(u_ps)


# fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
# is needed in order to force the spline fit to pass through all the input
# points.
# see: https://stackoverflow.com/questions/33962717/interpolating-a-closed-curve-using-scipy
tck, u = interpolate.splprep(
    [df_blade['x'], df_blade['y']], s=0, per=True, u=u)

# evaluate the spline fits for 1000 evenly spaced distance values
xi, yi = interpolate.splev(np.linspace(0, 1, 10000), tck)


# # plot the result
# fig, ax = plt.subplots(1, 1)
# ax.set_aspect('equal')
# plt.gca().set_aspect('equal')
# ax.plot(df_ss['x'], df_ss['y'], 'or')
# ax.plot(df_ps['x'], df_ps['y'], '.b')

# ax.plot(df_blade['x'], df_blade['y'], '.b')
# ax.plot(xi, yi, '-b')


# ax.plot(df_ss['x'], df_ss['y'] + pitch, 'or')
# ax.plot(df_ps['x'], df_ps['y'] + pitch, '.b')


# # create blocking due to wang 2013

# # x_le, y_le = df_ss.iloc[0]
# # x_te, y_te = df_ss.iloc[-1]

# ax.plot(x_le, y_le, '.g')
# ax.plot(x_te, y_te, '.g')

# ax.plot(x_le, y_le-0.5*pitch, '.g')
# ax.plot(x_te, y_te-0.5*pitch, '.g')

# plt.show()


# Blocking


nbb = 120
ninm = 10
scut = 20
ncut = 20

# helper

x_le, y_le = interpolate.splev(0, tck)
x_te, y_te = interpolate.splev(0.5, tck)

plt.plot(x_le, y_le, 'ob')
plt.plot(x_te, y_te, 'ob')

# hyperbolic tangent function (Thompson & Warsi 1982)

# rho = np.linspace(0, 1, num=nbb)

# eta = -1.2

# #b = 0.5 * np.log10((eta - 1.0) / (eta + 1.0))
# b = 2


# def s_der(rho):
#     return 1 - (1 - np.tanh(b * (1 - rho))) / np.tanh(b)


# alpha = 0.5

# s = np.where(
#     rho < alpha, alpha *
#     s_der(rho/alpha), (1 - s_der((1-rho)/(1-alpha)))
#     * alpha+s_der((1-rho)/(1-alpha))
# )


# def hyperbolic_tangent_fun(eta):
#   # eta should be between 0 and 1


def single_exponential_clustering(n, A=1.0):
    x = np.linspace(0, 1, num=n)
    y = (np.exp(A*x)-1)/(np.exp(A)-1)
    return x, y


# Roberts cluster function, see
# https://github.com/luohancfd/CFCFD-NG/blob/dev/lib/nm/source/fobject.cxx

def roberts_clustering(n, alpha=0.5, beta=2.0):
    # alpha = 0.5 cluster at both ends
    # alpha = 0.0 cluster toward t=1.0
    # stretching factor 1.0 < beta < +inf, closer to 1.0 gives stronger clustering
    x = np.linspace(0, 1, num=n)
    tmp = (beta + 1.0) / (beta - 1.0)
    tmp = np.power(tmp, ((x - alpha)/(1.0 - alpha)))
    tbar = (beta + 2.0 * alpha) * tmp - beta + 2.0 * alpha
    y = tbar / ((2.0 * alpha + 1.0) * (1.0 + tmp))
    return y


u_blade = roberts_clustering(nbb, 0.5, 1.01)

u_ss = u_blade * 0.5
u_ps = 0.5 + u_ss

u = np.append(u_ss, u_ps)

x_blade, y_blade = interpolate.splev(u, tck)

plt.plot(x_blade, y_blade, '.g')


# construct some chamber line approximation

s = np.asarray([0.2, 0.4, 0.6, 0.8])
s_ps = s * 0.5
s_ss = np.flip(0.5 + s_ps)

# print(s_ps)
# print(s_ss)

xc_ps, yc_ps = interpolate.splev(s_ps, tck)
xc_ss, yc_ss = interpolate.splev(s_ss, tck)

#print(xc_ps, yc_ps)
#print(xc_ss, yc_ss)

xc = 0.5 * (xc_ps + xc_ss)
yc = 0.5 * (yc_ps + yc_ss)

plt.plot(xc, yc, 'b')


# periodic lines

x_per_lower_0 = x_le
y_per_lower_0 = y_le - pitch / 2

x_per_upper_0 = x_le
y_per_upper_0 = y_le + pitch / 2

plt.plot(x_per_lower_0, y_per_lower_0, '.r')
plt.plot(x_per_upper_0, y_per_upper_0, '.r')

x_per_lower_1 = x_te
y_per_lower_1 = y_te - pitch / 2

x_per_upper_1 = x_te
y_per_upper_1 = y_te + pitch / 2

plt.plot(x_per_lower_1, y_per_lower_1, '.r')
plt.plot(x_per_upper_1, y_per_upper_1, '.r')

x_per_lower = np.insert(xc, 0, x_per_lower_0)
x_per_lower = np.append(x_per_lower, x_per_lower_1)

y_per_lower = np.insert(yc - 0.5 * pitch, 0, y_per_lower_0)
y_per_lower = np.append(y_per_lower, y_per_lower_1)

f_per_lower = interpolate.interp1d(x_per_lower, y_per_lower, kind='cubic')

x_per_lower_plot = np.linspace(x_per_lower[0], x_per_lower[-1])
y_per_lower_plot = f_per_lower(x_per_lower_plot)

plt.plot(x_per_lower, y_per_lower, 'ro')
plt.plot(x_per_lower_plot, y_per_lower_plot, 'r')
plt.plot(x_per_lower_plot, y_per_lower_plot + pitch, 'r')

# inm block

x_inm_lower_start = x_blade[-(ninm // 2) - 1]
y_inm_lower_start = y_blade[-(ninm // 2) - 1]

plt.plot(x_inm_lower_start, y_inm_lower_start, '.r')

x_inm_upper_start = x_blade[ninm // 2]
y_inm_upper_start = y_blade[ninm // 2]

plt.plot(x_inm_upper_start, y_inm_upper_start, '.r')

x_inm_lower_end = x_blade[-(ninm // 2) - 1] - 0.01
y_inm_lower_end = y_blade[-(ninm // 2) - 1] - 0.01

x_inm_upper_end = x_blade[ninm // 2] - 0.01
y_inm_upper_end = y_blade[ninm // 2] + 0.01

plt.plot([x_inm_lower_start, x_inm_lower_end], [
         y_inm_lower_start, y_inm_lower_end], 'r')
plt.plot([x_inm_upper_start, x_inm_upper_end], [
         y_inm_upper_start, y_inm_upper_end], 'r')
plt.plot([x_inm_lower_end, x_inm_upper_end], [
         y_inm_lower_end, y_inm_upper_end], 'r')

# exm block

x_exm_lower_start = x_blade[nbb + ninm//2]
y_exm_lower_start = y_blade[nbb + ninm//2]

x_exm_lower_end = x_exm_lower_start + 0.007
y_exm_lower_end = y_exm_lower_start - 0.025

x_exm_upper_start = x_blade[nbb - ninm//2 - 1]
y_exm_upper_start = y_blade[nbb - ninm//2 - 1]

x_exm_upper_end = x_exm_upper_start + 0.007
y_exm_upper_end = y_exm_upper_start + 0.001

plt.plot([x_exm_lower_start, x_exm_lower_end], [
         y_exm_lower_start, y_exm_lower_end], '-or')
plt.plot([x_exm_upper_start, x_exm_upper_end], [
         y_exm_upper_start, y_exm_upper_end], '-or')
plt.plot([x_exm_lower_end, x_exm_upper_end], [
         y_exm_lower_end, y_exm_upper_end], '-or')

# inl block

x_inl_1 = x_blade[-(ninm // 2) - 1 - scut]
y_inl_1 = y_blade[-(ninm // 2) - 1 - scut]

plt.plot([x_inl_1, x_per_lower_0], [y_inl_1, y_per_lower_0], 'r')
plt.plot([x_inm_lower_end, x_per_lower_0], [
         y_inm_lower_end, y_per_lower_0], 'r')


# print(u)


# mesh creation

# TFI

# Block inm
iMax_inm = ninm + 1
jMax_inm = ncut

X_inm = np.zeros(shape=(iMax_inm, jMax_inm))
Y_inm = np.zeros(shape=(iMax_inm, jMax_inm))

dx_i_jMax = (x_inm_lower_end - x_inm_upper_end) / (iMax_inm - 1)
dy_i_jMax = (y_inm_lower_end - y_inm_upper_end) / (iMax_inm - 1)

for i in range(iMax_inm):
    if -ninm//2 + i <= 0:
        X_inm[i, 0] = x_blade[ninm//2 - i]
        Y_inm[i, 0] = y_blade[ninm//2 - i]
    else:
        # fix as the LE coordinate is [0] as well as [-1]
        X_inm[i, 0] = x_blade[ninm//2 - i - 1]
        Y_inm[i, 0] = y_blade[ninm//2 - i - 1]

    X_inm[i, jMax_inm - 1] = x_inm_upper_end + dx_i_jMax * i
    Y_inm[i, jMax_inm - 1] = y_inm_upper_end + dy_i_jMax * i

dx_0_j = (x_inm_upper_end - x_inm_upper_start) / (jMax_inm - 1)
dy_0_j = (y_inm_upper_end - y_inm_upper_start) / (jMax_inm - 1)

dx_iMax_j = (x_inm_lower_end - x_inm_lower_start) / (jMax_inm - 1)
dy_iMax_j = (y_inm_lower_end - y_inm_lower_start) / (jMax_inm - 1)

for j in range(1, jMax_inm):
    X_inm[0, j] = x_inm_upper_start + dx_0_j * j
    Y_inm[0, j] = y_inm_upper_start + dy_0_j * j

    X_inm[iMax_inm - 1, j] = x_inm_lower_start + dx_iMax_j * j
    Y_inm[iMax_inm - 1, j] = y_inm_lower_start + dy_iMax_j * j

X = np.stack((X_inm, Y_inm), axis=-1)

tfi_linear_2d(X)


plt.plot(X[:, :, 0], X[:, :, 1], '.r')

for i in range(X.shape[0]):
    plt.plot(X[i, :, 0], X[i, :, 1], '-b')

for j in range(X.shape[1]):
    plt.plot(X[:, j, 0], X[:, j, 1], '-b')

plt.minorticks_on()
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.grid(b=True, which='minor', color='0.65', linestyle='--')

plt.axis('equal')
plt.show()
