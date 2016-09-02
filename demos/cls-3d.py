#!/usr/bin/env python2
# projectS and projectC were written by Gabriele Farina.

def projectS(rho, theta, z):
    p = np.array([np.sqrt(3.)*rho*(np.cos(theta) + np.sin(theta))/2.,
                  z + 1. + rho*(np.cos(theta) - np.sin(theta))/2.])
    p += np.array([1.5, 0.5])
    p /= 3.
    return p

def projectC(x, y, z):
    rho = np.sqrt(x**2 + y**2)
    if x == 0 and y == 0:
        theta = 0
    elif x >= 0:
        theta = np.arcsin(y/rho)
    else:
        theta = -np.arcsin(y/rho) + np.pi

    return projectS(rho, theta, z)
