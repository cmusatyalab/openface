# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 21/02/2017 """
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'cenk'

N = 10
x1 = np.random.rand(N)
y1 = np.random.rand(N)
plt.scatter(x1, y1, c='yellow')

x2 = np.random.rand(N)
y2 = np.random.rand(N)
plt.scatter(x2, y2, c='blue')

x3 = np.random.rand(N)
y3 = np.random.rand(N)
plt.scatter(x3, y3, c='black')

x4 = np.random.rand(N)
y4 = np.random.rand(N)
plt.scatter(x4, y4, c='red')

x5 = np.random.rand(N)
y5 = np.random.rand(N)
plt.scatter(x5, y5, c='green')

plt.legend(['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Predictions'], loc='upper left')

plt.show()
