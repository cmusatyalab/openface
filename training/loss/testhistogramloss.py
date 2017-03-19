# -*- coding: utf-8 -*-
""" Created by Cenk Bircanoğlu on 15/03/2017 """
import sys

import numpy as np

__author__ = 'cenk'


# assign points to grid bins
def getPlaces(x, grid):
    places_to_bins = dict()  # i of sorted x to j in grid
    bins_to_places = dict()

    for i in xrange(len(grid)):
        bins_to_places[i] = list()
    inx_sorted = np.argsort(x)

    ind = 1
    # find initial bucket :
    for i in xrange(len(grid)):
        if x[inx_sorted[0]] > grid[i]:
            ind = i + 1
        else:
            break

    x_start = 0
    while x[inx_sorted[x_start]] < grid[0]:
        x_start += 1

    for i in xrange(x_start, len(x)):

        while x[inx_sorted[i]] > grid[ind]:
            ind += 1
            if ind >= len(grid):
                return places_to_bins, bins_to_places
        places_to_bins[inx_sorted[i]] = ind
        bins_to_places[ind].append(inx_sorted[i])
    return places_to_bins, bins_to_places


# estimate the histogram using the assigments of points to grid bins
def getDistributionDensity(x, bins_to_places, grid, grid_delta):
    p = np.zeros_like(grid)
    for i in xrange(len(grid)):

        left_add = 0
        if i > 0:
            d_i_list_left = np.array(bins_to_places[i])
            left_dist = np.array([x[ii] for ii in d_i_list_left])
            left_add = sum(left_dist - grid[i - 1])

        right_add = 0
        if i < len(grid) - 1:
            d_i_list_right = np.array(bins_to_places[i + 1])
            right_dist = np.array([x[ii] for ii in d_i_list_right])
            right_add = sum(grid[i + 1] - right_dist)

        p[i] = (left_add + right_add)

    p /= len(x) * grid_delta
    return p


def calculateLossGradOverDistribution(distr_pos, distr_neg, L):
    grad_pos = np.dot(L, distr_neg)
    grad_neg = np.dot(distr_pos, L)
    return grad_pos, grad_neg


def calculateLossGradOverBinsForHist(d_pos, d_neg, grid_delta, grad_pos, grad_neg):
    grad_pos[1:] = (grad_pos[1:] - grad_pos[:-1])
    grad_pos /= grid_delta * len(d_pos)

    grad_neg[1:] = (grad_neg[1:] - grad_neg[:-1])
    grad_neg /= grid_delta * len(d_neg)

    return grad_pos, grad_neg


def getGradOverData(data, grad_over_bins, places_to_bins):
    grad = []
    for i in xrange(len(data)):
        print(places_to_bins[i])
        grad.append(grad_over_bins[places_to_bins[i]])

    return np.array(grad)


#######################################################################################################################
LOSS_SIMPLE = 'simple'
LOSS_LINEAR = 'linear'
LOSS_EXP = 'exp'

DISTR_TYPE_HIST = 'hist'
DISTR_TYPE_BETA = 'beta'


# Calculates probability of wrong order in pairs' similarities: positive pair less similar than negative one
# (this corresponds to 'simple' loss, other variants ('linear', 'exp') are generalizations that take into account
# not only the order but also the difference between the two similarity values).
# Can use histogram and beta-distribution to fit input data.
class DistributionLossLayer(object):
    def getL(self):
        L = np.ones((len(self.grid), len(self.grid)))
        for i in xrange(len(self.grid)):
            L[i] = self.grid[i] <= self.grid
        return L

    def setup(self, bottom, top):
        # np.seterr(all='raise')

        self.iteration = 0
        # parameters for the Histogram loss generalization variants
        self.alpha = 1
        self.margin = 0
        self.grid_delta = 0.01
        self.grid = np.array([i for i in np.arange(-1., 1. + self.grid_delta, self.grid_delta)])
        self.pos_label = 1
        self.neg_label = -1

    def reshape(self, bottom, top):
        ## bottom[0] is cosine similarities
        ## bottom[1] is pair labels

        top[0].reshape(1)

    def forward(self, bottom, top):
        bottom[0].data[bottom[0].data >= 1.] = 1.
        bottom[0].data[bottom[0].data <= -1.] = -1.

        self.pos_indecies = bottom[1].data == self.pos_label
        self.neg_indecies = bottom[1].data == self.neg_label

        self.d_pos = np.array(bottom[0].data[self.pos_indecies])
        self.d_neg = np.array(bottom[0].data[self.neg_indecies])

        self.places_to_bins_pos, self.bins_to_places_pos = getPlaces(self.d_pos, self.grid)
        self.places_to_bins_neg, self.bins_to_places_neg = getPlaces(self.d_neg, self.grid)

        self.distr_pos = getDistributionDensity(self.d_pos, self.bins_to_places_pos, self.grid, self.grid_delta)
        self.distr_neg = getDistributionDensity(self.d_neg, self.bins_to_places_neg, self.grid, self.grid_delta)

        L = self.getL()
        top[0].data[...] = np.dot(np.dot(self.distr_pos, L), self.distr_neg)

        sys.stdout.flush()
        self.iteration += 1

    def backward(self, top, propagate_down, bottom):
        L = self.getL()
        grad_pos_distr, grad_neg_distr = calculateLossGradOverDistribution(self.distr_pos, self.distr_neg, L)

        self.grad_pos_bin, self.grad_neg_bin = calculateLossGradOverBinsForHist(self.d_pos, self.d_neg,
                                                                                self.grid_delta, grad_pos_distr,
                                                                                grad_neg_distr)
        self.grad_pos = getGradOverData(self.d_pos, self.grad_pos_bin, self.places_to_bins_pos)
        self.grad_neg = getGradOverData(self.d_neg, self.grad_neg_bin, self.places_to_bins_neg)

        grad = np.zeros((len(self.grad_pos) + len(self.grad_neg), 1, 1, 1))
        grad[self.pos_indecies] = self.grad_pos
        grad[self.neg_indecies] = self.grad_neg
        bottom[0].diff[...] = grad


def python_net_file():
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("""name: 'pythonnet' force_backward: true
        input: 'data' input_shape { dim: 10 dim: 9 dim: 8 }
        layer { type: 'Python' name: 'one' bottom: 'data' top: 'one'
          python_param { module: 'test_python_layer' layer: 'SimpleLayer' } }
        layer { type: 'Python' name: 'two' bottom: 'one' top: 'two'
          python_param { module: 'test_python_layer' layer: 'SimpleLayer' } }
        layer { type: 'Python' name: 'three' bottom: 'two' top: 'three'
          python_param { module: 'test_python_layer' layer: 'SimpleLayer' } }""")
        return f.name


if __name__ == '__main__':

    def getL(grid):
        L = np.ones((len(grid), len(grid)))
        for i in xrange(len(grid)):
            L[i] = grid[i] <= grid
        return L


    d_pos = np.array([
        [19],
        [15],
        [1]
    ])
    d_neg = np.array([
        [7],
        [2],
        [15]
    ])

    d_pos = np.cos(d_pos).reshape(3)
    d_neg = np.cos(d_neg).reshape(3)
    print(d_pos, d_neg)
    grid = np.array([i for i in np.arange(-1., 1. + 0.2, 0.2)])
    places_to_bins_pos, bins_to_places_pos = getPlaces(d_pos, grid)
    places_to_bins_neg, bins_to_places_neg = getPlaces(d_neg, grid)
    dist_pos = getDistributionDensity(d_pos, bins_to_places_pos, grid, 0.2)
    dist_neg = getDistributionDensity(d_neg, bins_to_places_neg, grid, 0.2)

    L = getL(grid)
    bb = np.dot(np.dot(dist_pos, L), dist_neg)
    grad_pos, grad_neg = calculateLossGradOverDistribution(dist_pos, dist_neg, L)
    grad_pos_bin, grad_neg_bin = calculateLossGradOverBinsForHist(d_pos, d_neg, 0.2, grad_pos, grad_neg)
    grad_pos = getGradOverData(d_pos, grad_pos_bin, places_to_bins_pos)
    grad_neg = getGradOverData(d_neg, grad_neg_bin, places_to_bins_neg)
    print(grad_pos)
    print(grad_neg)
    pos_indecies = [0, 1, 2]
    neg_indecies = [3, 4, 5]
    grad = np.zeros((len(grad_pos) + len(grad_neg), 1))
    grad[pos_indecies] = grad_pos.reshape(3, 1)
    grad[neg_indecies] = grad_neg.reshape(3, 1)
    print(grad)
