#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STPPG: Spatio-Temporal Point Process Generator

References:
- https://www.jstatsoft.org/article/view/v053i02
- https://www.ism.ac.jp/editsec/aism/pdf/044_1_0001.pdf
- https://github.com/meowoodie/Spatio-Temporal-Point-Process-Simulator

Dependencies:
- Python 3.6.7
"""

import os, sys
sys.path.append('C:/Users/snwhg/Dropbox/Research/UH/hawkes/stpp_simulator')

import utils
import arrow
import numpy as np
from scipy.stats import norm

class BivariateHawkesLam(object):
    """Intensity of Spatio-temporal Hawkes point process"""
    def __init__(self, mu, kernel, maximum=1e+4):
        self.mu = mu
        self.kernel = [kernel[:2], kernel[2:]]
        self.maximum = maximum

    def value(self, t, his_t, s, his_s, point_type, his_type):
        """
        return the intensity value at (t, s).
        The last element of seq_t and seq_s is the location (t, s) that we are
        going to inspect. Prior to that are the past locations which have
        occurred.
        """
        if len(his_t) > 0:
            val = self.mu[point_type] + np.sum([self.kernel[point_type][his_type_i].nu(t, s, his_t_i, his_s_i) for (his_type_i, his_t_i, his_s_i) in zip(his_type, his_t, his_s)])
        else:
            val = self.mu[point_type]
        return val

    def upper_bound(self):
        """return the upper bound of the intensity value"""
        return self.maximum

    def __str__(self):
        return "Hawkes processes"

class BivariateSpatialTemporalPointProcess(object):
    """
    Marked Spatial Temporal Hawkes Process

    A stochastic spatial temporal points generator based on Hawkes process.
    """

    def __init__(self, lam):
        """
        Params:
        """
        # model parameters
        self.lam     = lam

    def _homogeneous_poisson_sampling(self, T=[0, 1], S=[[0, 1], [0, 1]]):
        
        _S     = [T] + S
        # sample the number of events from S
        n      = utils.lebesgue_measure(_S)
        N      = np.random.poisson(size=1, lam=self.lam.upper_bound() * n)
        # N      = np.random.poisson(size=1, lam=500 * n)
        # simulate spatial sequence and temporal sequence separately.
        points = [np.random.uniform(_S[i][0], _S[i][1], N[0]) for i in range(len(_S))]
        points = np.array(points).transpose()
        points = points[points[:, 0].argsort()]
        # points = [[ np.random.uniform(_S[i][0], _S[i][1], N[0]) for i in range(len(_S)) ], [ np.random.uniform(_S[i][0], _S[i][1], N[1]) for i in range(len(_S)) ]]
        # points = [np.array(point).transpose()  for point in points]
        # sort the sequence regarding the ascending order of the temporal sample.
        # points = [[point[point[:, 0].argsort()], np.repeat(i, point.shape[0])] for i, point in enumerate(points)]
        return points

    def _inhomogeneous_poisson_thinning(self, homo_points, verbose):
        """
        To generate a realization of an inhomogeneous Poisson process in S × T, this
        function uses a thining algorithm as follows. For a given intensity function
        lam(s, t):
        1. Define an upper bound max_lam for the intensity function lam(s, t)
        2. Simulate a homogeneous Poisson process with intensity max_lam.
        3. "Thin" the simulated process as follows,
            a. Compute p = lam(s, t)/max_lam for each point (s, t) of the homogeneous
            Poisson process
            b. Generate a sample u from the uniform distribution on (0, 1)
            c. Retain the locations for which u <= p.
        """
        retained_points = np.empty((0, homo_points.shape[1]))
        type_points = np.empty(0, dtype=np.int32)
        if verbose:
            print("[%s] generate %s samples from homogeneous poisson point process" % \
                (arrow.now(), homo_points.shape), file=sys.stderr)
        # thining samples by acceptance rate.
        for i in range(homo_points.shape[0]):
            # current time, location and generated historical times and locations.
            t     = homo_points[i, 0]
            s     = homo_points[i, 1:]
            his_t = retained_points[:, 0]
            his_s = retained_points[:, 1:]
            his_type = type_points[:]
            # thinning
            
            lam_value = [self.lam.value(t, his_t, s, his_s, 0, his_type), self.lam.value(t, his_t, s, his_s, 1, his_type)]
            if lam_value[0]<=0 or lam_value[1]<=0:
                raise ValueError("Nonpositive conditional intensity values obtained.")
            if lam_value[0]==lam_value[1]:
                point_type = np.random.randint(2)
            else:
                point_type = np.random.choice(np.arange(2), p = np.array(lam_value)/sum(lam_value))

            lam_bar   = self.lam.upper_bound()
            D         = np.random.uniform()
            lam_value = sum(lam_value)
            # - if lam_value is greater than lam_bar, then skip the generation process
            #   and return None.
            if lam_value > lam_bar:
                print("intensity %f is greater than upper bound %f." % (lam_value, lam_bar), file=sys.stderr)
                # print(i, retained_points, his_type)
                # print(retained_points.shape[0])
                return None, None
            # accept
            if lam_value >= D * lam_bar:
                # retained_points.append(homo_points[i])
                retained_points = np.concatenate([retained_points, homo_points[[i], :]], axis=0)
                type_points = np.append(type_points, point_type)
            # monitor the process of the generation
            if verbose and i != 0 and i % int(homo_points.shape[0] / 10) == 0:
                print("[%s] %d raw samples have been checked. %d samples have been retained." % \
                    (arrow.now(), i, retained_points.shape[0]), file=sys.stderr)
        # log the final results of the thinning algorithm
        if verbose:
            print("[%s] thining samples %s based on %s." % \
                (arrow.now(), retained_points.shape, self.lam), file=sys.stderr)
        return retained_points, type_points

    def generate(self, T=[0, 1], S=[[0, 1], [0, 1]], batch_size=10, min_n_points=5, verbose=True):
        """
        generate spatio-temporal points given lambda and kernel function
        """
        points_list = []
        point_types_list = []
        sizes       = []
        max_len     = 0
        b           = 0
        # generate inhomogeneous poisson points iterately
        while b < batch_size:
            homo_points = self._homogeneous_poisson_sampling(T, S)
            points, point_types = self._inhomogeneous_poisson_thinning(homo_points, verbose)
            if points is None or len(points) < min_n_points:
                continue
            max_len = points.shape[0] if max_len < points.shape[0] else max_len
            points_list.append(points)
            point_types_list.append(point_types)
            sizes.append(len(points))
            print("[%s] %d-th sequence is generated." % (arrow.now(), b+1), file=sys.stderr)
            b += 1
        # fit the data into a tensor
        data = np.zeros((batch_size, max_len, 3))
        data_types = np.zeros((batch_size, max_len))
        for b in range(batch_size):
            data[b, :points_list[b].shape[0]] = points_list[b]
            data_types[b, :point_types_list[b].shape[0]] = point_types_list[b]
        return data, data_types.astype(np.int64), sizes
