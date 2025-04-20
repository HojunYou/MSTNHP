import os, sys, pickle
sys.path.append(os.path.dirname(os.path.realpath(__name__)))

import numpy as np
import pandas as pd
import math

from mstppg import BivariateHawkesLam, BivariateSpatialTemporalPointProcess
from stppg import (
    SeparableExponentialKernel, 
    NonseparableKernel, 
    SeparableGaussianKernel, 
    SeparablePowerKernel,
    SeparableMixtureExponentialKernel,
    SeparableSinKernel
)


from utils import plot_spatio_temporal_points, plot_spatial_intensity

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0, metavar='N',
                        help='seed for random number (default: 0)')
    args = parser.parse_args()
    return args

def main(args):
    seed_value = args.seed
    np.random.seed(seed_value)
    np.set_printoptions(suppress=True)
    mu     = [.1, .1]

    ## First
    # kernel_00 = SeparableExponentialKernel(C=0.1, beta_t=0.1, beta_s = 2.0)
    # kernel_11 = SeparableExponentialKernel(C=0.1, beta_t=0.1, beta_s = 2.0)
    # kernel_01 = SeparableExponentialKernel(C=0.05, beta_t=0.5, beta_s = 4.0)
    # kernel_10 = SeparableExponentialKernel(C=0.05, beta_t=0.5, beta_s = 4.0)

    ## Second
    # kernel_00 = SeparableExponentialKernel(C=0.1, beta_t=0.2, beta_s = 2.0)
    # kernel_11 = SeparableExponentialKernel(C=0.1, beta_t=0.2, beta_s = 2.0)
    # kernel_01 = SeparableExponentialKernel(C=0.05, beta_t=0.5, beta_s = 8.0)
    # kernel_10 = SeparableExponentialKernel(C=0.05, beta_t=0.5, beta_s = 8.0)

    ## Third
    # kernel_00 = SeparableExponentialKernel(C=0.1, beta_t=0.1, beta_s = 4.0)
    # kernel_11 = SeparableExponentialKernel(C=0.1, beta_t=0.1, beta_s = 4.0)
    # kernel_01 = SeparableExponentialKernel(C=0.05, beta_t=0.5, beta_s = 8.0)
    # kernel_10 = SeparableExponentialKernel(C=0.05, beta_t=0.5, beta_s = 8.0)

    ### Fourth
    # kernel_00 = SeparableExponentialKernel(C=0.15, beta_t=0.1, beta_s = 4.0)
    # kernel_11 = SeparableExponentialKernel(C=0.15, beta_t=0.1, beta_s = 4.0)
    # kernel_01 = SeparableExponentialKernel(C=-0.02, beta_t=0.1, beta_s = 8.0)
    # kernel_10 = SeparableExponentialKernel(C=-0.02, beta_t=0.1, beta_s = 8.0)

    ### Sixth
    # kernel_00 = SeparableExponentialKernel(C=.15, beta_t=0.5, beta_s = 4.0)
    # kernel_11 = SeparableExponentialKernel(C=.15, beta_t=0.5, beta_s = 4.0)
    # kernel_01 = SeparableExponentialKernel(C=-.03, beta_t=0.5, beta_s = 8.0)
    # kernel_10 = SeparableExponentialKernel(C=-.03, beta_t=0.5, beta_s = 8.0)
    
    ### Seventh
#     kernel_00 = SeparableExponentialKernel(C=-0.03, beta_t=0.5, beta_s = 8.0)
#     kernel_11 = SeparableExponentialKernel(C=-0.03, beta_t=0.5, beta_s = 8.0)
#     kernel_01 = SeparableExponentialKernel(C=0.15, beta_t=0.5, beta_s = 4.0)
#     kernel_10 = SeparableExponentialKernel(C=0.15, beta_t=0.5, beta_s = 4.0)

    ### Nonseparable_short
    # kernel_00 = NonseparableKernel(C=0.15, beta_t=0.5, beta_s = 4.0)
    # kernel_11 = NonseparableKernel(C=0.15, beta_t=0.5, beta_s = 4.0)
    # kernel_01 = NonseparableKernel(C=-0.03, beta_t=0.5, beta_s = 8.0)
    # kernel_10 = NonseparableKernel(C=-0.03, beta_t=0.5, beta_s = 8.0)

    ### Nonseparable_short_2
    # kernel_00 = NonseparableKernel(C=-0.03, beta_t=0.5, beta_s = 8.0)
    # kernel_11 = NonseparableKernel(C=-0.03, beta_t=0.5, beta_s = 8.0)
    # kernel_01 = NonseparableKernel(C=0.15, beta_t=0.5, beta_s = 4.0)
    # kernel_10 = NonseparableKernel(C=0.15, beta_t=0.5, beta_s = 4.0)

    ### Nonseparable_short_3
    kernel_00 = NonseparableKernel(C=.15, beta_t=0.5, beta_s = 2.0)
    kernel_11 = NonseparableKernel(C=.15, beta_t=0.5, beta_s = 2.0)
    kernel_01 = NonseparableKernel(C=-.05, beta_t=0.5, beta_s = 4.0)
    kernel_10 = NonseparableKernel(C=-.05, beta_t=0.5, beta_s = 4.0)

    ### Nonseparable_short_4
    kernel_00 = NonseparableKernel(C=.3, beta_t=0.5, beta_s = 2.0)
    kernel_11 = NonseparableKernel(C=.3, beta_t=0.5, beta_s = 2.0)
    kernel_01 = NonseparableKernel(C=-.15, beta_t=0.5, beta_s = 4.0)
    kernel_10 = NonseparableKernel(C=-.15, beta_t=0.5, beta_s = 4.0)

    ### Nonseparable_mid
    # kernel_00 = NonseparableKernel(C=.15, beta_t=0.3, beta_s = 2.0)
    # kernel_11 = NonseparableKernel(C=.15, beta_t=0.3, beta_s = 2.0)
    # kernel_01 = NonseparableKernel(C=-.05, beta_t=0.3, beta_s = 2.0)
    # kernel_10 = NonseparableKernel(C=-.05, beta_t=0.3, beta_s = 2.0)

    ### Nonseparable_mid_2
    # kernel_00 = NonseparableKernel(C=-0.05, beta_t=0.2, beta_s = 4.0)
    # kernel_11 = NonseparableKernel(C=-0.05, beta_t=0.2, beta_s = 4.0)
    # kernel_01 = NonseparableKernel(C=.15, beta_t=0.2, beta_s = 4.0)
    # kernel_10 = NonseparableKernel(C=.15, beta_t=0.2, beta_s = 4.0)

    ### Nonseparable_mid_3
    kernel_00 = NonseparableKernel(C=0.25, beta_t=0.3, beta_s = 2.0)
    kernel_11 = NonseparableKernel(C=0.25, beta_t=0.3, beta_s = 2.0)
    kernel_01 = NonseparableKernel(C=-.1, beta_t=0.3, beta_s = 2.0)
    kernel_10 = NonseparableKernel(C=-.1, beta_t=0.3, beta_s = 2.0)

    ### Nonseparable_mid_4
    # kernel_00 = NonseparableKernel(C=0.25, beta_t=0.3, beta_s = 2.0)
    # kernel_11 = NonseparableKernel(C=0.25, beta_t=0.3, beta_s = 2.0)
    # kernel_01 = NonseparableKernel(C=.1, beta_t=0.3, beta_s = 2.0)
    # kernel_10 = NonseparableKernel(C=.1, beta_t=0.3, beta_s = 2.0)

    ### Nonseparable_mid_5
    # mu = [.03, .03]
    # kernel_00 = NonseparableKernel(C=0.25, beta_t=0.1, beta_s = 1.0)
    # kernel_11 = NonseparableKernel(C=0.25, beta_t=0.1, beta_s = 1.0)
    # kernel_01 = NonseparableKernel(C=.1, beta_t=0.1, beta_s = 1.0)
    # kernel_10 = NonseparableKernel(C=.1, beta_t=0.1, beta_s = 1.0)

    ### Nonseparable_mid_6
    # kernel_00 = NonseparableKernel(C=0.25, beta_t=0.2, beta_s = 1.0)
    # kernel_11 = NonseparableKernel(C=0.25, beta_t=0.2, beta_s = 1.0)
    # kernel_01 = NonseparableKernel(C=.1, beta_t=0.2, beta_s = 1.0)
    # kernel_10 = NonseparableKernel(C=.1, beta_t=0.2, beta_s = 1.0)

    ### Nonseparable_long
    # kernel_00 = NonseparableKernel(C=-0.15, beta_t=0.1, beta_s = 4.0)
    # kernel_11 = NonseparableKernel(C=-0.15, beta_t=0.1, beta_s = 4.0)
    # kernel_01 = NonseparableKernel(C=-.05, beta_t=0.1, beta_s = 4.0)
    # kernel_10 = NonseparableKernel(C=-.05, beta_t=0.1, beta_s = 4.0)

    ### Nonseparable_long_2
    kernel_00 = NonseparableKernel(C=0.2, beta_t=0.1, beta_s = 2.0)
    kernel_11 = NonseparableKernel(C=0.2, beta_t=0.1, beta_s = 2.0)
    kernel_01 = NonseparableKernel(C=-.1, beta_t=0.1, beta_s = 4.0)
    kernel_10 = NonseparableKernel(C=-.1, beta_t=0.1, beta_s = 4.0)
 
    ### Nonseparable_long_3
    # kernel_00 = NonseparableKernel(C=0.2, beta_t=0.1, beta_s = 2.0)
    # kernel_11 = NonseparableKernel(C=0.2, beta_t=0.1, beta_s = 2.0)
    # kernel_01 = NonseparableKernel(C=.1, beta_t=0.1, beta_s = 4.0)
    # kernel_10 = NonseparableKernel(C=.1, beta_t=0.1, beta_s = 4.0)
    
    ### Separable Zhang et al. (2020)
    kernel_00 = SeparablePowerKernel(C=0.15, alpha_t=.5, beta_t=1.3, beta_s=2.0)
    kernel_01 = SeparableExponentialKernel(C=0.03, beta_t=0.3, beta_s=2.0)
    kernel_10 = SeparableMixtureExponentialKernel(Cs=[0.05, 0.16], beta_ts=[0.2, 0.8], beta_s=2.0)
    kernel_11 = SeparableSinKernel(C=1.0, scale_t=8.0, beta_s=2.0)

    kernel = [kernel_00, kernel_01, kernel_10, kernel_11]
    lam    = BivariateHawkesLam(mu, kernel, maximum=5e+2)
    pp     = BivariateSpatialTemporalPointProcess(lam)
    # homo_points = pp._homogeneous_poisson_sampling(T= [0,100], S=[[-1., 1.], [-1., 1.]])

    t_domain = 100
    points, point_types, sizes = pp.generate(T=[0., t_domain], S=[[-1., 1.], [-1., 1.]], batch_size=20, verbose=True) 
    point_data = [points, point_types]
    dataname = 'bistpp_separable_zhang'
    with open(f'data/{dataname}_{seed_value}.pkl', 'wb') as f:
        pickle.dump(point_data, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)

