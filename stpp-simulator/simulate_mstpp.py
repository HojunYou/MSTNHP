
## Bivariate Hawkes
import os, sys, pickle
# sys.path.append(os.path.dirname(os.path.realpath(__name__)))

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

np.random.seed(0)
np.set_printoptions(suppress=True)

# parameters initialization
mu     = [.1, .1]
# kernel = StdDiffusionKernel(C=.05, beta=.1, sigma_x=.1, sigma_y=.1) # kernel type with different C and beta.
kernel_00 = SeparablePowerKernel(C=0.15, alpha_t=.5, beta_t=1.3, beta_s=2.0)
kernel_01 = SeparableExponentialKernel(C=0.03, beta_t=0.3, beta_s=2.0)
kernel_10 = SeparableMixtureExponentialKernel(Cs=[0.05, 0.16], beta_ts=[0.2, 0.8], beta_s=2.0)
kernel_11 = SeparableSinKernel(C=1.0, scale_t=8.0, beta_s=2.0)

# 2*math.pi*0.03/(0.1*4.0**2), 2*math.pi*0.03/(0.5*6.0**2)

kernel = [kernel_00, kernel_01, kernel_10, kernel_11]
# kernel = [kernel[:2], kernel[2:]]
# kernel = GaussianMixtureDiffusionKernel(n_comp=2, w=[0.5, 0.5], mu_x = [0, 0], mu_y=[0, 0], sigma_x = [0.1, 0.1], sigma_y = [0.1, 0.1], 
#                                         rho=[0.5, -0.5], beta=.1, C=.05)
# kernel = SeparableExponentialKernel(C=0.05, beta_t=0.1, beta_s = 2.0)
# kernel = SeparableExponentialKernel(C=0.4, beta_t=0.5, beta_s = 4.0)
lam    = BivariateHawkesLam(mu, kernel, maximum=5e+2)
pp     = BivariateSpatialTemporalPointProcess(lam)

### mle sixth with 900 sequences
# mu_mle = [0.10699480, 0.10544753]
# kernel_mle_00 = SeparableExponentialKernel(C=0.14932825, beta_t=0.58392197, beta_s = 6.82163218)
# kernel_mle_10 = SeparableExponentialKernel(C=-0.00884988, beta_t=0.58862926, beta_s = 2.72532088)
# kernel_mle_01 = SeparableExponentialKernel(C=-0.03067325, beta_t=0.53490585, beta_s = 4.75000992)
# kernel_mle_11 = SeparableExponentialKernel(C=0.13430747, beta_t=0.40686071, beta_s = 7.53366222)

# kernel_mle = [kernel_mle_00, kernel_mle_01, kernel_mle_10, kernel_mle_11]
# lam_mle    = BivariateHawkesLam(mu_mle, kernel_mle, maximum=5e+2)


t_domain = 100
points, point_types, sizes = pp.generate(T=[0., t_domain], S=[[-1., 1.], [-1., 1.]], batch_size=1000, verbose=True) 







point_data = [points, point_types]
with open('bistpp.pkl', 'wb') as f:
    pickle.dump(point_data, f)

points = points[0]
points = points[points[:,0]>0]
point_types = point_types[0]
# seq_t = [points[point_types==0, 0], points[point_types==1, 0]]
# seq_s = [points[point_types==0, 1:], points[point_types==1, 1:]]
seq_t = points[:,0]
seq_s = points[:,1:]

t_domain = 100
time_domain = [0, t_domain]
num_sample_t = 1000
t_span = np.linspace(time_domain[0]+(time_domain[1]-time_domain[0])/num_sample_t, time_domain[1], num_sample_t)
n_grid = 50
x_span = np.linspace(-1+2/(2*n_grid), 1-2/(2*n_grid), n_grid)
y_span = np.linspace(-1+2/(2*n_grid), 1-2/(2*n_grid), n_grid)

import torch
s_span = torch.cartesian_prod(torch.tensor(x_span), torch.tensor(y_span))
t_span = t_span[t_span>points[0,0]]
n_s = s_span.shape[0]
n_t = len(t_span)
t_points = torch.tensor(t_span.repeat(n_s), dtype=torch.float)
s_points = s_span.repeat(n_t, 1)
st_sample = torch.cat([t_points[:,None], s_points], dim=1).numpy()

def intensity_function(t, s, lam):
    sub_seq_t = seq_t[seq_t < t]
    sub_seq_s = seq_s[:len(sub_seq_t)]
    sub_seq_types = point_types[:len(sub_seq_t)]
    return [lam.value(t, sub_seq_t, s, sub_seq_s, 0, sub_seq_types), lam.value(t, sub_seq_t, s, sub_seq_s, 1, sub_seq_types)]

t_sample, s_sample = st_sample[:,0], st_sample[:,1:]
sample_intensity = [intensity_function(t, s, lam) for (t, s) in zip(t_sample, s_sample)]
sample_intensity = np.array(sample_intensity)
sample_intensity.min(), sample_intensity.max()
simulated_points = sample_intensity.sum()*t_domain/num_sample_t*(2/n_grid)**2
(simulated_points, points.shape[0])

import matplotlib.pyplot as plt
import pandas as pd

dataname = 'bistpp'
data_idx = 0
ns = n_grid
ns2 = ns**2
df = pd.DataFrame({'t': st_sample[:,0], 'x': st_sample[:,1], 'y': st_sample[:,2], 'intensity0': sample_intensity[:,0], 'intensity1': sample_intensity[:,1]})
# df = df[df['t']> 2]
df_np = df.to_numpy()
savepath = f'../../Meeting/2024/240508/{dataname}/True'
os.makedirs(savepath, exist_ok=True)
max_frame = len(df_np)//ns2
current_time = [seq_t[point_types==0][0], seq_t[point_types==1][0]]
next_time = [seq_t[point_types==0][1], seq_t[point_types==1][1]]
current_loc = [seq_s[point_types==0][0], seq_s[point_types==1][0]]
for i in range(0, max_frame//4):
    fig, ax = plt.subplots(nrows = 1, ncols=2, figsize=(10, 4))
    p0=ax[0].pcolor(df_np[i*ns2:(i+1)*ns2,1].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2,2].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2, -2].reshape(ns,-1).round(4), cmap=plt.cm.seismic, vmin=0.0, vmax = .4)
    p1=ax[1].pcolor(df_np[i*ns2:(i+1)*ns2,1].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2,2].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2, -1].reshape(ns,-1).round(4), cmap=plt.cm.seismic, vmin=0.0, vmax = .4)
    plt.colorbar(p1, ax = [ax[0], ax[1]], location='right', format='%.2f')
    # if df_np[i*ns2,0] > next_time[0]:
    #     next_time[0] = seq_t[np.where(seq_t[point_types==0]>current_time[0])[0][1]]
    #     current_idx = np.where(seq_t==next_time)[0][0]
    #     current_time = next_time
    #     current_loc = seq_s[current_idx]
    # elif df_np[i*ns2,0] > next_time[1]:
    # plt.scatter(current_loc[1], current_loc[0], c='black', s=20)
    plt.suptitle(f'time: {df_np[i*ns2, 0]:.2f}', fontsize=12)
    # plt.show()
    if i<10:
        savename = f'000{i}'
    elif i<100:
        savename = f'00{i}'
    elif i<1000:
        savename = f'0{i}'
    else:
        savename = f'{i}'
    plt.savefig(f'{savepath}/{data_idx}_{savename}.jpg', bbox_inches='tight')
    plt.close()


# for i in range(0, max_frame//3):
#     plt.pcolor(df_np[i*ns2:(i+1)*ns2,1].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2,2].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2, -1].reshape(ns,-1).round(4), cmap=plt.cm.seismic, vmin=0.0, vmax = .8)
#     plt.colorbar()
#     if df_np[i*ns2,0] > next_time:
#         next_time = seq_t[np.where(seq_t>current_time)[0][1]]
#         current_idx = np.where(seq_t==next_time)[0][0]
#         current_time = next_time
#         current_loc = seq_s[current_idx]
#     plt.scatter(current_loc[1], current_loc[0], c='black', s=20)
#     plt.title(f'time: {df_np[i*ns2, 0]:.2f}')
#     # plt.show()
#     if i<10:
#         savename = f'000{i}'
#     elif i<100:
#         savename = f'00{i}'
#     elif i<1000:
#         savename = f'0{i}'
#     else:
#         savename = f'{i}'
#     plt.savefig(f'{savepath}/{data_idx}_{savename}.jpg', bbox_inches='tight')
#     plt.close()




