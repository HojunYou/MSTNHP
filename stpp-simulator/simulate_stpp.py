import os, sys, pickle
sys.path.append(os.path.dirname(os.path.realpath(__name__)))

import numpy as np
import pandas as pd

from stppg import StdDiffusionKernel, GaussianDiffusionKernel, GaussianMixtureDiffusionKernel, HawkesLam, SpatialTemporalPointProcess, SeparableExponentialKernel
from utils import plot_spatio_temporal_points, plot_spatial_intensity, lebesgue_measure

np.random.seed(0)
np.set_printoptions(suppress=True)

# parameters initialization
mu     = .1
# kernel = StdDiffusionKernel(C=.05, beta=.1, sigma_x=.1, sigma_y=.1) # kernel type with different C and beta.
kernel = StdDiffusionKernel(C=.15, beta=.3, sigma_x=.3, sigma_y=.3) # kernel type with different C and beta.
# kernel = GaussianMixtureDiffusionKernel(n_comp=2, w=[0.5, 0.5], mu_x = [0, 0], mu_y=[0, 0], sigma_x = [0.1, 0.1], sigma_y = [0.1, 0.1], 
#                                         rho=[0.5, -0.5], beta=.1, C=.05)
# kernel = SeparableExponentialKernel(C=0.05, beta_t=0.1, beta_s = 2.0)
# kernel = SeparableExponentialKernel(C=0.4, beta_t=0.5, beta_s = 4.0)
lam    = HawkesLam(mu, kernel, maximum=2e+3)
pp     = SpatialTemporalPointProcess(lam)

t_domain = 100
points, sizes = pp.generate(T=[0., t_domain], S=[[-1., 1.], [-1., 1.]], batch_size=1, verbose=True) 
# points = np.load('first_event.npy')
# plot_spatial_intensity(lam, points[0], S=[[0., t_domain], [-1., 1.], [-1., 1.]], t_slots=100, grid_size=50, interval=400)


def convert_structure(df, begin_time, scale=1):
    df_list = []
    for i in range(len(df)):
        data_dict={}
        data_dict['time_since_start'] = (df['time'][i] - begin_time)/scale
        data_dict['location_from_origin'] = [df['x'][i], df['y'][i]]
        if i==0:
            data_dict['time_since_last_event'] = (df['time'][i] - begin_time)/scale
            data_dict['location_from_last_event'] = [df['x'][i], df['y'][i]]
        else:
            data_dict['time_since_last_event'] = df['time'][i] - df['time'][i-1]
            data_dict['location_from_last_event'] = [df['x'][i] - df['x'][i-1], df['y'][i] - df['y'][i-1]]
        data_dict['type_event'] = df['event_type'][i]
        df_list.append(data_dict)
    return df_list

point_list = [pd.DataFrame({'time': point[:sizes[i],0], 
                            'x': point[:sizes[i],1],
                            'y': point[:sizes[i],2],
                            'event_type': np.zeros(sizes[i], dtype=int)}) 
                            for i, point in enumerate(points)]
train_list = [convert_structure(point, begin_time=0) for point in point_list[:900]]
dev_list = [convert_structure(point, begin_time=0) for point in point_list[900:950]]
test_list = [convert_structure(point, begin_time=0) for point in point_list[950:]]

train_dict = {}
train_dict['train'] = train_list
train_dict['dim_process'] = 1
train_dict['test'] = []
train_dict['dev'] = []
train_dict['args'] = []

dev_dict = {}
dev_dict['train'] = []
dev_dict['dim_process'] = 1
dev_dict['test'] = []
dev_dict['dev'] = dev_list
dev_dict['args'] = []

test_dict = {}
test_dict['train'] = []
test_dict['dim_process'] = 1
test_dict['test'] = test_list
test_dict['dev'] = []
test_dict['args'] = []

save_path = '../easytpp/examples/data/stpp_exp_2/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, 'points.pkl'), 'wb') as f:
    pickle.dump(points, f)

with open(os.path.join(save_path, 'train.pkl'), 'wb') as f:
    pickle.dump(train_dict, f)

with open(os.path.join(save_path, 'dev.pkl'), 'wb') as f:
    pickle.dump(dev_dict, f)

with open(os.path.join(save_path, 'test.pkl'), 'wb') as f:
    pickle.dump(test_dict, f)


# save_path = '../data/stpp_exp_2/'
# with open(os.path.join(save_path, 'points.pkl'), 'rb') as f:
#     points = pickle.load(f)

points = points[0]
points = points[points[:,0]>0]
seq_t, seq_s = points[:, 0], points[:, 1:]
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
    return lam.value(t, sub_seq_t, s, sub_seq_s)

t_sample, s_sample = st_sample[:,0], st_sample[:,1:]
sample_intensity = [intensity_function(t, s, lam) for (t, s) in zip(t_sample, s_sample)]
sample_intensity = np.array(sample_intensity)
sample_intensity.min(), sample_intensity.max()
simulated_points = sample_intensity.sum()*100/1000*(2/50)**2
(simulated_points, points.shape[0])

import matplotlib.pyplot as plt
import pandas as pd

dataname = 'stpp_short'
data_idx = 0
ns = n_grid
ns2 = ns**2
df = pd.DataFrame({'t': st_sample[:,0], 'x': st_sample[:,1], 'y': st_sample[:,2], 'intensity': sample_intensity})
df = df[df['t']> 2]
df_np = df.to_numpy()
savepath = f'../../Meeting/2024/240424/{dataname}/True'
os.makedirs(savepath, exist_ok=True)
max_frame = len(df_np)//ns2
current_time = seq_t[0]
next_time = seq_t[1]
current_loc = seq_s[0]
for i in range(0, max_frame//3):
    plt.pcolor(df_np[i*ns2:(i+1)*ns2,1].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2,2].reshape(ns,-1).round(4), df_np[i*ns2:(i+1)*ns2, -1].reshape(ns,-1).round(4), cmap=plt.cm.seismic, vmin=0.0, vmax = .8)
    plt.colorbar()
    if df_np[i*ns2,0] > next_time:
        next_time = seq_t[np.where(seq_t>current_time)[0][1]]
        current_idx = np.where(seq_t==next_time)[0][0]
        current_time = next_time
        current_loc = seq_s[current_idx]
    plt.scatter(current_loc[1], current_loc[0], c='black', s=20)
    plt.title(f'time: {df_np[i*ns2, 0]:.2f}')
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


