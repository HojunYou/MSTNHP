import pickle
import numpy as np
import pandas as pd
import os

dataset_id = 'separable_zhang'
start_seed= 0
n_data = 20
end_seed = start_seed+n_data
bistpp = []

for i in range(start_seed, end_seed):
    with open(f'./data/bistpp_{dataset_id}_{i}.pkl', 'rb') as f:
        bistpp.append(pickle.load(f))

max_len = max([x[0].shape[1] for x in bistpp])
total_len = sum([x[0].shape[0] for x in bistpp])
bistpp_data = [np.zeros((total_len, max_len, 3)), 
               2*np.ones((total_len, max_len), dtype=np.int64)]

for j in range(len(bistpp)):
    for i, bi_data in enumerate(bistpp[j][0]):
        if len(np.where(bi_data[:,0]==0)[0]) ==0 :
            continue    
        else:
            end_idx = np.where(bi_data[:,0]==0)[0][0]
        bistpp[j][1][i,end_idx:] = 2

# for i, bi_data in enumerate(bistpp[1][0]):
#     if len(np.where(bi_data[:,0]==0)[0]) ==0 :
#         continue    
#     else:
#         end_idx = np.where(bi_data[:,0]==0)[0][0]
#     bistpp[1][1][i,end_idx:] = 2
batch_size = total_len//n_data
for i in range(len(bistpp)):
    start_idx = i*batch_size
    end_idx = (i+1)*batch_size
    bistpp_data[0][start_idx:end_idx, :bistpp[i][0].shape[1], :] = bistpp[i][0]
    bistpp_data[1][start_idx:end_idx, :bistpp[i][0].shape[1]] = bistpp[i][1]


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

sizes = [np.where(point[:,0]==0)[0][0] if len(np.where(point[:,0]==0)[0])>0 else point.shape[0] for point in bistpp_data[0]]

point_list = [pd.DataFrame({'time': point[:sizes[i],0], 
                            'x': point[:sizes[i],1],
                            'y': point[:sizes[i],2],
                            'event_type': bistpp_data[1][i][:sizes[i]]}) 
                            for i, point in enumerate(bistpp_data[0])]

train_len = int(.9*len(point_list))
dev_len = int(.05*len(point_list))
train_list = [convert_structure(point, begin_time=0) for point in point_list[:train_len]]
dev_list = [convert_structure(point, begin_time=0) for point in point_list[train_len:train_len+dev_len]]
test_list = [convert_structure(point, begin_time=0) for point in point_list[train_len+dev_len:]]


train_dict = {}
train_dict['train'] = train_list
train_dict['dim_process'] = 2
train_dict['test'] = []
train_dict['dev'] = []
train_dict['args'] = []

dev_dict = {}
dev_dict['train'] = []
dev_dict['dim_process'] = 2
dev_dict['test'] = []
dev_dict['dev'] = dev_list
dev_dict['args'] = []

test_dict = {}
test_dict['train'] = []
test_dict['dim_process'] = 2
test_dict['test'] = test_list
test_dict['dev'] = []
test_dict['args'] = []

save_path = f'./data/bistpp_{dataset_id}/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

with open(os.path.join(save_path, 'points.pkl'), 'wb') as f:
    pickle.dump(bistpp_data, f)

# with open(os.path.join(save_path, 'points.pkl'), 'wb') as f:
#     pickle.dump(points, f)

with open(os.path.join(save_path, 'train.pkl'), 'wb') as f:
    pickle.dump(train_dict, f)

with open(os.path.join(save_path, 'dev.pkl'), 'wb') as f:
    pickle.dump(dev_dict, f)

with open(os.path.join(save_path, 'test.pkl'), 'wb') as f:
    pickle.dump(test_dict, f)

