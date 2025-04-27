
import argparse
import pandas as pd
import numpy as np
import pickle
import torch
from datetime import datetime
import os, sys

sys.path.append(sys.path.append(os.getcwd()))

from THP.preprocess.Dataset import EventData, collate_fn

def datetime_to_julian(dt, origin):
    origin_date = datetime.strptime(origin, '%Y-%m-%d')
    days = (dt - origin_date).days
    return days

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

def load_data(name, dict_name):
    with open(name, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        num_types = data['dim_process']
        data = data[dict_name]
        return data, int(num_types)

print('[Info] Loading train data...')
data_name = 'retweet'
train_data, num_types = load_data(os.path.join('./Hawkes_data/', 'data_'+data_name, 'train.pkl'), 'train')
num_types
data_lens = [len(inst) for inst in train_data]
data_lens = np.array(data_lens)
train_data[0]

## Nigerian dataset

data_15 = [train_data[i] for i, data_len in enumerate(data_lens) if data_len==15]
np.unique(data_lens, return_counts=True)

nigeria = pd.read_feather('./Hawkes_data/nigeria.feather')
nigeria['event_type'] = nigeria['event_type']-1
#Convert nigeria time (in julian) column to datetime and segment the dataframe by 3 months from 2011-01-01 to 2018-01-01
# nigeria['date'] = pd.to_datetime(nigeria['time'], unit='D', origin=pd.Timestamp('1970-01-02'))
nigeria = nigeria[(nigeria['date'] >= '2011-01-01') & (nigeria['date'] < '2018-01-01')]
nigeria = nigeria.reset_index(drop=True)
nigeria['date'] = pd.to_datetime(nigeria['date'], format='%Y-%m-%d')
nigeria['event_type'] = nigeria['event_type'].astype('int')

quarter_date = pd.date_range(start='2010-12-31', end='2018-01-31', freq='3M')
quarter_julian = (datetime_to_julian(quarter_date, '1970-01-01')+1).astype('float')

nigeria_list = []
for i in range(len(quarter_date)-1):
    nigeria_list.append(nigeria[(nigeria['date'] > quarter_date[i]) & (nigeria['date'] <= quarter_date[i+1])].reset_index(drop=True))

nigeria_data = [convert_structure(nigeria_list[i], quarter_julian[i]) for i in range(len(nigeria_list)-1)]

nigeria_dict = {}
nigeria_dict['train'] = nigeria_data
nigeria_dict['dim_process'] = 2
nigeria_dict['test'] = []
nigeria_dict['dev'] = []
nigeria_dict['test1'] = []
nigeria_dict['args'] = []

nigeria_test_data = [convert_structure(nigeria_list[-1], quarter_julian[-2])]
nigeria_test_dict={}
nigeria_test_dict['train'] = []
nigeria_test_dict['dim_process'] = 2
nigeria_test_dict['test'] = nigeria_test_data
nigeria_test_dict['dev'] = []

nigeria_dev_data = [convert_structure(nigeria_list[-1], quarter_julian[-2])]
nigeria_dev_dict={}
nigeria_dev_dict['train'] = []
nigeria_dev_dict['dim_process'] = 2
nigeria_dev_dict['test'] = []
nigeria_dev_dict['dev'] = nigeria_dev_data

with open(os.path.join('./Hawkes_data/', 'data_nigeria', 'train.pkl'), 'wb') as f:
    pickle.dump(nigeria_dict, f)

with open(os.path.join('./Hawkes_data/', 'data_nigeria', 'test.pkl'), 'wb') as f:
    pickle.dump(nigeria_test_dict, f)

with open(os.path.join('./Hawkes_data/', 'data_nigeria', 'dev.pkl'), 'wb') as f:
    pickle.dump(nigeria_dev_dict, f)


nigeria['time'] = np.round(100*(nigeria['time']-quarter_julian[0])/(nigeria.time.max()-quarter_julian[0]), 4)

num_samples = 1000
# retention_p = .5
retention_p_1 = .1
retention_p_2 = .5
np.random.seed(1)
# retention = np.random.binomial(1, retention_p, size=(num_samples, len(nigeria)))
retention_1 = np.random.binomial(1, retention_p_1, size=(num_samples, sum(nigeria.event_type==0)))
retention_2 = np.random.binomial(1, retention_p_2, size=(num_samples, sum(nigeria.event_type==1)))

thinned_nigeria = []
for i in range(num_samples):
    thinned_sample = nigeria[nigeria.event_type==0].iloc[retention_1[i]==1].reset_index(drop=True)
    thinned_sample = pd.concat([thinned_sample, nigeria[nigeria.event_type==1].iloc[retention_2[i]==1].reset_index(drop=True)], axis=0)
    thinned_sample = thinned_sample.sort_values(by=['time']).reset_index(drop=True)
    # thinned_sample = nigeria[retention[i]==1].reset_index(drop=True)
    thinned_nigeria.append(thinned_sample)
    

thinned_nigeria_data = [convert_structure(thinned_nigeria[i], 0) for i in range(len(thinned_nigeria))]
thinned_nigeria_dict = {}
thinned_nigeria_dict['train'] = thinned_nigeria_data
thinned_nigeria_dict['dim_process'] = 2
thinned_nigeria_dict['test'] = []
thinned_nigeria_dict['dev'] = []
# thinned_nigeria_dict['test1'] = []
# thinned_nigeria_dict['args'] = []

# with open(os.path.join('./Hawkes_data/', 'data_thinned_nigeria/', str(int(10*retention_p)), 'train.pkl'), 'wb') as f:
with open(os.path.join('./Hawkes_data/', 'data_thinned_nigeria/', str(int(10*retention_p_1))+'_'+str(int(10*retention_p_2)), 'train.pkl'), 'wb') as f:
    pickle.dump(thinned_nigeria_dict, f)

np.random.seed(2)
test_num_samples = 100
dev_num_samples = 100
# test_retention = np.random.binomial(1, retention_p, size=(num_samples, len(nigeria)))
# dev_retention = np.random.binomial(1, retention_p, size=(num_samples, len(nigeria)))
test_retention_1 = np.random.binomial(1, retention_p_1, size=(test_num_samples, sum(nigeria.event_type==0)))
test_retention_2 = np.random.binomial(1, retention_p_2, size=(test_num_samples, sum(nigeria.event_type==1)))
dev_retention_1 = np.random.binomial(1, retention_p_1, size=(dev_num_samples, sum(nigeria.event_type==0)))
dev_retention_2 = np.random.binomial(1, retention_p_2, size=(dev_num_samples, sum(nigeria.event_type==1)))


thinned_nigeria_test = []
for i in range(test_num_samples):
    # thinned_sample = nigeria[test_retention[i]==1].reset_index(drop=True)
    thinned_sample = nigeria[nigeria.event_type==0].iloc[test_retention_1[i]==1].reset_index(drop=True)
    thinned_sample = pd.concat([thinned_sample, nigeria[nigeria.event_type==1].iloc[test_retention_2[i]==1].reset_index(drop=True)], axis=0)
    thinned_sample = thinned_sample.sort_values(by=['time']).reset_index(drop=True)
    thinned_nigeria_test.append(thinned_sample)


thinned_nigeria_test_data = [convert_structure(thinned_nigeria_test[i], 0) for i in range(len(thinned_nigeria_test))]
thinned_nigeria_test_dict = {}
thinned_nigeria_test_dict['train'] = []
thinned_nigeria_test_dict['dim_process'] = 2
thinned_nigeria_test_dict['test'] = thinned_nigeria_test_data
thinned_nigeria_test_dict['dev'] = []

# with open(os.path.join('./Hawkes_data/', 'data_thinned_nigeria/', str(int(10*retention_p)), 'test.pkl'), 'wb') as f:
with open(os.path.join('./Hawkes_data/', 'data_thinned_nigeria/', str(int(10*retention_p_1))+'_'+str(int(10*retention_p_2)), 'test.pkl'), 'wb') as f:
    pickle.dump(thinned_nigeria_test_dict, f)

thinned_nigeria_dev = []
for i in range(dev_num_samples):
    # thinned_sample = nigeria[dev_retention[i]==1].reset_index(drop=True)
    thinned_sample = nigeria[nigeria.event_type==0].iloc[dev_retention_1[i]==1].reset_index(drop=True)
    thinned_sample = pd.concat([thinned_sample, nigeria[nigeria.event_type==1].iloc[dev_retention_2[i]==1].reset_index(drop=True)], axis=0)
    thinned_sample = thinned_sample.sort_values(by=['time']).reset_index(drop=True)
    thinned_nigeria_dev.append(thinned_sample)

thinned_nigeria_dev_data = [convert_structure(thinned_nigeria_dev[i], 0) for i in range(len(thinned_nigeria_dev))]
thinned_nigeria_dev_dict = {}
thinned_nigeria_dev_dict['train'] = []
thinned_nigeria_dev_dict['dim_process'] = 2
thinned_nigeria_dev_dict['test'] = []
thinned_nigeria_dev_dict['dev'] = thinned_nigeria_dev_data

# with open(os.path.join('./Hawkes_data/', 'data_thinned_nigeria/', str(int(10*retention_p)), 'dev.pkl'), 'wb') as f:
with open(os.path.join('./Hawkes_data/', 'data_thinned_nigeria/', str(int(10*retention_p_1))+'_'+str(int(10*retention_p_2)), 'dev.pkl'), 'wb') as f:
    pickle.dump(thinned_nigeria_dev_dict, f)

# train_data = EventData(nigeria_data)
# trainloader = torch.utils.data.DataLoader(
#         train_data,
#         num_workers=2,
#         batch_size=4,
#         collate_fn=collate_fn,
#         shuffle=True,
#         drop_last=False
#     )
# batch = next(iter(trainloader))


### Transform batched data to a raw data.

from datetime import datetime
import pandas as pd

dateparse = lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
d=pd.read_csv('./Hawkes_data/911.csv',
    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],
    dtype={'lat':str,'lng':str,'desc':str,'zip':str,
                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 
     parse_dates=['timeStamp'],date_parser=dateparse)

d.timeStamp = pd.DatetimeIndex(d.timeStamp)
d=d[(d.timeStamp >= "2016-01-01 00:00:00")]
d.timeStamp.describe()
d["title"].value_counts()
d['type'] = d["title"].apply(lambda x: x.split(':')[0])
d.type.describe()
d.type.value_counts()
d.reset_index(drop=True, inplace=True)


start_2016 = '2016-01-01'
end_2016 = '2017-01-01'
week_2016 = pd.date_range(start=start_2016, end=end_2016, freq='W-MON')
d_2016 = [d[(d['timeStamp'] > week_2016[i]) & (d['timeStamp'] <= week_2016[i+1])].reset_index(drop=True) for i in range(len(week_2016)-1)]


for j in range(int((len(week_2016)-1)/4)):
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(12, 9))
    for i in range(4*j, 4*(j+1)):
        fig_i = i%4
        weekly_df = d_2016[i].copy()
        ax[fig_i].scatter(weekly_df.timeStamp, weekly_df.type, s=1)
        ax[fig_i].set_xlabel('Time')
        ax[fig_i].set_ylabel('Event type')
        ax[fig_i].set_title(str(week_2016[i])[:10])
    plt.savefig('../../../Meeting/2023/231009/911_2016_'+str(j)+'.jpg')
    plt.close()

data_name = 'conttime'
train_data, num_types = load_data(os.path.join('./Hawkes_data/', 'data_'+data_name, 'train.pkl'), 'train')
num_types
data_lens = [len(inst) for inst in train_data]
data_lens = np.array(data_lens)




### Convert to a raw data
def divide_df_by_event_type(df):
    df_event_list = df.event_type.unique().tolist()
    df_dict = {}
    for i in range(len(df_event_list)):
        df_dict[df_event_list[i]] = df[df.event_type==df_event_list[i]].copy().reset_index(drop=True)
        df_dict[df_event_list[i]]['event_counts'] = df_dict[df_event_list[i]].index+1

    return df_dict
    

import matplotlib.pyplot as plt
train_data_sequence = [pd.DataFrame([[train_event['time_since_start'], train_event['type_event']] for train_event in train_chunk], columns=['time', 'event_type']) for train_chunk in train_data]
train_data_dicts = [divide_df_by_event_type(train_sequence) for train_sequence in train_data_sequence]
last_time_events = [train_seq.iloc[-1].time for train_seq in train_data_sequence]
last_time_events = np.array(last_time_events)

color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, ax = plt.subplots(3, 3, figsize=(12, 9))
for train_i in range(18, 27):
    fig_i = train_i%9
    group_i = train_i//9
    row_i = fig_i//3
    col_i = fig_i%3
    event_types = train_data_sequence[train_i]['event_type']
    colors = [color_palette[event_type] for event_type in event_types]
    for event_type in np.unique(event_types):
        ax[row_i,col_i].plot(train_data_dicts[train_i][event_type]['time']/10000, train_data_dicts[train_i][event_type]['event_counts'], color=color_palette[event_type])
    ax[row_i, col_i].legend([str(event_type) for event_type in np.unique(event_types)], fontsize=8, loc='upper right')
    # ax[row_i,col_i].scatter(train_data_dicts[train_i][], train_data_sequence[train_i]['event_type'], color=colors, s=10)
    # ax[row_i,col_i].axline((0,0), slope=1, color='red')
    # ax[row_i,col_i].set_title(str(quarter_date[train_i])[:10])
    
    if col_i==0:
        ax[row_i,col_i].set_ylabel('Event counts')
    if row_i==2:
        ax[row_i,col_i].set_xlabel('Event time')
# plt.show()
# plt.close()
# Save retweet plot for current group to meeting directory
plt.savefig('../../../Meeting/2023/231009/Retweet_'+str(group_i+1)+'.jpg')
plt.close()





#### GTD dataset preprocessing
import pandas as pd
import math
from datetime import datetime, date
import numpy as np
import os, sys
sys.path.append(sys.path.append(os.getcwd()))

# Load and preprocess GTD data: merge CSVs, filter for Pakistan, standardize group labels, and prepare events
gtd = pd.read_csv('./gtd/globalterrorismdb_0522dist.csv', encoding='ISO-8859-1')  ### 1970-2020
gtd_recent = pd.read_csv('./gtd/globalterrorismdb_2021Jan-June_1222dist.csv', encoding='ISO-8859-1') ### 2021.01.-2021.06.
gtd = gtd[['iyear', 'imonth', 'iday', 'latitude', 'longitude', 'country_txt', 'gname']]
gtd_recent = gtd_recent[['iyear', 'imonth', 'iday', 'latitude', 'longitude', 'country_txt', 'gname']]
gtd = pd.concat([gtd, gtd_recent], axis=0)
gtd.reset_index(drop=True, inplace=True)

country_counts = gtd.country_txt.value_counts()
country_counts = country_counts[country_counts>1000] ## Colombia, Phillipines

### Select Pakistan
target_country = 'Pakistan'
gtd_small = gtd[gtd.country_txt==target_country]
gtd_small = gtd_small.reset_index(drop=True)
gtd_small.gname.value_counts()
### Merge all Baloch groups into 'Baloch'
Baloch_Group = ['Baloch Republican Army (BRA)', 'Baloch Liberation Front (BLF)', 'Baloch Liberation Army (BLA)']
gtd_small.loc[gtd_small.gname.isin(Baloch_Group),['gname']] = 'Baloch'

### Select top 2 groups
top_2 = gtd_small.gname.value_counts()[:3].index
top_2 = top_2.tolist()
### If 'Unknown' is in the top 2, remove it
if 'Unknown' in top_2:
    unknown_index = top_2.index('Unknown')
    top_2.pop(unknown_index)
else:
    top_2 = top_2[:2]

gtd_small = gtd_small[(gtd_small.gname==top_2[0]) | (gtd_small.gname==top_2[1])]
gtd_small.groupby(['iyear', 'gname']).size()

# gtd_small = gtd_small[gtd_small.gname == 'Tehrik-i-Taliban Pakistan (TTP)']

gtd_small = gtd_small[(gtd_small.iyear>=2008) & (gtd_small.iyear<2021)]
# gtd_small.groupby(['iyear', 'gname']).size().to_frame().reset_index().pivot(index='iyear', columns='gname', values=0).fillna(0).to_csv('gtd/Pakistan_gname.csv')
gtd_small.reset_index(drop=True, inplace=True)
gtd_pk = gtd_small.copy()
gtd_pk = gtd_pk[~gtd_pk.latitude.isna()]
assert gtd_pk.isna().sum().sum()==0, "Missing values in the dataset"

# min_long, max_long = math.floor(gtd_pk.longitude.min()), math.ceil(gtd_pk.longitude.max())
# min_lat, max_lat = math.floor(gtd_pk.latitude.min()), math.ceil(gtd_pk.latitude.max())

import matplotlib.pyplot as plt
import geopandas
from shapely.geometry import Point, Polygon

###Plot pakistan map before gtd_pk
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
pakistan_shp = world[(world.name=='Pakistan')]
pakistan_polygon = pakistan_shp.geometry.values[0]
pakistan_coords = list(pakistan_polygon.exterior.coords)
pakistan_longitudes = [x[0] for x in pakistan_coords]
pakistan_latitudes = [x[1] for x in pakistan_coords]

min_long, max_long = math.floor(min(pakistan_longitudes)), math.ceil(max(pakistan_longitudes))
min_lat, max_lat = math.floor(min(pakistan_latitudes)), math.ceil(max(pakistan_latitudes))

scaled_pakistan_coords = [(2*(x-min_long)/(max_long-min_long)-1, 2*(y-min_lat)/(max_lat-min_lat)-1) for x, y in pakistan_coords]
scaled_pakistan_polygon = Polygon(scaled_pakistan_coords)

# with open('./data/gtd_pakistan/pakistan_polygon.pkl', 'wb') as f:
#     pickle.dump(scaled_pakistan_polygon, f)

geopandas.GeoSeries([scaled_pakistan_polygon]).plot()
test_point = Point(0, 0)
test_point.within(scaled_pakistan_polygon)
# groups = gtd_pk.groupby('gname')
fig, ax = plt.subplots(figsize=(10, 10))
scaled_pakistan_polygon.plot(ax=ax, color='white', edgecolor='black')
plt.show()
plt.close()

## Convert latitude and longitude to [-1, 1]
gtd_pk['x'] = 2*(gtd_pk.longitude-min_long)/(max_long-min_long)-1
gtd_pk['y'] = 2*(gtd_pk.latitude-min_lat)/(max_lat-min_lat)-1
gtd_pk.loc[gtd_pk.iday==0, 'iday']=1
gtd_pk[gtd_pk.iday==1]
gtd_pk['time'] = gtd_pk.apply(lambda x: datetime(x.iyear, x.imonth, x.iday), axis=1)
gtd_pk.drop_duplicates(subset=['time'], keep='first', inplace=True)
gtd_pk.reset_index(drop=True, inplace=True)
gtd_pk['event_type'] = gtd_pk.gname.apply(lambda x: top_2.index(x))
gtd_pk = gtd_pk[['time', 'x', 'y', 'event_type']]

gtd_pk = gtd_pk.sort_values('time').reset_index(drop=True)
# gtd_pk_train = gtd_pk[gtd_pk.time.dt.year<2020]
# gtd_pk_train = gtd_pk.copy()

## Convert time scale to [0,100]
year_list = gtd_pk.time.dt.year.unique()
gtd_list = [gtd_pk[gtd_pk.time.dt.year==year].reset_index(drop=True) for year in year_list]
## Convert time scale to [0,100]
begin_time = [datetime(year-1, 12, 31) for year in year_list]
end_time = [datetime(year+1, 1, 1) for year in year_list]
gtd_time_list = [gtd_df.time.apply(lambda x: 100*(x-begin_time[i]).days/(end_time[i]-begin_time[i]).days) for i, gtd_df in enumerate(gtd_list)]
begin_time_train = datetime(2007, 12, 31)
end_time_train = datetime(2021, 1, 1) # datetime(2020, 1, 1)
# gtd_pk_train['time'] = gtd_pk_train.time.apply(lambda x: 100*(x-begin_time_train).days/(end_time_train-begin_time_train).days)
gtd_list = [gtd_df.assign(time=gtd_time_list[i]) for i, gtd_df in enumerate(gtd_list)]

# Helper: convert event DataFrame into list of dicts for STPP model consumption
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

train_list = [convert_structure(point, begin_time=0) for point in gtd_list]
test_list = [convert_structure(gtd_list[-1], begin_time=0)]

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
dev_dict['dev'] = test_list
dev_dict['args'] = []

test_dict = {}
test_dict['train'] = []
test_dict['dim_process'] = 2
test_dict['test'] = test_list
test_dict['dev'] = []
test_dict['args'] = []

import os, pickle

# Ensure output directory exists for processed GTD data
save_path = './data/gtd_pakistan/'
os.makedirs(save_path, exist_ok=True)

# Save GTD data to pickle files
with open(os.path.join(save_path, 'points.pkl'), 'wb') as f:
    pickle.dump(gtd_list, f)
# gtd_pk.to_feather('./data/gtd_pakistan/gtd_pakistan.feather')

with open(os.path.join(save_path, 'train.pkl'), 'wb') as f:
    pickle.dump(train_dict, f)

with open(os.path.join(save_path, 'dev.pkl'), 'wb') as f:
    pickle.dump(dev_dict, f)

with open(os.path.join(save_path, 'test.pkl'), 'wb') as f:
    pickle.dump(test_dict, f)

import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime

# Visualize spatial distribution of training events by type
plt.scatter(gtd_pk_train.x, gtd_pk_train.y, c=gtd_pk_train.event_type); plt.show()
target_year = 2010

figure_path = '../Meeting/2024/240620/gtd_pakistan/'+str(target_year)
os.makedirs(figure_path, exist_ok=True)
cmap = plt.get_cmap('coolwarm')

date_list = []
start_date = datetime(target_year, 1, 1)
end_date = start_date+timedelta(days=365)
for i in range(365):
    date_list.append(start_date+timedelta(days=i))

for date_i, current_date in enumerate(date_list):
    history_startdate = current_date-timedelta(days=10)
    history_i = gtd_pk[(gtd_pk['time']>=history_startdate) & (gtd_pk['time']<=current_date)]
    for row_i in range(history_i.shape[0]):
        history_date = history_i.iloc[row_i]['time']
        history_alpha = 1-(current_date-history_date).days/10
        color_type = 'g' if history_i.iloc[row_i]['event_type']==0 else 'r'
        plt.scatter(history_i.iloc[row_i]['x'], history_i.iloc[row_i]['y'], c=color_type, alpha=history_alpha)
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title(str(current_date))
    plt.savefig(os.path.join(figure_path, str(current_date.year)+'_'+str(date_i)+'.jpg'), bbox_inches='tight')
    plt.close()
    


for sample_i in range(gtd_pk.shape[0]//10):
    current_sample = gtd_pk.iloc[sample_i]
    date_difference = timedelta(days=10)
    history_date = current_sample['time']-date_difference
    history_i = gtd_pk[(gtd_pk['time']>=history_date) & (gtd_pk['time']<current_sample['time'])]
    plt.scatter(gtd_pk.iloc[sample_i]['x'], gtd_pk.iloc[sample_i]['y'], c=cmap(gtd_pk.iloc[row_i]['event_type']))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title(str(gtd_pk.iloc[row_i]['time']))

    plt.savefig(os.path.join(figure_path, 'sample_'+str(sample_i)+'.jpg'), bbox_inches='tight')
    plt.close()
