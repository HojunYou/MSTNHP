
# Resolve project imports and legacy paths before loading dependencies.
if __package__:
    from ._paths import configure_paths
else:
    from _paths import configure_paths
configure_paths()

import argparse, numpy as np, torch, pickle, shapely.geometry as sg
from easy_tpp.config_factory import Config
from easy_tpp.runner import STPPRunner as Runner
from easy_tpp.preprocess import STPPDataLoader, STPPDataset, STEventTokenizer
from easy_tpp.preprocess.dataset import get_data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default='configs/experiment_config_gtd_pakistan.yaml',
                    help='Dir of configuration yaml to train and evaluate the model. ')

parser.add_argument('--experiment_id', type=str, required=False, default='STNHP_eval',
                    help='Experiment id in the config file.')

parser.add_argument('--ckpt_dir', type=str, required=False, default='./checkpoints/223690_22794185409536_250526-135535/models/saved_model',
                        help='Path to model')

parser.add_argument('--polygon',      required=False,    default='./data/gtd_pakistan/nigeria_polygon_norm.pkl',
                       help='pickled Shapely Polygon')


parser.add_argument('--device',      default='cuda:0' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()


config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
## HJ: Create a model runner from the configuration and train the model
model_runner = Runner.build_from_config(config)

## HJ: Load model from the checkpoint
n_comp = model_runner.model_wrapper.model.n_comps



model_runner.model_wrapper.model.load_state_dict(torch.load(args.ckpt_dir, map_location=args.device),
                                           strict=False)

model = model_runner.model_wrapper.model.to(args.device)
model.eval()

## HJ: get data and model configurations
data_config = model_runner.runner_config.data_config
model_backend = model_runner.runner_config.base_config.backend

## HJ: Load train dataset
split = 'test'
data_dir = data_config.get_data_dir(split)

print(f"data_dir: {data_dir}")

data_source_type = data_dir.split('.')[-1]
data_loader = STPPDataLoader(data_config, model_backend)  # data_loader = TPPDataLoader(data_config, model_backend)
data = data_loader.build_input_from_pkl(data_dir, split)
dataset = STPPDataset(data)  # dataset = TPPDataset(data)
sample_size = len(dataset)
tokenizer = STEventTokenizer(data_config.data_specs)  # tokenizer = EventTokenizer(data_config.data_specs)
batch_size = 1  # 13

## HJ: Load train dataloader
train_loader = get_data_loader(dataset, model_backend, tokenizer, batch_size=batch_size, shuffle=False, space=True)

with open(args.polygon, 'rb') as f:
    polygon = pickle.load(f)

# ------ 1.  true spatial area -----------------
if isinstance(polygon, sg.Polygon):
    area = polygon.area
else:                               # MultiPolygon
    area = sum(p.area for p in polygon.geoms)

#### Prediction #####
def get_upperbound_stnhp(model, batch):
    time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = batch
    dtimespace_for_bound = model.make_multiple_dtimespace_loss_samples(time_delta_seqs[:, 1:], space_seqs[:, :-1])
    dtime_for_bound, dspace_for_bound = dtimespace_for_bound[..., 0, 0], dtimespace_for_bound[..., :, :, 1:]
    dspace_for_bound = torch.norm(dspace_for_bound, dim=-1, p=2)
    intensities_for_bound = model.compute_intensities_at_sample_spacetimes(
        time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, dtime_for_bound, dspace_for_bound, compute_last_step_only=False
    )
    intensity_upperbound = intensities_for_bound.sum(dim=-1).max(dim=-1)[0]*model.event_sampler.over_sample_rate
    return intensity_upperbound.detach().cpu().numpy()

def generate_homogeneous_points(time_delta_seq, intensity_upperbound, seed_i=0):

    B, L = time_delta_seq.shape
    T = time_delta_seq + 5
    T = T.tolist()
    S = [[-1, 1], [-1, 1]]
    npoints = np.array([[area*t_i for t_i in T_i] for T_i in T])
    poisson_lam = intensity_upperbound*npoints
    np.random.seed(seed_i)
    N = np.random.poisson(lam=poisson_lam)
    # simulate spatial sequence and temporal sequence separately.
    homo_points = [[np.random.uniform([0, -1,-1], [T[n_i][i], 1, 1], (N[n_i][i], 3)) for i in range(L)] for n_i in range(B)]
    homo_points = [[point[point[:,0].argsort()] for point in points] for points in homo_points]
    return homo_points

def get_space_components(space_seq, homo_points_s):

    B, L, _ = space_seq.shape
    space_samples = [
        [homo_seq_s[:,None,:].repeat(1, 3, 1) for homo_seq_s in homo_point_s] for homo_point_s in homo_points_s
    ]
    # space_samples = space_samples.repeat(1, 1, 1, 3, 1)
    for i in range(B):
        for j in range(len(space_samples[i])):
            if j==0:
                space_samples[i][j][:,:2,:]=0
                space_samples[i][j][:,2,:] -= space_seq[i,j,:][None,:]
            elif j==1:
                space_samples[i][j][:,0,:] = 0
                space_samples[i][j][:,1:,:] -= space_seq[i,j-1:j+1,...][None,:,:]
            else:
                space_samples[i][j] -= space_seq[i,j-2:j+1, ...][None,:,:]

    space_samples = [
        [space_seq.norm(p=2, dim=-1) for space_seq in space_sample] for space_sample in space_samples
    ]

    return space_samples

def get_prediction(predicted_intensity, intensity_upperbound, homo_points_t, homo_points_s, type_seqs):

    predicted_intensity_sum = predicted_intensity.sum(-1)
    B, L, _ = predicted_intensity_sum.shape
    end_points = np.argmax(type_seqs[:,1:]==2, axis=1)
    zero_idx = np.where(end_points==0)[0]
    end_points[zero_idx] = L
    pred_type_seqs = np.zeros((B, L))
    pred_time_seqs = np.zeros((B, L))
    pred_space_seqs = np.zeros((B, L, 2))

    prob = np.random.rand(*predicted_intensity_sum.shape)
    retained_idx = np.where(predicted_intensity_sum > prob*intensity_upperbound[...,None])
    retained_idx = [idx.tolist() for idx in retained_idx]
    retained_idx = list(zip(*retained_idx))

    retained_idx_dict = {(k[0], k[1]):k[2] for k in retained_idx[::-1]}

    ## get an idx from retained_idx_dict and get values from time_samples with the corresponding index.
    ## then insert the value to pred_time_seqs
    for k, v in retained_idx_dict.items():
        pred_time = homo_points_t[k[0]][k[1]][v]
        pred_space = homo_points_s[k[0]][k[1]][v]
        pred_type = predicted_intensity[k[0], k[1], v].argmax().item()
        pred_type_seqs[k[0], k[1]] = pred_type
        pred_time_seqs[k[0], k[1]] = pred_time
        pred_space_seqs[k[0], k[1]] = pred_space

    all_retained = (pred_time_seqs!=0).all().item()
    if not all_retained:
        not_retained = np.array(np.where(pred_time_seqs==0)).transpose()
        for i in range(B):
            not_retained_i_cols = not_retained[not_retained[:,0]==i,1]
            if not not_retained_i_cols.any():
                continue
            if (not_retained_i_cols>=end_points[i]).all():
                continue
            for c in not_retained_i_cols:
                if c < end_points[i]:
                    pred_time_seqs[i,c] = homo_points_t[i][c][-1]
    # assert all_retained, 'There are some points not retained.'
    pad_idx = np.where(type_seqs[:,1:]==2.0)
    pad_idx = [idx.tolist() for idx in pad_idx]
    pad_idx = list(zip(*pad_idx))
    for pad_i in pad_idx:
        pred_time_seqs[pad_i] = 2.0
        pred_type_seqs[pad_i] = 2
        pred_space_seqs[pad_i] = 0

    total_len = len(type_seqs[:,1:].flatten())-len(pad_idx)

    return pred_time_seqs, pred_space_seqs, pred_type_seqs, total_len

def prediction_stnhp(model, batch, seed=0):

    time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, _, _, _ = batch
    time_seq, time_delta_seq, type_seq, space_seq, space_delta_seq = time_seqs[:, :-1], time_delta_seqs[:, 1:], type_seqs[:, :-1], space_seqs[:, :-1], space_delta_seqs[:, 1:]
    B, L = time_seq.size()
    nt, ns = model.loss_integral_num_sample_per_step, model.spatial_npoints
    model.loss_integral_num_sample_per_step = 50
    model.spatial_npoints = 10
    intensity_upperbound = get_upperbound_stnhp(model, batch)
    homo_points = generate_homogeneous_points(time_delta_seq, intensity_upperbound, seed_i=seed)
    homo_points_t = [
        [torch.Tensor(homo_seq[:,0]) for j, homo_seq in enumerate(homo_point)] for i, homo_point in enumerate(homo_points)
    ]
    homo_points_s = [
        [torch.Tensor(homo_seq[:,1:]) for homo_seq in homo_point] for homo_point in homo_points
    ]
    space_samples = get_space_components(space_seq, homo_points_s)
    max_len = max([len(homo_points_t[i][j]) for i in range(B) for j in range(L)])
    pred_sample_intensity = np.zeros((B,L,max_len,2))
    for i in range(B):
        for j in range(L):
            len_points = homo_points_t[i][j].shape[0]
            pred_sample_intensity[i,j,:len_points] = model.compute_intensities_at_sample_spacetimes(
                time_seqs[i][None,...],
                time_delta_seqs[i][None,...],
                space_seqs[i][None,...],
                space_delta_seqs[i][None,...],
                type_seqs[i][None,...],
                homo_points_t[i][j][None,None,...], space_samples[i][j][None, None,...]
            )[0,j].detach().cpu().numpy()

    pred_time_delta_seqs, pred_space_seqs, pred_type_seqs, total_len =\
        get_prediction(pred_sample_intensity, intensity_upperbound, homo_points_t, homo_points_s, type_seqs.numpy())

    time_pred_diff = (time_delta_seqs[:,1:].numpy()-pred_time_delta_seqs)**2
    space_pred_diff = np.sum((space_seqs[:,1:].numpy()-pred_space_seqs)**2, -1)
    type_pred_diff = type_seqs[:,1:].numpy()!=pred_type_seqs

    return time_pred_diff, space_pred_diff, type_pred_diff, total_len


import os, logging, sys
os.makedirs("logs", exist_ok=True)
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train.log", mode="w")

    ],
)
logger = logging.getLogger(__name__)




import time
from datetime import datetime
start_time = time.time()
## start_time to hour:min:seconds
logger.info(f"Start time: {datetime.now()}")

time_errors = []
type_errors = []
space_time_errors = []
space_errors = []

n_rep = 2

for i in range(n_rep):
    total_len = 0
    time_pred_diff = 0
    total_time_pred_diff = []
    space_pred_diff = 0
    type_pred_diff = 0
    torch.manual_seed(i)
    for batch_idx, batch in enumerate(train_loader):
        batch = tuple(t.to(args.device) for t in batch.values())
        batch_time_pred_diff, batch_space_pred_diff, batch_type_pred_diff, batch_len = prediction_stnhp(model, batch, seed=i)
        time_pred_diff+=batch_time_pred_diff.sum()
        space_pred_diff+=batch_space_pred_diff.sum()
        type_pred_diff+=batch_type_pred_diff.sum()
        total_len+=batch_len
        logger.info(f"Batch time prediction diff: {np.sqrt(batch_time_pred_diff.sum()/batch_len).item()}")
        logger.info(f"Batch space prediction diff: {np.sqrt(batch_space_pred_diff.sum()/batch_len).item()}")
        logger.info(f"Batch type prediction diff: {batch_type_pred_diff.sum()/batch_len}")
        logger.info(f"Batch {batch_idx}/{len(train_loader)} done.")
        if np.sqrt(batch_time_pred_diff.sum()/batch_len).item()>20:
            break

    end_time = time.time()
    logger.info(f"Time prediction diff: {np.sqrt(time_pred_diff/total_len).item()}")
    logger.info(f"Space prediction diff: {np.sqrt(space_pred_diff/total_len).item()}")
    logger.info(f"Type prediction diff: {type_pred_diff/total_len}")
    logger.info(f"Space-time prediction diff: {np.sqrt((time_pred_diff+space_pred_diff).sum()/total_len).item()}")
    logger.info(f"End time: {datetime.now()}")
    logger.info(f"Time elapsed: {end_time-start_time}")

    type_errors.append(type_pred_diff/total_len)
    time_errors.append(np.sqrt(time_pred_diff/total_len).item())
    space_errors.append(np.sqrt(space_pred_diff/total_len).item())
    space_time_errors.append(np.sqrt((time_pred_diff+space_pred_diff).sum()/total_len).item())

type_errors = np.array(type_errors)
time_errors = np.array(time_errors)
space_time_errors = np.array(space_time_errors)
space_errors = np.array(space_errors)
logger.info(f"Distribution of type errors: {np.mean(type_errors), np.std(type_errors)}")
logger.info(f"Distribution of time errors: {np.mean(time_errors), np.std(time_errors)}")
logger.info(f"Distribution of space errors: {np.mean(space_errors), np.std(space_errors)}")
logger.info(f"Distribution of space-time errors: {np.mean(space_time_errors), np.std(space_time_errors)}")

pred_dict = {}
pred_dict['type_errors'] = type_errors
pred_dict['time_errors'] = time_errors
pred_dict['space_errors'] = space_errors
pred_dict['space_time_errors'] = space_time_errors


