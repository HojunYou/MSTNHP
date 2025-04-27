import torch
from torch import nn
import numpy as np
import shapely
import pickle

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class ContTimeSpaceLSTMCell(nn.Module):
    """LSTM Cell in Neural Hawkes Process, NeurIPS'17.
    """

    def __init__(self, hidden_dim, beta=1.0, n_comps=3):
        """Initialize the continuous LSTM cell.

        Args:
            hidden_dim (int): dim of hidden state.
            beta (float, optional): beta in nn.Softplus. Defaults to 1.0.
        """
        super(ContTimeSpaceLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_comps = n_comps  # HJ: the number of spatial components (J) in the paper
        self.init_dense_layer(hidden_dim, bias=True, beta=beta)

    def init_dense_layer(self, hidden_dim, bias, beta, custom_init=True):
        """Initialize linear layers given Equations (5a-6c) in the paper.

        Args:
            hidden_dim (int): dim of hidden state.
            bias (bool): whether to use bias term in nn.Linear.
            beta (float): beta in nn.Softplus.
        """

        self.layer_input = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_forget = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_output = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_input_bar = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_forget_bar = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.layer_pre_c = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        ## HJ: n_comps from spatial components and 1 from the time component
        self.layer_decay = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * (self.n_comps+1), bias=bias), 
            nn.Softplus(beta=beta)
        )
        if custom_init:
            unif_bound = np.sqrt(3) / np.sqrt(2*hidden_dim)
            # nn.init.kaiming_uniform_(layer.weight, a=unif_bound)
            nn.init.uniform_(self.layer_decay[0].weight, a=.0, b=unif_bound)
            nn.init.constant_(self.layer_decay[0].bias, 0.0)

    def forward(self, x_i, hidden_i_minus, cell_i_minus, cell_bar_i_minus):
        """Update the continuous-time LSTM cell.
        Args:
            x_i (tensor): event embedding vector at t_i and s_i.
            hidden_i_minus (tensor): hidden state at t_i- and s_i
            cell_i_minus (tensor): cell state at t_i- and s_i
            cell_bar_i_minus (tensor): cell bar state at t_{i-1} and s_i

        Returns:
            list: cell state, cell bar state, decay and output at t_i
        """
        x_i_ = torch.cat((x_i, hidden_i_minus), dim=1)
        
        gate_input = torch.nn.Sigmoid()(self.layer_input(x_i_)) # update input gate - Equation (5a)
        gate_forget = torch.nn.Sigmoid()(self.layer_forget(x_i_)) # update forget gate - Equation (5b)
        gate_output = torch.nn.Sigmoid()(self.layer_output(x_i_)) # update output gate - Equation (5d)
        gate_input_bar = torch.nn.Sigmoid()(self.layer_input_bar(x_i_)) # update input bar - similar to Equation (5a)
        gate_forget_bar = torch.nn.Sigmoid()(self.layer_forget_bar(x_i_)) # update forget bar - similar to Equation (5b)
        gate_pre_c = torch.tanh(self.layer_pre_c(x_i_)) # update gate z - Equation (5c)

        # update gate decay - Equation (6c)
        gate_decay = self.layer_decay(x_i_)
        ## HJ: reshape gate_decay to [batch_size, hidden_dim, spatial_components+time_component]
        gate_decay = gate_decay.view(-1, self.hidden_dim, self.n_comps+1) 
        
        cell_i = gate_forget * cell_i_minus + gate_input * gate_pre_c # update cell state to t_i+ - Equation (6a)
        cell_bar_i = gate_forget_bar * cell_bar_i_minus + gate_input_bar * gate_pre_c # update cell state bar - Equation (6b)

        return cell_i, cell_bar_i, gate_decay, gate_output

    def decay(self, cell_i, cell_bar_i, gate_decay, gate_output, dtime, dspace):
        """Cell and hidden state decay according to Equation (7).

        Args:
            cell_i (tensor): cell state at t_i.
            cell_bar_i (tensor): cell bar state at t_i.
            gate_decay (tensor): gate decay state at t_i.
            gate_output (tensor): gate output state at t_i.
            dtime (tensor): delta time to decay.

        Returns:
            list: list of cell and hidden state tensors after the decay.
        """
        # HJ:  Concatenate dtime and dspace to [B, 3]
        dx = torch.cat((dtime, dspace), dim=-1)
        # gate_decay times dx when gate_decay has dimension [B, hidden_size, 3] and dx has [B, 3]
        # c_t = cell_bar_i + (cell_i - cell_bar_i) * torch.exp(-torch.bmm(gate_decay, dx[...,None])).squeeze(-1)
        c_t = cell_bar_i + (cell_i - cell_bar_i) * torch.exp(-(gate_decay*dx[...,None,:]).sum(-1))
        h_t = gate_output * torch.tanh(c_t)

        return c_t, h_t

class STNHP(TorchBaseModel):
    """Torch implementation of The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process,
       NeurIPS 2017, https://arxiv.org/abs/1612.09328.
    """

    def __init__(self, model_config):
        """Initialize the NHP model.

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(STNHP, self).__init__(model_config)
        self.beta = model_config.model_specs.get('beta', 1.0)
        self.bias = model_config.model_specs.get('bias', False)
        ## HJ: n_comps means the number of spatial components 
        self.n_comps = model_config.n_comps if model_config.n_comps is not None else 3
        ## HJ: how many spatial points will be used to calculate the integral
        self.spatial_npoints = model_config.spatial_npoints if model_config.spatial_npoints is not None else 10
        ## HJ: this is a unit area to calculate numerical integral
        self.unit_area = 4/(self.spatial_npoints**2)
        self.rnn_cell = ContTimeSpaceLSTMCell(self.hidden_size, beta=self.beta, n_comps=self.n_comps)

        self.layer_intensity = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_event_types, self.bias),
            nn.Softplus(self.beta)
        )
        
        ## HJ: Load the polygon to consider territorial constraints.
        self.polygon = None
        self.polygon_path = model_config.polygon_path if model_config.polygon_path is not None else None
        if self.polygon_path is not None:
            with open(self.polygon_path, 'rb') as f:
                self.polygon = pickle.load(f)
            print(f"Polygon is loaded from {self.polygon_path}")
        ## HJ: index tensor to check if the spatial points are inside a given polygon.
        self.inside_polygon_index = None

    def init_state(self, batch_size):
        """Initialize hidden and cell states.

        Args:
            batch_size (int): size of batch data.

        Returns:
            list: list of hidden states, cell states and cell bar states.
        """
        h_t, c_t, c_bar = torch.zeros(batch_size,
                                      3 * self.hidden_size,
                                      device=self.device).chunk(3, dim=1)
        return h_t, c_t, c_bar

    def forward(self, batch, **kwargs):
        """Call the model.

        Args:
            batch (tuple, list): batch input.

        Returns:
            list: hidden states, [batch_size, seq_len, hidden_dim], states right before the event happens;
                  stacked decay states,  [batch_size, max_seq_length, 4, hidden_dim], states right after
                  the event happens.
        """
        time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, batch_non_pad_mask, _, type_mask = batch

        all_hiddens = []
        all_outputs = []
        all_cells = []
        all_cell_bars = []
        all_decays = []

        max_steps = kwargs.get('max_steps', None)
        max_decay_time = kwargs.get('max_decay_time', 5.0)
        max_decay_space = kwargs.get('max_decay_space', [5.0])
        # last event has no time label
        max_seq_length = max_steps if max_steps is not None else type_seqs.size(1) - 1
        batch_size = len(type_seqs)
        h_t, c_t, c_bar_i = self.init_state(batch_size)

        # if only one event, then we dont decay
        # HJ: didn't change anything since there is no case of max_seq_length == 1
        if max_seq_length == 1:
            types_sub_batch = type_seqs[:, 0]
            x_t = self.layer_type_emb(types_sub_batch)
            cell_i, c_bar_i, decay_i, output_i = self.rnn_cell(x_t, h_t, c_t, c_bar_i)
            # Append all output
            all_outputs.append(output_i)
            all_decays.append(decay_i)
            all_cells.append(cell_i)
            all_cell_bars.append(c_bar_i)
            all_hiddens.append(h_t)
        else:
            # Loop over all events
            for i in range(max_seq_length):
                if i == type_seqs.size(1) - 1:
                    dt = torch.ones_like(time_delta_seqs[:, i]) * max_decay_time
                    ds = torch.ones_like(space_delta_seqs[:, i]) * max_decay_space
                ### Important ###
                elif i<=self.n_comps-2:
                    ## HJ: the case when we don't have 'n_comps' previous spatial locations
                    ## HJ: We add zeros to the beginning of the spatial location (n_comps-i-1) times
                    dt = time_delta_seqs[:, i + 1]
                    ## Calculate spatial location difference
                    ds = space_seqs[:,(i+1):(i+2)]-space_seqs[:,:(i+1)]
                    ## Calculate the distance of the spatial location
                    ds = torch.norm(ds, p=2, dim=-1)
                    ds = torch.cat([torch.zeros_like(ds[:,:1])]*(self.n_comps-1-i)+[ds], dim=1)
                else:
                    ## HJ: the case when we have 'n_comps' previous spatial locations
                    dt = time_delta_seqs[:, i + 1]  # need to carefully check here
                    ds = space_seqs[:,(i+1):(i+2)]-space_seqs[:,(i+1-(self.n_comps)):(i+1)]
                    ds = torch.norm(ds, p=2, dim=-1)
                
                types_sub_batch = type_seqs[:, i]
                x_t = self.layer_type_emb(types_sub_batch)

                # cell_i  (batch_size, process_dim)
                cell_i, c_bar_i, decay_i, output_i = self.rnn_cell(x_t, h_t, c_t, c_bar_i)

                # c_t, h_t: [batch_size, hidden_dim] # States decay - Equation (7) in the paper
                # HJ: we give dt and ds to decay function
                c_t, h_t = self.rnn_cell.decay(cell_i, c_bar_i, decay_i, output_i, dt[:, None], ds)

                # Append all output
                all_outputs.append(output_i)
                all_decays.append(decay_i)
                all_cells.append(cell_i)
                all_cell_bars.append(c_bar_i)
                all_hiddens.append(h_t)

        # (batch_size, max_seq_length, hidden_dim)
        cell_stack = torch.stack(all_cells, dim=1)
        cell_bar_stack = torch.stack(all_cell_bars, dim=1)
        decay_stack = torch.stack(all_decays, dim=1)
        output_stack = torch.stack(all_outputs, dim=1)
        # [batch_size, max_seq_length, hidden_dim]
        hiddens_stack = torch.stack(all_hiddens, dim=1)
        # [batch_size, max_seq_length, 4, hidden_dim]
        # HJ: Since decay_stack has a different dimension from the others due to spatial components, we need to stack them separately
        output_states_stack = torch.stack((cell_stack, cell_bar_stack, output_stack), dim=2)

        return hiddens_stack, decay_stack, output_states_stack

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            list: loglike loss, num events.
        """
        time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, batch_non_pad_mask, _, type_mask = batch

        hiddens_ti, decay_states, output_states = self.forward(batch)
        # Num of samples in each batch and num of event time point in the sequence
        batch_size, seq_len, _ = hiddens_ti.size()

        # Lambda(t) right before each event time point
        # lambda_at_event - [batch_size, num_times=max_len-1, num_event_types]
        # Here we drop the last event because it has no delta_time label (can not decay)
        lambda_at_event = self.layer_intensity(hiddens_ti)

        # HJ: Generate spatio-temporal grid points for numerical integration
        interval_st_sample = self.make_multiple_dtimespace_loss_samples(time_delta_seqs[:,1:], space_seqs[:,:-1])
        interval_t_sample, interval_s_sample = interval_st_sample[:,:,:,0,0], interval_st_sample[:,:,:,:,1:]
        # HJ: Convert difference to distance (s-s_i -> ||s-s_i||)
        interval_s_sample = torch.norm(interval_s_sample, dim=-1, p=2)
        
        # [batch_size, num_times = max_len - 1, num_mc_sample, hidden_size]
        state_t_sample = self.compute_states_at_sample_points(decay_states, output_states, interval_t_sample, interval_s_sample)

        # [batch_size, num_times = max_len - 1, num_mc_sample, event_num]
        lambda_t_sample = self.layer_intensity(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=batch_non_pad_mask[:, 1:],
                                                                        lambda_type_mask=type_mask[:, 1:])

        # (num_samples, num_times)
        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events

    def compute_states_at_sample_points(self, decays, output_states, sample_dtimes, sample_dspaces):
        """Compute the states at sampling times.

        Args:
            decay_states (tensor): states right after the events.
            sample_dtimes (tensor): delta times in sampling.

        Returns:
            tensor: hiddens states at sampling times.
        """
        # cells (batch_size, num_times, hidden_dim)
        cells, cell_bars, outputs = output_states.unbind(dim=-2)
        # h_ts shape (batch_size, num_times, num_mc_sample, hidden_dim)
        # cells[:, :, None, :]  (batch_size, num_times, 1, hidden_dim)
        # HJ: We give sample_dtimes and sample_dspaces to decay function
        _, h_ts = self.rnn_cell.decay(cells[:, :, None, :],
                                      cell_bars[:, :, None, :],
                                      decays[:, :, None, :],
                                      outputs[:, :, None, :],
                                      sample_dtimes[..., None],
                                      sample_dspaces)

        return h_ts
    
    def compute_intensities_at_sample_spacetimes(self, time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, sample_dtimes, sample_dspaces, **kwargs):
        """Compute the intensity at sampled times, not only event times.
        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_sample], sampled inter-event timestamps.
        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type.
        """
        compute_last_step_only = kwargs.get('compute_last_step_only', False)
        input_ = time_seqs, time_delta_seqs, space_seqs, space_delta_seqs, type_seqs, None, None, None
        # forward to the last but one event
        hiddens_ti, decays, output_states = self.forward(input_, **kwargs)

        # Num of samples in each batch and num of event time point in the sequence
        batch_size, seq_len, _ = hiddens_ti.size()

        # update the states given last event
        # cells (batch_size, num_times, hidden_dim)
        cells, cell_bars, outputs = output_states.unbind(dim=-2)

        if compute_last_step_only:
            interval_t_sample = sample_dtimes[:, -1:, :, None]
            interval_s_sample = sample_dspaces[:, -1:, ...]
            _, h_ts = self.rnn_cell.decay(cells[:, -1:, None, :],
                                          cell_bars[:, -1:, None, :],
                                          decays[:, -1:, None, :],
                                          outputs[:, -1:, None, :],
                                          interval_t_sample,
                                          interval_s_sample)

            # [batch_size, 1, num_mc_sample, num_event_types]
            sampled_intensities = self.layer_intensity(h_ts)

        else:
            # interval_t_sample - [batch_size, num_times, num_mc_sample, 1]
            # interval_t_sample = sample_dtimes[..., None]
            # Use broadcasting to compute the decays at all time steps
            # at all sample points
            # h_ts shape (batch_size, num_times, num_mc_sample, hidden_dim)
            # cells[:, :, None, :]  (batch_size, num_times, 1, hidden_dim)
            _, h_ts = self.rnn_cell.decay(cells[:, :, None, :],
                                          cell_bars[:, :, None, :],
                                          decays[:, :, None, :],
                                          outputs[:, :, None, :],
                                          sample_dtimes[..., None],
                                          sample_dspaces)

            # [batch_size, num_times, num_mc_sample, num_event_types]
            sampled_intensities = self.layer_intensity(h_ts)

        return sampled_intensities
    
    # def make_dspace_loss_samples(self, space_seq):
    #     # HJ: Generates spatial grid points for numerical integration
    #     # HJ: This function is used when we have only one spatial component (n_comps=1)
    #     B, L, _ = space_seq.size()
    #     space_xlower_limits = -1 + 1/self.spatial_npoints - space_seq[:,:,0]
    #     space_xupper_limits = 1 - 1/self.spatial_npoints - space_seq[:,:,0]
    #     space_ylower_limits = -1 + 1/self.spatial_npoints - space_seq[:,:,1]
    #     space_yupper_limits = 1 - 1/self.spatial_npoints - space_seq[:,:,1]
    #     x_coords = torch.linspace(0, 1, self.spatial_npoints).to(space_seq.device)
    #     y_coords = torch.linspace(0, 1, self.spatial_npoints).to(space_seq.device)
    #     x_coords = x_coords.view(1, 1, -1) * (space_xupper_limits[:,:,None] - space_xlower_limits[:,:,None]) + space_xlower_limits[:,:,None]
    #     y_coords = y_coords.view(1, 1, -1) * (space_yupper_limits[:,:,None] - space_ylower_limits[:,:,None]) + space_ylower_limits[:,:,None]
    #     x_coords = x_coords.view(-1, self.spatial_npoints)
    #     y_coords = y_coords.view(-1, self.spatial_npoints)
    #     sampled_dspace = torch.stack([torch.cartesian_prod(x_coord, y_coord) for x_coord, y_coord in zip(x_coords, y_coords)], dim=0).to(space_seq.device)
    #     sampled_dspace = sampled_dspace.view(B, L, -1, 2)
    #     return sampled_dspace
    
    def make_multiple_dspace_loss_samples(self, space_seq):
        # HJ: Generates spatial grid points for numerical integration
        # HJ: This function is used when we have multiple spatial components (n_comps>1)
        B, L, _ = space_seq.size()

        ## HJ: We need to generate spatial grid points for each spatial component
        ## HJ: We generate end points of the spatial grid points for each spatial component
        space_xlower_limit_single = -1 + 1/self.spatial_npoints - space_seq[:,:,0]
        space_xupper_limit_single = 1 - 1/self.spatial_npoints - space_seq[:,:,0]
        space_ylower_limit_single = -1 + 1/self.spatial_npoints - space_seq[:,:,1]
        space_yupper_limit_single = 1 - 1/self.spatial_npoints - space_seq[:,:,1]

        zeros = [torch.zeros(B, i).to(space_seq.device) for i in range(1, self.n_comps)]
        ## HJ: Generating Lower and Upper Limits for Multiple Components.
        space_xlower_limits = torch.cat([
            torch.cat([zeros[i-1], space_xlower_limit_single[:,:-i]], dim=-1)[...,None] for i in range(self.n_comps-1, 0, -1)
        ]+[space_xlower_limit_single[...,None]], dim=-1)

        space_xupper_limits = torch.cat([
            torch.cat([zeros[i-1], space_xupper_limit_single[:,:-i]], dim=-1)[...,None] for i in range(self.n_comps-1, 0, -1)
        ]+[space_xupper_limit_single[...,None]], dim=-1)

        space_ylower_limits = torch.cat([
            torch.cat([zeros[i-1], space_ylower_limit_single[:,:-i]], dim=-1)[...,None] for i in range(self.n_comps-1, 0, -1)
        ]+[space_ylower_limit_single[...,None]], dim=-1)

        space_yupper_limits = torch.cat([
            torch.cat([zeros[i-1], space_yupper_limit_single[:,:-i]], dim=-1)[...,None] for i in range(self.n_comps-1, 0, -1)
        ]+[space_yupper_limit_single[...,None]], dim=-1)

        ## HJ: Generating Spatial Grid Points for Multiple Components
        x_coords = torch.linspace(0, 1, self.spatial_npoints).to(self.device)
        y_coords = torch.linspace(0, 1, self.spatial_npoints).to(self.device)
        x_coords = x_coords.view(1, 1, -1, 1) * (space_xupper_limits[...,None,:] - space_xlower_limits[...,None,:]) + space_xlower_limits[:,:,None]
        y_coords = y_coords.view(1, 1, -1, 1) * (space_yupper_limits[...,None,:] - space_ylower_limits[...,None,:]) + space_ylower_limits[:,:,None]
        x_coords = x_coords.view(-1, self.spatial_npoints, space_xlower_limits.shape[2])
        y_coords = y_coords.view(-1, self.spatial_npoints, space_xlower_limits.shape[2])

        # with open('./data/gtd_pakistan/pakistan_polygon.pkl', 'rb') as f:
        #     polygon = pickle.load(f)

        def multiple_cartesian(x_points, y_points):
            cartesians = []
            for i in range(x_points.shape[1]):
                cartesians.append(torch.cartesian_prod(x_points[:,i], y_points[:,i]))
            return torch.stack(cartesians, dim=1)

        sampled_dspace = torch.stack([multiple_cartesian(x_coord, y_coord) for x_coord, y_coord in zip(x_coords, y_coords)], dim=0).to(self.device)
        sampled_dspace = sampled_dspace.view(B, L, -1, space_xlower_limits.shape[2], 2)
        if self.polygon is not None:
            if self.inside_polygon_index is None:
                self.falls_inside_polygon()
            sampled_dspace = sampled_dspace[:,:,self.inside_polygon_index]

        return sampled_dspace

    def falls_inside_polygon(self):
        grid_xlower_limit = -1 + 1/self.spatial_npoints
        grid_xupper_limit = 1 - 1/self.spatial_npoints
        grid_ylower_limit = -1 + 1/self.spatial_npoints
        grid_yupper_limit = 1 - 1/self.spatial_npoints
        grid_x = torch.linspace(grid_xlower_limit, grid_xupper_limit, self.spatial_npoints).to(self.device)
        grid_y = torch.linspace(grid_ylower_limit, grid_yupper_limit, self.spatial_npoints).to(self.device)
        grid_xy = torch.cartesian_prod(grid_x, grid_y).cpu().numpy()
        self.inside_polygon_index = []
        for i in range(grid_xy.shape[0]):
            if self.polygon.contains(shapely.geometry.Point(grid_xy[i])):
                self.inside_polygon_index.append(i)
        self.inside_polygon_index = torch.tensor(self.inside_polygon_index).to(self.device)

    # def make_dtimespace_loss_samples(self, time_delta_seq, space_seq):
    #     # HJ: Generates spatio-temporal grid points for numerical integration
    #     # HJ: This function is used when we have only one spatial component (n_comps=1)
    #     time_samples = self.make_dtime_loss_samples(time_delta_seq)
    #     space_samples = self.make_dspace_loss_samples(space_seq)
    #     B, L, n_t = time_samples.size()
    #     n_s = space_samples.shape[2]
    #     time_samples = time_samples.view(-1, n_t)
    #     space_samples = space_samples.view(-1, n_s, 2)
    #     assert time_samples.size(0) == space_samples.size(0), "Time and space samples should have the same batch * seq_length size"
    #     st_samples = torch.stack([self._make_3dsamples(time_sample, space_sample) for time_sample, space_sample in zip(time_samples, space_samples)], dim=0)
    #     st_samples = st_samples.to(time_samples.device)
    #     st_samples = st_samples.view(B, L, n_t*n_s, -1)
    #     return st_samples
    
    def make_multiple_dtimespace_loss_samples(self, time_delta_seq, space_seq):
        # HJ: Generates spatio-temporal grid points for numerical integration
        # HJ: This function is used when we have multiple spatial components (n_comps>1)
        time_samples = self.make_dtime_loss_samples(time_delta_seq)
        space_samples = self.make_multiple_dspace_loss_samples(space_seq)
        B, L, n_t = time_samples.size()
        n_s = space_samples.shape[2]
        time_samples = time_samples.view(-1, n_t)
        space_samples = space_samples.view(-1, n_s, space_samples.shape[3], 2)
        assert time_samples.size(0) == space_samples.size(0), "Time and space samples should have the same batch * seq_length size"
        st_samples = []
        for i in range(space_samples.shape[2]):
            st_sample = torch.stack([self._make_3dsamples(time_sample, space_sample[:,i,:]) for time_sample, space_sample in zip(time_samples, space_samples)], dim=0)
            st_sample = st_sample.to(self.device)
            st_sample = st_sample.view(B, L, n_t*n_s, 1, -1)
            st_samples.append(st_sample)
        st_samples = torch.cat(st_samples, dim=3)
        return st_samples
    
    def _make_3dsamples(self, time_sample, space_sample):
        # HJ: Computes cartesian product of time_sample and space_sample to generate spatio-temporal grid points
        n_t = time_sample.shape[0]
        n_s = space_sample.shape[0]
        time_sample_repeated = torch.tensor(time_sample.cpu().numpy().repeat(n_s)).to(time_sample.device)
        space_sample_repeated = torch.tensor(np.tile(space_sample.cpu().numpy(), (n_t, 1))).to(space_sample.device)
        st_sample = torch.cat([time_sample_repeated[:,None], space_sample_repeated], dim=1)
        return st_sample
    
    def _make_equal_3dsamples(self, time_seq, time_sample, space_sample):
        # HJ: Same as _make_3dsamples but for equal time intervals
        device = time_sample.device
        time_sample = time_sample.cpu().numpy()
        time_seq_np = time_seq.cpu().numpy()
        time_seq_end = np.argmax(time_seq_np)+1
        time_seq = time_seq_np[:time_seq_end]
        space_sample = space_sample[:time_seq_end]
        time_order = np.searchsorted(time_sample, time_seq)
        st_samples = []
        for i, time_idx in enumerate(time_order[:-1]):
            n_t = time_order[i+1]-time_order[i]
            n_s = space_sample[i].shape[0]
            time_samplepoints = time_sample[time_order[i]:time_order[i+1]]
            space_samplepoints = space_sample[i]
            time_samplepoints_repeated = torch.tensor(time_samplepoints.repeat(n_s), dtype=torch.float).to(device)
            space_samplepoints_repeated = space_samplepoints.repeat(n_t, 1)
            st_sample = torch.cat([time_samplepoints_repeated[:,None], space_samplepoints_repeated], dim=1)
            st_samples.append(st_sample)

        return st_samples

    # def make_equal_dtimespace_samples(self, time_seqs, time_domain, space_seqs, num_sample_t = 500):
    #     # HJ: Generates spatio-temporal grid points with equally-spaced time points for numerical integration
    #     # HJ: This function is used when we have only one spatial component (n_comps=1)
    #     start_times = time_seqs[:,0]
    #     time_samples = torch.linspace(time_domain[0], time_domain[1], num_sample_t)
    #     space_samples = self.make_dspace_loss_samples(space_seqs)
    #     time_sample_list = [time_samples[start_time<time_samples] for start_time in start_times]
    #     B, L, n_s, _ = space_samples.size()
            
    #     st_sample_list = [self._make_equal_3dsamples(time_seq, time_sample, space_sample) for time_seq, time_sample, space_sample in zip(time_seqs, time_sample_list, space_samples)]
    #     return st_sample_list
    
    def make_multiple_equal_dtimespace_samples(self, time_seqs, time_domain, space_seqs, num_sample_t = 500):
        # HJ: Generates spatio-temporal grid points with equally-spaced time points for numerical integration
        # HJ: This function is used when we have multiple spatial components (n_comps>1)
        start_times = time_seqs[:,0]
        time_samples = torch.linspace(time_domain[0], time_domain[1], num_sample_t)
        space_samples = self.make_multiple_dspace_loss_samples(space_seqs)
        time_sample_list = [time_samples[start_time<time_samples] for start_time in start_times]
        B, L, n_s, n_comps, _ = space_samples.size()
        st_equal_samples = []
        for i in range(n_comps):
            st_equal_sample = [self._make_equal_3dsamples(time_seq, time_sample, space_sample) for time_seq, time_sample, space_sample in zip(time_seqs, time_sample_list, space_samples[:,:,:,i,:])]
            st_equal_samples.append(st_equal_sample)
            
        st_equal_sample_list = []
        for i in range(len(st_equal_samples[0])):
            st_samples = [st_equal_sample[i] for st_equal_sample in st_equal_samples]
            st_sample_list = []
            for j in range(len(st_samples[0])):
                st_sample_list.append(torch.stack([st_samples[k][j] for k in range(len(st_samples))], dim=1))
            st_equal_sample_list.append(st_sample_list)

        return st_equal_sample_list

    def compute_loglikelihood(self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask, lambda_type_mask):
        # HJ: Compute the log likelihood of the model.
        event_lambdas = torch.sum(lambda_at_event * lambda_type_mask, dim=-1) + self.eps
        event_lambdas = event_lambdas.masked_fill_(~seq_mask, 1.0) # mask the pad event
        event_ll = torch.log(event_lambdas) # [batch_size, seq_len)

        lambdas_total_samples = lambdas_loss_samples.sum(dim=-1)
        non_event_ll = lambdas_total_samples.sum(dim=-1) * time_delta_seq * self.unit_area * seq_mask/self.loss_integral_num_sample_per_step
        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]

        return event_ll, non_event_ll, num_events

    def predict_one_step_at_every_event(self, batch):
        """One-step prediction for every event in the sequence.
        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].
        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seq, time_delta_seq, space_seq, space_delta_seq, event_seq, batch_non_pad_mask, _, type_mask = batch
        # remove the last event, as the prediction based on the last event has no label
        # time_delta_seq should start from 1, because the first one is zero
        time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, 1:], event_seq[:, :-1]
        # [batch_size, seq_len]
        dtime_boundary = time_delta_seq + self.event_sampler.dtime_max

        # [batch_size, seq_len, num_sample]
        accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(time_seq,
                                                                              time_delta_seq,
                                                                              event_seq,
                                                                              dtime_boundary,
                                                                              self.compute_intensities_at_sample_times)

        # [batch_size, seq_len]
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)

        # [batch_size, seq_len, 1, event_num]
        intensities_at_times = self.compute_intensities_at_sample_times(time_seq,
                                                                        time_delta_seq,
                                                                        event_seq,
                                                                        dtimes_pred[:, :, None],
                                                                        max_steps=event_seq.size()[1])

        # [batch_size, seq_len, event_num]
        intensities_at_times = intensities_at_times.squeeze(dim=-2)

        types_pred = torch.argmax(intensities_at_times, dim=-1)

        return dtimes_pred, types_pred