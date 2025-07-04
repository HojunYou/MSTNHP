from easy_tpp.preprocess.dataset import TPPDataset, STPPDataset
from easy_tpp.preprocess.dataset import get_data_loader
from easy_tpp.preprocess.event_tokenizer import EventTokenizer, STEventTokenizer
from easy_tpp.utils import load_pickle, py_assert


class TPPDataLoader:
    def __init__(self, data_config, backend, **kwargs):
        """Initialize the dataloader

        Args:
            data_config (EasyTPP.DataConfig): data config.
            backend (str): backend engine, e.g., tensorflow or torch.
        """
        self.data_config = data_config
        self.num_event_types = data_config.data_specs.num_event_types
        self.backend = backend
        self.kwargs = kwargs

    def build_input_from_pkl(self, source_dir: str, split: str):
        data = load_pickle(source_dir)

        py_assert(data["dim_process"] == self.num_event_types,
                  ValueError,
                  "inconsistent dim_process in different splits?")

        source_data = data[split]
        time_seqs = [[x["time_since_start"] for x in seq] for seq in source_data]
        type_seqs = [[x["type_event"] for x in seq] for seq in source_data]
        time_delta_seqs = [[x["time_since_last_event"] for x in seq] for seq in source_data]

        input_dict = dict({'time_seqs': time_seqs, 'time_delta_seqs': time_delta_seqs, 'type_seqs': type_seqs})
        return input_dict

    def get_loader(self, split='train', **kwargs):
        """Get the corresponding data loader.

        Args:
            split (str, optional): denote the train, valid and test set. Defaults to 'train'.
            num_event_types (int, optional): num of event types in the data. Defaults to None.

        Raises:
            NotImplementedError: the input of 'num_event_types' is inconsistent with the data.

        Returns:
            EasyTPP.DataLoader: the data loader for tpp data.
        """
        data_dir = self.data_config.get_data_dir(split)
        data_source_type = data_dir.split('.')[-1]

        if data_source_type == 'pkl':
            data = self.build_input_from_pkl(data_dir, split)
            dataset = TPPDataset(data)
            tokenizer = EventTokenizer(self.data_config.data_specs)
            loader = get_data_loader(dataset,
                                     self.backend,
                                     tokenizer,
                                     batch_size=self.kwargs['batch_size'],
                                     shuffle=self.kwargs['shuffle'],
                                     **kwargs)
        else:
            raise NotImplementedError

        return loader

    def train_loader(self, **kwargs):
        """Return the train loader

        Returns:
            EasyTPP.DataLoader: data loader for train set.
        """
        return self.get_loader('train', **kwargs)

    def valid_loader(self, **kwargs):
        """Return the valid loader

        Returns:
            EasyTPP.DataLoader: data loader for valid set.
        """
        return self.get_loader('dev', **kwargs)

    def test_loader(self, **kwargs):
        """Return the test loader

        Returns:
            EasyTPP.DataLoader: data loader for test set.
        """
        return self.get_loader('test', **kwargs)
    

class STPPDataLoader(TPPDataLoader):
    def __init__(self, data_config, backend, **kwargs):
        """Initialize the dataloader

        Args:
            data_config (EasyTPP.DataConfig): data config.
            backend (str): backend engine, e.g., tensorflow or torch.
        """
        super().__init__(data_config, backend, **kwargs)

    def build_input_from_pkl(self, source_dir: str, split: str):
        data = load_pickle(source_dir)

        py_assert(data["dim_process"] == self.num_event_types,
                  ValueError,
                  "inconsistent dim_process in different splits?")

        source_data = data[split]
        time_seqs = [[x["time_since_start"] for x in seq] for seq in source_data]
        space_seqs = [[x["location_from_origin"] for x in seq] for seq in source_data]
        type_seqs = [[x["type_event"] for x in seq] for seq in source_data]
        
        time_delta_seqs = [[x["time_since_last_event"] for x in seq] for seq in source_data]
        space_delta_seqs = [[x["location_from_last_event"] for x in seq] for seq in source_data]

        input_dict = dict({'time_seqs': time_seqs, 'time_delta_seqs': time_delta_seqs, 'space_seqs': space_seqs, 'space_delta_seqs': space_delta_seqs, 'type_seqs': type_seqs})
        return input_dict

    def get_loader(self, split='train', **kwargs):
        """Get the corresponding data loader.

        Args:
            split (str, optional): denote the train, valid and test set. Defaults to 'train'.
            num_event_types (int, optional): num of event types in the data. Defaults to None.

        Raises:
            NotImplementedError: the input of 'num_event_types' is inconsistent with the data.

        Returns:
            EasyTPP.DataLoader: the data loader for tpp data.
        """
        data_dir = self.data_config.get_data_dir(split)
        data_source_type = data_dir.split('.')[-1]

        if data_source_type == 'pkl':
            data = self.build_input_from_pkl(data_dir, split)
            dataset = STPPDataset(data)
            tokenizer = STEventTokenizer(self.data_config.data_specs)
            loader = get_data_loader(dataset,
                                     self.backend,
                                     tokenizer,
                                     space=True,
                                     batch_size=self.kwargs['batch_size'],
                                     shuffle=self.kwargs['shuffle'],
                                     **kwargs)
        else:
            raise NotImplementedError

        return loader

    def train_loader(self, **kwargs):
        """Return the train loader

        Returns:
            EasyTPP.DataLoader: data loader for train set.
        """
        return self.get_loader('train', **kwargs)

    def valid_loader(self, **kwargs):
        """Return the valid loader

        Returns:
            EasyTPP.DataLoader: data loader for valid set.
        """
        return self.get_loader('dev', **kwargs)

    def test_loader(self, **kwargs):
        """Return the test loader

        Returns:
            EasyTPP.DataLoader: data loader for test set.
        """
        return self.get_loader('test', **kwargs)
