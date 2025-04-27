from collections import OrderedDict

from easy_tpp.runner.base_runner import Runner
from easy_tpp.runner.tpp_runner import TPPRunner
from easy_tpp.preprocess import STPPDataLoader
from easy_tpp.utils import RunnerPhase, logger, MetricsHelper, MetricsTracker, concat_element, save_pickle
from easy_tpp.utils.const import Backend

@Runner.register(name='std_stpp')
class STPPRunner(TPPRunner):

    def __init__(self, runner_config, unique_model_dir=False, **kwargs):
        super(STPPRunner, self).__init__(runner_config, unique_model_dir, **kwargs)
        skip_data_loader = kwargs.get('skip_data_loader', False)
        if not skip_data_loader:
            # build data reader
            data_config = self.runner_config.data_config
            backend = self.runner_config.base_config.backend
            kwargs = self.runner_config.trainer_config.get_yaml_config()
            self._data_loader = STPPDataLoader(
                data_config=data_config,
                backend=backend,
                **kwargs
            )
        
        self.metrics_tracker = MetricsTracker()
        if self.runner_config.trainer_config.metrics is not None:
            self.metric_functions = self.runner_config.get_metric_functions()

        self._init_model()

        pretrain_dir = self.runner_config.model_config.pretrained_model_dir
        if pretrain_dir is not None:
            self._load_model(pretrain_dir)