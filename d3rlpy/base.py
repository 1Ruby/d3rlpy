import numpy as np
import copy
import json
import gym

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Union
from typing import Sequence
from collections import defaultdict
from tqdm.auto import tqdm
from .preprocessing import create_scaler, Scaler
from .augmentation import create_augmentation, AugmentationPipeline, DrQPipeline
from .augmentation import DrQPipeline
from .dataset import Episode, MDPDataset, Transition, TransitionMiniBatch
from .logger import D3RLPyLogger
from .metrics.scorer import NEGATED_SCORER
from .context import disable_parallel
from .gpu import Device
from .optimizers import OptimizerFactory
from .encoders import EncoderFactory, create_encoder_factory
from .q_functions import QFunctionFactory, create_q_func_factory
from .argument_utils import check_scaler, ScalerArg
from .online.utility import get_action_size_from_env


class ImplBase(metaclass=ABCMeta):
    @abstractmethod
    def save_model(self, fname: str) -> None:
        pass

    @abstractmethod
    def load_model(self, fname: str) -> None:
        pass

    @property
    def observation_shape(self) -> Sequence[int]:
        pass

    @property
    def action_size(self) -> int:
        pass


def _serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in params.items():
        if isinstance(value, Device):
            params[key] = value.get_id()
        elif isinstance(value, (Scaler, EncoderFactory, QFunctionFactory)):
            params[key] = {
                "type": value.get_type(),
                "params": value.get_params(),
            }
        elif isinstance(value, OptimizerFactory):
            params[key] = value.get_params()
        elif isinstance(value, AugmentationPipeline):
            aug_types = value.get_augmentation_types()
            aug_params = value.get_augmentation_params()
            params[key] = {"params": value.get_params(), "augmentations": []}
            for aug_type, aug_param in zip(aug_types, aug_params):
                params[key]["augmentations"].append(
                    {"type": aug_type, "params": aug_param}
                )
    return params


def _deseriealize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in params.items():
        if key == "scaler" and params["scaler"]:
            scaler_type = params["scaler"]["type"]
            scaler_params = params["scaler"]["params"]
            scaler = create_scaler(scaler_type, **scaler_params)
            params[key] = scaler
        elif key == "augmentation" and params["augmentation"]:
            augmentations = []
            for param in params[key]["augmentations"]:
                aug_type = param["type"]
                aug_params = param["params"]
                augmentation = create_augmentation(aug_type, **aug_params)
                augmentations.append(augmentation)
            params[key] = DrQPipeline(augmentations, **params[key]["params"])
        elif "optim_factory" in key:
            params[key] = OptimizerFactory(**value)
        elif "encoder_factory" in key:
            params[key] = create_encoder_factory(
                value["type"], **value["params"]
            )
        elif key == "q_func_factory":
            params[key] = create_q_func_factory(
                value["type"], **value["params"]
            )
    return params


class LearnableBase:
    """Algorithm base class.

    All algorithms have the shared interfaces same as scikit-learn.

    Attributes:
        batch_size (int): the batch size of training.
        scaler (d3rlpy.preprocessing.Scaler): preprocessor
        augmentation (list(str or d3rlpy.augmentation.base.Augmentation)):
            list of data augmentations.
        use_gpu (d3rlpy.gpu.Device): GPU device.
        impl (d3rlpy.base.ImplBase): implementation object.
        eval_results_ (collections.defaultdict): evaluation results.
        loss_history_ (collections.defaultdict): history of loss values.
        active_logger_ (d3rlpy.logger.D3RLPyLogger): active logger during fit method.

    """

    _batch_size: int
    _n_frames: int
    _n_steps: int
    _gamma: float
    _scaler: Optional[Scaler]
    _impl: Optional[ImplBase]
    _eval_results: DefaultDict[str, List[float]]
    _loss_history: DefaultDict[str, List[float]]
    _active_logger: Optional[D3RLPyLogger]

    def __init__(
        self,
        batch_size: int,
        n_frames: int,
        n_steps: int,
        gamma: float,
        scaler: ScalerArg,
    ):
        self._batch_size = batch_size
        self._n_frames = n_frames
        self._n_steps = n_steps
        self._gamma = gamma
        self._scaler = check_scaler(scaler)

        self._impl = None
        self._eval_results = defaultdict(list)
        self._loss_history = defaultdict(list)
        self._active_logger = None

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        # propagate property updates to implementation object
        if hasattr(self, "_impl") and self._impl and hasattr(self._impl, name):
            setattr(self._impl, name, value)

    @classmethod
    def from_json(cls, fname: str, use_gpu: bool = False) -> "LearnableBase":
        """Returns algorithm configured with json file.

        The Json file should be the one saved during fitting.

        .. code-block:: python

            from d3rlpy.algos import Algo

            # create algorithm with saved configuration
            algo = Algo.from_json('d3rlpy_logs/<path-to-json>/params.json')

            # ready to load
            algo.load_model('d3rlpy_logs/<path-to-model>/model_100.pt')

            # ready to predict
            algo.predict(...)

        Args:
            fname (str): file path to `params.json`.
            use_gpu (bool, int or d3rlpy.gpu.Device):
                flag to use GPU, device ID or device.

        Returns:
            d3rlpy.base.LearnableBase: algorithm.

        """
        with open(fname, "r") as f:
            params = json.load(f)

        observation_shape = tuple(params["observation_shape"])
        action_size = params["action_size"]
        del params["observation_shape"]
        del params["action_size"]

        # reconstruct objects from json
        params = _deseriealize_params(params)

        # overwrite use_gpu flag
        params["use_gpu"] = use_gpu

        algo = cls(**params)
        algo.create_impl(observation_shape, action_size)
        return algo

    def set_params(self, **params: Dict[str, Any]) -> "LearnableBase":
        """Sets the given arguments to the attributes if they exist.

        This method sets the given values to the attributes including ones in
        subclasses. If the values that don't exist as attributes are
        passed, they are ignored.
        Some of scikit-learn utilities will use this method.

        .. code-block:: python

            algo.set_params(batch_size=100)

        Args:
            **params: arbitrary inputs to set as attributes.

        Returns:
            d3rlpy.algos.base.AlgoBase: itself.

        """
        for key, val in params.items():
            assert hasattr(self, key)
            setattr(self, key, val)
        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Returns the all attributes.

        This method returns the all attributes including ones in subclasses.
        Some of scikit-learn utilities will use this method.

        .. code-block:: python

            params = algo.get_params(deep=True)

            # the returned values can be used to instantiate the new object.
            algo2 = AlgoBase(**params)

        Args:
            deep (bool): flag to deeply copy objects such as `impl`.

        Returns:
            dict: attribute values in dictionary.

        """
        rets = {}
        for key in dir(self):
            # remove magic properties
            if key[:2] == "__":
                continue
            # remove protected properties
            if key[-1] == "_":
                continue
            # pick scalar parameters
            value = getattr(self, key)
            if np.isscalar(value):
                rets[key] = value
            elif isinstance(value, object) and not callable(value):
                if deep:
                    rets[key] = copy.deepcopy(value)
                else:
                    rets[key] = value
        return rets

    def save_model(self, fname: str) -> None:
        """Saves neural network parameters.

        .. code-block:: python

            algo.save_model('model.pt')

        Args:
            fname (str): destination file path.

        """
        assert self._impl is not None
        self._impl.save_model(fname)

    def load_model(self, fname: str) -> None:
        """Load neural network parameters.

        .. code-block:: python

            algo.load_model('model.pt')

        Args:
            fname (str): source file path.

        """
        assert self._impl is not None
        self._impl.load_model(fname)

    def fit(
        self,
        episodes: List[Episode],
        n_epochs: int = 1000,
        save_metrics: bool = True,
        experiment_name: Optional[str] = None,
        with_timestamp: bool = True,
        logdir: str = "d3rlpy_logs",
        verbose: bool = True,
        show_progress: bool = True,
        tensorboard: bool = True,
        eval_episodes: Optional[List[Episode]] = None,
        save_interval: int = 1,
        scorers: Optional[
            Dict[str, Callable[[Any, List[Episode]], float]]
        ] = None,
        shuffle: bool = True,
    ) -> None:
        """Trains with the given dataset.

        .. code-block:: python

            algo.fit(episodes)

        Args:
            episodes (list(d3rlpy.dataset.Episode)): list of episodes to train.
            n_epochs (int): the number of epochs to train.
            save_metrics (bool): flag to record metrics in files. If False,
                the log directory is not created and the model parameters are
                not saved during training.
            experiment_name (str): experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp (bool): flag to add timestamp string to the last of
                directory name.
            logdir (str): root directory name to save logs.
            verbose (bool): flag to show logged information on stdout.
            show_progress (bool): flag to show progress bar for iterations.
            tensorboard (bool): flag to save logged information in tensorboard
                (additional to the csv data)
            eval_episodes (list(d3rlpy.dataset.Episode)):
                list of episodes to test.
            save_interval (int): interval to save parameters.
            scorers (list(callable)):
                list of scorer functions used with `eval_episodes`.
            shuffle (bool): flag to shuffle transitions on each epoch.

        """

        transitions = []
        for episode in episodes:
            transitions += episode.transitions

        # initialize scaler
        if self._scaler:
            self._scaler.fit(episodes)

        # instantiate implementation
        if self._impl is None:
            action_size = transitions[0].get_action_size()
            observation_shape = tuple(transitions[0].get_observation_shape())
            self.create_impl(
                self._process_observation_shape(observation_shape), action_size
            )

        # setup logger
        logger = self._prepare_logger(
            save_metrics,
            experiment_name,
            with_timestamp,
            logdir,
            verbose,
            tensorboard,
        )

        # add reference to active logger to algo class during fit
        self._active_logger = logger

        # save hyperparameters
        self._save_params(logger)

        # refresh evaluation metrics
        self._eval_results = defaultdict(list)

        # refresh loss history
        self._loss_history = defaultdict(list)

        # hold original dataset
        env_transitions = transitions

        # training loop
        total_step = 0
        for epoch in range(n_epochs):

            # data augmentation
            new_transitions = self._generate_new_data(env_transitions)
            if new_transitions:
                transitions = env_transitions + new_transitions

            # shuffle data
            if shuffle:
                indices = np.random.permutation(np.arange(len(transitions)))
            else:
                indices = np.arange(len(transitions))

            # dict to add incremental mean losses to epoch
            epoch_loss = defaultdict(list)

            n_iters = len(transitions) // self._batch_size
            range_gen = tqdm(
                range(n_iters),
                disable=not show_progress,
                desc="Epoch %d" % int(epoch),
            )

            for itr in range_gen:
                with logger.measure_time("step"):
                    # pick transitions
                    with logger.measure_time("sample_batch"):
                        sampled_transitions = []
                        head_index = itr * self.batch_size
                        tail_index = head_index + self.batch_size
                        for index in indices[head_index:tail_index]:
                            sampled_transitions.append(transitions[index])

                        batch = TransitionMiniBatch(
                            transitions=sampled_transitions,
                            n_frames=self._n_frames,
                            n_steps=self._n_steps,
                            gamma=self._gamma,
                        )

                    # update parameters
                    with logger.measure_time("algorithm_update"):
                        loss = self.update(epoch, total_step, batch)

                    # record metrics
                    for name, val in zip(self._get_loss_labels(), loss):
                        if val is not None:
                            logger.add_metric(name, val)
                            epoch_loss[name].append(val)

                    # update progress postfix with losses
                    if itr % 10 == 0:
                        mean_loss = {
                            k: np.mean(v) for k, v in epoch_loss.items()
                        }
                        range_gen.set_postfix(mean_loss)

                    total_step += 1

            # save loss to loss history dict
            self._loss_history["epoch"].append(epoch)
            self._loss_history["step"].append(total_step)
            for name in self._get_loss_labels():
                if name in epoch_loss:
                    self._loss_history[name].append(np.mean(epoch_loss[name]))

            if scorers and eval_episodes:
                self._evaluate(eval_episodes, scorers, logger)

            # save metrics
            logger.commit(epoch, total_step)

            # save model parameters and greedy policy
            if epoch % save_interval == 0:
                logger.save_model(epoch, self)

        # drop reference to active logger since out of fit there is no active
        # logger
        self._active_logger = None

    def create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        """Instantiate implementation objects with the dataset shapes.

        This method will be used internally when `fit` method is called.

        Args:
            observation_shape (tuple): observation shape.
            action_size (int): dimension of action-space.

        """
        raise NotImplementedError

    def build_with_dataset(self, dataset: MDPDataset) -> None:
        """Instantiate implementation object with MDPDataset object.

        Args:
            dataset (d3rlpy.dataset.MDPDataset): dataset.

        """
        observation_shape = dataset.get_observation_shape()
        self.create_impl(
            self._process_observation_shape(observation_shape),
            dataset.get_action_size(),
        )

    def build_with_env(self, env: gym.Env) -> None:
        """Instantiate implementation object with OpenAI Gym object.

        Args:
            env (gym.Env): gym-like environment.

        """
        observation_shape = env.observation_space.shape
        self.create_impl(
            self._process_observation_shape(observation_shape),
            get_action_size_from_env(env),
        )

    def _process_observation_shape(
        self, observation_shape: Sequence[int]
    ) -> Sequence[int]:
        if len(observation_shape) == 3:
            n_channels = observation_shape[0]
            image_size = observation_shape[1:]
            # frame stacking for image observation
            observation_shape = (self._n_frames * n_channels, *image_size)
        return observation_shape

    def update(
        self, epoch: int, total_step: int, batch: TransitionMiniBatch
    ) -> List[float]:
        """Update parameters with mini-batch of data.

        Args:
            epoch (int): the current number of epochs.
            total_step (int): the current number of total iterations.
            batch (d3rlpy.dataset.TransitionMiniBatch): mini-batch data.

        Returns:
            list: loss values.

        """
        raise NotImplementedError

    def _generate_new_data(
        self, transitions: List[Transition]
    ) -> Optional[List[Transition]]:
        """Returns generated transitions for data augmentation.

        This method is called at the beginning of every epoch.

        Args:
            transitions (list(d3rlpy.dataset.Transition)): list of transitions.

        Returns:
            list(d3rlpy.dataset.Transition): list of new transitions.

        """
        return None

    def _get_loss_labels(self) -> List[str]:
        raise NotImplementedError

    def _prepare_logger(
        self,
        save_metrics: bool,
        experiment_name: Optional[str],
        with_timestamp: bool,
        logdir: str,
        verbose: bool,
        tensorboard: bool,
    ) -> D3RLPyLogger:
        if experiment_name is None:
            experiment_name = self.__class__.__name__

        logger = D3RLPyLogger(
            experiment_name,
            save_metrics=save_metrics,
            root_dir=logdir,
            verbose=verbose,
            tensorboard=tensorboard,
            with_timestamp=with_timestamp,
        )

        return logger

    def _evaluate(
        self,
        episodes: List[Episode],
        scorers: Dict[str, Callable[[Any, List[Episode]], float]],
        logger: D3RLPyLogger,
    ) -> None:
        for name, scorer in scorers.items():
            # evaluation with test data
            test_score = scorer(self, episodes)

            # higher scorer's scores are better in scikit-learn.
            # make it back to its original sign here.
            if scorer in NEGATED_SCORER:
                test_score *= -1

            # logging metrics
            logger.add_metric(name, test_score)

            # store metric locally
            if test_score is not None:
                self._eval_results[name].append(test_score)

    def _save_params(self, logger: D3RLPyLogger) -> None:
        assert self._impl

        # get hyperparameters without impl
        params = {}
        with disable_parallel():
            for k, v in self.get_params(deep=False).items():
                if isinstance(v, (ImplBase, LearnableBase)):
                    continue
                params[k] = v

        # save algorithm name
        params["algorithm"] = self.__class__.__name__

        # save shapes
        params["observation_shape"] = self._impl.observation_shape
        params["action_size"] = self._impl.action_size

        # serialize objects
        params = _serialize_params(params)

        logger.add_params(params)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def scaler(self) -> Optional[Scaler]:
        return self._scaler

    @property
    def impl(self) -> ImplBase:
        assert self._impl is not None
        return self._impl

    @property
    def observation_shape(self) -> Sequence[int]:
        assert self._impl is not None
        return self._impl.observation_shape

    @property
    def action_size(self) -> int:
        assert self._impl is not None
        return self._impl.action_size
