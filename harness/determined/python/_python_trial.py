import logging
import pathlib
import random
from abc import abstractmethod
from typing import Any, Dict, Optional, cast

import numpy as np

import determined as det
from determined import horovod, util, workload
from determined.python import PythonTrialContext
from determined_common import check


class PythonTrialController(det.TrialController):
    def __init__(self, trial_inst: det.Trial, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        check.is_instance(trial_inst, PythonTrial, "PythonTrialController needs an PythonTrial")
        self.trial = cast(PythonTrial, trial_inst)
        self.context = cast(PythonTrialContext, self.context)
        # self.callbacks = self.trial.build_callbacks()

        # If a load path is provided load weights and restore the data location.
        self._load()

    @staticmethod
    def pre_execute_hook(env: det.EnvContext, hvd_config: horovod.HorovodContext) -> None:
        PythonTrialController._set_random_seeds(env.trial_seed)

    @staticmethod
    def _set_random_seeds(seed: int) -> None:
        # Set identical random seeds on all training processes.
        # When using horovod, each worker will start at a unique
        # offset in the dataset, ensuring it's processing a unique
        # training batch.
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def from_trial(*args: Any, **kwargs: Any) -> det.TrialController:
        return PythonTrialController(*args, **kwargs)

    @staticmethod
    def from_native(*args: Any, **kwargs: Any) -> det.TrialController:
        raise NotImplementedError("PythonTrial only supports the Trial API")

    def _evaluate_full_dataset_defined(self) -> bool:
        return util.is_overridden(self.trial.evaluate_full_dataset, PythonTrial)

    def run(self) -> None:
        for w, args, response_func in self.workloads:
            if w.kind == workload.Workload.Kind.RUN_STEP:
                response_func(
                    util.wrap_metrics(
                        self._train_for_step(w.step_id, w.num_batches, w.total_batches_processed),
                        self.context.get_stop_requested(),
                    )
                )
            elif w.kind == workload.Workload.Kind.COMPUTE_VALIDATION_METRICS:
                response_func(
                    util.wrap_metrics(
                        self._compute_validation_metrics(), self.context.get_stop_requested()
                    )
                )
            elif w.kind == workload.Workload.Kind.CHECKPOINT_MODEL:
                check.eq(len(args), 1)
                check.is_instance(args[0], pathlib.Path)
                path = cast(pathlib.Path, args[0])
                response_func(self._save(path))
            elif w.kind == workload.Workload.Kind.TERMINATE:
                response_func(workload.Skipped())
                break
            else:
                raise AssertionError("Unexpected workload: {}".format(w.kind))

    def _train_for_step(
        self, step_id: int, num_batches: int, total_batches_processed: int
    ) -> workload.Response:
        check.gt(step_id, 0)

        per_batch_metrics = []
        for _ in range(num_batches):
            tr_metrics = self.trial.train_some()
            per_batch_metrics.append(tr_metrics)
        check.is_instance(
            tr_metrics,
            dict,
            "train_batch() must return a dictionary "
            f"mapping string names to Tensor metrics, got {type(tr_metrics)}",
        )

        metrics = det.util.make_metrics(None, per_batch_metrics)

        logging.debug(f"Done training step: {num_batches} batches in {num_batches} batches.")

        return metrics

    def _compute_validation_metrics(self) -> workload.Response:
        metrics = {}  # type: Optional[Dict[str, Any]]
        check.true(self._evaluate_full_dataset_defined())

        metrics = self.trial.evaluate_full_dataset()

        check.is_instance(
            metrics, dict, f"eval() must return a dictionary, got {type(metrics)}."
        )
        # for callback in self.callbacks.values():
        #     logging.warning(
        #         "on_validation_step_end is now deprecated, please use on_validation_end instead"
        #     )
        #     callback.on_validation_step_end(cast(Dict[str, Any], metrics))
        #
        # for callback in self.callbacks.values():
        #     callback.on_validation_end(cast(Dict[str, Any], metrics))

        num_inputs = None  # TODO: figure this out
        return {"num_inputs": num_inputs, "validation_metrics": metrics}

    def _load(self) -> None:
        if not self.load_path:
            return
        self.trial.load(self.load_path)

    def _save(self, path: pathlib.Path) -> workload.Response:
        path.mkdir(parents=True, exist_ok=True)

        # The model code is the current working directory.
        util.write_user_code(path)

        self.trial.save(path)

        return cast(
            workload.Response,
            {
                "framework": "python",  # type: ignore
                "format": "user-defined",
            },
        )


class PythonTrial(det.Trial):
    """"""

    trial_controller_class = PythonTrialController
    trial_context_class = PythonTrialContext

    @abstractmethod
    def __init__(self, context: PythonTrialContext) -> None:
        """"""
        pass

    @abstractmethod
    def train_some(self) -> Dict[str, Any]:
        """
        Arguments:
            num_batches: number of batches to process
        Returns:
            torch.Tensor or Dict[str, Any]:
                training metrics to return.
        """
        pass

    # def build_callbacks(self) -> Dict[str, _callback.PyTorchCallback]:
    #     """
    #     Defines a dictionary of string names to callbacks to be used during
    #     training and/or validation.
    #
    #     The string name will be used as the key to save and restore callback
    #     state for any callback that defines :meth:`load_state_dict` and :meth:`state_dict`.
    #     """
    #     return {}

    def evaluate_full_dataset(self) -> Dict[str, Any]:
        """
        The metrics returned from this function must be JSON-serializable.

        Arguments:
            None
        """
        pass

    def save(self, path) -> None:
        """"""
        pass

    def load(self, path) -> None:
        """"""
        pass
