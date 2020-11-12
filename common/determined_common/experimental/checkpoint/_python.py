import pathlib
from typing import Any, Dict, Union, cast


from determined import experimental, util
from determined.python import PythonTrial, PythonTrialContext


def load_model(
    ckpt_dir: pathlib.Path, metadata: Dict[str, Any], **kwargs: Any
) -> PythonTrial:

    trial_cls, trial_context = experimental._load_trial_on_local(
        ckpt_dir.joinpath("code"),
        managed_training=False,
        config=metadata["experiment_config"],
        hparams=metadata["hparams"],
    )

    trial_context = cast(PythonTrialContext, trial_context)
    trial = cast(PythonTrial, trial_cls(trial_context))
    trial.load(ckpt_dir)
    return trial
