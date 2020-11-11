from typing import Any

import determined as det


class PythonTrialContext(det.TrialContext):
    """Contains runtime information for any Determined workflow that uses the ``PyTorch`` API.
    1. Functionalities inherited from :class:`determined.TrialContext`, including getting
       the runtime information and properly handling training data in distributed training.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
