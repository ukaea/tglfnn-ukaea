import pathlib
import pickle
from typing import Any, Literal, Mapping


def load(machine: Literal["step", "multimachine"] = "multimachine") -> Mapping[str, Any]:
    if machine == "step":
        pickle_file = pathlib.Path(__file__).parent / "weights" / "step.pkl"
        if not pickle_file.exists():
            raise FileNotFoundError(
                f"Pickle file for 'step' version of tglfnn-ukaea not found at {pickle_file}."
                " This may be because you installed the public version of the package, which"
                " does not include the 'step' model weights due to licensing restrictions."
            )
    elif machine == "multimachine":
        pickle_file = pathlib.Path(__file__).parent / "weights" / "multimachine.pkl"
    else:
        raise ValueError(
            f"Unknown machine type: '{machine}' (must be 'step' or 'multimachine')"
        )

    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    return data
