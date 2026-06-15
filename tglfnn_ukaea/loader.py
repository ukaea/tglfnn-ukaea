import pathlib
import pickle
from typing import Any, Literal, Mapping


def load(
    machine: Literal["step", "multimachine"] = "multimachine",
) -> Mapping[str, Any]:
    if machine not in ["step", "multimachine"]:
        raise ValueError(
            f"Unknown machine type: '{machine}' (must be 'step' or 'multimachine')"
        )

    pickle_file = pathlib.Path(__file__).parent / "weights" / f"{machine}.pkl"
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    return data
