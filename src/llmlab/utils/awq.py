from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Tuple


def prepare_awq_calib_data(
    raw_data: Any,
) -> Tuple[Any, Optional[str]]:
    """
    Convert calibration samples into a form accepted by AutoAWQ.

    Returns (data, text_column):
      * list[str] -> (list[str], None)
      * datasets.Dataset / DatasetDict -> (list[str], None)
      * str / Path (HF dataset name or file path) -> (raw_data, None)
      * None -> (None, None)
    """

    if raw_data is None:
        return None, None

    # Already a list-like container (list/tuple etc.) but not string/path.
    if isinstance(raw_data, Iterable) and not isinstance(raw_data, (str, bytes, Path)):
        sample_list = list(raw_data)
        if not sample_list:
            raise ValueError("AWQ calibration samples list is empty.")
        # Ensure each element is string or list of ints per AWQ contract.
        if not all(isinstance(x, (str, bytes, list)) for x in sample_list):
            raise ValueError(
                "Calibration samples must be strings or token ID lists. "
                "Received unsupported element types."
            )
        # bytes are acceptable but convert to string for convenience.
        sample_list = [x.decode("utf-8") if isinstance(x, bytes) else x for x in sample_list]
        return sample_list, None

    # Hugging Face datasets support
    column_names = getattr(raw_data, "column_names", None)
    if column_names is not None:
        try:
            import datasets  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'datasets' package is required to use Dataset objects as calibration data. "
                "Install it via `pip install datasets`."
            ) from exc

        if isinstance(column_names, dict):  # DatasetDict
            if len(raw_data) == 0:
                raise ValueError("Provided DatasetDict has no splits for AWQ calibration.")
            first_split = next(iter(raw_data.keys()))
            dataset = raw_data[first_split]
            column_names = dataset.column_names
        else:
            dataset = raw_data

        text_column = "text" if "text" in column_names else column_names[0]
        samples = dataset[text_column]
        return list(samples), None

    # Path or HF dataset identifier -> defer to AWQ, text_column not used.
    if isinstance(raw_data, (str, Path)):
        return raw_data, None

    raise TypeError(
        "Unsupported AWQ calibration data type. Provide a list of strings, "
        "a Hugging Face dataset, a dataset name/path, or None."
    )
