from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Tuple, Optional


def prepare_awq_calib_data(
    raw_data: Any,
    default_text_column: str = "text",
) -> Tuple[Any, Optional[str]]:
    """
    Convert calibration samples into the format expected by AutoAWQ.

    Accepts:
      * list[str] → converts to datasets.Dataset({"text": [...]})
      * datasets.Dataset / DatasetDict → returned as-is with the appropriate text column
      * filesystem path / HF dataset name → returned as-is (handled downstream)
      * None → returns (None, default_text_column)

    Returns a tuple of (processed_data, text_column_name). The text column
    may be None when the downstream consumer does not require it.
    """

    if raw_data is None:
        return None, default_text_column

    # Case 1: simple list of strings
    if isinstance(raw_data, Iterable) and not isinstance(raw_data, (str, bytes, Path)):
        sample_list = list(raw_data)
        if not sample_list:
            raise ValueError("AWQ calibration samples list is empty.")
        try:
            from datasets import Dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "datasets is required to build an AWQ calibration dataset from a list of strings. "
                "Install it via `pip install datasets`."
            ) from exc
        dataset = Dataset.from_dict({default_text_column: sample_list})
        return dataset, default_text_column

    # Case 2: already a HF Dataset / DatasetDict
    column_names = getattr(raw_data, "column_names", None)
    if column_names is not None:
        # DatasetDict exposes a dict of splits -> columns
        if isinstance(column_names, dict):
            if len(raw_data) == 0:
                raise ValueError("Provided DatasetDict has no splits for AWQ calibration.")
            first_split = next(iter(raw_data.keys()))
            dataset = raw_data[first_split]
            text_column = default_text_column
            if text_column not in dataset.column_names:
                text_column = dataset.column_names[0]
            return dataset, text_column

        # Single Dataset
        text_column = default_text_column
        if text_column not in column_names:
            text_column = column_names[0]
        return raw_data, text_column

    # Case 3: assume it's a path or HF dataset identifier; let AWQ handle it.
    return raw_data, default_text_column
