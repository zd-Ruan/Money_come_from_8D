"""Qlib feature handler for the Alpha191 factor family.

The Alpha191 factors are precomputed by :mod:`quant_pipeline.alpha191` and
dumped into a Qlib provider as per-feature ``.day.bin`` files (one column per
factor, ``alpha001``..``alpha191``).  This handler is a genuine
``DataHandlerLP`` subclass (same construction pattern as the stock
Alpha158/Alpha360 handlers) that loads those columns through the standard Qlib
``QlibDataLoader``, applying the same processor pipeline as the Alpha360
handler used by the reference runs (ProcessInf -> ZScoreNorm -> Fillna on
features; DropnaLabel -> CSZScoreNorm on the label).

This module is imported lazily by the pipeline runner only when
``features.mode == "alpha191"`` (i.e. inside the run environment where qlib is
already importable), so the top-level qlib import is safe.

Usage
-----
Set ``features.mode: alpha191`` in a pipeline config; the runner constructs
``Alpha191(**handler_kwargs)`` with the usual instruments / start_time /
end_time / fit_* / label / filter_pipe arguments.
"""

from __future__ import annotations

from typing import Any

from qlib.contrib.data.handler import check_transform_proc
from qlib.data.dataset.handler import DataHandlerLP

from quant_pipeline.alpha191 import alpha191_registry

_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]
_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]


class Alpha191(DataHandlerLP):
    """Qlib handler exposing the 191 Alpha191 features as a DataHandlerLP."""

    def __init__(
        self,
        instruments: str = "csi500",
        start_time: str | None = None,
        end_time: str | None = None,
        freq: str = "day",
        infer_processors: list[dict[str, Any]] | None = None,
        learn_processors: list[dict[str, Any]] | None = None,
        fit_start_time: str | None = None,
        fit_end_time: str | None = None,
        process_type: str = DataHandlerLP.PTYPE_A,
        filter_pipe: list[dict[str, Any]] | None = None,
        inst_processors: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        if infer_processors is None:
            infer_processors = _DEFAULT_INFER_PROCESSORS
        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        fields, names = self.get_feature_config()
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (fields, names),
                    "label": kwargs.pop("label", None),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    @staticmethod
    def get_feature_config() -> tuple[list[str], list[str]]:
        names = [factor.name for factor in alpha191_registry()]
        fields = [f"${name}" for name in names]
        return fields, names
