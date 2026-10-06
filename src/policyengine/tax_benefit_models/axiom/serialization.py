"""Exact JSON transport for source-only Axiom entity tables.

This codec preserves primitive table values, dtypes, indices, and effective
MicroDataFrame weights. It is transport code, not a rules evaluator.
"""

import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from microdf import MicroDataFrame


class EntityTableJSONCodec:
    """Serialize one country's weighted entity tables without dtype inference."""

    def __init__(self, *, schema: str, country_label: str) -> None:
        self.schema = schema
        self.country_label = country_label

    def _encode_cell(self, value: Any) -> Any:
        if value is pd.NA:
            return {"_type": "na"}
        if value is pd.NaT:
            return {"_type": "nat"}
        if isinstance(value, Decimal):
            return {"_type": "decimal", "value": str(value)}
        if isinstance(value, pd.Timestamp):
            return {"_type": "timestamp", "value": value.isoformat()}
        if isinstance(value, (datetime, date)):
            return {"_type": type(value).__name__, "value": value.isoformat()}
        if isinstance(value, np.generic):
            return self._encode_cell(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return {"_type": "float", "value": str(value)}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise ValueError(
            f"Unsupported {self.country_label} entity-table JSON value: "
            f"{type(value).__name__}."
        )

    def _decode_cell(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        kind = value.get("_type")
        if kind == "na":
            return pd.NA
        if kind == "nat":
            return pd.NaT
        decoders = {
            "decimal": Decimal,
            "timestamp": pd.Timestamp,
            "datetime": datetime.fromisoformat,
            "date": date.fromisoformat,
            "float": float,
        }
        if kind not in decoders:
            raise ValueError(
                f"Unsupported {self.country_label} JSON scalar tag {kind!r}."
            )
        return decoders[kind](value["value"])

    def _encode_dtype(self, dtype: Any) -> dict[str, Any]:
        if isinstance(dtype, pd.CategoricalDtype):
            return {
                "kind": "category",
                "categories": self._encode_index(dtype.categories),
                "ordered": dtype.ordered,
            }
        if isinstance(dtype, pd.StringDtype):
            return {
                "kind": "string",
                "storage": dtype.storage,
                "na_value": self._encode_cell(dtype.na_value),
            }
        return {"kind": "pandas", "name": str(dtype)}

    def _decode_dtype(self, payload: dict[str, Any]) -> Any:
        if payload["kind"] == "category":
            return pd.CategoricalDtype(
                categories=self._decode_index(payload["categories"]),
                ordered=payload["ordered"],
            )
        if payload["kind"] == "string":
            na_value = self._decode_cell(payload.get("na_value", {"_type": "na"}))
            if na_value is pd.NA:
                return pd.StringDtype(storage=payload["storage"])
            if not isinstance(na_value, float) or not math.isnan(na_value):
                raise ValueError(
                    f"Invalid {self.country_label} JSON string missing-value "
                    "descriptor."
                )
            try:
                return pd.StringDtype(storage=payload["storage"], na_value=na_value)
            except TypeError as exc:
                legacy_dtype = pd.StringDtype(storage=payload["storage"])
                if self._encode_cell(legacy_dtype.na_value) == self._encode_cell(
                    na_value
                ):
                    return legacy_dtype
                raise ValueError(
                    "This pandas version cannot restore NaN-semantics string data."
                ) from exc
        if payload["kind"] != "pandas":
            raise ValueError(f"Unsupported {self.country_label} JSON dtype descriptor.")
        return pd.api.types.pandas_dtype(payload["name"])

    def _encode_index(self, index: pd.Index) -> dict[str, Any]:
        if isinstance(index, pd.MultiIndex):
            return {
                "kind": "multi",
                "levels": [self._encode_index(level) for level in index.levels],
                "codes": [code.tolist() for code in index.codes],
                "names": [self._encode_cell(name) for name in index.names],
            }
        name = self._encode_cell(index.name)
        if isinstance(index, pd.RangeIndex):
            return {
                "kind": "range",
                "start": index.start,
                "stop": index.stop,
                "step": index.step,
                "name": name,
            }
        return {
            "kind": "index",
            "values": [self._encode_cell(value) for value in index],
            "dtype": self._encode_dtype(index.dtype),
            "name": name,
        }

    def _decode_index(self, payload: dict[str, Any]) -> pd.Index:
        if payload["kind"] == "multi":
            return pd.MultiIndex(
                levels=[self._decode_index(level) for level in payload["levels"]],
                codes=payload["codes"],
                names=[self._decode_cell(name) for name in payload["names"]],
                verify_integrity=True,
            )
        name = self._decode_cell(payload["name"])
        if payload["kind"] == "range":
            return pd.RangeIndex(
                payload["start"], payload["stop"], payload["step"], name=name
            )
        if payload["kind"] != "index":
            raise ValueError(f"Unsupported {self.country_label} JSON index descriptor.")
        return pd.Index(
            [self._decode_cell(value) for value in payload["values"]],
            dtype=self._decode_dtype(payload["dtype"]),
            name=name,
        )

    def encode_table(self, value: MicroDataFrame) -> dict[str, Any]:
        """Encode a MicroDataFrame to JSON-compatible primitive values."""

        table = pd.DataFrame(value)
        if not table.columns.is_unique or not all(
            isinstance(name, str) for name in table.columns
        ):
            raise ValueError(
                f"{self.country_label} JSON entity tables need unique string "
                "column names."
            )
        return {
            "schema": self.schema,
            "columns": list(table.columns),
            "columns_name": self._encode_cell(table.columns.name),
            "dtypes": [self._encode_dtype(dtype) for dtype in table.dtypes],
            "index": self._encode_index(table.index),
            "data": [
                [self._encode_cell(value) for value in row]
                for row in table.itertuples(index=False, name=None)
            ],
        }

    def decode_table(self, payload: dict[str, Any], entity: str) -> MicroDataFrame:
        """Restore a weighted MicroDataFrame from encoded primitive values."""

        if payload.get("schema") != self.schema:
            raise ValueError(
                f"Unsupported {self.country_label} entity-table JSON schema."
            )
        columns = payload["columns"]
        if (
            not all(isinstance(name, str) for name in columns)
            or len(set(columns)) != len(columns)
            or len(columns) != len(payload["dtypes"])
        ):
            raise ValueError(
                f"Invalid {self.country_label} entity-table JSON columns/dtypes."
            )
        table = pd.DataFrame(
            [[self._decode_cell(value) for value in row] for row in payload["data"]],
            index=self._decode_index(payload["index"]),
            columns=columns,
            dtype=object,
        ).astype(
            {
                name: self._decode_dtype(dtype)
                for name, dtype in zip(columns, payload["dtypes"])
            }
        )
        table.columns.name = self._decode_cell(payload["columns_name"])
        weight = f"{entity}_weight"
        if weight not in table:
            raise ValueError(
                f"{self.country_label} JSON {entity} table must contain "
                f"effective {weight}."
            )
        return MicroDataFrame(table, weights=weight)
