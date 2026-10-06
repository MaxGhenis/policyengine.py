"""NZ compatibility wrappers over the shared exact Axiom table codec."""

from typing import Any

import pandas as pd  # noqa: F401 - retained for the pilot's monkeypatch seam
from microdf import MicroDataFrame

from policyengine.tax_benefit_models.axiom import EntityTableJSONCodec

TABLE_SCHEMA = "policyengine/nz-entity-table/1"
_CODEC = EntityTableJSONCodec(schema=TABLE_SCHEMA, country_label="NZ")


def _decode_dtype(payload: dict[str, Any]) -> Any:
    """Retain the pilot's tested private compatibility seam."""

    return _CODEC._decode_dtype(payload)


def encode_table(value: MicroDataFrame) -> dict[str, Any]:
    """Encode an NZ entity table without dtype or scalar inference."""

    return _CODEC.encode_table(value)


def decode_table(payload: dict[str, Any], entity: str) -> MicroDataFrame:
    """Restore an NZ entity table and its effective MicroDataFrame weight."""

    return _CODEC.decode_table(payload, entity)
