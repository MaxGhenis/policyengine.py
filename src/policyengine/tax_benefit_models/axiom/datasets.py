"""Shared Frame and MicroSeries plumbing for source-only Axiom datasets."""

from functools import cache
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from microdf import MicroDataFrame


@cache
def shared_entity_schema(group_entities: tuple[str, ...]) -> Any:
    """Return one lazily imported schema instance for an entity tuple."""

    try:
        from microcosm.frame.schema import EntitySchema
    except ImportError as error:
        raise ImportError(
            "Source-only Axiom datasets require a compatible microcosm-frame install."
        ) from error
    return EntitySchema(group_entities=group_entities)


def frame_from_household_tables(
    tables: dict[str, pd.DataFrame],
    *,
    entities: Sequence[str],
    runtime_loader: Callable[[], tuple[Any, Any, Any, Any]],
    country_label: str,
    allow_effective_weights: bool = False,
    weight_kind: Literal["design", "calibrated"] = "design",
    metadata: Optional[dict[str, Any]] = None,
    table_validator: Optional[Callable[[dict[str, pd.DataFrame]], None]] = None,
) -> Any:
    """Construct a household-weighted Frame without trusting derived weights.

    Country wrappers supply their entity tuple, schema/runtime loader, and any
    additional nesting validator. The household weight remains the sole stored
    weight vector; Frame resolves effective weights for all other entities.
    """

    declared = tuple(entities)
    if set(tables) != set(declared):
        raise ValueError(
            f"{country_label} requires exactly the entity tables {declared}."
        )
    copies = {entity: pd.DataFrame(table).copy() for entity, table in tables.items()}
    effective: dict[str, np.ndarray] = {}
    for entity, table in copies.items():
        for column in list(table.columns):
            if not column.endswith("_weight"):
                continue
            owner = column.removesuffix("_weight")
            if owner != entity or (
                owner != "household" and not allow_effective_weights
            ):
                raise ValueError(
                    f"{country_label} household_weight must be the sole stored "
                    "weight vector."
                )
            if owner != "household":
                effective[owner] = table.pop(column).to_numpy()
    if table_validator is not None:
        table_validator(copies)
    Frame, WeightKind, Weights, schema = runtime_loader()
    if "household_weight" not in copies["household"]:
        raise ValueError(f"{country_label} household table has no household_weight.")
    values = copies["household"].pop("household_weight").to_numpy()
    person = copies["person"]
    strata = person["support_stratum"] if "support_stratum" in person else None
    frame = Frame(
        copies,
        schema,
        {
            "household": Weights(
                values=values,
                kind=(
                    WeightKind.CALIBRATED
                    if weight_kind == "calibrated"
                    else WeightKind.DESIGN
                ),
            )
        },
        strata,
        metadata=metadata,
    )
    for entity, prior in effective.items():
        resolved = frame.resolve_weights(entity).values
        if not np.array_equal(prior, resolved):
            raise ValueError(
                f"{country_label} effective {entity}_weight differs from "
                "Frame-resolved household weights."
            )
    return frame


def weighted_tables_from_frame(
    frame: Any, entities: Sequence[str]
) -> dict[str, MicroDataFrame]:
    """Expose Frame-resolved entity weights as weighted MicroDataFrames."""

    tables: dict[str, MicroDataFrame] = {}
    for entity in entities:
        table = frame.table(entity).copy()
        column = f"{entity}_weight"
        table[column] = frame.resolve_weights(entity).values
        tables[entity] = MicroDataFrame(table, weights=column)
    return tables
