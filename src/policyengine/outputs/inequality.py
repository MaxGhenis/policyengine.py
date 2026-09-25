"""Inequality analysis output types."""

import warnings
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from microdf import MicroSeries
from pydantic import ConfigDict

from policyengine.core import Output, Simulation

# Start of the UserWarning microdf's MicroSeries.gini() emits when any value
# is negative.
_MICRODF_NEGATIVE_VALUES_WARNING = r"gini\(\) called on data containing negative values"


class USInequalityPreset(str, Enum):
    """Preset configurations for US inequality analysis."""

    STANDARD = "standard"
    CBO_COMPARABLE = "cbo_comparable"


def _weighted_inequality(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Calculate the weighted Gini coefficient and income shares.

    Uses microdf's ``MicroSeries.gini``, ``top_10_pct_share``,
    ``top_1_pct_share`` and ``bottom_50_pct_share``. The record that
    straddles a share cutoff contributes only the part of its weight that
    lies on the requested side. Negative values are included as they are.

    microdf returns NaN for some degenerate inputs, so this module's own
    contract is applied first: the Gini is 0.0 when there are no records,
    the weights sum to zero or total weighted income is zero; the shares are
    0.0 when there are no records, the weights sum to zero or less, or total
    weighted income is zero or less.

    Args:
        values: Array of income values
        weights: Array of weights

    Returns:
        Dict with ``gini``, ``top_10_share``, ``top_1_share`` and
        ``bottom_50_share``.
    """
    total_weight = weights.sum()
    total_income = np.sum(values * weights)
    series = MicroSeries(values, weights=weights)

    if len(values) == 0 or total_weight == 0 or total_income == 0:
        gini = 0.0
    else:
        with warnings.catch_warnings():
            # Negative incomes are real and have always been included unclipped.
            warnings.filterwarnings(
                "ignore",
                message=_MICRODF_NEGATIVE_VALUES_WARNING,
                category=UserWarning,
            )
            gini = float(series.gini())

    if len(values) == 0 or total_weight <= 0 or total_income <= 0:
        return {
            "gini": gini,
            "top_10_share": 0.0,
            "top_1_share": 0.0,
            "bottom_50_share": 0.0,
        }

    return {
        "gini": gini,
        "top_10_share": float(series.top_10_pct_share()),
        "top_1_share": float(series.top_1_pct_share()),
        "bottom_50_share": float(series.bottom_50_pct_share()),
    }


def _series_for_entity(
    simulation: Simulation, variable_name: str, target_entity: str, data: pd.DataFrame
) -> pd.Series:
    """Return a variable series aligned to the requested entity."""
    variable = simulation.tax_benefit_model_version.get_variable(variable_name)

    if variable.entity != target_entity:
        mapped = simulation.output_dataset.data.map_to_entity(
            variable.entity,
            target_entity,
            columns=[variable_name],
        )
        return mapped[variable_name]

    return data[variable_name]


class Inequality(Output):
    """Single inequality measure result - represents one database row.

    This is a single-simulation output type that calculates inequality
    metrics for a given income variable, optionally filtered by
    demographic variables.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    simulation: Simulation
    income_variable: str
    entity: str = "household"
    weight_multiplier_variable: Optional[str] = None
    equivalization_variable: Optional[str] = None
    equivalization_power: float = 0.0

    # Optional demographic filters
    filter_variable: Optional[str] = None
    filter_variable_eq: Optional[Any] = None
    filter_variable_leq: Optional[Any] = None
    filter_variable_geq: Optional[Any] = None

    # Results populated by run()
    gini: Optional[float] = None
    top_10_share: Optional[float] = None
    top_1_share: Optional[float] = None
    bottom_50_share: Optional[float] = None

    def run(self):
        """Calculate inequality metrics."""
        # Get target entity data
        target_entity = self.entity
        data = getattr(self.simulation.output_dataset.data, target_entity)

        income_series = _series_for_entity(
            self.simulation, self.income_variable, target_entity, data
        )

        # Get weights
        weight_col = f"{target_entity}_weight"
        if weight_col in data.columns:
            weights = data[weight_col]
        else:
            weights = pd.Series(np.ones(len(income_series)), index=income_series.index)

        if self.weight_multiplier_variable is not None:
            weight_multiplier = _series_for_entity(
                self.simulation,
                self.weight_multiplier_variable,
                target_entity,
                data,
            )
            weights = weights * weight_multiplier

        # Apply demographic filter if specified
        if self.filter_variable is not None:
            filter_series = _series_for_entity(
                self.simulation, self.filter_variable, target_entity, data
            )

            # Build filter mask
            mask = filter_series.notna()
            if self.filter_variable_eq is not None:
                mask &= filter_series == self.filter_variable_eq
            if self.filter_variable_leq is not None:
                mask &= filter_series <= self.filter_variable_leq
            if self.filter_variable_geq is not None:
                mask &= filter_series >= self.filter_variable_geq

            # Apply mask
            income_series = income_series[mask]
            weights = weights[mask]

        equivalization_arr = None
        if self.equivalization_variable is not None and self.equivalization_power != 0:
            equivalization_series = _series_for_entity(
                self.simulation,
                self.equivalization_variable,
                target_entity,
                data,
            )
            if self.filter_variable is not None:
                equivalization_series = equivalization_series[mask]
            equivalization_arr = pd.to_numeric(
                equivalization_series, errors="coerce"
            ).to_numpy(dtype=float)

        # Convert to numpy arrays
        values = pd.to_numeric(income_series, errors="coerce").to_numpy(dtype=float)
        weights_arr = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)

        # Remove invalid values
        valid_mask = ~np.isnan(values) & ~np.isnan(weights_arr)
        if equivalization_arr is not None:
            valid_mask &= ~np.isnan(equivalization_arr) & (equivalization_arr > 0)

        values = values[valid_mask]
        weights_arr = weights_arr[valid_mask]
        if equivalization_arr is not None:
            values = values / np.power(
                equivalization_arr[valid_mask], self.equivalization_power
            )

        measures = _weighted_inequality(values, weights_arr)
        self.gini = measures["gini"]
        self.top_10_share = measures["top_10_share"]
        self.top_1_share = measures["top_1_share"]
        self.bottom_50_share = measures["bottom_50_share"]


# Default income variables for each country
UK_INEQUALITY_INCOME_VARIABLE = "equiv_hbai_household_net_income"
US_INEQUALITY_INCOME_VARIABLE = "household_net_income"


def calculate_uk_inequality(
    simulation: Simulation,
    income_variable: str = UK_INEQUALITY_INCOME_VARIABLE,
    filter_variable: Optional[str] = None,
    filter_variable_eq: Optional[Any] = None,
    filter_variable_leq: Optional[Any] = None,
    filter_variable_geq: Optional[Any] = None,
) -> Inequality:
    """Calculate inequality metrics for a UK simulation.

    Args:
        simulation: The simulation to analyse
        income_variable: Income variable to use (default: equiv_hbai_household_net_income)
        filter_variable: Optional variable to filter by
        filter_variable_eq: Filter for exact match
        filter_variable_leq: Filter for less than or equal
        filter_variable_geq: Filter for greater than or equal

    Returns:
        Inequality object with Gini and income share metrics
    """
    inequality = Inequality(
        simulation=simulation,
        income_variable=income_variable,
        entity="household",
        filter_variable=filter_variable,
        filter_variable_eq=filter_variable_eq,
        filter_variable_leq=filter_variable_leq,
        filter_variable_geq=filter_variable_geq,
    )
    inequality.run()
    return inequality


def calculate_us_inequality(
    simulation: Simulation,
    income_variable: str = US_INEQUALITY_INCOME_VARIABLE,
    preset: Union[USInequalityPreset, str] = USInequalityPreset.STANDARD,
    filter_variable: Optional[str] = None,
    filter_variable_eq: Optional[Any] = None,
    filter_variable_leq: Optional[Any] = None,
    filter_variable_geq: Optional[Any] = None,
) -> Inequality:
    """Calculate inequality metrics for a US simulation.

    Args:
        simulation: The simulation to analyse
        income_variable: Income variable to use (default: household_net_income)
        preset: Optional preset for weighting/equivalization
        filter_variable: Optional variable to filter by
        filter_variable_eq: Filter for exact match
        filter_variable_leq: Filter for less than or equal
        filter_variable_geq: Filter for greater than or equal

    Returns:
        Inequality object with Gini and income share metrics
    """
    preset = USInequalityPreset(preset)
    inequality_kwargs = {}

    if preset == USInequalityPreset.CBO_COMPARABLE:
        inequality_kwargs = {
            "weight_multiplier_variable": "household_count_people",
            "equivalization_variable": "household_count_people",
            "equivalization_power": 0.5,
        }

    inequality = Inequality(
        simulation=simulation,
        income_variable=income_variable,
        entity="household",
        **inequality_kwargs,
        filter_variable=filter_variable,
        filter_variable_eq=filter_variable_eq,
        filter_variable_leq=filter_variable_leq,
        filter_variable_geq=filter_variable_geq,
    )
    inequality.run()
    return inequality
