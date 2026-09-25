"""Tests for inequality analysis output type."""

import os
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from microdf import MicroDataFrame, MicroSeries

from policyengine.core import Simulation
from policyengine.outputs.inequality import (
    UK_INEQUALITY_INCOME_VARIABLE,
    US_INEQUALITY_INCOME_VARIABLE,
    Inequality,
    USInequalityPreset,
    _weighted_inequality,
    calculate_us_inequality,
)
from policyengine.tax_benefit_models.uk import (
    PolicyEngineUKDataset,
    UKYearData,
    uk_latest,
)


class _FakeOutputData(SimpleNamespace):
    def map_to_entity(self, source_entity, target_entity, columns):
        raise AssertionError("Unexpected map_to_entity() call in household-level test")


class _FakeTaxBenefitModelVersion:
    def get_variable(self, name):
        return SimpleNamespace(entity="household", name=name)


def _make_household_simulation(household_df: pd.DataFrame) -> Simulation:
    output_dataset = SimpleNamespace(data=_FakeOutputData(household=household_df))
    return Simulation.model_construct(
        output_dataset=output_dataset,
        tax_benefit_model_version=_FakeTaxBenefitModelVersion(),
    )


def test_gini_perfect_equality():
    """Test Gini coefficient with perfect equality (all same income)."""
    values = np.array([100.0, 100.0, 100.0, 100.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    gini = _weighted_inequality(values, weights)["gini"]
    assert abs(gini) < 0.01  # Should be ~0


def test_gini_perfect_inequality():
    """Test Gini coefficient with extreme inequality."""
    # One person has all income, others have none
    values = np.array([0.0, 0.0, 0.0, 1000.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    gini = _weighted_inequality(values, weights)["gini"]
    assert gini > 0.7  # Should be high


def test_gini_moderate_inequality():
    """Test Gini coefficient with moderate inequality."""
    values = np.array([10.0, 20.0, 30.0, 40.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])
    gini = _weighted_inequality(values, weights)["gini"]
    # Moderate inequality should be between 0.1 and 0.4
    assert 0.1 < gini < 0.4


def test_gini_weighted():
    """Test Gini coefficient with different weights."""
    values = np.array([100.0, 200.0])
    weights = np.array([3.0, 1.0])  # 3 people with 100, 1 with 200
    gini = _weighted_inequality(values, weights)["gini"]
    # Should reflect that 75% have lower income
    assert 0.05 < gini < 0.3


def test_inequality_basic():
    """Test basic inequality calculation."""
    person_df = MicroDataFrame(
        pd.DataFrame(
            {
                "person_id": [0, 1, 2, 3],
                "benunit_id": [0, 0, 1, 1],
                "household_id": [0, 0, 1, 1],
                "person_weight": [1.0, 1.0, 1.0, 1.0],
            }
        ),
        weights="person_weight",
    )

    benunit_df = MicroDataFrame(
        pd.DataFrame(
            {
                "benunit_id": [0, 1],
                "benunit_weight": [1.0, 1.0],
            }
        ),
        weights="benunit_weight",
    )

    # Two households with different incomes
    household_df = MicroDataFrame(
        pd.DataFrame(
            {
                "household_id": [0, 1],
                "household_weight": [1.0, 1.0],
                "equiv_hbai_household_net_income": [20000.0, 80000.0],
            }
        ),
        weights="household_weight",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.h5")

        dataset = PolicyEngineUKDataset(
            name="Test",
            description="Test dataset",
            filepath=filepath,
            year=2024,
            data=UKYearData(
                person=person_df, benunit=benunit_df, household=household_df
            ),
        )

        simulation = Simulation(
            dataset=dataset,
            tax_benefit_model_version=uk_latest,
            output_dataset=dataset,
        )

        inequality = Inequality(
            simulation=simulation,
            income_variable="equiv_hbai_household_net_income",
            entity="household",
        )
        inequality.run()

        # Check Gini is calculated
        assert inequality.gini is not None
        assert 0 <= inequality.gini <= 1

        # Check income shares are calculated
        assert inequality.top_10_share is not None
        assert inequality.top_1_share is not None
        assert inequality.bottom_50_share is not None

        # With 2 households of equal weight, one with 20k and one with 80k:
        # Top 50% (1 hh) has 80k, bottom 50% (1 hh) has 20k
        # Total = 100k, so bottom 50% share = 20%
        assert abs(inequality.bottom_50_share - 0.2) < 0.01


def test_inequality_income_shares():
    """Test income share calculations with more households."""
    person_df = MicroDataFrame(
        pd.DataFrame(
            {
                "person_id": list(range(10)),
                "benunit_id": list(range(10)),
                "household_id": list(range(10)),
                "person_weight": [1.0] * 10,
            }
        ),
        weights="person_weight",
    )

    benunit_df = MicroDataFrame(
        pd.DataFrame(
            {
                "benunit_id": list(range(10)),
                "benunit_weight": [1.0] * 10,
            }
        ),
        weights="benunit_weight",
    )

    # 10 households with incomes 10k, 20k, ..., 100k
    household_df = MicroDataFrame(
        pd.DataFrame(
            {
                "household_id": list(range(10)),
                "household_weight": [1.0] * 10,
                "equiv_hbai_household_net_income": [
                    10000.0,
                    20000.0,
                    30000.0,
                    40000.0,
                    50000.0,
                    60000.0,
                    70000.0,
                    80000.0,
                    90000.0,
                    100000.0,
                ],
            }
        ),
        weights="household_weight",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.h5")

        dataset = PolicyEngineUKDataset(
            name="Test",
            description="Test dataset",
            filepath=filepath,
            year=2024,
            data=UKYearData(
                person=person_df, benunit=benunit_df, household=household_df
            ),
        )

        simulation = Simulation(
            dataset=dataset,
            tax_benefit_model_version=uk_latest,
            output_dataset=dataset,
        )

        inequality = Inequality(
            simulation=simulation,
            income_variable="equiv_hbai_household_net_income",
            entity="household",
        )
        inequality.run()

        # Total income = 550k
        # Top 10% (1 hh with 100k) = 100k/550k = 18.2%
        assert abs(inequality.top_10_share - 100000 / 550000) < 0.02

        # Bottom 50% (5 hh with 10k-50k) = 150k/550k = 27.3%
        assert abs(inequality.bottom_50_share - 150000 / 550000) < 0.02


def test_inequality_variable_defaults():
    """Test default income variables for UK and US."""
    assert UK_INEQUALITY_INCOME_VARIABLE == "equiv_hbai_household_net_income"
    assert US_INEQUALITY_INCOME_VARIABLE == "household_net_income"


def test_inequality_supports_weight_multiplier_and_equivalization():
    """Test custom person-weighting and square-root equivalization."""
    simulation = _make_household_simulation(
        pd.DataFrame(
            {
                "household_weight": [2.0, 1.0],
                "household_net_income": [60_000.0, 120_000.0],
                "household_count_people": [1.0, 4.0],
            }
        )
    )

    inequality = Inequality(
        simulation=simulation,
        income_variable="household_net_income",
        entity="household",
        weight_multiplier_variable="household_count_people",
        equivalization_variable="household_count_people",
        equivalization_power=0.5,
    )
    inequality.run()

    adjusted_values = np.array([60_000.0, 60_000.0])
    adjusted_weights = np.array([2.0, 4.0])

    assert inequality.gini == pytest.approx(
        MicroSeries(adjusted_values, weights=adjusted_weights).gini()
    )


def test_calculate_us_inequality_cbo_comparable_preset_is_optional():
    """Test the optional US preset without changing default behaviour."""
    simulation = _make_household_simulation(
        pd.DataFrame(
            {
                "household_weight": [1.0, 1.0],
                "household_market_income": [50_000.0, 100_000.0],
                "household_net_income": [40_000.0, 80_000.0],
                "household_count_people": [1.0, 4.0],
            }
        )
    )

    standard = calculate_us_inequality(
        simulation, income_variable="household_market_income"
    )
    cbo_comparable = calculate_us_inequality(
        simulation,
        income_variable="household_market_income",
        preset=USInequalityPreset.CBO_COMPARABLE,
    )

    assert standard.gini == pytest.approx(
        MicroSeries(np.array([50_000.0, 100_000.0]), weights=np.ones(2)).gini()
    )
    assert cbo_comparable.gini == pytest.approx(0.0)
    assert cbo_comparable.gini < standard.gini


def test_calculate_us_inequality_rejects_unknown_preset():
    """Test validation of preset names."""
    simulation = _make_household_simulation(
        pd.DataFrame(
            {
                "household_weight": [1.0],
                "household_net_income": [10_000.0],
                "household_count_people": [1.0],
            }
        )
    )

    with pytest.raises(ValueError, match="not_a_preset"):
        calculate_us_inequality(simulation, preset="not_a_preset")


def test_inequality_weighted():
    """Test inequality with weighted households."""
    person_df = MicroDataFrame(
        pd.DataFrame(
            {
                "person_id": [0, 1],
                "benunit_id": [0, 1],
                "household_id": [0, 1],
                "person_weight": [1.0, 1.0],
            }
        ),
        weights="person_weight",
    )

    benunit_df = MicroDataFrame(
        pd.DataFrame(
            {
                "benunit_id": [0, 1],
                "benunit_weight": [1.0, 1.0],
            }
        ),
        weights="benunit_weight",
    )

    # Two households: one with weight 9 (30k), one with weight 1 (100k)
    household_df = MicroDataFrame(
        pd.DataFrame(
            {
                "household_id": [0, 1],
                "household_weight": [9.0, 1.0],
                "equiv_hbai_household_net_income": [30000.0, 100000.0],
            }
        ),
        weights="household_weight",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.h5")

        dataset = PolicyEngineUKDataset(
            name="Test",
            description="Test dataset",
            filepath=filepath,
            year=2024,
            data=UKYearData(
                person=person_df, benunit=benunit_df, household=household_df
            ),
        )

        simulation = Simulation(
            dataset=dataset,
            tax_benefit_model_version=uk_latest,
            output_dataset=dataset,
        )

        inequality = Inequality(
            simulation=simulation,
            income_variable="equiv_hbai_household_net_income",
            entity="household",
        )
        inequality.run()

        # Total weighted income = 9*30k + 1*100k = 370k
        # 90% of weight (9 hh) has 270k = 73% of income
        # So top 10% (1 hh with weight 1) has 100k/370k = 27%
        assert abs(inequality.top_10_share - 100000 / 370000) < 0.02


ZERO_SHARES = {"top_10_share": 0.0, "top_1_share": 0.0, "bottom_50_share": 0.0}


def _microdf_measures(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """microdf's own answers, with its negative-value warning allowed."""
    series = MicroSeries(values, weights=weights)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return {
            "gini": series.gini(),
            "top_10_share": series.top_10_pct_share(),
            "top_1_share": series.top_1_pct_share(),
            "bottom_50_share": series.bottom_50_pct_share(),
        }


@pytest.mark.parametrize("n_records", [1, 3, 7])
def test_constant_incomes_give_population_shares(n_records):
    """Constant incomes: each top or bottom x% of people holds exactly x%.

    The record straddling each cutoff contributes only its weight on the
    requested side, so the answer does not depend on how the weight is split
    into records.
    """
    values = np.full(n_records, 25_000.0)
    weights = np.full(n_records, 2.5)

    measures = _weighted_inequality(values, weights)

    assert measures["top_10_share"] == pytest.approx(0.10, abs=1e-12)
    assert measures["top_1_share"] == pytest.approx(0.01, abs=1e-12)
    assert measures["bottom_50_share"] == pytest.approx(0.50, abs=1e-12)
    assert measures["gini"] == pytest.approx(0.0, abs=1e-12)


def test_inequality_run_constant_incomes_give_population_shares():
    """The public output reports the same proportional split."""
    simulation = _make_household_simulation(
        pd.DataFrame(
            {
                "household_weight": [1.0, 1.0, 1.0],
                "household_net_income": [40_000.0, 40_000.0, 40_000.0],
            }
        )
    )

    inequality = Inequality(
        simulation=simulation,
        income_variable="household_net_income",
        entity="household",
    )
    inequality.run()

    assert inequality.top_10_share == pytest.approx(0.10, abs=1e-12)
    assert inequality.bottom_50_share == pytest.approx(0.50, abs=1e-12)


def test_gini_unequal_weights_matches_lorenz_trapezoid():
    """Unequal weights: the Gini is 1 - 2 x the trapezoid area under the Lorenz curve."""
    values = np.array([10_000.0, 30_000.0, 120_000.0])
    weights = np.array([800.0, 1_200.0, 50.0])

    cumulative_population = np.concatenate([[0.0], np.cumsum(weights) / weights.sum()])
    cumulative_income = np.concatenate(
        [[0.0], np.cumsum(values * weights) / np.sum(values * weights)]
    )
    lorenz_area = np.sum(
        np.diff(cumulative_population)
        * (cumulative_income[1:] + cumulative_income[:-1])
        / 2
    )

    gini = _weighted_inequality(values, weights)["gini"]

    assert gini == pytest.approx(1 - 2 * lorenz_area, abs=1e-12)
    assert gini == pytest.approx(0.28293, abs=1e-5)


def test_record_straddling_top_10_cutoff_is_split_by_weight():
    """Only the part of the straddling record's weight above the cutoff counts."""
    # Total weight 10, so the top 10% is 1 unit of the richest record's 2.
    values = np.array([10.0, 20.0, 30.0])
    weights = np.array([5.0, 3.0, 2.0])

    measures = _weighted_inequality(values, weights)

    assert measures["top_10_share"] == pytest.approx(30.0 / 170.0, abs=1e-12)
    # The top 1% is 0.1 units of that record.
    assert measures["top_1_share"] == pytest.approx(3.0 / 170.0, abs=1e-12)


def test_record_straddling_bottom_50_cutoff_is_split_by_weight():
    """The bottom half takes the lowest record and half of the middle one."""
    values = np.array([10.0, 20.0, 30.0])
    weights = np.array([1.0, 1.0, 1.0])

    measures = _weighted_inequality(values, weights)

    assert measures["bottom_50_share"] == pytest.approx(20.0 / 60.0, abs=1e-12)


def test_no_records_return_zeros():
    """No records: every measure is 0.0, not microdf's NaN."""
    measures = _weighted_inequality(np.array([]), np.array([]))

    assert measures == {"gini": 0.0, **ZERO_SHARES}


def test_inequality_run_with_empty_filter_returns_zeros():
    """A filter that matches no households reports 0.0 for every measure."""
    simulation = _make_household_simulation(
        pd.DataFrame(
            {
                "household_weight": [1.0, 2.0],
                "household_net_income": [20_000.0, 50_000.0],
                "household_count_people": [1.0, 2.0],
            }
        )
    )

    inequality = calculate_us_inequality(
        simulation,
        filter_variable="household_count_people",
        filter_variable_eq=9.0,
    )

    assert inequality.gini == 0.0
    assert inequality.top_10_share == 0.0
    assert inequality.top_1_share == 0.0
    assert inequality.bottom_50_share == 0.0


def test_zero_total_weight_returns_zeros():
    """Weights summing to zero: every measure is 0.0."""
    # Total weighted income is -10, so only the weight guard applies.
    measures = _weighted_inequality(np.array([10.0, 20.0]), np.array([1.0, -1.0]))

    assert measures == {"gini": 0.0, **ZERO_SHARES}


def test_negative_total_weight_zeroes_shares():
    """Weights summing below zero: shares are 0.0; the Gini is microdf's."""
    # Total weighted income is 8,000, so only the weight guard applies.
    values = np.array([10_000.0, 1_000.0])
    weights = np.array([1.0, -2.0])

    measures = _weighted_inequality(values, weights)

    assert {k: measures[k] for k in ZERO_SHARES} == ZERO_SHARES
    assert measures["gini"] == _microdf_measures(values, weights)["gini"]


@pytest.mark.parametrize(
    "values",
    [np.array([0.0, 0.0, 0.0]), np.array([-100.0, 50.0, 50.0])],
    ids=["all_zero", "negatives_cancel"],
)
def test_zero_total_income_returns_zeros(values):
    """Zero total weighted income: every measure is 0.0, not NaN."""
    measures = _weighted_inequality(values, np.array([1.0, 1.0, 1.0]))

    assert measures == {"gini": 0.0, **ZERO_SHARES}


def test_negative_total_income_zeroes_shares():
    """Negative total weighted income: shares are 0.0; the Gini is microdf's."""
    values = np.array([-50_000.0, 10_000.0])
    weights = np.array([1.0, 1.0])

    measures = _weighted_inequality(values, weights)

    assert {k: measures[k] for k in ZERO_SHARES} == ZERO_SHARES
    assert measures["gini"] == _microdf_measures(values, weights)["gini"]


def test_negative_incomes_are_included_without_escaping_warning():
    """Negative incomes stay in, unclipped; microdf's warning about them is silenced."""
    values = np.array([-5_000.0, 10_000.0, 30_000.0, 80_000.0])
    weights = np.array([1.0, 2.0, 1.5, 0.5])
    simulation = _make_household_simulation(
        pd.DataFrame({"household_weight": weights, "household_net_income": values})
    )

    # microdf itself warns on this input.
    with pytest.warns(UserWarning, match="negative values"):
        MicroSeries(values, weights=weights).gini()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inequality = Inequality(
            simulation=simulation,
            income_variable="household_net_income",
            entity="household",
        )
        inequality.run()

    expected = _microdf_measures(values, weights)
    assert inequality.gini == expected["gini"]
    assert inequality.top_10_share == expected["top_10_share"]
    assert inequality.top_1_share == expected["top_1_share"]
    assert inequality.bottom_50_share == expected["bottom_50_share"]


def test_other_microdf_warnings_are_not_silenced(monkeypatch):
    """Only the negative-values warning is suppressed."""

    def gini_with_other_warning(self, negatives=None):
        warnings.warn("some other microdf warning", UserWarning)
        return 0.5

    monkeypatch.setattr(MicroSeries, "gini", gini_with_other_warning)

    with pytest.warns(UserWarning, match="some other microdf warning"):
        _weighted_inequality(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
