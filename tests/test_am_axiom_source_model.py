"""Boundary and opt-in integration tests for the Armenia Axiom surface.

No test substitutes Python policy logic for RuleSpec. The ordinary tests cover
fail-closed transport contracts. The opt-in test executes a neutral plumbing
fixture through the real Axiom runtime; it does not encode Armenian law.
"""

import base64
import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest
from microdf import MicroDataFrame
from pydantic import ValidationError

from policyengine.core import (
    Dataset,
    Simulation,
    TaxBenefitModel,
    TaxBenefitModelVersion,
)
from policyengine.outputs import Aggregate, AggregateType
from policyengine.tax_benefit_models.am import (
    AMYearData,
    AxiomArmeniaModelVersion,
    MicrocosmReleaseReceipt,
    PopulaceArmeniaDataset,
)

PERIOD = {"start": "2024-01-01", "end": "2024-12-31", "kind": "year"}


class OverlayRejectingModelVersion(TaxBenefitModelVersion):
    """Non-executable model used only to exercise the shared overlay gate."""

    accepts_policyengine_overlays = False


def test_policy_dict_is_rejected_before_policyengine_parameter_compilation():
    model = OverlayRejectingModelVersion(
        model=TaxBenefitModel(id="signed-artifact-model"), version="content-addressed"
    )
    dataset = Dataset(name="fixture", description="fixture", year=2024)

    with pytest.raises(
        ValueError, match="does not accept PolicyEngine policy dictionaries"
    ):
        Simulation(
            dataset=dataset,
            tax_benefit_model_version=model,
            policy={"gov.example.rate": 0.2},
        )


def test_armenia_model_requires_a_rule_module_before_loading_runtime(tmp_path):
    with pytest.raises(FileNotFoundError, match="signed Armenia RuleSpec module"):
        AxiomArmeniaModelVersion(
            rulespec_root=str(tmp_path),
            module="am/statutes/missing.yaml",
            trusted_apply_public_key="not-a-key",
            policy_period=PERIOD,
        )


def test_unsigned_rule_module_fails_before_loading_runtime(tmp_path):
    module = tmp_path / "am/statutes/integration.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\nmodule: {summary: unsigned}\nrules: []\n")

    with pytest.raises(ValueError, match="signed apply manifest"):
        AxiomArmeniaModelVersion(
            rulespec_root=str(tmp_path),
            module="am/statutes/integration.yaml",
            trusted_apply_public_key="not-a-key",
            policy_period=PERIOD,
        )


def test_test_yaml_cannot_be_selected_as_an_executable_module(tmp_path):
    module = tmp_path / "am/statutes/integration.test.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("cases: []\n")

    with pytest.raises(ValueError, match="executable .yaml source"):
        AxiomArmeniaModelVersion(
            rulespec_root=str(tmp_path),
            module="am/statutes/integration.test.yaml",
            trusted_apply_public_key="not-a-key",
            policy_period=PERIOD,
        )


def test_another_resolvable_jurisdiction_namespace_is_refused(tmp_path):
    module = tmp_path / "am/statutes/integration.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\nmodule: {summary: exact AM}\nrules: []\n")
    variant = tmp_path / "am-local/statutes/override.yaml"
    variant.parent.mkdir(parents=True)
    variant.write_text(
        "format: rulespec/v1\nmodule: {summary: unsigned variant}\nrules: []\n"
    )

    with pytest.raises(ValueError, match="Unsupported RuleSpec jurisdiction"):
        AxiomArmeniaModelVersion(
            rulespec_root=str(tmp_path),
            module="am/statutes/integration.yaml",
            trusted_apply_public_key="not-a-key",
            policy_period=PERIOD,
        )


def test_source_dataset_requires_certification_coordinates():
    with pytest.raises(ValidationError, match="certified Microcosm release"):
        PopulaceArmeniaDataset(
            name="uncertified",
            description="No release coordinates",
            year=2024,
            weight_kind="design",
        )


def test_release_manifest_hash_is_checked_before_h5_or_runtime(tmp_path):
    artifact = tmp_path / "candidate.h5"
    artifact.write_bytes(b"not an hdf5 artifact")
    manifest = tmp_path / "release_manifest.json"
    manifest.write_text("{}")
    in_memory = AMYearData(
        person=MicroDataFrame(
            pd.DataFrame(
                {
                    "person_id": [1],
                    "person_household_id": [1],
                    "person_weight": [1.0],
                }
            ),
            weights="person_weight",
        ),
        household=MicroDataFrame(
            pd.DataFrame({"household_id": [1], "household_weight": [1.0]}),
            weights="household_weight",
        ),
    )
    dataset = PopulaceArmeniaDataset(
        name="candidate",
        description="Invalid certification fixture",
        filepath=str(artifact),
        year=2024,
        weight_kind="design",
        release_manifest_path=str(manifest),
        release_manifest_sha256="0" * 64,
        source_sha256="1" * 64,
        release_receipt=MicrocosmReleaseReceipt(
            release_manifest_sha256="0" * 64,
            build_id="untrusted",
            dataset="populace_am_2024",
            artifact_path="populace_am_2024.h5",
            artifact_sha256="1" * 64,
            artifact_revision="untrusted",
        ),
        data=in_memory,
    )

    with pytest.raises(ValueError, match="release manifest SHA-256"):
        dataset.to_frame()


@pytest.mark.skipif(
    os.environ.get("RUN_AM_AXIOM_INTEGRATION") != "1",
    reason="requires compatible source-only Microcosm, Axiom, and receipt runtimes",
)
def test_real_axiom_runtime_registers_variables_and_preserves_weights(tmp_path):
    receipt_sign = pytest.importorskip("receipt.sign")
    pytest.importorskip("axiom_rules_engine")
    pytest.importorskip("axiom_rules_engine_dense")
    axiom_adapter = pytest.importorskip("microcosm.frame.adapters.axiom")
    AxiomEntityTableDataset = axiom_adapter.AxiomEntityTableDataset

    root = tmp_path / "rulespec-am"
    module_relative = Path("am/statutes/integration.yaml")
    module = root / module_relative
    module.parent.mkdir(parents=True)
    module.write_text(
        """format: rulespec/v1
module:
  summary: Neutral PolicyEngine/Axiom transport fixture; not Armenian law.
rules:
  - name: reported_income
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: AMD
    versions:
      - effective_from: '2024-01-01'
        formula: income
"""
    )
    test_relative = Path("am/statutes/integration.test.yaml")
    companion = root / test_relative
    companion.write_text(
        """cases:
  - name: transport identity
    period: 2024
    input: {income: 1}
    output: {reported_income: 1}
"""
    )

    private_key, public_key = receipt_sign.generate_signing_keypair()
    applied = [
        {"path": str(path), "sha256": sha256((root / path).read_bytes()).hexdigest()}
        for path in (module_relative, test_relative)
    ]
    manifest_payload = {
        "schema_version": "axiom-encode/applied-rulespec/v5",
        "backend": "integration-test",
        "applied_files": applied,
    }
    unsigned = json.dumps(
        manifest_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    domain = b"axiom-encode/external-signer-sign/v2\x00apply_ed25519\x00"
    signature = receipt_sign.sign_payload(private_key, unsigned, domain=domain)
    manifest_payload["signature"] = {
        "algorithm": "ed25519-domain-v1",
        "key_id": f"sha256:{receipt_sign.raw_public_key_sha256(public_key)}",
        "value": base64.b64encode(signature).decode("ascii"),
    }
    signed_manifest = root / ".axiom/encoding-manifests/am/statutes/integration.json"
    signed_manifest.parent.mkdir(parents=True)
    signed_manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "PolicyEngine tests"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-qm", "Add signed fixture"],
        check=True,
    )
    _, untrusted_public_key = receipt_sign.generate_signing_keypair()
    with pytest.raises(ValueError, match="was not signed by the pinned key"):
        AxiomArmeniaModelVersion(
            rulespec_root=str(root),
            module=str(module_relative),
            trusted_apply_public_key=untrusted_public_key.decode(),
            policy_period=PERIOD,
        )

    tables = {
        "person": pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_household_id": [10, 20],
                "income": [100.0, 200.0],
            }
        ),
        "household": pd.DataFrame(
            {"household_id": [10, 20], "household_weight": [2.0, 5.0]}
        ),
    }
    artifact = tmp_path / "populace_am_2024.h5"
    AxiomEntityTableDataset(tables=tables, time_period=2024).save(artifact)
    artifact_sha = sha256(artifact.read_bytes()).hexdigest()
    release = {
        "schema_version": 1,
        "data_package": {"name": "microcosm-data", "version": "source"},
        "compatible_model_packages": [],
        "compatible_core_packages": [],
        "default_datasets": {"national": "populace_am_2024"},
        "build": {"build_id": "populace-am-test-release"},
        "artifacts": {
            "populace_am_2024": {
                "kind": "microdata",
                "path": "populace_am_2024.h5",
                "repo_id": "policyengine/populace-am",
                "revision": "populace-am-test-release",
                "sha256": artifact_sha,
            }
        },
    }
    release_path = tmp_path / "release_manifest.json"
    release_path.write_text(json.dumps(release))
    release_sha = sha256(release_path.read_bytes()).hexdigest()

    dataset = PopulaceArmeniaDataset(
        name="signed-source-fixture",
        description="Neutral transport fixture; not an Armenian population",
        filepath=str(artifact),
        year=2024,
        weight_kind="design",
        release_manifest_path=str(release_path),
        release_manifest_sha256=release_sha,
    )
    model = AxiomArmeniaModelVersion(
        rulespec_root=str(root),
        module=str(module_relative),
        trusted_apply_public_key=public_key.decode(),
        policy_period=PERIOD,
    )

    variable = model.get_variable("reported_income")
    assert variable.entity == "person"
    assert variable.data_type is float
    assert variable.definition_period == "year"
    assert variable.unit == "AMD"
    assert model.rulespec_artifact["trusted_apply_key_id"].startswith("sha256:")
    assert model.relation_contract == []
    restored = AxiomArmeniaModelVersion.model_validate_json(model.model_dump_json())
    assert restored.id == model.id
    assert len(restored.variables) == 1
    assert restored.get_variable("reported_income").unit == "AMD"

    simulation = Simulation(dataset=dataset, tax_benefit_model_version=restored)
    simulation.run()
    output = simulation.output_dataset
    assert isinstance(output.data, AMYearData)
    assert output.data.person["reported_income"].sum() == 1200.0
    aggregate = Aggregate(
        simulation=simulation,
        variable="reported_income",
        aggregate_type=AggregateType.SUM,
    )
    aggregate.run()
    assert aggregate.result == 1200.0
    household_aggregate = Aggregate(
        simulation=simulation,
        variable="reported_income",
        entity="household",
        aggregate_type=AggregateType.SUM,
    )
    household_aggregate.run()
    assert household_aggregate.result == 1200.0
    receipt = output.metadata["policyengine_axiom_runs"][-1]
    assert receipt["country"] == "am"
    assert receipt["current_law_complete"] is False
    assert receipt["managed_population"] is False
    assert receipt["dataset_release"]["schema"] == (
        "policyengine/microcosm-release-input/1"
    )
    assert receipt["dataset_release"]["artifact_sha256"] == artifact_sha
    assert receipt["rulespec_artifact"]["module"] == str(module_relative)
    assert len(receipt["runtime_provenance"]["runtime"]["axiom_dense_sha256"]) == 64
