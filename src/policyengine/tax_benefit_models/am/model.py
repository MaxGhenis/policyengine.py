"""Source-only Armenia execution through the real Axiom dense runtime."""

import json
import subprocess
from copy import deepcopy
from datetime import date
from hashlib import sha256
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import numpy as np
import pandas as pd
from pydantic import Field

from policyengine.core import TaxBenefitModel, TaxBenefitModelVersion, Variable
from policyengine.tax_benefit_models.axiom import (
    shared_entity_schema,
    verify_signed_rulespec_tree,
)

from . import datasets as am_datasets
from .datasets import PopulaceArmeniaDataset, year_data_from_frame

if TYPE_CHECKING:
    from policyengine.core import Simulation


class AxiomArmenia(TaxBenefitModel):
    """Identity for signed Armenian RuleSpec artifacts executed by Axiom."""

    id: str = "axiom-rulespec-am"
    description: str = (
        "Signed Armenia RuleSpec executed by Axiom over Microcosm entity tables."
    )


am_model = AxiomArmenia()


class AxiomArmeniaModelVersion(TaxBenefitModelVersion):
    """One content-identified, source-only Armenia RuleSpec execution model."""

    country_code: ClassVar[str] = "am"
    accepts_policyengine_overlays: ClassVar[bool] = False

    rulespec_root: str
    module: str
    trusted_apply_public_key: str
    policy_period: dict[str, str]
    runtime_provenance: dict[str, Any]
    rulespec_artifact: dict[str, Any]
    variable_contract: list[dict[str, Any]]
    relation_contract: list[dict[str, Any]]
    entity_variables: dict[str, list[str]] = Field(default_factory=dict)
    variables: list[Variable] = Field(default_factory=list, exclude=True)

    def __init__(self, **kwargs: Any) -> None:
        root = Path(kwargs["rulespec_root"]).expanduser().resolve()
        module = _canonical_module(kwargs["module"])
        module_path = root / module
        if not module_path.is_file() or module_path.is_symlink():
            raise FileNotFoundError(
                f"A signed Armenia RuleSpec module was not found at {module}."
            )
        period = _canonical_period(kwargs["policy_period"])
        artifact = verify_signed_rulespec_tree(
            root,
            jurisdiction="am",
            module=module,
            trusted_apply_public_key=kwargs["trusted_apply_public_key"],
        )
        variable_contract, relation_contract = _discover_contract(root, module)
        if not variable_contract:
            raise ValueError(
                "The selected Armenia RuleSpec module has no derived Person or "
                "Household outputs."
            )
        provenance = _build_runtime_provenance(root)
        version_payload = {
            "rulespec_artifact": artifact,
            "runtime_provenance": provenance,
            "policy_period": period,
            "variable_contract": variable_contract,
            "relation_contract": relation_contract,
        }
        version = sha256(
            json.dumps(version_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        saved_model = kwargs.pop("model", None)
        if saved_model is not None:
            if isinstance(saved_model, TaxBenefitModel):
                saved_model_id = saved_model.id
            elif isinstance(saved_model, dict):
                saved_model_id = saved_model.get("id")
            else:
                saved_model_id = None
            if saved_model_id != am_model.id:
                raise ValueError(
                    "The saved Armenia model identity is not axiom-rulespec-am."
                )
        identity = {
            "module": module,
            "policy_period": period,
            "runtime_provenance": provenance,
            "rulespec_artifact": artifact,
            "variable_contract": variable_contract,
            "relation_contract": relation_contract,
            "entity_variables": _entity_variables(variable_contract),
            "version": version,
            "id": f"{am_model.id}@{version}",
        }
        for field, current in identity.items():
            if field in kwargs and kwargs[field] != current:
                raise ValueError(
                    f"The saved Armenia model {field} differs from the signed "
                    "artifact or current runtime; construct a new model version."
                )
        kwargs.update(identity)
        kwargs["model"] = am_model
        kwargs["variables"] = []
        kwargs.pop("variables_by_name", None)
        kwargs["rulespec_root"] = str(root)
        super().__init__(**kwargs)
        for contract in self.variable_contract:
            data_type = {
                "float": float,
                "int": int,
                "bool": bool,
                "str": str,
            }.get(contract["dtype"])
            if data_type is None:
                raise ValueError(
                    "Axiom returned an unsupported PolicyEngine dtype for "
                    f"{contract['name']!r}: {contract['dtype']!r}."
                )
            self.add_variable(
                Variable(
                    id=f"{self.id}:{contract['name']}",
                    name=contract["name"],
                    label=contract["name"].replace("_", " "),
                    tax_benefit_model_version=self,
                    entity=contract["entity"],
                    data_type=data_type,
                    definition_period=contract["period"],
                    unit=contract["unit"],
                    metadata={
                        "source_only": True,
                        "rulespec_module": self.module,
                        "rulespec_id": contract["rulespec_id"],
                        "rulespec_source": contract["source"],
                        "axiom_entity": contract["axiom_entity"],
                        "axiom_dtype": contract["axiom_dtype"],
                        "axiom_period": contract["axiom_period"],
                    },
                )
            )

    def _requested_outputs(self, simulation: "Simulation") -> list[str]:
        requested = {
            entity: list(names) for entity, names in self.entity_variables.items()
        }
        for entity, names in simulation.extra_variables.items():
            if entity not in requested:
                raise ValueError(
                    f"Unknown Armenia entity {entity!r} in Simulation.extra_variables."
                )
            for name in names:
                variable = self.get_variable(name)
                if variable.entity != entity:
                    raise ValueError(
                        f"Armenia variable {name!r} belongs to {variable.entity!r}, "
                        f"not {entity!r}."
                    )
                if name not in requested[entity]:
                    requested[entity].append(name)
        return [
            name
            for entity in am_datasets.ENTITIES
            for name in requested.get(entity, [])
        ]

    def run(self, simulation: "Simulation") -> "Simulation":
        for name in ("policy", "dynamic", "scoping_strategy"):
            if getattr(simulation, name) is not None:
                raise ValueError(
                    f"The Armenia Axiom model does not support Simulation.{name}; "
                    "select a separately signed RuleSpec artifact instead."
                )
        root = Path(self.rulespec_root)
        artifact = verify_signed_rulespec_tree(
            root,
            jurisdiction="am",
            module=self.module,
            trusted_apply_public_key=self.trusted_apply_public_key,
        )
        if artifact != self.rulespec_artifact:
            raise RuntimeError(
                "The signed Armenia RuleSpec tree changed; construct a new model version."
            )
        if _build_runtime_provenance(root) != self.runtime_provenance:
            raise RuntimeError(
                "The Armenia RuleSpec/Axiom runtime changed; construct a new model version."
            )
        if _discover_contract(root, self.module) != (
            self.variable_contract,
            self.relation_contract,
        ):
            raise RuntimeError(
                "The Armenia Axiom metadata contract changed; construct a new model version."
            )
        if self.relation_contract:
            raise NotImplementedError(
                "The Armenia Axiom adapter cannot execute modules with dense "
                "cross-entity relations until Microcosm supports relation batches."
            )

        dataset = simulation.dataset
        if not isinstance(dataset, PopulaceArmeniaDataset):
            raise TypeError("AxiomArmeniaModelVersion requires PopulaceArmeniaDataset.")
        if dataset.is_output_dataset:
            raise ValueError(
                "An Armenia output dataset cannot be reused as rules input."
            )
        if int(self.policy_period["start"][:4]) != dataset.year:
            raise ValueError(
                "The Armenia dataset year must match the explicit Axiom period start year."
            )
        frame = dataset.to_frame()
        if dataset.release_receipt is None or dataset.source_sha256 is None:
            raise RuntimeError(
                "Armenia input lacks an authenticated Microcosm release receipt."
            )
        input_frame_sha256 = _frame_sha256(frame)
        outputs = self._requested_outputs(simulation)
        stored = {
            name for entity in frame.entities for name in frame.table(entity).columns
        }
        formula_owned = set(self.variables_by_name)
        if formula_owned.intersection(stored):
            raise ValueError(
                "Armenia input contains formula-owned Axiom outputs: "
                f"{sorted(formula_owned.intersection(stored))}."
            )

        AxiomEngine, AxiomPeriod, schema = _load_axiom_adapter()
        engine = AxiomEngine(
            root / self.module,
            schema=schema,
            rulespec_roots=(root,),
        )
        materialized = engine.materialize(
            frame, outputs, AxiomPeriod(**self.policy_period)
        )
        contract_by_name = {item["name"]: item for item in self.variable_contract}
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        for name in outputs:
            contract = contract_by_name[name]
            values = np.asarray(materialized[name])
            expected = frame.n(contract["entity"])
            if values.shape != (expected,):
                raise ValueError(
                    f"Axiom returned shape {values.shape} for Armenia output "
                    f"{name!r}; expected {(expected,)}."
                )
            if contract["dtype"] in {"float", "int"}:
                try:
                    valid = np.isfinite(values.astype(float)).all()
                except (TypeError, ValueError, OverflowError):
                    valid = False
                if not valid:
                    raise ValueError(f"Axiom returned non-finite output {name!r}.")
            elif pd.isna(values).any():
                raise ValueError(f"Axiom returned null output {name!r}.")
            tables[contract["entity"]][name] = values

        tables["household"]["household_weight"] = frame.weights_for("household").values
        output_frame = am_datasets.frame_from_tables(
            tables,
            weight_kind=dataset.weight_kind,
            metadata=frame.metadata,
        )
        metadata = deepcopy(dataset.metadata)
        previous = metadata.get("policyengine_axiom_runs", [])
        if not isinstance(previous, list):
            raise ValueError("Armenia policyengine_axiom_runs metadata must be a list.")
        metadata["policyengine_axiom_runs"] = [
            *previous,
            {
                "schema": "policyengine/axiom-execution/1",
                "country": "am",
                "dataset_release": dataset.release_receipt.model_dump(
                    mode="json", by_alias=True
                ),
                "input_artifact_sha256": dataset.source_sha256,
                "input_frame_sha256": input_frame_sha256,
                "weight_kind": dataset.weight_kind,
                "policy_period": deepcopy(self.policy_period),
                "model_version": self.version,
                "output_variables": outputs,
                "variable_contract": deepcopy(self.variable_contract),
                "relation_contract": deepcopy(self.relation_contract),
                "rulespec_artifact": deepcopy(self.rulespec_artifact),
                "runtime_provenance": deepcopy(self.runtime_provenance),
                "current_law_complete": False,
                "managed_population": False,
            },
        ]
        simulation.output_dataset = PopulaceArmeniaDataset(
            id=simulation.id,
            name=dataset.name,
            description=dataset.description,
            filepath=None,
            year=dataset.year,
            is_output_dataset=True,
            data=year_data_from_frame(output_frame),
            metadata=metadata,
            source_sha256=dataset.source_sha256,
            weight_kind=dataset.weight_kind,
            release_receipt=dataset.release_receipt,
        )
        return simulation

    def save(self, simulation: "Simulation") -> None:
        """Keep source-only Armenia outputs in memory; never overwrite inputs."""

    def load(self, simulation: "Simulation") -> None:
        raise FileNotFoundError(
            "Source-only Armenia simulations are recomputed, not persisted."
        )


def _canonical_module(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("Armenia RuleSpec module must be a relative string path.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or ".." in path.parts
        or any(part in ("", ".") for part in path.parts)
    ):
        raise ValueError("Armenia RuleSpec module must be a canonical relative path.")
    return path.as_posix()


def _canonical_period(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"start", "end", "kind"}:
        raise ValueError(
            "Armenia policy_period must contain exactly start, end, and kind."
        )
    period = {name: value[name] for name in ("start", "end", "kind")}
    if not isinstance(period["kind"], str) or not period["kind"].strip():
        raise ValueError("Armenia policy_period.kind must be a non-empty string.")
    try:
        start = date.fromisoformat(period["start"])
        end = date.fromisoformat(period["end"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Armenia policy period dates must use ISO YYYY-MM-DD."
        ) from exc
    if start.isoformat() != period["start"] or end.isoformat() != period["end"]:
        raise ValueError("Armenia policy period dates must use ISO YYYY-MM-DD.")
    if start > end:
        raise ValueError("Armenia policy period start must not follow its end.")
    if (end - start).days + 1 > 366:
        raise ValueError("Armenia policy periods cannot span more than 366 days.")
    return period


def _load_axiom_adapter() -> tuple[Any, Any, Any]:
    try:
        from microcosm.frame.adapters.axiom import AxiomEngine, AxiomPeriod
    except ImportError as error:
        raise ImportError(
            "The Armenia model requires compatible source installs of "
            "microcosm-frame and AxiomPeriod."
        ) from error
    return AxiomEngine, AxiomPeriod, shared_entity_schema(("household",))


def _discover_contract(
    root: Path, module: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from axiom_rules_engine import CompiledDenseProgram
    except ImportError as error:
        raise ImportError(
            "The Armenia model requires the real Axiom Python and dense runtimes."
        ) from error
    AxiomEngine, _, schema = _load_axiom_adapter()
    engine = AxiomEngine(root / module, schema=schema, rulespec_roots=(root,))
    metadata_by_name: dict[str, Any] = {}
    relations: dict[tuple[str, str], dict[str, Any]] = {}
    for frame_entity, axiom_entity in (
        ("person", "Person"),
        ("household", "Household"),
    ):
        try:
            program = CompiledDenseProgram.from_file(
                root / module,
                rulespec_roots=(root,),
                entity=axiom_entity,
            )
        except ValueError as exc:
            missing = (
                "dense compilation could not find derived outputs for entity "
                f"`{axiom_entity}`"
            )
            if str(exc) == missing:
                continue
            raise
        for item in program.derived_metadata:
            if item.entity == "Scalar":
                continue
            if item.entity not in {"Person", "Household"}:
                raise ValueError(
                    "The Armenia adapter supports only Person and Household "
                    f"derived outputs; {item.name!r} belongs to {item.entity!r}."
                )
            if item.name.endswith(("_id", "_weight")):
                raise ValueError(
                    "Axiom-derived Armenia outputs must not overwrite structural "
                    f"Frame columns: {item.name!r}."
                )
            if item.name in metadata_by_name and metadata_by_name[item.name] != item:
                raise ValueError(
                    f"Axiom returned conflicting metadata for {item.name!r}."
                )
            metadata_by_name[item.name] = item
        for relation in program.relations:
            relations[(frame_entity, relation.key)] = {
                "entity": frame_entity,
                "key": relation.key,
                "name": relation.name,
                "current_slot": relation.current_slot,
                "related_slot": relation.related_slot,
                "related_inputs": list(relation.related_inputs),
                "execution_supported": False,
            }
    contract: list[dict[str, Any]] = []
    for name, item in sorted(metadata_by_name.items()):
        metadata = engine.variable_metadata(name)
        contract.append(
            {
                "name": name,
                "entity": metadata.entity,
                "dtype": metadata.dtype,
                "period": metadata.period,
                "unit": item.unit,
                "rulespec_id": item.id,
                "source": item.source,
                "axiom_entity": item.entity,
                "axiom_dtype": item.dtype,
                "axiom_period": item.period,
            }
        )
    return contract, [relations[key] for key in sorted(relations)]


def _entity_variables(contract: list[dict[str, Any]]) -> dict[str, list[str]]:
    result = {entity: [] for entity in am_datasets.ENTITIES}
    for item in contract:
        result[item["entity"]].append(item["name"])
    return result


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_path(name: str) -> Path:
    spec = find_spec(name)
    if spec is None or spec.origin is None or not Path(spec.origin).is_file():
        raise ImportError(f"Cannot locate source-only runtime module {name!r}.")
    return Path(spec.origin)


def _package_sha256(name: str) -> str:
    origin = _module_path(name)
    if origin.suffix != ".py":
        return _file_sha256(origin)
    root = origin.parent
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _optional_version(distribution: str) -> Optional[str]:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _build_runtime_provenance(root: Path) -> dict[str, Any]:
    """Identify committed signed sources plus every executed runtime layer."""

    try:
        if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
            raise ValueError("rulespec_root must be the rulespec-am repository root.")
        dirty = _git(
            root,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "am",
            ".axiom/encoding-manifests/am",
        )
        if dirty:
            raise RuntimeError(
                "Commit the signed Armenia RuleSpec sources and apply manifests "
                "before constructing a content-identified model."
            )
        return {
            "rulespec": {
                "repository": "TheAxiomFoundation/rulespec-am",
                "commit": _git(root, "rev-parse", "HEAD"),
                "am_tree": _git(root, "rev-parse", "HEAD:am"),
                "apply_manifest_tree": _git(
                    root, "rev-parse", "HEAD:.axiom/encoding-manifests/am"
                ),
            },
            "runtime": {
                "policyengine_version": importlib_metadata.version("policyengine"),
                "policyengine_am_sha256": _package_sha256(
                    "policyengine.tax_benefit_models.am"
                ),
                "policyengine_axiom_sha256": _package_sha256(
                    "policyengine.tax_benefit_models.axiom"
                ),
                "policyengine_core_sha256": _package_sha256("policyengine.core"),
                "policyengine_outputs_sha256": _package_sha256("policyengine.outputs"),
                "pandas_version": importlib_metadata.version("pandas"),
                "numpy_version": importlib_metadata.version("numpy"),
                "microdf_version": importlib_metadata.version("microdf-python"),
                "microdf_sha256": _package_sha256("microdf"),
                "microcosm_frame_version": importlib_metadata.version(
                    "microcosm-frame"
                ),
                "microcosm_frame_sha256": _package_sha256("microcosm.frame"),
                "axiom_python_version": importlib_metadata.version(
                    "axiom-rules-engine"
                ),
                "axiom_python_sha256": _package_sha256("axiom_rules_engine"),
                "axiom_dense_version": _optional_version("axiom-rules-engine-dense"),
                "axiom_dense_sha256": _file_sha256(
                    _module_path("axiom_rules_engine_dense")
                ),
                "receipt_version": importlib_metadata.version("receipt"),
                "receipt_sha256": _package_sha256("receipt"),
            },
        }
    except (
        subprocess.CalledProcessError,
        importlib_metadata.PackageNotFoundError,
        ModuleNotFoundError,
    ) as error:
        raise ImportError(
            "The Armenia model requires committed rulespec-am plus compatible "
            "source installs of Microcosm, Axiom Python/dense, and receipt."
        ) from error


def _frame_sha256(frame: Any) -> str:
    """Fingerprint actual tables and the authoritative household weights."""

    digest = sha256()
    for entity in am_datasets.ENTITIES:
        table = frame.table(entity)
        header = [entity, list(table.columns), [str(dtype) for dtype in table.dtypes]]
        digest.update(json.dumps(header, separators=(",", ":")).encode())
        digest.update(
            pd.util.hash_pandas_object(table, index=True)
            .to_numpy(dtype="<u8")
            .tobytes()
        )
    digest.update(frame.weights_for("household").values.astype("<f8").tobytes())
    return digest.hexdigest()
