"""Certified-input boundary for source-only Armenian Microcosm frames."""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from microdf import MicroDataFrame
from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from policyengine.core import Dataset, YearData
from policyengine.tax_benefit_models.axiom import (
    EntityTableJSONCodec,
    frame_from_household_tables,
    shared_entity_schema,
    weighted_tables_from_frame,
)

ENTITIES = ("person", "household")
CODEC = EntityTableJSONCodec(
    schema="policyengine/am-entity-table/1", country_label="Armenia"
)


class AMYearData(YearData):
    """Person and household tables with Frame-resolved effective weights."""

    person: MicroDataFrame
    household: MicroDataFrame

    @field_serializer("person", "household", when_used="json")
    def serialize_entity_table(self, value: MicroDataFrame) -> dict[str, Any]:
        return CODEC.encode_table(value)

    @field_validator("person", "household", mode="before")
    @classmethod
    def restore_entity_table(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, dict):
            return CODEC.decode_table(value, info.field_name)
        return value

    @property
    def entity_data(self) -> dict[str, MicroDataFrame]:
        return {entity: getattr(self, entity) for entity in ENTITIES}


class MicrocosmReleaseReceipt(BaseModel):
    """Exact local release coordinates authenticated before HDF5 loading."""

    receipt_schema: Literal["policyengine/microcosm-release-input/1"] = Field(
        default="policyengine/microcosm-release-input/1",
        serialization_alias="schema",
    )
    country: Literal["am"] = "am"
    release_manifest_sha256: str
    build_id: str
    dataset: str
    artifact_path: str
    artifact_sha256: str
    artifact_revision: str
    artifact_repository: Literal["policyengine/populace-am"] = (
        "policyengine/populace-am"
    )


def _load_frame_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        from microcosm.frame import Frame, WeightKind, Weights
    except ImportError as error:
        raise ImportError(
            "The Armenia adapter requires a compatible source install of "
            "microcosm-frame."
        ) from error
    return Frame, WeightKind, Weights, shared_entity_schema(("household",))


def _load_dataset_reader() -> Any:
    try:
        from microcosm.frame.adapters.axiom import AxiomEntityTableDataset
    except ImportError as error:
        raise ImportError(
            "The Armenia adapter requires Microcosm's Axiom entity-table HDF5 reader."
        ) from error
    return AxiomEntityTableDataset


def _validate_household_nesting(tables: dict[str, pd.DataFrame]) -> None:
    required = {
        "person": ("person_id", "person_household_id"),
        "household": ("household_id", "household_weight"),
    }
    for entity, columns in required.items():
        for column in columns:
            if column not in tables[entity]:
                raise ValueError(f"Armenia {entity} table is missing {column!r}.")
            if tables[entity][column].isna().any():
                raise ValueError(f"Armenia {entity}.{column} must not contain nulls.")
    for entity in ENTITIES:
        identifier = f"{entity}_id"
        if tables[entity][identifier].duplicated().any():
            raise ValueError(f"Armenia {identifier} values must be unique.")
    household_ids = set(tables["household"]["household_id"])
    membership_ids = set(tables["person"]["person_household_id"])
    if not membership_ids.issubset(household_ids):
        raise ValueError("Armenia person_household_id contains an unknown household.")
    weights = tables["household"]["household_weight"]
    if (
        not pd.api.types.is_numeric_dtype(weights.dtype)
        or not np.isfinite(weights.to_numpy(dtype=float)).all()
        or (weights < 0).any()
    ):
        raise ValueError("Armenia household weights must be finite and non-negative.")


def frame_from_tables(
    tables: dict[str, pd.DataFrame],
    *,
    allow_effective_weights: bool = False,
    weight_kind: Literal["design", "calibrated"],
    metadata: Optional[dict[str, Any]] = None,
) -> Any:
    """Validate Armenia's entity structure and construct a weighted Frame."""

    return frame_from_household_tables(
        tables,
        entities=ENTITIES,
        runtime_loader=_load_frame_runtime,
        country_label="Armenia",
        allow_effective_weights=allow_effective_weights,
        weight_kind=weight_kind,
        metadata=metadata,
        table_validator=_validate_household_nesting,
    )


def year_data_from_frame(frame: Any) -> AMYearData:
    """Expose effective weights through PolicyEngine MicroDataFrames."""

    return AMYearData(**weighted_tables_from_frame(frame, ENTITIES))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


class PopulaceArmeniaDataset(Dataset):
    """A local, manifest-bound Armenia HDF5 input for the Axiom adapter.

    This class does not download or select a managed population release. A
    caller must supply both a local HDF5 artifact and the exact Microcosm
    release manifest digest that authenticates it. Derived output datasets are
    kept in memory and cannot be reused as rules input.
    """

    data: Optional[AMYearData] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_sha256: Optional[str] = None
    weight_kind: Literal["design", "calibrated"]
    release_manifest_path: Optional[str] = None
    release_manifest_sha256: Optional[str] = None
    release_receipt: Optional[MicrocosmReleaseReceipt] = None

    @model_validator(mode="after")
    def require_certified_source_coordinates(self) -> "PopulaceArmeniaDataset":
        if self.is_output_dataset:
            return self
        if not self.filepath or not self.release_manifest_path:
            raise ValueError(
                "An Armenia source dataset requires a certified Microcosm release "
                "HDF5 path and release manifest coordinates."
            )
        try:
            _require_sha256(
                self.release_manifest_sha256, label="release manifest SHA-256"
            )
        except ValueError as exc:
            raise ValueError(
                "An Armenia source dataset requires a certified Microcosm release "
                "manifest SHA-256."
            ) from exc
        return self

    def _load_release_receipt(self, artifact_path: Path) -> MicrocosmReleaseReceipt:
        assert self.release_manifest_path is not None
        assert self.release_manifest_sha256 is not None
        manifest_path = Path(self.release_manifest_path)
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("Armenia release manifest must be a regular file.")
        manifest_bytes = manifest_path.read_bytes()
        actual_manifest_sha = sha256(manifest_bytes).hexdigest()
        if actual_manifest_sha != self.release_manifest_sha256:
            raise ValueError("Armenia release manifest SHA-256 does not match.")
        try:
            manifest = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Armenia release manifest is not valid JSON.") from exc
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("Armenia requires Microcosm release manifest schema 1.")
        build = manifest.get("build")
        build_id = build.get("build_id") if isinstance(build, dict) else None
        if not isinstance(build_id, str) or not build_id:
            raise ValueError("Armenia release manifest has no build.build_id.")
        expected_dataset = f"populace_am_{self.year}"
        defaults = manifest.get("default_datasets")
        if (
            not isinstance(defaults, dict)
            or defaults.get("national") != expected_dataset
        ):
            raise ValueError(f"Armenia release default must be {expected_dataset!r}.")
        artifacts = manifest.get("artifacts")
        artifact = (
            artifacts.get(expected_dataset) if isinstance(artifacts, dict) else None
        )
        if not isinstance(artifact, dict):
            raise ValueError("Armenia release manifest has no national artifact.")
        expected_path = f"{expected_dataset}.h5"
        if (
            artifact.get("kind") != "microdata"
            or artifact.get("path") != expected_path
            or artifact.get("repo_id") != "policyengine/populace-am"
            or artifact.get("revision") != build_id
            or artifact_path.name != expected_path
        ):
            raise ValueError(
                "Armenia release manifest does not bind the canonical populace-am artifact."
            )
        artifact_sha = _require_sha256(
            artifact.get("sha256"), label="Armenia artifact SHA-256"
        )
        if _file_sha256(artifact_path) != artifact_sha:
            raise ValueError(
                "Armenia HDF5 does not match the release artifact SHA-256."
            )
        return MicrocosmReleaseReceipt(
            release_manifest_sha256=actual_manifest_sha,
            build_id=build_id,
            dataset=expected_dataset,
            artifact_path=expected_path,
            artifact_sha256=artifact_sha,
            artifact_revision=build_id,
        )

    def load(self) -> None:
        if self.filepath is None:
            raise ValueError(
                "Cannot load an Armenia source dataset without a filepath."
            )
        path = Path(self.filepath)
        if not path.is_file() or path.is_symlink():
            raise ValueError("Armenia HDF5 input must be a regular file.")
        before = path.stat()
        receipt = self._load_release_receipt(path)
        if (
            self.source_sha256 is not None
            and self.source_sha256 != receipt.artifact_sha256
        ):
            raise ValueError(
                "Armenia source SHA-256 differs from the authenticated release artifact."
            )
        if self.release_receipt is not None and self.release_receipt != receipt:
            raise ValueError(
                "Armenia saved release receipt differs from the authenticated manifest."
            )
        with pd.HDFStore(path, mode="r") as store:
            keys = {key.lstrip("/") for key in store.keys()}
            if keys != {*ENTITIES, "_time_period"}:
                raise ValueError(
                    "Armenia HDF5 must contain person, household, and _time_period only."
                )
            period = store["_time_period"]
            if (
                len(period) != 1
                or not pd.api.types.is_integer_dtype(period.dtype)
                or period.iloc[0] != self.year
            ):
                raise ValueError(f"Armenia dataset period must equal {self.year}.")
            attrs = store.get_storer("_time_period").attrs
            encoded = getattr(attrs, "policyengine_metadata_json", None)
            if isinstance(encoded, bytes):
                encoded = encoded.decode()
            metadata = (
                json.loads(encoded) if encoded is not None else dict(self.metadata)
            )
            if not isinstance(metadata, dict):
                raise ValueError("Armenia dataset metadata must be a JSON object.")
            if self.metadata and self.metadata != metadata:
                raise ValueError(
                    "Armenia dataset metadata differs from its HDF5 artifact."
                )
        reader = _load_dataset_reader()(file_path=path)
        if reader.time_period != self.year:
            raise ValueError(f"Armenia dataset period must equal {self.year}.")
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError("Armenia input artifact changed while it was loaded.")
        frame = frame_from_tables(
            reader.tables, weight_kind=self.weight_kind, metadata=metadata
        )
        self.data = year_data_from_frame(frame)
        self.metadata = metadata
        self.source_sha256 = receipt.artifact_sha256
        self.release_receipt = receipt

    def to_frame(self) -> Any:
        """Reauthenticate and reload sources before returning a fresh Frame.

        Source datasets intentionally ignore previously loaded in-memory data:
        each call rechecks the release manifest and exact HDF5 bytes. Output
        datasets remain memory-only and cannot be reused as rules input.
        """

        if not self.is_output_dataset:
            self.load()
        elif self.data is None:
            raise ValueError("An Armenia output dataset has no in-memory data.")
        assert self.data is not None
        return frame_from_tables(
            self.data.entity_data,
            allow_effective_weights=True,
            weight_kind=self.weight_kind,
            metadata=self.metadata,
        )
