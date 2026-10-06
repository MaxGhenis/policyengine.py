"""Verification helpers for externally signed RuleSpec source trees."""

import base64
import binascii
import json
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

APPLY_SCHEMA = "axiom-encode/applied-rulespec/v5"
APPLY_ALGORITHM = "ed25519-domain-v1"
APPLY_DOMAIN = b"axiom-encode/external-signer-sign/v2\x00apply_ed25519\x00"
AUTHORITATIVE_ROOTS = ("legislation", "policies", "regulations", "statutes")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _is_regular_below(root: Path, path: Path) -> bool:
    """Return whether ``path`` is a non-symlinked regular file below ``root``."""

    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, ValueError):
        return False
    return path.is_file() and resolved == path


def _safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a repository-relative string path.")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{label} must stay below the RuleSpec repository root.")
    if path.as_posix() != value or any(part in ("", ".") for part in path.parts):
        raise ValueError(f"{label} must use a canonical POSIX path.")
    return path


def _trusted_key_bytes(value: str) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A non-empty trusted apply public key is required.")
    encoded = value.strip().encode()
    if encoded.startswith(b"-----BEGIN PUBLIC KEY-----"):
        return encoded + (b"\n" if not encoded.endswith(b"\n") else b"")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError(
            "The trusted apply public key must be PEM or base64-encoded raw Ed25519 material."
        ) from exc
    if len(raw) != 32:
        raise ValueError("A base64 trusted apply public key must decode to 32 bytes.")
    return raw


def verify_signed_rulespec_tree(
    root: Path,
    *,
    jurisdiction: str,
    module: str,
    trusted_apply_public_key: str,
) -> dict[str, Any]:
    """Verify closed signed coverage for one jurisdiction's RuleSpec sources.

    The trust root is supplied explicitly by the caller. This consumer accepts
    only current ``axiom-encode/applied-rulespec/v5`` manifests and verifies
    their domain-separated Ed25519 signatures through the real ``receipt``
    package. Every executable YAML source must be covered at its exact digest.
    """

    root = root.resolve()
    module_path = _safe_relative_path(module, label="RuleSpec module")
    if module_path.suffix != ".yaml" or module_path.name.endswith(".test.yaml"):
        raise ValueError(
            "RuleSpec module must be an executable .yaml source, not a test or "
            "alternate-extension file."
        )
    allowed_prefixes = {
        PurePosixPath(jurisdiction, directory) for directory in AUTHORITATIVE_ROOTS
    }
    if not any(
        prefix == module_path.parent or prefix in module_path.parents
        for prefix in allowed_prefixes
    ):
        raise ValueError(
            f"RuleSpec module must be below {jurisdiction}/"
            f"{{{','.join(AUTHORITATIVE_ROOTS)}}}; programs are not an executable trust surface."
        )
    absolute_module = root / module_path
    if not _is_regular_below(root, absolute_module):
        raise FileNotFoundError(
            f"A signed {jurisdiction.upper()} RuleSpec module was not found at {module_path}."
        )

    programs = root / jurisdiction / "programs"
    if programs.is_symlink() or (
        programs.is_dir()
        and any(
            path.suffix.lower() in {".yaml", ".yml"} for path in programs.rglob("*")
        )
    ):
        raise ValueError(
            f"{jurisdiction}/programs is not an executable signed-source surface."
        )

    authored: dict[str, str] = {}
    for directory in AUTHORITATIVE_ROOTS:
        source_root = root / jurisdiction / directory
        if not source_root.is_dir():
            continue
        if source_root.is_symlink():
            raise ValueError(
                f"RuleSpec authority root must not be a symlink: {source_root}."
            )
        for path in sorted(source_root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"RuleSpec source path must not be a symlink: {path}.")
            if path.is_dir() or path.suffix.lower() not in {".yaml", ".yml"}:
                continue
            if path.suffix != ".yaml":
                raise ValueError(
                    f"Executable RuleSpec sources must use the exact .yaml suffix: {path}."
                )
            if not _is_regular_below(root, path):
                raise ValueError(f"RuleSpec source must not be a symlink: {path}.")
            relative = path.relative_to(root).as_posix()
            if relative.endswith(".test.yaml"):
                continue
            authored[relative] = _file_sha256(path)

    # The resolver can load any jurisdiction namespace beneath a configured
    # RuleSpec root. This consumer deliberately supports only the exact
    # Armenia namespace, so another resolvable authority tree cannot enter the
    # compiled closure without signed coverage.
    for namespace in sorted(root.iterdir()):
        if namespace.name == jurisdiction or not namespace.is_dir():
            continue
        for directory in AUTHORITATIVE_ROOTS:
            source_root = namespace / directory
            if not source_root.is_dir():
                continue
            executable = [
                path
                for path in source_root.rglob("*.yaml")
                if not path.name.endswith(".test.yaml")
            ]
            if executable:
                raise ValueError(
                    "Unsupported RuleSpec jurisdiction namespace has executable "
                    f"sources in the configured root: {namespace.name!r}."
                )
    if not authored:
        raise ValueError(
            f"No executable RuleSpec YAML exists below {jurisdiction}'s authoritative roots."
        )
    if module_path.as_posix() not in authored:
        raise ValueError(
            "The selected RuleSpec module is not an executable source in the "
            f"signed {jurisdiction.upper()} authority tree: {module}."
        )

    manifest_root = root / ".axiom" / "encoding-manifests" / jurisdiction
    manifests = sorted(manifest_root.rglob("*.json")) if manifest_root.is_dir() else []
    if not manifests:
        raise ValueError(
            f"No signed apply manifest covers the {jurisdiction.upper()} RuleSpec tree."
        )

    try:
        from receipt.sign import (
            KeyringSpec,
            KeySpec,
            raw_public_key_sha256,
            verify_threshold,
        )
    except ImportError as exc:
        raise ImportError(
            "Signed RuleSpec verification requires the Axiom receipt package."
        ) from exc

    public_key = _trusted_key_bytes(trusted_apply_public_key)
    fingerprint = raw_public_key_sha256(public_key)
    key_id = f"sha256:{fingerprint}"
    keyring = KeyringSpec(
        keys=(
            KeySpec(
                key_id=key_id,
                fingerprint=fingerprint,
                scheme="raw-sha256",
            ),
        ),
        threshold=1,
    )

    coverage: dict[str, str] = {}
    manifest_receipts: list[dict[str, Any]] = []
    for manifest_path in manifests:
        if not _is_regular_below(root, manifest_path):
            raise ValueError(
                f"Signed apply manifest must not be a symlink: {manifest_path}."
            )
        try:
            payload = json.loads(manifest_path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid signed apply manifest: {manifest_path}."
            ) from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != APPLY_SCHEMA
        ):
            raise ValueError(
                f"Unsupported signed apply manifest schema at {manifest_path}."
            )
        signature = payload.get("signature")
        if not isinstance(signature, dict) or set(signature) != {
            "algorithm",
            "key_id",
            "value",
        }:
            raise ValueError(
                f"Malformed signed apply manifest signature at {manifest_path}."
            )
        if signature["algorithm"] != APPLY_ALGORITHM:
            raise ValueError(
                f"Unsupported apply signature algorithm at {manifest_path}."
            )
        if signature["key_id"] != key_id:
            raise ValueError(
                f"Signed apply manifest at {manifest_path} was not signed by the pinned key."
            )
        try:
            raw_signature = base64.b64decode(signature["value"], validate=True)
        except (TypeError, binascii.Error) as exc:
            raise ValueError(
                f"Malformed apply signature bytes at {manifest_path}."
            ) from exc
        unsigned = dict(payload)
        del unsigned["signature"]
        verify_threshold(
            _canonical_json(unsigned),
            {key_id: raw_signature},
            {key_id: public_key},
            keyring,
            domain=APPLY_DOMAIN,
            label=manifest_path.relative_to(root).as_posix(),
            allow_legacy=False,
        )

        applied = payload.get("applied_files")
        if not isinstance(applied, list) or not applied:
            raise ValueError(
                f"Signed apply manifest has no applied files: {manifest_path}."
            )
        applied_receipts: list[dict[str, str]] = []
        for item in applied:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                raise ValueError(f"Malformed applied-file entry at {manifest_path}.")
            relative = _safe_relative_path(
                item["path"], label="Signed apply manifest path"
            )
            if (
                len(relative.parts) < 3
                or relative.parts[0] != jurisdiction
                or relative.parts[1] not in AUTHORITATIVE_ROOTS
                or relative.suffix != ".yaml"
            ):
                raise ValueError(
                    "Signed apply manifest paths must be exact .yaml files below "
                    f"{jurisdiction}/{{{','.join(AUTHORITATIVE_ROOTS)}}}."
                )
            expected = item["sha256"]
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)
            ):
                raise ValueError(f"Invalid applied-file SHA-256 at {manifest_path}.")
            path = root / relative
            if not _is_regular_below(root, path):
                raise ValueError(
                    f"Signed RuleSpec file is missing or non-regular: {relative}."
                )
            actual = _file_sha256(path)
            if actual != expected:
                raise ValueError(f"Signed RuleSpec file digest changed: {relative}.")
            name = relative.as_posix()
            if name in coverage and coverage[name] != actual:
                raise ValueError(
                    f"Conflicting signed coverage for RuleSpec file: {name}."
                )
            coverage[name] = actual
            applied_receipts.append({"path": name, "sha256": actual})
        manifest_receipts.append(
            {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": _file_sha256(manifest_path),
                "key_id": key_id,
                "applied_files": applied_receipts,
            }
        )

    uncovered = sorted(
        name for name, digest in authored.items() if coverage.get(name) != digest
    )
    if uncovered:
        raise ValueError(
            "Every executable RuleSpec source needs exact signed apply manifest "
            f"coverage; uncovered: {uncovered}."
        )
    if coverage.get(module_path.as_posix()) != authored[module_path.as_posix()]:
        raise ValueError(
            f"The selected RuleSpec module lacks signed coverage: {module}."
        )

    tree_digest = sha256()
    for name, digest in sorted(authored.items()):
        tree_digest.update(name.encode())
        tree_digest.update(b"\x00")
        tree_digest.update(digest.encode())
        tree_digest.update(b"\x00")
    return {
        "schema": "policyengine/signed-rulespec-source/1",
        "jurisdiction": jurisdiction,
        "module": module_path.as_posix(),
        "module_sha256": authored[module_path.as_posix()],
        "source_tree_sha256": tree_digest.hexdigest(),
        "trusted_apply_key_id": key_id,
        "manifests": manifest_receipts,
    }
