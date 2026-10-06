# Armenia Axiom source model

PolicyEngine provides an opt-in Armenia model at
`policyengine.tax_benefit_models.am`. It runs a selected RuleSpec-AM module
through the real Axiom dense runtime over person and household tables loaded by
Microcosm. It does not contain a parallel Python implementation of Armenian
policy.

This surface is fail-closed. Constructing a model requires all of the following:

- a committed RuleSpec-AM checkout containing the selected module;
- exact `axiom-encode/applied-rulespec/v5` coverage for every executable
  Armenian RuleSpec source;
- an independently pinned Ed25519 apply public key, which the `receipt` package
  uses to verify each domain-separated manifest signature;
- compatible source installs of `microcosm-frame`, `axiom-rules-engine`, its
  native dense extension, and `receipt`; and
- a local `populace_am_YEAR.h5` whose bytes and canonical repository coordinates
  match an explicitly pinned Microcosm release-manifest SHA-256.

The trust key is a consumer input. Do not copy it from the manifest being
verified. A selected module below `am/programs` is refused because that tree is
not an executable signed-source authority. The adapter also refuses executable
authority trees under any namespace other than exact `am`, because Axiom could
otherwise resolve unsigned source outside the closed signed coverage set.

## Usage

```python
from policyengine import Simulation
from policyengine.tax_benefit_models.am import (
    AxiomArmeniaModelVersion,
    PopulaceArmeniaDataset,
)

dataset = PopulaceArmeniaDataset(
    name="local Armenia release",
    description="Locally supplied, release-bound Microcosm artifact",
    filepath="/secure/releases/populace_am_2024.h5",
    year=2024,
    weight_kind="calibrated",
    release_manifest_path="/secure/releases/release_manifest.json",
    release_manifest_sha256="<independently pinned lowercase SHA-256>",
)
model = AxiomArmeniaModelVersion(
    rulespec_root="/secure/checkouts/rulespec-am",
    module="am/statutes/<selected-module>.yaml",
    trusted_apply_public_key="<independently pinned Ed25519 public key PEM>",
    policy_period={
        "start": "2024-01-01",
        "end": "2024-12-31",
        "kind": "year",
    },
)

simulation = Simulation(dataset=dataset, tax_benefit_model_version=model)
simulation.run()
```

The model compiles the signed module with Axiom, derives its registered
PolicyEngine `Variable` objects from Axiom's entity, dtype, period, unit, ID,
and source metadata, and materializes those outputs through Microcosm. The
output uses `AMYearData`, so PolicyEngine aggregates remain weighted
`MicroSeries` operations and cross-entity analysis continues through
`map_to_entity`; callers do not manually multiply by person or household
weights.

Each output carries a self-contained receipt with the signed RuleSpec tree,
runtime package and native-extension identities, exact input frame and HDF5
hashes, Microcosm release coordinates, explicit period, output metadata, and
household weight kind.

For source datasets, every `to_frame()` call deliberately discards any prior
in-memory table value, reauthenticates the release manifest and HDF5 digest, and
reloads the entity tables. This makes the file-and-manifest pair the trust
boundary rather than mutable Python state. Callers must explicitly label the
household weights as `design` or `calibrated`; the adapter never infers that
certification claim.

## Verification

The real native execution test is opt-in because the Axiom dense extension is
not yet a managed PolicyEngine dependency:

```shell
POLICYENGINE_SKIP_COUNTRY_IMPORTS=1 \
RUN_AM_AXIOM_INTEGRATION=1 \
uv run --no-sync pytest --noconftest -q tests/test_am_axiom_source_model.py
```

It skips unless compatible source-only Microcosm, Axiom Python/native, and
receipt runtimes are installed. The fixture is a signed neutral identity rule,
not Armenian law.

## Deliberate limits

This interface does not select or download a managed Armenia dataset. It does
not state that a certified `populace-am` release currently exists, and every run
receipt records `managed_population: false`. It also does not state that the
selected module represents complete Armenian current law; receipts record
`current_law_complete: false`.

PolicyEngine policy and dynamic dictionaries are rejected. A counterfactual is
a separately authored and signed RuleSpec artifact, not a PolicyEngine
parameter overlay. Modules that declare cross-entity dense relations still
stop at the Axiom adapter's explicit relation-batch `NotImplementedError` until
Microcosm supplies that bridge. These constraints keep missing runtime
capability visible instead of replacing Axiom with handwritten policy logic.
