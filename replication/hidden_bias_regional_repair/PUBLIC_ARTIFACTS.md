# Public Artifact Index

This file records the public-facing code locations for the hidden-bias
regional-repair project. Generated CSVs, large fitted banks, and run logs are
not part of this public code release.

## CUI Published Source

The Cui published experiments are present in this public repository:

- `scripts/build_frozen_cui_published_entry.py`
- `scripts/section4_cui_published_experiments.py`

Current public SHAs:

- CUI entry builder:
  `99327ea3aaa222ef8fe2b031c3613a064db551cb116059cfe73c6fd9116211f9`
- CUI experiment adapter:
  `c4fa4e83dd6112ecdd2c4dfce1fc6cd3d3e0c4a97d4f48b093c78f4f6ba31d70`

These are the public entry points for the CUI published experiments.

## Canonical Region-Gated Repair Source

The canonical source for the 2026-08-17 low-response/high-response ablation is:

- `scripts/validated_reference_transfer_canonical_region_if_library.py`

SHA:

`af471e392a2eed8f68ab5a88d03bc3bf221b8a53005337530f7d3a96cb6a236b`

This source gates the deployed `if_library` residual correction by the selected
analysis region:

`candidate_outcome = base_outcome + gamma * correction * analysis_region`

That is the required source for controlled low-response versus high-response
comparisons. Older runs where the deployed correction ignored the region are
not valid high-response ablation results.

## Generated Data

Generated CSVs, fitted-entry banks, and run logs are not committed to this
public code repository.
