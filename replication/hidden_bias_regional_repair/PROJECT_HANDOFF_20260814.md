# Project handoff: paper, experiments, banks, and migration

Last updated: 2026-08-14 14:00 UTC

This is the primary resumption document for `dml2` or any successor driver.
It explains the scientific project, separates public evidence from exploratory
work, records the current fitted-value banks, and identifies every durable
repository. Chat history and devvm-local paths are not sources of truth.

## What we are doing and why

The paper studies missing-outcome mean estimation under low response and weak
overlap. Observable validation criteria can compare fit and score behavior,
but they cannot generally rank the hidden bias of competing orthogonal
estimators. The constructive response is therefore not to keep selecting among
experts. It is to offer a zero-containing repair path and deploy a repair only
when held-out observable evidence supports the move.

The August 12 paper release established a certified, reproducible Section 4
result for the frozen expert-specific interfaces. The August 14 work asks a
harder follow-up question: can one public repair function and one safety rule
operate uniformly across AIPW, TMLE, C-TMLE, faithful Cui selective ML, and Ma
DR-BC on a common MAR estimand? To answer that honestly, we are building a
complete fitted-value bank before inspecting full-bank features. The bank is
an exploratory observable atlas, not a new paper result.

## Where everything lives

| Surface | Purpose | Durable location |
|---|---|---|
| Public replication code | Algorithms, generators, tests, protocols | `https://github.com/atavory/ney` |
| Paper Overleaf | Manuscript prose, generated paper tables, paper status | project `69edff47028a983c95b7fcc2` |
| Data Overleaf | Accepted raw CSVs, provenance, aggregators, verification, compact handoffs | project `69e603c91cd3e08b56a20933` |
| Mailbox/control plane | Cross-driver messages and heartbeats | `aai_research_tlv/tree/atavory/dml_reference_transfer/canonical_20260809_v1/` |
| Frozen Cartesian source | v3 source bundle and hashes | `aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/source/` |
| Completed v3 local partition | dml Kang--Schafer/alignment rows | `aai_research_tlv/tree/atavory/dml_reference_transfer/unified_cartesian_20260814/dml_ks_alignment_v3/` |
| Full fitted-value bank | Manifests, remote shard, and evacuation slices | `aai_research_tlv/tree/atavory/dml_reference_transfer/full_expert_bank_handoff_20260814/` |
| Cui published control bank | Handoff, statuses, local archive, and eventual remote archive | `aai_research_tlv/tree/atavory/dml_reference_transfer/cui_published_expert_bank_handoff_20260814/` |
| Devvm evacuation inventory | Cross-reference for everything removed from the dying host | `aai_research_tlv/tree/atavory/dml_reference_transfer/devvm50798_evacuation_20260814/` |

Large JSON/NPZ fitted-value banks belong on Manifold, not in Git or Overleaf.
The data Overleaf stores compact inventories, CSV summaries, verification
reports, and source hashes that point to those Manifold objects. The paper
Overleaf stores only manuscript-facing material and a concise status pointer.

## Public-facing versus exploratory

### Certified and public-facing

The frozen August 12 Section 4 release remains authoritative:

- data bundle: `support_csv/dml_section4_release_20260812_v1/` in the data
  Overleaf;
- sensitivity atlas: `support_csv/dml_section4_c_atlas_20260812_v2/`;
- figures: `support_csv/dml_section4_c_atlas_figures_20260812_v3/`;
- public replication starting point: GitHub `atavory/ney`;
- paper tracker: `PROJECT_STATUS.md` in the paper Overleaf.

Those artifacts passed the fail-closed coverage/provenance workflow. They may
support manuscript claims. Their values must not be silently replaced by any
August 14 exploratory result.

### Exploratory and not paper-facing

The unified Cartesian baseline, direction pilot, full fitted-value bank, and
Cui control bank are research infrastructure. They may diagnose mechanisms and
suggest one uniform rule, but they cannot enter the abstract, paper tables, or
conclusions. A rule selected with these banks must be hash-frozen and evaluated
on a fresh-seed full Cartesian confirmation bank before it becomes evidence.

## Experiment lineages and why each exists

| Lineage | Purpose | Current interpretation |
|---|---|---|
| Certified Section 4 release | Support the submitted paper's frozen residual/projection interfaces and `c` atlas | Public-facing and unchanged |
| Historical residual-v2 and regional-residual-v2 runs | Diagnose global versus regional constructions and published benchmark behavior | Retained history; not the universal-rule experiment |
| Complete residual matrix | Compare adapters broadly and expose estimator-specific behavior | Diagnostic; motivated a common API |
| Unified Cartesian v2 | First attempt at a single function across experts | Invalidated after 263 jobs because proposal construction discarded `RepairParameters`; no row is eligible |
| Unified Cartesian v3 baseline | Five experts across 34 common MAR cells with one frozen rule | Exploratory measurement; uses historical `balanced_mse`, not theorem-aligned `aipw_variance` |
| Frozen direction pilot | Test whether repeated whole-pipeline fit direction reproduces for AIPW/TMLE | Failed its immutable 0.70 threshold; recorded negative diagnostic |
| Full fitted-value bank | Freeze every expert/cell realization and 20 repeated cross-fits for observable feature analysis | Complete the entire 16,320-entry bank before analysis; truth remains sealed |
| Cui published control bank | Add the two published Cui scenarios where faithful Cui is expected to be an important control | 3,840 entries, split 1,918 dml / 1,922 dml2; must be complete before any whole-bank conclusion |
| Exact paired knob grid | Identify effects of SE threshold and shrink constant without cross-lineage confounding | Future: SE `{1,2.83}` by `c={0,1,2,3,4,5,6,8}` on fixed rows/seeds |
| Uniform safety variants | Test move bounds, ESS/max-weight/kurtosis trust gates, robust gates, and symmetric winsorization | Future; parameters must be identical across experts |
| Fresh-seed confirmation | Test the single rule chosen from the exploratory atlas | Mandatory before new paper claims |

The native Ma published DiD study remains a separate-estimand external study.
It must not be pooled into the common missing-outcome-mean Cartesian matrix.

## Frozen bank organization

The full bank crosses five experts with all 34 common MAR cells: eight
Kang--Schafer cells, ten alignment cells, twelve real-covariate cells, and four
nonlinear-MAR anchor cells. It contains 170 expert/cell pairs and 96 dataset
realizations per pair, for 16,320 entries.

Each entry stores inspectable JSON plus compressed NPZ, never a pickled model.
It freezes the full/reference fit, folds, reference score and endpoint,
candidate path, selected candidate, shrink weight, seeds, source/configuration
hashes, and 20 whole-pipeline repeated-crossfit fits. `delete_blocks=0`, so the
full bank supports repeated-crossfit direction reproducibility but not genuine
leave-block-out sensitivity.

The immutable `development` and `validation` labels are provenance fields.
The first atlas deliberately analyzes observable behavior across the complete
bank and all five experts. It is not a confirmatory expert holdout. Simulation
truth files are sealed during this analysis.

Full-bank manifest SHA-256:
`5b1e62644be799cddfa6f49e57dddd318ecc2c6a1a4f316bdb2a4eb3be847377`.
Split SHA-256:
`61aa6e4f62fb4ac032a8a88ace80e9b07233676167872e24e1c9a6ed7e34a69e`.

The Cui control bank contains eight cells, 768 dataset realizations, and five
experts, for 3,840 entries. Manifest SHA-256:
`1639ee39a1570068e6f0f817ae9120579d4f55cea6e911aa6cb8c831772a1f28`.
Split SHA-256:
`7417dd46de852f63674db89b1df1b1297ac11b185011f33c9a07127d35461687`.

## Execution state at 2026-08-14 14:00 UTC

- Full bank, dml owner: 9,046/9,234 complete, 64 active, zero failures.
- Full bank, dml2 owner: 7,086/7,086 present and audited. The remote 1,790-entry
  completion archive is published with SHA-256
  `7a859c1b6c6c6e15c2dde7b39bd6080457414349dda6ed3daa9d70ffadf3553f`.
- A mistaken duplicate dml2 attempt was stopped at 950/1,790 after the remote
  archive was verified. It is not authoritative.
- Cui bank, dml owner: 1,918/1,918 complete. Its Manifold archive is
  `dml_cui_published_dml_1918_entries_20260814.tar.zst`, independently
  round-trip verified with SHA-256
  `ce14308a2156f2dfb9f6a2e13e04ca5605524c3ea4218cd1a37821a46fc663d9`.
- Cui bank, dml2 owner: 1,292/1,922 complete, 64 active, zero failures at
  13:58 UTC. Status is published beside the handoff.
- The current host is being retired. Its bank entries are on `/dev/shm`; swap
  pressure has triggered repeated OOM kills. Manifold durability and verified
  inventories take priority over retaining convenient local copies.

This is a timestamped execution snapshot, not the final completion
certificate. The final inventory must replace counts with exact terminal
coverage and archive hashes.

## Scientific hard rules

1. Do not mix certified August 12 paper values with August 14 exploratory
   banks.
2. Do not inspect simulation truth while constructing observable atlas
   features or a candidate rule.
3. Do not aggregate or compare full-bank behavior until all 170 expert/cell
   pairs and all required entries pass exact coverage.
4. Do not call the `balanced_mse` v3 baseline validation of the population
   observable-risk theorem.
5. Do not relabel the failed direction pilot, invalid v2 rows, duplicate-run
   rows, or partial Cui bank as evidence.
6. Apply every safety feature and threshold identically across experts; no
   estimator-specific rescue exceptions.
7. Freeze and hash the selected rule before generating a fresh-seed
   confirmation bank.
8. Paper numbers exist only after the seven-step acceptance rule in data
   Overleaf `REPRODUCIBILITY.md` passes.

## What dml2 owns after this devvm is retired

1. Finish the 1,922-entry Cui shard and publish a terminal status, archive, and
   SHA-256 to the Cui handoff prefix.
2. Ingest the final dml full-bank slices and verify that their union with the
   7,086 dml2 entries covers the 16,320-entry manifest exactly once.
3. Publish a final artifact inventory CSV and restore instructions; never rely
   on a local path from the retired host.
4. Preserve the public/certified boundary in GitHub and both Overleaf projects.
5. Only after exact full-bank and Cui-bank coverage, run an observable-only
   atlas summarizer and publish its code, configuration, and hashes.
6. Treat any proposed rule as exploratory, freeze it, and request a new-seed
   confirmation rather than editing the existing paper release.

## Resumption order

1. Read this document, `UNIFIED_REPAIR_RESEARCH_LOG_20260814.md`,
   `unified_cartesian_protocol_20260814.md`, and
   `EXPERT_CV_API_AND_DATASETS_20260814.md`.
2. Read data Overleaf `REPRODUCIBILITY.md` and paper Overleaf
   `PROJECT_STATUS.md`.
3. Verify Manifold objects against `artifact_inventory.csv` and adjacent
   checksum files.
4. Recompute manifest coverage from identities, not directory size or chat
   counts.
5. Resume only the explicitly incomplete shard or publication step.
