# Frozen expert-bank direction pilot

> Historical status (2026-08-14): this locked pilot was completed and failed
> its prespecified `C >= 0.70` criterion. It is retained unchanged as an audit
> record. By subsequent user decision, it does not gate construction of the
> full bank. The full bank is completed first and used as an exploratory
> observable atlas across all datasets and experts; any confirmatory test of a
> later frozen rule uses new independent seeds.

Frozen before pilot fitting or inspection. This pilot asks one question only:
can whole-pipeline direction reproducibility distinguish the historically
well-aimed AIPW/Kang--Schafer repair from the historically randomly aimed
TMLE/Kang--Schafer repair using observables alone?

## Units and split

- Eight Kang--Schafer strata: `cc`, `ci`, `ic`, and `ii`, each at
  `n in {200,1000}`.
- Twelve fresh dataset realizations per stratum, shared by AIPW and TMLE:
  96 datasets and 192 algorithm-by-dataset bank entries.
- Within each stratum, the 12 identities are ranked by
  `SHA256(salt|dataset_identity)`. The first six are development and the last
  six validation. The enumerated split JSON and its hash are authoritative.
- The validation bank is not queried until the artifact code and primary
  statistic pass development checks and their hashes are frozen.

## Frozen fitted values

Each bank entry stores no executable estimator or pickle. JSON plus compressed
NPZ contains the observed dataset, fold IDs, reference influence score,
initial and reference outcome fits, fitted propensity, every candidate score
and endpoint for gamma `{0,.25,.5,1}`, selected gamma, shrink weight, seeds,
configuration, software versions, and exact source hash. Simulation truth is
retained in a separately named, hash-covered evaluation NPZ and is inaccessible
to the observable diagnostic.

For each entry, fit the complete expert-plus-candidate pipeline once, then
refit it under 20 independently seeded repeated cross-fits and 10 fixed
delete-block perturbations. The repeated cross-fits define the primary
direction statistic. Delete-block sensitivity is secondary and cannot change
the pilot decision.

## One primary observable statistic

For repeated-crossfit moves `d_b`, with an exact stand-down represented by
zero, define

`S = |sum_b sign(d_b)| / 20`.

Thus one isolated nonzero move scores `0.05`, not `1`; consistent nonzero
direction scores near one; sign instability and stand-down score near zero.
No magnitude, ordinary SE, propensity diagnostic, truth, or expert identity
enters `S`.

The sole validation decision statistic is paired concordance across the 48
untouched validation datasets:

`C = mean[1{S_AIPW>S_TMLE} + 0.5*1{S_AIPW=S_TMLE}]`.

The resampling architecture passes this pilot if `C >= 0.70`. All eight
stratum-specific values, activation counts, and the full distribution of `S`
will be reported regardless. No alternative stability statistic may replace
`S` after validation is opened. Score-vector correlation and delete-block
sensitivity are explicitly secondary mechanism diagnostics.

## Frozen fitting procedure

- Common global residual repair; gamma `{0,.25,.5,1}`; nominal 1-SE historical
  selector; positive-part shrinkage `c=2`.
- Estimated propensity with fixed floor `0.05`; XGBoost 3.4.0; three inner
  folds; historical `balanced_mse` selection loss.
- These choices reproduce the procedure being diagnosed; the pilot does not
  endorse them as the final paper procedure.

If the primary validation criterion fails, direction resampling is not an
adequate safety certificate and the full 57-million-observation bank is not
launched. If it passes, the same frozen-value schema and a newly declared
full-matrix split are used before testing any final gate.
