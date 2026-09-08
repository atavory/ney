# Selected-Gamma Theorem Simulation

This finite-support simulation checks the one-standard-error weighted-residual
selector against the selected-candidate oracle inequality.  The theorem
object is fresh-candidate risk after selection.  The same-sample squared
error is reported only as a diagnostic.

## Summary

- correct_clear, n=500: fresh violations 0/6000; same-sample over RHS 1582/6000; selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000; mean fresh risk 0.00741016; mean same-sample squared error 0.00744867; mean RHS 0.00949617.
- correct_clear, n=2000: fresh violations 0/3000; same-sample over RHS 866/3000; selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000; mean fresh risk 0.00185254; mean same-sample squared error 0.00182863; mean RHS 0.00211065.
- correct_clear, n=8000: fresh violations 0/1000; same-sample over RHS 315/1000; selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000; mean fresh risk 0.000463135; mean same-sample squared error 0.000467651; mean RHS 0.000496607.
- correct_near_tie, n=500: fresh violations 0/6000; same-sample over RHS 1872/6000; selected shares gamma0=0.824, gamma0.25=0.077, gamma0.5=0.072, gamma1=0.026; mean fresh risk 0.00983015; mean same-sample squared error 0.0100661; mean RHS 0.0103924.
- correct_near_tie, n=2000: fresh violations 0/3000; same-sample over RHS 949/3000; selected shares gamma0=0.811, gamma0.25=0.131, gamma0.5=0.058, gamma1=0.000; mean fresh risk 0.00245606; mean same-sample squared error 0.00241093; mean RHS 0.00252696.
- correct_near_tie, n=8000: fresh violations 0/1000; same-sample over RHS 300/1000; selected shares gamma0=0.832, gamma0.25=0.166, gamma0.5=0.002, gamma1=0.000; mean fresh risk 0.000613898; mean same-sample squared error 0.000561635; mean RHS 0.000622715.
- misspecified_tradeoff, n=500: fresh violations 0/6000; same-sample over RHS 306/6000; selected shares gamma0=0.000, gamma0.25=0.001, gamma0.5=0.003, gamma1=0.996; mean fresh risk 0.00755176; mean same-sample squared error 0.00758306; mean RHS 0.0287081.
- misspecified_tradeoff, n=2000: fresh violations 0/3000; same-sample over RHS 151/3000; selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000; mean fresh risk 0.0019015; mean same-sample squared error 0.00190202; mean RHS 0.00699317.
- misspecified_tradeoff, n=8000: fresh violations 0/1000; same-sample over RHS 64/1000; selected shares gamma0=0.000, gamma0.25=0.000, gamma0.5=0.000, gamma1=1.000; mean fresh risk 0.000489657; mean same-sample squared error 0.000456958; mean RHS 0.00174025.
