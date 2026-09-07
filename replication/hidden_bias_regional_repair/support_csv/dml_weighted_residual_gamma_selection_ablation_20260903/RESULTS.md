# Weighted-Residual Gamma Selection Ablation

This diagnostic keeps the fitted residual direction and upstream fits
fixed, then selects gamma by held-out responder weighted residual
loss instead of centered score loss.  The gamma grid and one-SE
stand-down rule are otherwise unchanged.

## Method Readout

- aipw: current 25.316%, residual-selected 25.525% [18.776%, 32.061%], delta 0.209% [-0.306%, 0.785%], activation 27.4% vs 27.7%, harm 10.1% vs 10.6%, gamma changed 5.3%
- ctmle: current 0.297%, residual-selected 9.637% [7.211%, 12.149%], delta 9.340% [6.682%, 12.139%], activation 26.6% vs 0.5%, harm 8.5% vs 0.3%, gamma changed 26.5%
- cui_selective_ml: current 16.393%, residual-selected 17.073% [10.585%, 23.482%], delta 0.680% [-0.863%, 2.241%], activation 35.2% vs 32.7%, harm 14.5% vs 12.6%, gamma changed 15.1%
- ma_dr_bc: current 11.596%, residual-selected 12.951% [6.637%, 18.979%], delta 1.354% [-4.352%, 7.483%], activation 27.4% vs 22.3%, harm 10.2% vs 8.1%, gamma changed 26.6%
- tmle: current -24.977%, residual-selected -21.372% [-34.086%, -10.534%], delta 3.605% [1.137%, 6.753%], activation 24.7% vs 24.7%, harm 13.0% vs 13.5%, gamma changed 4.7%

## Primary Average

- Primary methods: current 17.203%, residual-selected 18.741% [14.768%, 22.782%], delta 1.538% [-0.504%, 3.913%], gamma changed 18.3%.
