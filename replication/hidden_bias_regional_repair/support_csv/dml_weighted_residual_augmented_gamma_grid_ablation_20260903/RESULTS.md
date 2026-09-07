# Weighted-Residual Selector Augmented Gamma Grid Ablation

This diagnostic keeps the same fitted residual directions and selects
gamma by held-out weighted residual loss after adding gamma=0.75 to
the candidate grid.  The current comparator is the four-point
weighted-residual selector, not the older centered-score selector.

## Method Readout

- aipw: current 25.525%, augmented 26.229% [19.377%, 32.577%], delta 0.704% [-0.558%, 2.070%], activation 27.4% vs 27.4%, harm 10.2% vs 10.1%, gamma changed 7.7%, selected 0.75 7.7%
- ctmle: current 9.637%, augmented 10.105% [7.555%, 12.790%], delta 0.468% [-0.153%, 1.067%], activation 26.6% vs 26.6%, harm 8.4% vs 8.5%, gamma changed 11.3%, selected 0.75 11.3%
- cui_selective_ml: current 17.073%, augmented 16.471% [10.142%, 22.696%], delta -0.602% [-1.519%, 0.256%], activation 35.2% vs 35.2%, harm 14.6% vs 14.5%, gamma changed 7.3%, selected 0.75 7.3%
- ma_dr_bc: current 12.951%, augmented 13.595% [7.577%, 19.555%], delta 0.645% [-0.868%, 2.093%], activation 27.4% vs 27.4%, harm 10.1% vs 10.2%, gamma changed 7.7%, selected 0.75 7.7%
- tmle: current -21.372%, augmented -21.404% [-33.288%, -10.896%], delta -0.032% [-2.781%, 2.847%], activation 24.7% vs 24.7%, harm 13.1% vs 13.0%, gamma changed 6.9%, selected 0.75 6.9%

## Primary Average

- Primary methods: current 18.741%, augmented 19.302% [15.240%, 23.260%], delta 0.561% [-0.252%, 1.400%], selected 0.75 8.5%.
