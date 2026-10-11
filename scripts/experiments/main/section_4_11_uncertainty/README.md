# 4.11 Sensitivity to Input Uncertainty and Predictive Uncertainty

This section separates sensitivity to reconstructed inputs, variability caused
by unobserved recovery time and predictive coverage on independent events.

## Reconstructed-input sensitivity: Table 11

A frozen model is evaluated on the same 500 reference events with unchanged
simulator targets. Input-share and baseline-capacity noise use log-normal scales
of 0.10, 0.20 and 0.30. Missing-edge rates are 5%, 10% and 20%, with removed
shares not reallocated. Day-0 damage noise uses Gaussian scales of 0.05, 0.10
and 0.20, clipped to the physical damage range. Each stochastic condition uses
ten draws and 2,000 paired event-bootstrap resamples. The joint condition
combines share and capacity scales of 0.20, 10% edge omission and damage scale
0.10.

[input_uncertainty.py](input_uncertainty.py) contains the main input-perturbation
analysis. [missing_supplier_sensitivity.py](missing_supplier_sensitivity.py)
contains complementary edge-omission procedures that retain at least one
supplier per buyer. Enabling share reallocation redistributes removed share mass
among retained suppliers. This complementary
condition differs from the omission setting in Table 11.

[evaluate_share_ablation_simulator_sensitivity.py](evaluate_share_ablation_simulator_sensitivity.py)
examines input-share structure and simulator-label sensitivity. Uniform model
inputs with fixed targets, perturbed simulator labels and omitted supplier
relations address separate comparisons; their results are kept distinct.

## Unobserved recovery-time variability: Supplementary Table S18

[evaluate_recovery_floor_event_conformal.py](evaluate_recovery_floor_event_conformal.py)
contains repeated-simulation procedures for 100 fixed Day-0 states and 30
recovery-time draws per state. Recovery time remains unavailable to the model.
Conditional simulator variance is compared with model MSE, resampling complete
Day-0 states with all their associated draws.

## Independent event-level coverage: Table 12

[run_independent_conformal.py](run_independent_conformal.py) contains the
independent-event coverage analysis. The trained models remain fixed. A new
500-event calibration set and a separate 500-event test set are generated
independently of training and model selection.

Each calibration event contributes one maximum absolute residual across all
prediction nodes for peak loss, and across all nodes and ten key dates for
trajectory prediction. Separate 90% intervals are calibrated for the two targets.
An event is covered only when its complete prediction population is covered.
Test confidence intervals resample complete events with the calibrated radii
fixed. Separate target coverage does not imply joint 90% coverage of both targets.

[predictive_uncertainty.py](predictive_uncertainty.py) supplies an additional
ensemble and validation-calibrated analysis. Its pooled residual calibration
does not reproduce the independent event-simultaneous criterion of Table 12.

See [Supplementary Note S7](../../si/note_s7_robustness_and_uncertainty/README.md)
for Tables S16-S19 and the complete uncertainty protocols.
