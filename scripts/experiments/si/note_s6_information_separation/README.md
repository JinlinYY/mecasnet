# Supplementary Note S6: Predictive Information Set and Training-Inference Separation

Tables S10-S11 distinguish information observed at prediction time from
quantities available only for supervision or simulator-based evaluation.

Predictive inputs comprise the pre-disruption graph and firm attributes, direct
shock indicators, observed Day-0 damage, the mean observed direct-shock damage
and prescribed key dates. Future production trajectories, true peak loss,
true cascade extent, trough labels and simulator recovery time are not observed
predictive inputs.

[Static and event data](../../../../src/mecasnet/data.py) define the model
observations and targets. [factory.py](../../../../src/mecasnet/factory.py)
specifies the manuscript model settings. [model_v3.py](../../../../src/mecasnet/model_v3.py)
defines prediction from the observed information, while
[losses.py](../../../../src/mecasnet/losses.py) relates predictions to the
supervised targets during training.

The distinction matters for uncertainty analysis: future simulator recovery
time can vary among events with identical Day-0 observations. The corresponding
conditional-variance experiment is described in
[Supplementary Note S7](../note_s7_robustness_and_uncertainty/README.md).
