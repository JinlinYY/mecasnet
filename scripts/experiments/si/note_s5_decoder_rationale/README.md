# Supplementary Note S5: Mechanistic Rationale for the Structured Trajectory Decoder

This note relates the structured trajectory family to chemical-domain supply
constraints, inventory buffering, decline fronts and bounded recovery behavior.
Table S9 summarizes the rationale for the decoder components.

| Scientific quantity | Corresponding calculation |
|---|---|
| Buffered decline front | `_collapse_shape` in [model_v3.py](../../../../src/mecasnet/model_v3.py) |
| Bounded recovery family | `_recovery_shape` in the same file |
| Component production trajectory | `_reconstruct_trajectory` |
| Three-component lower envelope | `_reconstruct_trajectory_trimodal` |

The structured branch gives a compact temporal representation. The final
trajectory also combines recurrent and direct predictions. Learned component
parameters therefore describe the prediction model and should not be treated
as uniquely identified physical parameters of the simulator.

[Section 4.6](../../main/section_4_06_ablation/README.md) evaluates changes to
the decline and recovery representations. The learned-parameter analysis is
available in [Section 4.7](../../main/section_4_07_trajectory_reconstruction/README.md).
The chemical simulator equations and recovery-horizon diagnostics additionally
require the original simulator and corresponding reference trajectories.
