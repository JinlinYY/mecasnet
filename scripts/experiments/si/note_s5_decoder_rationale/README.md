# Supplementary Note S5: structured decoder rationale

The mathematical components in this note map to:

| Mechanism | Implementation |
|---|---|
| Super-Gaussian decline | `src/mecasnet/model_v3.py::_collapse_shape` |
| Bounded recovery family | `src/mecasnet/model_v3.py::_recovery_shape` |
| Component trajectory | `_reconstruct_trajectory` |
| Three-component lower envelope | `_reconstruct_trajectory_trimodal` |
| Fixed paper-profile settings | `src/mecasnet/factory.py::apply_profile` |

The supporting learned-parameter diagnostic is located in the main-manuscript
Section 4.7 folder. Decoder parameters describe one prediction branch; final
trajectories also include recurrent/direct fusion and a late correction, so the
parameters are not uniquely identified simulator constants.
