# Main-manuscript experiments

The directories follow the experimental sections of the current manuscript.
Each section describes its scientific question, comparison design, evaluation
population and relationship to the Supplementary Information.

| Section | Manuscript title | Study directory | Tables or figures |
|---|---|---|---|
| 4.1 | Evaluation Metrics | [Evaluation metrics](section_4_01_evaluation_metrics/) | Metric definitions |
| 4.2 | Implementation Details | [Study settings](section_4_02_implementation_details/) | Repeated-partition and training settings |
| 4.3 | Overall Cascade Prediction Performance | [Overall performance](section_4_03_overall_performance/) | Table 2 |
| 4.4 | Event-Split Robustness and Topology-Shift Evaluation | [Split and topology robustness](section_4_04_split_and_topology/) | Table 3 |
| 4.5 | Propagation-Depth and Network-Position Analysis | [Propagation structure](section_4_05_propagation_depth/) | Tables 4-5 |
| 4.6 | Ablation Study | [Architecture ablation](section_4_06_ablation/) | Table 6 |
| 4.7 | Damage-Trajectory Reconstruction in the Henriet Chemical Domain | [Trajectory reconstruction](section_4_07_trajectory_reconstruction/) | Figures 4-5 |
| 4.8 | Repair-Oriented Screening and Resilience Analysis | [Repair screening](section_4_08_repair_screening_and_resilience/) | Figure 6; Supplementary Tables S24-S26 |
| 4.9 | Computational Efficiency and Practical Value | [Computational comparisons](section_4_09_efficiency/) | Tables 7-9; Supplementary Tables S20-S23 |
| 4.10 | Sensitivity to the Cascade-Definition Threshold | [Threshold sensitivity](section_4_10_threshold_sensitivity/) | Table 10 |
| 4.11 | Sensitivity to Input Uncertainty and Predictive Uncertainty | [Sensitivity and coverage](section_4_11_uncertainty/) | Tables 11-12 |

Sections 4.1 and 4.2 describe definitions and common study settings. Their
corresponding calculations are included in the model training and evaluation
code. The other directories contain analyses associated with the respective
scientific questions. Reproduction requirements are described in the
[experimental reproduction guide](../REPRODUCTION.md).
