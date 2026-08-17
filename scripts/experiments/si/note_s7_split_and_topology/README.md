# Supplementary Note S7: split and topology robustness

Supplementary Note S7 expands the experiment reported in main-manuscript
Section 4.4. The executable scripts and their required run order are kept in:

```text
scripts/experiments/main/section_4_04_split_and_topology/
```

See that directory's README for event-neighbor auditing, similar-event
exclusion, static block design, regenerated reduced-network requirements, and
frozen-checkpoint evaluation. The standard split is event-disjoint but shares
one static graph; the reduced-graph analysis is topology modification, not
fully node-disjoint inductive transfer.
