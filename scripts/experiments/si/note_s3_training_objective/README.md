# Supplementary Note S3: training objective parameters

This note maps directly to package configuration and losses:

| SI quantity | Implementation |
|---|---|
| Loss coefficients | `src/mecasnet/config.py::Config` |
| Key-date weights | `Config.kf_weights` |
| Severity-dependent focal weights | `src/mecasnet/losses.py::focal_weights` |
| Composite objective | `src/mecasnet/losses.py::total_loss` |

The default focal weight is `0.1 + 5 * true_peak_loss`, with a minimum of `2.5`
for directly shocked firms. The ten key-date weights are documented in
`docs/training.md`.

No standalone runner is needed: these terms are exercised by every
`mecasnet-train` call and stored indirectly through the command/config record.
