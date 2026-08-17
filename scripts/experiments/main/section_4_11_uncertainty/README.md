# Section 4.11: input and predictive uncertainty

The two workflows answer different questions and should not be combined into a
single interval claim.

## Reconstructed-input sensitivity

```bash
python scripts/experiments/main/section_4_11_uncertainty/input_uncertainty.py \
  --data-root /path/to/authorized/data \
  --checkpoint /path/to/legacy-y8-seed0.pt \
  --output-dir runs/main/section_4_11/input \
  --n-test 500 --replicates 10 --bootstrap 2000 --device cuda
```

## Ensemble and conformal predictive uncertainty

```bash
python scripts/experiments/main/section_4_11_uncertainty/predictive_uncertainty.py \
  --data-root /path/to/authorized/data \
  --checkpoints /path/to/seed0.pt /path/to/seed1.pt /path/to/seed2.pt \
                /path/to/seed3.pt /path/to/seed4.pt \
  --output-dir runs/main/section_4_11/predictive \
  --n-val 500 --n-test 500 --alpha 0.10 --device cuda
```

The first workflow measures sensitivity to perturbed network/damage inputs.
The second uses validation events for conformal calibration and test events for
coverage. Raw ensemble spread is not automatically a calibrated interval.
