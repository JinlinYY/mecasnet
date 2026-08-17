# Supplementary Note S2: cross-simulator evaluation

The Chemical and NEEQ models are trained independently. This experiment tests
whether the architecture can be trained in another domain; it is not direct
parameter transfer between networks.

Use the standard training entry point with the authorized NEEQ data root and a
separate output directory:

```bash
mecasnet-train \
  --profile paper --variant MeCaSNet \
  --data-root /path/to/authorized/neeq-data \
  --epochs 60 --seeds 5 --seed-start 0 --save-ckpt \
  --out-dir runs/si/s2/neeq-mecasnet
```

Record the actual event counts and split manifest used for the domain. Do not
reuse a Chemical-domain checkpoint unless the experiment is explicitly defined
as transfer and feature compatibility has been established.
