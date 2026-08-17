# Data contract

MeCaSNet consumes one static directed network and independently generated
disruption events. The code does not infer the supply network or run a cascade
simulator.

The node order is the central identity contract: every static array and every
event array must use exactly the same zero-based node ordering. Raw firm
identifiers are neither required nor accepted by the model.

## Directory layout

```text
DATA_ROOT/
├── static_meta.pkl
└── events/
    ├── event_000000.npz
    ├── event_000001.npz
    └── event_XXXXXX.npz
```

Event IDs do not have to be consecutive, but each filename must end in an
integer after `event_`. That integer is used for splitting, manifests, and
checkpoint audit records.

> Security: `static_meta.pkl` uses Python pickle and `shock_meta` is loaded from
> an object-containing NumPy archive. Both mechanisms can execute or construct
> Python objects during deserialization. Load only files produced by a trusted
> source.

## Static network

`static_meta.pkl` must deserialize to a Python mapping with:

| Key | Shape/type | Meaning |
|---|---|---|
| `V` | integer | Number of firms/nodes |
| `E` | integer | Number of directed supply relations |
| `n_sec` | integer | Number of sector categories |
| `sectors` | `(V,)`, integer | Zero-based sector index per firm |
| `P_ini` | `(V,)`, float | Positive business-as-usual production |
| `edge_src` | `(E,)`, integer | Supplier node index |
| `edge_dst` | `(E,)`, integer | Customer node index |
| `A` | `(E,)`, float | Positive baseline flow on each relation |

For edge `e`, `edge_src[e] -> edge_dst[e]`. The loader derives the customer
input share as `A[e] / P_ini[edge_dst[e]]` and the supplier output share as
`A[e] / P_ini[edge_src[e]]`.

### Orientation example

If firm 4 supplies a baseline flow of 20 to firm 9, whose baseline production
is 100, encode:

```python
edge_src[e] = 4
edge_dst[e] = 9
A[e] = 20.0
P_ini[9] = 100.0
```

The model input share for this edge is `20 / 100 = 0.2`. Reversing `src` and
`dst` changes the scientific meaning of forward material propagation and
reverse demand feedback; it is not a harmless transpose.

### Loader validation

`StaticNetwork` rejects:

- nonpositive `V` or `n_sec`, and negative `E`;
- arrays whose lengths disagree with `V` or `E`;
- sector indices outside `[0,n_sec)`;
- nonfinite or nonpositive `P_ini`;
- edge indices outside `[0,V)`;
- nonfinite or nonpositive flows.

Parallel edges are accepted by the sparse construction but should be aggregated
before export unless multiple relations between the same ordered pair have a
documented meaning. Self-loops are not explicitly prohibited; include them only
if they are part of the intended supply representation.

### Derived static features

The loader computes features rather than expecting them on disk:

| Feature | Transformation |
|---|---|
| Sector | one-hot of `sectors`, unless `use_sector_oh=False` |
| In-degree | network-standardized `log(1 + in_degree)` |
| Out-degree | network-standardized `log(1 + out_degree)` |
| Baseline production | network-standardized `log(1 + P_ini)` |
| Reverse PageRank | network-standardized log score |
| Connectivity | indicator for the largest strongly connected component |

Therefore `Fv = n_sec + 5` with sector one-hot features and `Fv = 5` without
them. A checkpoint can be loaded only when its expected input width matches the
network feature width.

## Event files

Each `event_XXXXXX.npz` must contain:

| Key | Shape/type | Role |
|---|---|---|
| `shock_mask` | `(V,)`, binary | Firms directly disrupted at Day 0 |
| `delta0` | `(V,)`, float in `[0,1]` | Day-0 capacity-loss ratio |
| `key_days` | `(K,)`, integer | Prescribed prediction dates |
| `u_keyframes` | `(K,V)`, float | Actual/baseline production ratio |
| `peak_loss_node` | `(V,)`, float | Producer-defined peak loss; document full-trajectory versus key-date calculation |
| `cum_loss_node` | `(V,)`, float | Validated compatibility field; not used by the default objective |
| `shock_meta` | scalar object mapping | Generation metadata used only where allowed |

`aggregate_bau_loss_curve` is optional and read only for stratified audit output.
The manuscript profile uses
`[0, 5, 10, 20, 30, 50, 70, 100, 150, 199]`.

### Event semantics

- `shock_mask[v]=1` means node `v` is directly disrupted at Day 0.
- `delta0[v]` is the observed capacity-loss fraction. It should normally be
  zero outside the direct-shock set.
- `u_keyframes[k,v]` is actual production divided by business-as-usual
  production; values are expected to be physically interpretable near `[0,1]`.
- `peak_loss_node[v]` is the target peak loss from the dataset producer. State
  whether this uses the full simulator trajectory or only the ten key dates.
- `cum_loss_node` is required for compatibility even though it is not part of
  the default public training objective.

The current validator requires exact shapes, exact key-date agreement, a binary
shock mask, finite `delta0` in `[0,1]`, and finite target arrays. It deliberately
does not silently clip target trajectories; fix invalid values at the data
generation stage.

### `shock_meta`

`shock_meta` must be a scalar object containing a mapping, even in minimal mode.
The paper predictor does not read simulator-only recovery/mode/tier fields. They
may still be present for audit stratification, but are exposed only when
`include_audit_metadata=True`.

Legacy `clean` and `full` modes recognize fields such as
`recovery_per_target`, `delta_per_target`, `mode`, `tier`, `delta`, and
`recovery_days`. These modes are not the final manuscript information set.

## Information boundary

With `event_scalars_mode="minimal"`, the model receives only the mean observed
`delta0` among directly shocked nodes. Tier, mode, simulator recovery time,
future production/inventory/order states, and target-derived labels are excluded
from the forward input.

The encoder tensor remains 12-dimensional for checkpoint compatibility:

```text
event_scalars = [0, 0, mean_direct_damage, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

Target trajectories are used only after prediction to form training losses and
evaluation subsets. In particular, focal weights, reachability labels, trough
dates, and the cascade mask are not inference inputs.

## Event-local reach subgraph

For an event, the loader follows supplier-to-customer edges from the directly
shocked set for at most `reach_hops=5`. Let the resulting static indices be
`reach_idx` and their count be `Nr`. The batch contains only those nodes and
edges induced between them.

Consequences:

- batch node indices range from `0` to `Nr-1`, not `0` to `V-1`;
- `reach_idx[local_v]` maps a local prediction back to the static network;
- batch size is fixed to one event because `Nr` and edge count vary;
- the metrics in `mecasnet.runner` use nodes in these reach subgraphs;
- nodes outside a reach subgraph are not automatically included as `u=1` in
  the implemented metrics.

The reach extraction is a computational and scientific scope choice. If a new
dataset exhibits effects beyond five directed hops, increase `reach_hops` and
report the change.

## Batch dictionary returned by the loader

| Key | Shape/type | Description |
|---|---|---|
| `event_id` | integer | ID parsed from the filename |
| `reach_idx` | `(Nr,)` long | Local-to-static node mapping |
| `Nr` | integer | Local node count |
| `x_v` | `(Nr,Fv)` float | Derived static node features |
| `shock_mask` | `(Nr,)` float | Local direct-shock indicator |
| `delta0` | `(Nr,)` float | Local Day-0 damage |
| `P_ini` | `(Nr,)` float | Local baseline production |
| `sectors` | `(Nr,)` long | Local sector index |
| `edge_src`, `edge_dst` | `(Er,)` long | Local directed endpoints |
| `edge_a` | `(Er,)` float | Customer input share |
| `edge_outshare` | `(Er,)` float | Supplier output share |
| `key_days` | `(K,)` long | Prediction dates |
| `u_keyframes` | `(K,Nr)` float | Training/evaluation target |
| `peak_loss` | `(Nr,)` float | Peak-loss target |
| `is_reach` | `(Nr,)` float | Auxiliary label using threshold 0.001 |
| `trough_day_target` | `(Nr,)` long | Argmin key-date index |
| `event_scalars` | `(12,)` float | Profile-controlled event descriptor |
| `shock_hop_oh` | `(Nr,5)` float | Capped undirected shock distance; not used by default paper encoder |

With `include_audit_metadata=True`, mode, tier, and aggregate BAU-loss audit
fields are added as non-tensor metadata.

## Minimal trusted-data example

The following creates a structurally valid three-node example. It is for schema
testing, not a scientifically meaningful cascade:

```python
from pathlib import Path
import pickle
import numpy as np

root = Path("example_data")
(root / "events").mkdir(parents=True, exist_ok=True)

static = {
    "V": 3,
    "E": 2,
    "n_sec": 2,
    "sectors": np.array([0, 1, 1], dtype=np.int64),
    "P_ini": np.array([100.0, 80.0, 60.0], dtype=np.float32),
    "edge_src": np.array([0, 1], dtype=np.int64),
    "edge_dst": np.array([1, 2], dtype=np.int64),
    "A": np.array([20.0, 15.0], dtype=np.float32),
}
with (root / "static_meta.pkl").open("wb") as stream:
    pickle.dump(static, stream)

days = np.array([0, 5, 10, 20, 30, 50, 70, 100, 150, 199], dtype=np.int32)
u = np.ones((len(days), 3), dtype=np.float32)
u[:, 0] = np.array([0.6, 0.65, 0.72, 0.80, 0.86, 0.92, 0.96, 0.98, 0.99, 1.0])
u[:, 1] = np.array([1.0, 0.98, 0.93, 0.88, 0.86, 0.89, 0.93, 0.97, 0.99, 1.0])

np.savez(
    root / "events" / "event_000000.npz",
    shock_mask=np.array([1, 0, 0], dtype=np.uint8),
    delta0=np.array([0.4, 0.0, 0.0], dtype=np.float32),
    key_days=days,
    u_keyframes=u,
    peak_loss_node=1.0 - u.min(axis=0),
    cum_loss_node=(1.0 - u).mean(axis=0),
    shock_meta=np.array({"source": "schema-example"}, dtype=object),
)
```

Load it with:

```python
from mecasnet import Config
from mecasnet.data import CascadeEventDataset, StaticNetwork

cfg = Config(data_root="example_data", event_scalars_mode="minimal")
net = StaticNetwork(cfg)
batch = CascadeEventDataset(cfg, net, [0])[0]
print(net.V, net.E, net.Fv, batch["Nr"], batch["u_keyframes"].shape)
```

## Validation recommendations

- Compare a few encoded edges with the source system to verify direction.
- Check `peak_loss_node` against the stated full-trajectory or key-date rule.
- Inspect reach-subgraph sizes and the fraction of affected nodes outside five hops.
- Hash the static file and ordered event IDs for every reported run.
- Store train/validation/test IDs in a manifest and confirm the sets are disjoint.
- Record the dataset generator version, simulator parameters, and random seeds.
- Do not publish protected raw identifiers or reversible derived attributes.
