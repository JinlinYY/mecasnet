# Data contract

MeCaSNet consumes one static directed network and independently generated
disruption events. The code does not infer the supply network or run a cascade
simulator.

## Directory layout

```text
DATA_ROOT/
├── static_meta.pkl
└── events/
    └── event_XXXXXX.npz
```

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

## Event files

Each `event_XXXXXX.npz` must contain:

| Key | Shape/type | Role |
|---|---|---|
| `shock_mask` | `(V,)`, binary | Firms directly disrupted at Day 0 |
| `delta0` | `(V,)`, float in `[0,1]` | Day-0 capacity-loss ratio |
| `key_days` | `(K,)`, integer | Prescribed prediction dates |
| `u_keyframes` | `(K,V)`, float | Actual/baseline production ratio |
| `peak_loss_node` | `(V,)`, float | Maximum loss over the simulator trajectory |
| `cum_loss_node` | `(V,)`, float | Mean or cumulative node-loss label used by the dataset |
| `shock_meta` | scalar object mapping | Generation metadata used only where allowed |

`aggregate_bau_loss_curve` is optional and read only for stratified audit output.
The manuscript profile uses
`[0, 5, 10, 20, 30, 50, 70, 100, 150, 199]`.

## Information boundary

With `event_scalars_mode="minimal"`, the model receives only the mean observed
`delta0` among directly shocked nodes. Tier, mode, simulator recovery time,
future production/inventory/order states, and target-derived labels are excluded
from the forward input.

Target trajectories are used only after prediction to form training losses and
evaluation subsets. In particular, focal weights, reachability labels, trough
dates, and the cascade mask are not inference inputs.

## Validation recommendations

- Verify all edge indices lie in `[0,V)` and all arrays have consistent lengths.
- Require finite, positive `P_ini` and non-negative flows.
- Hash the static file and ordered event IDs for every reported run.
- Store train/validation/test IDs in a manifest and confirm the sets are disjoint.
- Do not publish protected raw identifiers or reversible derived attributes.
