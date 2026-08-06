# Oracle Contact Graph Ablation

All four experiments use the same:

- `oracle_role_residual_kv`
- Oracle object and hand masks
- `contact_target_weight=1.0`
- prompts, seed, denoising steps, and KV support

Only the contact relation changes:

| Config | Relation |
| --- | --- |
| `no_graph.env` | No relation residual |
| `distance_only.env` | Spatial top-K edges with distance weights |
| `shuffled.env` | Same nodes, edge count, and confidence; shuffled hand endpoints |
| `source_qk.env` | Spatial top-K edges weighted by source Q/K affinity |

The graph is directed from object contact queries to nearby hand keys. The
relation residual is applied to transformer blocks `[10, 20)`:

```text
target_output += strength * object_confidence
                 * (source_relation_message - target_relation_message)
```

Run one experiment:

```bash
bash run_907_contact_graph.sh \
  experiments/907_contact_graph/source_qk.env
```

Run all experiments sequentially:

```bash
bash run_907_contact_graph_all.sh
```

Use `STEP=1` for a runtime smoke test and `STEP=15` for the comparison.
Graph structure for each causal block is saved under the experiment's
`roles/` directory.
