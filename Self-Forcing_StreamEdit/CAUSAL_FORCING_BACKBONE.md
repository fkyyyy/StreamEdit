# Causal Forcing Backbone

The editing pipeline supports these backbone presets:

- `causal_forcing_framewise`: official frame-wise 4-step model.
- `causal_forcing_plus_plus_2step`: official frame-wise 2-step model,
  with a 4-step schedule for the first generated block.

Download one of the official checkpoints:

```bash
hf download zhuhz22/Causal-Forcing \
  framewise/causal_forcing.pt \
  --local-dir checkpoints

hf download zhuhz22/Causal-Forcing \
  causal-forcing++/framewise-2step.pt \
  --local-dir checkpoints
```

Run the default 2-step backbone with target identity memory:

```bash
bash run_907_bayes_causal_forcing.sh
```

Run the 4-step backbone:

```bash
BACKBONE=causal_forcing_framewise \
bash run_907_bayes_causal_forcing.sh
```

Override the checkpoint location when it is outside this directory:

```bash
CHECKPOINT_PATH=/absolute/path/to/framewise-2step.pt \
bash run_907_bayes_causal_forcing.sh
```

Run customized reference editing:

```bash
ROUTING_MODE=hand_role_bayes_flow_customized_kv \
REFERENCE_IMAGE=/absolute/path/to/aligned_edited_first_frame.png \
bash run_907_bayes_causal_forcing.sh
```

The startup log must report:

```text
BACKBONE_CONFIG name=causal_forcing_plus_plus_2step
EDIT_BACKBONE ... frames_per_block=1 steps=2 first_block_steps=4
BACKBONE_CHECKPOINT_LOADED name=causal_forcing_plus_plus_2step
```
