# Published experiment artifacts

This directory contains the compact, reviewable outputs produced by the
repository's Full-method diagnostics and main-ablation runners.

## Included results

- `main_ablation/`: the four requested ablation configurations, logs, summary
  tables, metrics, and final edited videos.
- `ours_responsibility_visualization/`: the final responsibility pipeline,
  role and permission maps, velocity-control figure, selected keyframes,
  metadata, metrics, diagnostics archive, and edited video.
- `ours_single_case_diagnostics/final_color_v4/`: the final color-rendered
  mechanism dashboard, all published cross/self-attention head images,
  operation-specific permissions, velocity-update PCA figures, anchor plots,
  multi-frame views, metadata, metrics, and edited video.

The diagnostic capture manifest and automatic case selection metadata are
stored directly under `ours_single_case_diagnostics/`.

## Generation code

The corresponding implementation is versioned in the repository at:

- `run_ours_responsibility_visualization.sh`
- `run_ours_single_case_diagnostics.sh`
- `tools/visualize_ours_responsibility.py`
- `tools/visualize_single_case_diagnostics.py`
- `tools/visualize_single_case_multiframe.py`
- `utils/mechanism_attention_diagnostics.py`
- `run_main_ablation.sh`
- `tools/finalize_main_ablation.py`

## Deliberately excluded local captures

Raw tensor dumps and superseded rendering iterations remain local. The full
local artifact tree is about 6.4 GB, including about 5.7 GB of `.pt` tensors;
several individual tensors exceed GitHub's 100 MB object limit. These raw
captures are reproducible with the runners above and are not required to view
the committed figures or videos. Paper rewrite previews are also outside the
experiment-artifact scope.
