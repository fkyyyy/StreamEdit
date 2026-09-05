# M1 immutable delta-V bank: framewise audit

## Scope

- M1: `M1-immutable-deltaV-bank.mp4`
- Controls: `L0-local-baseline.mp4` and `965a-wallet-streamgve.mp4`
- Same input video, prompts, seed, chunk protocol, and 81 RGB frames.
- `phone_mask.mp4` is used only as an offline moving evaluation support. It is
  not an inference input.
- Causal-block entry RGB frames are 9, 21, 33, 45, 57, and 69.

## Main result

M1 preserves the native brightness/energy behavior, but does not remove the
source-screen ghost. Its apparent light shake is overwhelmingly shared with L0
and 965a; M1 adds only a small motion deviation.

### Brown shell brightness

The metric is the median luma of pixels classified as brown inside the moving
offline phone support. It deliberately excludes the blue-white spot.

| Method | F00-F20 | F32-F52 | F60-F80 | late vs early |
|---|---:|---:|---:|---:|
| L0 | 41.02 | 49.24 | 41.63 | +1.47% |
| 965a | 41.02 | 49.28 | 41.67 | +1.60% |
| M1 | 40.35 | 48.61 | 40.51 | +0.39% |

Thus M1 does not show recursive shell darkening. Relative to the native controls,
however, M1 is slightly darker rather than brighter: its all-frame brown-shell
luma is about 0.92 lower than L0 on average. The largest deficits are F56
(-3.41), F67 (-3.34), and F55 (-2.57). The perceived improvement is therefore
best stated as recovery from the strongly darkened suppression/P-series runs,
not improvement over native StreamGVE.

### Blue-white source-screen ghost

| Method | F00-F20 | F32-F52 | F60-F80 |
|---|---:|---:|---:|
| L0 | 5.86% | 19.38% | 30.07% |
| 965a | 5.86% | 19.38% | 30.08% |
| M1 | 5.85% | 19.39% | 30.05% |

M1 and L0 cool-fraction timelines have Pearson correlation 0.9985 and mean
absolute difference only 0.0050. The visible sequence is:

- F00-F20: stable brown wallet, only a weak smooth highlight.
- F22-F32: the first local blue-white spot becomes visible.
- F33-F44: the spot grows but remains localized.
- F45-F56: the central source-like region expands sharply; F45 changes from
  14.8% at F44 to 24.2%.
- F57-F68: the large pale area persists and peaks at F64-F65 near 49%.
- F69-F80: source-calendar-like horizontal structures become visible; this is
  structured source appearance, not merely a scalar brightness error.

The immutable residual therefore does not suppress the native leakage path.
It changes M1 versus L0 most around F53-F60, but the spot trajectory and its
chunk-aligned transitions remain effectively the native trajectory.

### Motion and light shake

Motion is evaluated in two ways inside a dilated moving GT support: temporal L1
and the step of the source-relative edit centroid after subtracting GT phone
motion.

| Method | boundary temporal L1 | nonboundary temporal L1 | boundary centroid step | nonboundary centroid step |
|---|---:|---:|---:|---:|
| L0 | 0.03578 | 0.02887 | 2.926 px | 1.963 px |
| 965a | 0.03577 | 0.02890 | 2.918 px | 1.961 px |
| M1 | 0.03601 | 0.02888 | 2.999 px | 1.984 px |

M1 versus L0 has temporal-L1 correlation 0.9987 and centroid-step correlation
0.9963. Their largest motion events occur at the same frames: F31, F44, and
F68-F70. M1 raises the boundary centroid step by only 0.073 px on average and
the nonboundary step by 0.021 px. There is a measurable M1-specific increase
around F57 and F69, but it is secondary. The observed shake is mainly inherited
from native StreamGVE/source motion, not created by the immutable bank.

## What the M1 diagnostics say

- The bank really persists: block 0 freezes 544 tokens over four layers and
  blocks 1-6 all read it. This is not the old one-block lifetime bug.
- Nearly every dynamic-SOG owner query is admitted. Mean similarity stays near
  0.62-0.63 while the threshold is only 0.35.
- Effective mean gate is only about 0.082-0.087 because
  `0.20 * (similarity - 0.35) / (1 - 0.35)` is small.
- The applied correction RMS rises from roughly 0.035 in block 1 to roughly
  0.042 in block 6.
- Roughly 44-62% of admitted residuals hit the RMS cap. The bank residual is
  energetic but then applied with a small scalar gate.

This combination is both permissive and weak: almost all owner queries read a
globally searched top-k bank, yet the actual correction is only about 6-8% of
native output RMS. It can tint the native result but cannot counteract an
already present source-valued contribution. Hard top-k changes and a moving
owner gate can account for the small additional deviation at F57/F69, but they
do not explain the dominant shake shared by all three videos.

## Important exactness caveat

The log says block 0 is `native_exact=1`, meaning the M1 attention operator is
not read before the bank freezes. The independently rendered M1 video is not
pixel-identical to L0, however: M1-vs-L0 ROI MAE over F00-F08 is 1.66/255 and
F00 itself is 0.54/255. Separate-run GPU nondeterminism or encoding can explain
some difference, but L0 and 965a are substantially closer during the early
period. The code should therefore claim an exact tensor fallback at the M1
operator, not proven end-to-end bitwise equality of the rendered first block.
This enabled-path side effect is worth an explicit code review.

## Recommended next ablation

Do not tune M1 strength upward. A stronger blind additive delta-V is likely to
increase appearance bias and address-switch artifacts without removing the
native source component. The next controlled experiment should keep the same
write-once bank and native attention, but make the read a closed-loop
counterfactual error correction:

1. Retrieve the frozen desired first-block target-minus-source response.
2. Estimate the current block target-minus-source response under the same
   clean-source address.
3. Inject only their bounded difference, not the full frozen delta again.
4. Add confidence based on top-1/top-2 margin or assignment entropy and log
   candidate switching per query. Low-confidence or unstable queries must
   exactly abstain.
5. Keep the memory values immutable. Any temporal state should stabilize only
   the read assignment, never update the appearance bank.

This directly targets drift/leakage error while retaining native brightness. A
separate one-variable ablation should then test temporally stable addressing; it
should not be mixed into the first closed-loop experiment.

## Artifacts

- `M1_framewise_comparison.csv`: all 81 frame metrics.
- `M1_framewise_summary.json`: period, boundary, motion, and pairwise summaries.
- `M1_framewise_timelines.png`: M1/L0/965a timelines.
- `M1_vs_baselines_boundaries.jpg`: same-frame boundary windows.
- `M1_vs_baselines_differences.jpg`: visual and amplified residual comparison.
- `M1_top_motion_windows.jpg`: highest source-relative motion frames.
- `M1_wallet_crop_all_frames.jpg`: all 81 M1 wallet crops.
