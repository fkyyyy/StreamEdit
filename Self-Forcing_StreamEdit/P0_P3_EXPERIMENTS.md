# P0-P3: source residual and attention-value ablations

This note records the controlled P0-P3 experiment chain for the wallet edit.
The experiments study the coupling between useful source geometry/illumination
and unwanted source-phone appearance.  They are ablations on the native
StreamGVE path, not a finished identity-memory method.

## Shared protocol

All four experiments use the same source video, prompts, trigger words, seed,
15 denoising steps, 21-frame rollout chunks, one overlap block, and the native
StreamGVE attention/KV history.  They use the default dynamic SOG foreground
mask.  No external object mask, phone mask, hand mask, optical flow, manual
trajectory, factorized routing, soft modulation, or identity anchor is used at
inference time.

The external phone mask is used only by offline visualization and metric
scripts.  It is never passed to generation.

## Experiment matrix

| Experiment | Change from predecessor | Intended question | Current visual result |
| --- | --- | --- | --- |
| P0 | Project the source reconstruction residual away from components opposing the target edit direction | Can velocity-space antagonistic projection remove source identity while retaining geometry? | Geometry is usable, but the wallet darkens and a blue-white source-screen spot appears in later chunks. |
| P1 | Limit later projection removal energy to the corresponding first-block denoising-step budget | Is excessive removal in later chunks the cause of darkening? | Brightness improves only slightly and source-like spots become stronger. Do not continue tuning this budget. |
| P2 | Restore the application-weighted norm of the P0 projected residual with a positive per-sample, per-latent-frame scalar | Is loss of residual magnitude the cause of darkening? | Negative result. It follows P0 closely, retains the spot, and late brown-shell luma is slightly worse than P0. |
| P3 | Replace the exact source-background attention output contribution on automatic foreground queries with an RMS-matched target-value contribution | Can source appearance be removed at the attention output while preserving its useful energy and native addressing? | Negative result. The wallet is slightly darker and the blue-white spot remains. The replacement reduces the measured cool-colored area versus P0/P2, but does not remove the visible artifact and worsens the brown-shell brightness trend. |

## P0: antagonistic source-residual projection

Run:

```bash
bash run_P0_projected_residual.sh
```

The native velocity formula is

```text
source_residual = v_gt - v_src
v_t = v_trg + bg_mask * source_residual
```

P0 defines the target edit direction as `v_trg - v_src`.  At each token it
removes only the component of `source_residual` whose dot product with the edit
direction is negative, then uses the original soft SOG `bg_mask`:

```text
safe_residual = remove_antagonistic_source_residual(
    source_residual, v_trg - v_src
)
v_t = v_trg + bg_mask * safe_residual
```

This does not modify attention or KV.  The result shows that an orthogonal or
non-antagonistic velocity component is not necessarily appearance-free.

## P1: causal first-block removal-energy budget

Run:

```bash
bash run_P1_causal_energy_budget.sh
```

P1 starts from P0.  For every denoising-step index, block 0 records the
fraction of application-weighted source-residual energy removed by P0.  Later
blocks may remove less, but are scaled back when they exceed the corresponding
block-0 fraction.  The state is a scalar per sample and denoising step; it is
not a spatial or appearance memory.

The budget was active in the wallet run: later projection scales were roughly
0.54-0.66, while the applied removed fraction stayed near 0.0286.  Nevertheless
brightness improved by only about one to two luma points and the blue-white
source-like region grew.  This shows that reducing removal restores source
appearance together with energy.

## P2: norm-preserving projected residual

Run:

```bash
bash run_P2_norm_preserving_projection.sh
```

P2 returns to the full P0 projection and does not use the P1 budget.  It
measures the norm after the residual is multiplied by the native soft
`bg_mask`, then scales the projected residual by one positive scalar per
sample and latent frame:

```text
scale = ||bg_mask * source_residual|| / ||bg_mask * safe_residual||
calibrated_residual = clamp(scale, max=4) * safe_residual
```

Positive scaling retains the projected direction.  A 4x guard prevents a
nearly zero safe direction from amplifying numerical noise.

In the wallet run the actual scale was only about 1.01-1.06 and never hit the
guard, but the late brown-shell luma still fell by about 10.3 percent versus
8.8 percent for P0.  The source-like spot first became unambiguously visible
at frame 45, peaked near frame 64, collapsed when the source phone screen
became dark at frames 67-68, and returned with UI-like structure at frame 69.
This rejects the hypothesis that a scalar norm deficit is the main cause.

## P3: counterfactual source-background attention output

Run:

```bash
bash run_P3_counterfactual_source_bg_output.sh
```

P3 starts from P0 and leaves the native source-background K/V segment in the
attention sequence.  It therefore preserves the native query, keys, softmax
denominator, addressing, and mutable KV history.  During the late injection
steps, it evaluates two additional V-only attention contributions for the
existing source-background key segment:

```text
C_src = Attention(Q, K_all, V_only_source_segment)
C_trg = Attention(Q, K_all, V_only_target_segment)
```

Because `Q`, `K_all`, and the complete softmax denominator are identical,
these are exact query-conditioned contributions of the selected segment.  On
automatic SOG foreground queries only, P3 applies

```text
alpha = rms(C_src) / rms(C_trg)
O_p3 = O_native - C_src + clamp(alpha, max=4) * C_trg
```

Background queries are returned bit-for-bit from the native output.  If the
target contribution is numerically degenerate, P3 abstains and preserves the
native output for that query.  P3 does not suppress source values globally,
does not drop the source K/V pair, and does not write a new memory bank.

The runtime log prefix is `COUNTERFACTUAL_SOURCE_BG_OUTPUT`.  Important fields
are foreground and active coverage, source/raw-target/matched-target RMS, scale,
capped fraction, degenerate fraction, and correction RMS.

The rendered 81-frame result is also negative.  Visual inspection shows a
slightly darker wallet and a persistent blue-white source-screen-like spot.
Using the external phone mask strictly for offline measurement, brown-shell
median luma falls from 38.90 over frames 0-20 to 31.98 over frames 60-80, a
17.8 percent decrease.  The cool-spot diagnostic first crosses its automatic
threshold at frame 26 and the artifact remains visibly structured in later
blocks.  The measured late cool-spot fraction is 0.068, substantially below
P0's 0.240, so P3 suppresses some of the cool-colored area, but it does not
eliminate the perceptually obvious spot.

The darkening is not explained by an instantaneous jump at every causal-block
boundary.  The largest collapse occurs inside block 1: brown-shell median luma
falls from 55.08 at frame 9 to 28.18 at frame 20.  Later blocks remain at the
lower level while the spot changes shape and intensity.  Thus exact
counterfactual replacement of the current source-background segment is neither
an appearance-complete intervention nor an appearance-safe energy anchor.  It
removes part of the measured cool contribution, but other paths still preserve
the artifact and the RMS-matched target contribution changes the wallet's
appearance/brightness trajectory.

P3 is not first-chunk identity transport.  A later canonical-memory experiment
would freeze the first block's target-minus-source appearance residual, align
it to later clean-source geometry, and read it as a separate bounded residual.
That mechanism is intentionally outside this P0-P3 ablation chain.

## Evaluation gate

Inspect all 81 frames, with special attention to frames 32-80 and causal-block
starts 33, 45, 57, and 69.  A useful P3 result must satisfy all of the following:

1. The blue-white source-screen spot and UI-like structure are reduced.
2. Brown-shell brightness does not progressively collapse.
3. Wallet outline, stitching, pose, hands, and background remain stable.
4. The improvement persists across chunk boundaries instead of resetting for
   only one block.

P3 leaves a visible spot while also darkening the wallet.  The result therefore
supports the second failure case: leakage enters through paths outside the
isolated current source-background segment, and RMS matching alone does not
make the substituted target contribution appearance-safe.  This run should be
kept as a diagnostic negative result rather than tuned as the next solution.
