# Debug Session: reference-source-regression
- **Status**: [OPEN]
- **Issue**: Customized reference editing still regresses to the source object after a few blocks, with no visible improvement after commit b9162fe.
- **Debug Server**: Disabled; remote host cannot reach the local collector
- **Log File**: GitHub artifact `outputs/907_reference_regression_debug`

## Reproduction Steps
1. Run `hand_role_bayes_flow_customized_kv` with the aligned Coca-Cola first-frame reference.
2. Observe that the first blocks contain the target can.
3. Observe that later blocks regress to the white source bottle and blue cap.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Expected Evidence |
|----|------------|------------|--------|-------------------|
| H1 | Commitment updates belief after the Bayes action was already computed | High | Low | Rejected: region-level Bayes action equals the post-commit action |
| H2 | Velocity routing consumes a stale pre-commit preserve map | High | Low | Rejected: saved Bayes action exactly matches the final control belief |
| H3 | Reference transport loses effective object support over time | Medium | Low | Confirmed: effective support falls from 0.0239 to 0.0016 and edit support reaches zero |
| H4 | Source residual dominates the target field after commitment weakens | High | Medium | Confirmed: object-region contribution/target rises to 0.60-0.68 as preserve reaches 0.82-0.89 |
| H5 | Reference state is overwritten by generic hand-trigger state before fixed-budget pruning | High | Low | Confirmed as a secondary bug, but rejected as the visual root cause by post-fix output |
| H6 | Hidden reference prefill is the common cause of source regression | High | Medium | Rejected: prompt-only has the same identity drift and late blue-cap regression |
| H7 | The target velocity itself becomes source-like after early blocks | High | Medium | Confirmed: global target/source gap is only 3-9% of target magnitude after block 1; object-region gap is 8-18% in late blocks |
| H8 | Scalar current Q/K blending collapses the target branch toward source dynamics | High | Medium | Strongly supported: the first denoising step uses only 13% target Q/K and the observed target/source field gap collapses; causal ablation pending |
| H9 | Residual routing cannot exactly preserve background when the target prompt leaks outside the object | High | Low | Confirmed algebraically and by maps: at preserve=1 the routed field is `v_gt + (v_trg-v_src)`; block-0 background target/source gap is 31.7% of target magnitude |

## Log Evidence
Instrumentation added for:
- H3: reference support size, write strength, and hand contact.
- H1/H3: preserve action before and after commitment on the transported region.
- H2/H4: final preserve action and velocity contribution on the same region.

Evidence from commit `0b4d31c`:

- Bootstrap selection is valid: 43 tokens, 2.76% coverage, mean active write
  0.983, reference evidence 8.0.
- Block 0 active reference region works: commitment effective 0.486 and
  preserve action drops from 0.425 to 0.296; top-43 preserve is 0.191.
- In block 1, effective commitment drops to 0.149 on active tokens and
  preserve action rises to 0.771; top-43 preserve is 0.821.
- In blocks 4 and 5, global effective commitment is only 0.0016, edit support
  is 0.0094 then 0, and top-43 preserve is 0.815 then 0.867.
- Bayes preserve equals the action recomputed from final post-commit belief in
  every block, rejecting stale-belief routing.
- During block 0, generic trigger expands state support from the 43-token
  reference budget to 266/296/347 tokens. The next block prunes that mixed
  state back to 43 tokens, and transported commitment falls from about 0.88
  to 0.28-0.30.
- Compared with the old local-transport run, late top-43 preserve increased
  from about 0.54-0.55 to 0.82-0.87 after the previous-only/budget change.
- Commit `a616a32` independently reproduces the same trajectory: 569 saved
  arrays differ from `0b4d31c` by at most 1.19e-7. This excludes randomness
  and the server environment as explanations.

Post-fix evidence from commit `266c54d`:

- The state-separation patch changed the internal trajectory, so it was
  executed correctly.
- Block-0 transported posterior stayed at 1.0 and top-region preserve fell to
  0.18, but the rendered video was visually unchanged.
- The isolated reference precision then collapsed: mean transport was 0.0009
  in block 1 and exactly zero from block 2 onward.
- Therefore reference/trigger state pollution is real but is not the visual
  root cause. Stronger block-0 commitment is insufficient to alter generation.
- The state-separation behavior is reverted rather than accumulated.

Ablation evidence from commit `37557f8`:

- `prompt_only` has no reference or customized bootstrap, but still changes
  target identity across blocks and regresses to the source blue cap. This
  rejects hidden reference prefill as the common root cause.
- Prompt-only edit-token coverage is 5.79%, 10.19%, 9.64%, 11.97%, 8.35%,
  2.74%, and finally 0%. Object-region preserve action rises to 0.817 and
  0.891 in the last two blocks.
- Prompt-only object-region source contribution relative to target velocity
  rises from 0.336 in block 3 to 0.598 and 0.683 in blocks 5 and 6.
- After block 1, the global target/source velocity gap is only 3.2%-8.8% of
  target magnitude; in the inferred object region it is 8.1%-17.8% in the
  late blocks. The nominal target branch has already become source-like.
- In block 0 background, preserve action is 0.999 but the target/source gap
  is still 31.7% of target magnitude. The current residual equation leaves
  this prompt delta untouched, explaining Coca-Cola text in the pan.
- Prefill-only changes early field magnitudes but follows the same late
  support and source-contribution trajectory. It is not a persistent target
  force.
- The overlap rollout overwrites prefill-only
  `block_001_hand_role_debug.npz`; that artifact is the final overlap call,
  not the original block 1.

## Verification Conclusion

The failure is not reference-specific. There are two distinct mechanisms:

1. Scalar source-heavy current Q/K blending and source-latent initialization
   make the target branch converge toward source dynamics.
2. The residual router is not an exact source endpoint. Even full preserve
   leaves `v_trg-v_src`, so globally leaked target semantics remain in the
   background.

Late hand/object support collapse then raises preserve action inside the
object and exposes source identity. No additional memory-strength fix is
justified. The next causal test should remove source Q/K blending only on
current edit-support tokens while leaving background behavior unchanged.

## Next Instrumentation

- Add an opt-in spatial Q/K ablation: current edit-support tokens use pure
  target Q/K while background retains the original timestep blend.
- Keep the existing velocity maps for pre/post comparison.
- Fix rollout debug indexing before using overlap artifacts quantitatively.
