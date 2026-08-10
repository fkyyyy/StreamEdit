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
| H4 | Source residual dominates the target field after commitment weakens | High | Medium | Pending direct field maps; active-region preserve rises to 0.77-0.87 |
| H5 | Reference state is overwritten by generic hand-trigger state before fixed-budget pruning | High | Low | Confirmed as a secondary bug, but rejected as the visual root cause by post-fix output |
| H6 | Hidden reference prefill changes temporal initialization without supplying a persistent target force | High | Medium | Pending prompt-only versus prefill-only ablation |
| H7 | The target velocity itself remains source-like despite reference KV/identity support | High | Medium | Pending target-to-source and target-to-exact-source field maps |

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

## Verification Conclusion

The decisive failure is now upstream or downstream of commitment: either the
hidden reference prefill changes causal initialization without producing a
persistent target field, the model's target velocity is already source-like,
or source initialization/residual routing overwhelms the target field. The
next evidence must compare prompt-only and reference-prefill-only executions
and save velocity magnitudes spatially. No additional memory-strength fix is
justified before those results.

## Next Instrumentation

- Save target velocity, source residual, routed source contribution, final
  routed velocity, target-source gap, and target-exact-source gap maps.
- Run `hand_role_bayes_flow_identity_kv` with the Coca-Cola prompt and no
  reference.
- Run the identical mode with only hidden first-frame prefill enabled.
