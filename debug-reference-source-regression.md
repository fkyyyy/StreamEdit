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
| H4 | Source residual dominates the target field after commitment weakens | Medium | Medium | Partially confirmed: active-region preserve rises to 0.77-0.87; field magnitudes were not saved |
| H5 | Reference state is overwritten by generic hand-trigger state before fixed-budget pruning | High | Low | Confirmed: block-0 state expands from 43 to 347 tokens and 95-100% of its top state overlaps trigger support |

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

## Verification Conclusion
The bootstrap component and identity prior are not the failure. The reference
commitment and generic hand-trigger commitment share one state. Online trigger
expands and overwrites the reference state in block 0; fixed-budget pruning in
the next block retains a mixed 43-token subset rather than an authoritative
reference track. The resulting commitment collapse restores a strong source
residual. Reference and online trigger states must be separated before any
further routing-strength changes.

## Minimal Fix

- In customized reference mode, transported reference posterior and precision
  are now the only values written back to the persistent reference track.
- Generic hand-trigger evidence is combined with reference evidence only for
  the current control action and cannot overwrite the reference track.
- The non-reference commitment path retains the original weighted update.
- Added a regression test in which a reference token and a disjoint hand
  trigger are both active; only the reference token may persist to the next
  frame.
