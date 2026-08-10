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
| H7 | The target velocity itself becomes source-like after early blocks | High | Medium | Confirmed: global target/source gap is only 3-9% of target magnitude after block 1 |
| H8 | Removing current Q/K source blend can restore identity without harming motion | High | Medium | Rejected: edit gap increases 2-2.6x, but hand/action become autonomously generated |
| H9 | Residual routing exactly preserves background when preserve action is one | High | Low | Rejected: it leaves `v_trg-v_src`; block-0 background gap is 31.7% of target magnitude |
| H10 | Q/K ablation changed source role inference rather than target generation | Medium | Low | Rejected: source attention, hand probability, and object posterior are bit-identical |
| H11 | Target-side writeback amplifies the Q/K intervention over blocks | High | Low | Confirmed: identity-support map diverges after block 0 and reaches max difference 0.694 |

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

Ablation evidence from commits `37557f8` and `820c573`:

- Prompt-only has no reference or customized bootstrap, but still changes
  target identity and regresses to the source blue cap. Hidden prefill is not
  the common root cause.
- Prompt-only edit-token coverage falls from 5.79%-11.97% to 2.74% and then
  zero. Object-region preserve rises to 0.817 and 0.891.
- Object-region source contribution relative to target velocity rises to
  0.598 and 0.683 in the final two blocks.
- Q/K unblending causally increases object-region target/source field gap:
  block 2 rises from 0.105 to 0.271 and block 3 from 0.133 to 0.328.
- It also increases the hand-region gap: block 2 rises from 0.050 to 0.144
  and block 3 from 0.084 to 0.202. Between 18% and 36% of Q/K edit support
  overlaps the hand in active blocks.
- Source attention, hand probability, and object posterior are exactly equal
  between prompt-only and Q/K-unblended runs in every block. The source role
  detector did not drift.
- Target-side identity support diverges after the first block; its maximum
  map difference reaches 0.647 in block 3 and 0.694 in block 4. Generated
  target output is written back and amplifies the intervention.
- User verification reports complete autonomous generation: source hand and
  action are no longer preserved. The pure-target Q/K ablation is rejected
  and reverted.

## Verification Conclusion

Current Q/K source blending is not simply an unwanted source bias. It is the
motion and hand-geometry anchor. Removing it increases target semantics, but
destroys source motion because the inferred edit support overlaps the hand
and self-attention propagates the intervention nonlocally.

The remaining failure must be solved without changing current Q/K. Appearance
control should operate on target V or on the semantic velocity residual
`v_trg-v_src`, while source Q/K continues to anchor motion. Target-side
writeback must not consolidate an autonomously generated observation as new
identity evidence.

## Next Instrumentation

- Keep the original scalar current Q/K blend.
- Do not use generated prompt-only identity as authoritative evidence.
- Design the next causal test on appearance-only V/semantic-delta control,
  with target writeback disabled for the test so feedback cannot confound it.
