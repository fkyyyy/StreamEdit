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
| H12 | Latest poor reference output came from a prompt/script mismatch | Medium | Low | Rejected: local and remote scripts have identical SHA and use the Coca-Cola prompt |
| H13 | Static first-frame trajectory anchoring is inactive because logs show zero strength | High | Low | Rejected: only step 0 is logged; the remaining 14 steps execute and produce up to 0.34 single-step pull |
| H14 | Self-reference anchoring only marks the inferred object | High | Low | Rejected: it zeros preserve action for all 6240 cached tokens in chunk 1 and 9360 in chunk 2 |
| H15 | Reference KV persistence has consistent dynamics across rollout chunks | High | Low | Rejected: target/source cosine drops from about 0.99 to 0.7598 at chunk 2 |
| H16 | Text-only block 0 is generated under the same identity constraint as later blocks | High | Low | Rejected: block-0 identity read support is exactly zero; memory is created only after generation |
| H17 | Text-only identity memory preserves the first generated identity as an anchor | High | Low | Rejected: later blocks update the prototype with gains 0.429, 0.256, and 0.250 |
| H18 | Identity support remains aligned with the inferred object over time | High | Low | Rejected: identity-top to object-top overlap falls from 52.4% to 21.4-28.6% |
| H19 | Identity support sufficiently releases source preservation | High | Low | Rejected: preserve action on identity-top tokens remains 0.64-0.79 and late object preserve reaches 0.89 |

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

Latest remote evidence from commit `347195e` on top of `cdb4111`:

- The remote branch is 26 commits ahead of the last controlled baseline
  `4c14854`. It contains multiple interleaved code and output commits for
  reference KV persistence, identity velocity override, trajectory anchoring,
  key reweighting, belief-cache overrides, and self-reference anchoring.
- `ref_target_latent` is the single aligned first-frame target latent. It is
  expanded to every generated frame and applied after every denoising step
  using `0.4 * identity_mask * (1 - timestep)`.
- Logging occurs only at denoising step 0, where progress is exactly zero.
  Using the observed block-0 mask mean/peak, the 15-step cumulative pull is
  approximately 0.34 globally on the dilated support and 0.93 at its peak.
  Block 5 reaches approximately 0.36 mean and 0.94 peak cumulative pull.
- Self-reference anchoring sets `preserve_action=0` for every cached token up
  to `local_end_index`, not just object tokens. The runtime log reports 6240
  affected tokens in chunk 1 and 9360 in chunk 2.
- Reference keys are multiplied by 2.0 only when the persisted cache is
  re-injected in chunk 2. At that boundary, target/source velocity cosine
  drops from 0.9903 in the preceding block to 0.7598, while target/source gap
  jumps from 0.3893 to 2.9963.
- The underlying Bayes route remains source-heavy: preserve action is
  0.925-0.962, late edit support reaches zero, and object-region preserve is
  about 0.90-0.95. The new anchors bypass rather than repair this route.
- `block_001_hand_role_debug.npz` is again overwritten by the overlap chunk,
  so its apparent block index is not globally aligned.

Text-only identity evidence from `907_reference_causal_ablation/prompt_only`:

- Block 0 has zero identity read support and zero identity edit tokens. Its
  generated target KV initializes identity memory only after the block is
  complete, with update gain 1.0 and evidence 0.3361.
- Blocks 1, 2, and 3 update that prototype with gains 0.4291, 0.2564, and
  0.2503. By block 3, the original block-0 observation accounts for only
  about one third of accumulated evidence.
- Identity support grows from mean 0.0197 in block 1 to about 0.056 later,
  so frame 0 and frames 10/20 are generated under structurally different
  conditions.
- Only 52.4%, 46.6%, and 47.2% of top identity tokens overlap top inferred
  object tokens in blocks 1-3. The ratio falls to 21.4-28.6% in blocks 4-6.
- Preserve action on top identity tokens is still 0.639-0.794. Late inferred
  object preserve action rises to 0.817 and 0.891, exposing source identity.

## Verification Conclusion

The latest reference path is not a controlled implementation of one method.
It combines a static latent trajectory pull, global first-block cache
reclassification, reference-key norm scaling, cross-chunk KV reinjection,
ordinary Bayes routing, and generated-target writeback. These mechanisms have
different support domains and activate differently across rollout chunks.
The poor result is therefore expected and cannot validate any single idea.

The last controlled baseline remains `4c14854`. Reference work should restart
from that code state while retaining later output commits only as failed
ablation evidence. Source Q/K must remain the motion anchor; reference should
control appearance through one isolated mechanism at a time.

For text-only editing, the current slow identity memory cannot guarantee
first-to-later identity consistency. Block 0 is unconditioned, then becomes a
low-evidence observation that is continuously averaged with later generated
identities. The read support also drifts away from the inferred object while
source preservation remains strong. Fixing only memory strength or only
routing cannot remove this initialization asymmetry.

## Next Instrumentation

- Preserve the 26 remote commits; do not rewrite branch history.
- Create a clean code baseline equivalent to `4c14854` while keeping the
  latest Coca-Cola script and all output artifacts.
- Reintroduce only reference KV persistence first, without trajectory anchor,
  key scaling, self-reference anchoring, or source-KV cache reclassification.
- Fix global debug block indexing before comparing cross-chunk behavior.
- For text-only identity, causally separate initialization from online
  adaptation: compare a frozen block-0 prototype against the current moving
  average while keeping Q/K and Bayes routing unchanged.
- Save prototype-to-block-0 key/value drift per layer before changing the
  identity update rule.
