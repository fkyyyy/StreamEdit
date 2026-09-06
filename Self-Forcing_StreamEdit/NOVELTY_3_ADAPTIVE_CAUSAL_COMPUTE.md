# Novelty 3: Belief-Aware Adaptive Causal Computation for Long-Horizon Video Editing

## 1. Status

This document records the proposed third contribution of StreamEdit. It is a
research and implementation plan, not a description of functionality that is
already implemented.

The recommended direction is not to treat long-video support or fixed 2-step /
4-step inference as an isolated contribution. Instead, StreamEdit should use
the uncertainty already produced by its role inference and memory system to
adapt causal inference compute and trigger recovery. Long-video editing is the
main setting in which this mechanism should be evaluated.

Proposed contribution:

> **Belief-aware adaptive causal computation dynamically selects a 2-step fast
> path, a 4-step correction path, or event-triggered re-anchoring according to
> token-role uncertainty and memory health, enabling efficient and stable
> long-horizon streaming video editing.**

## 2. Unified Method Story

The three contributions should be presented as one closed-loop method rather
than three independent modules.

1. **Role perception -- who should be edited?**
   Hand information, model-native counterfactual velocity, internal attention,
   and temporal evidence infer a posterior over token roles:
   `object`, `contact`, `hand`, and `background`.
2. **Role-conditioned state control -- what should be changed and remembered?**
   The same role posterior controls current velocity routing and KV / delta-V
   memory read, write, and abstention.
3. **Belief-aware computation -- when should more computation or recovery be
   used?**
   Role uncertainty, edit-preserve conflict, temporal reliability, role drift,
   and memory mismatch select the causal denoising budget and recovery action.

The resulting loop is:

```text
hand prior + counterfactual velocity + model evidence
                         |
                         v
          token-role belief q_t(z) and uncertainty
                         |
             +-----------+-----------+
             |                       |
             v                       v
       S1 velocity control      M2 memory control
       edit / preserve          read / write / abstain
             |                       |
             +-----------+-----------+
                         |
                         v
           generated latent and memory health
                         |
                         v
       2-step / 4-step / recovery and re-anchoring
                         |
                         v
                next observation q_(t+1)(z)
```

This gives the paper a single framing:

> **Closed-loop belief-controlled streaming video editing.**

## 3. Formalization

Let the belief state for block `t` be

```text
b_t = {q_t(z), U_t, R_t, C_t, D_t, M_t}
```

where:

- `q_t(z)` is the posterior over object/contact/hand/background roles;
- `U_t` is role entropy or uncertainty;
- `R_t` is temporal and counterfactual-evidence reliability;
- `C_t` is edit-preserve conflict;
- `D_t` is temporal role drift;
- `M_t` summarizes memory health, age, retrieval agreement, and contamination
  risk.

An interpretable risk score can initially be used instead of a learned policy:

```text
risk_t = alpha * role_entropy
       + beta  * edit_preserve_conflict
       + gamma * (1 - temporal_reliability)
       + delta * role_drift
       + eta   * memory_mismatch
```

The block-level causal-compute action is

```text
2-step fast path             if risk_t < tau_low
4-step correction path       if tau_low <= risk_t < tau_high
4-step + recovery/re-anchor  if risk_t >= tau_high
```

The controller should use asymmetric hysteresis:

- escalation to 4-step or recovery is immediate;
- returning to 2-step requires at least two consecutive low-risk blocks;
- a recovery block is read-only and cannot update M2 memory;
- memory writing resumes only after two consecutive reliable observations;
- all thresholds should be normalized or calibrated online rather than tuned
  to one video.

## 4. Actions and State Machine

### 4.1 Fast state: stable belief

Use 2-step inference when role confidence is high, edit-preserve conflict is
low, and retrieved memory agrees with the current counterfactual direction.

- S1 uses normal role-conditioned velocity routing.
- M2 reads normally.
- Only the stable object core may write.
- This is the default low-cost path.

### 4.2 Correction state: ambiguous or fast motion

Use 4-step inference when motion, contact, occlusion, or role entropy increases.

- Spend extra denoising computation on the entire block.
- Keep M2 reads active.
- Tighten the memory-write criterion.
- Re-estimate the role posterior after correction before allowing a write.

### 4.3 Recovery state: unreliable observation or memory mismatch

Use 4-step inference plus event-triggered re-anchoring when the current belief
is unreliable or conflicts strongly with trusted history.

- Freeze all M2 writes.
- Read only from high-confidence, low-age owner memory.
- Reduce the owner-area budget to the reliable rigid core.
- Reconstruct the current owner state from trusted historical slots.
- Resume normal operation only after temporal consensus is restored.

The current temporal-recovery behavior provides the starting point: an
unreliable observation already reduces owner extent, permits memory reads, and
sets memory writes to zero. Novelty 3 extends this behavior to adaptive compute
and explicit re-anchoring.

## 5. Connection to Existing S1 and M2

The compute controller must consume existing model-native signals instead of
introducing a separate RGB optical-flow pipeline or external object mask.
Candidate signals already available or directly derivable include:

- role entropy and object/contact/hand/background posterior;
- temporal evidence reliability and visibility recovery;
- edit belief, preserve belief, and their conflict;
- owner support area and its change from transported history;
- query-cycle confidence and missing-frame count;
- delta-V retrieval similarity and current/retrieved direction mismatch;
- pre-consensus versus accepted M2 write support;
- closed-loop delta-V residual magnitude.

The same posterior must remain the common interface across all contributions:

```text
q_t(z) -> S1 velocity routing
q_t(z) -> M2 read/write/abstention
q_t(z), M_t -> causal compute schedule and recovery
```

This dependency is important to the novelty claim. Without role inference, the
system does not know where uncertainty occurs. Without role-conditioned memory,
the controller cannot distinguish a transient observation error from persistent
identity drift. Without adaptive computation, difficult blocks receive the same
budget as easy blocks and long-horizon errors accumulate unnecessarily.

## 6. Why Long Video Is an Evaluation, Not the Mechanism

Long-video support alone is not a sufficiently specific algorithmic
contribution. It should demonstrate the value of the closed-loop controller:

- fixed low-step inference should be efficient but drift under occlusion and
  repeated interaction;
- fixed high-step inference should be stable but expensive;
- adaptive inference should approach the cost of the low-step baseline while
  approaching or exceeding the consistency of the high-step baseline;
- event-triggered recovery should stop isolated role errors from contaminating
  later blocks.

The target result is:

> Adaptive 2/4-step inference achieves long-horizon edit and identity
> consistency comparable to or better than fixed 4-step inference at an average
> computation closer to fixed 2-step inference.

## 7. Recommended Implementation Order

### Phase A: instrumentation and replayable risk analysis

1. Aggregate the existing token-level signals into one record per rollout
   block.
2. Save the proposed risk terms, selected state, memory health, and expected
   compute action without changing inference.
3. Replay existing runs offline to check whether known failure blocks receive
   higher risk than stable blocks.

### Phase B: block-level 2/4-step switching

1. Add a block-level scheduler around the causal inference loop.
2. Implement 2-step and 4-step schedules with shared endpoints and compatible
   latent scaling.
3. Keep the schedule fixed within a block to preserve batching efficiency.
4. Record actual NFE, wall-clock time, selected action, and transition reason.

### Phase C: event-triggered recovery and re-anchoring

1. Reuse the existing recovery read-only policy.
2. Select trusted M2 slots using confidence, age, and retrieval agreement.
3. Re-anchor only the rigid object core; contact tokens remain read-only.
4. Require temporal consensus before exiting recovery or resuming writes.

### Phase D: long-horizon evaluation

1. Evaluate videos substantially longer than the current short cook example.
2. Include repeated occlusion, fast interaction, object release/re-grasp, and
   multiple interaction episodes.
3. Compare quality, drift, contamination, recovery, NFE, runtime, and memory
   growth.

Token-level asynchronous denoising should not be the first implementation. A
Transformer still processes all tokens together, so token-specific step counts
may add complexity without real wall-clock savings. Block-level switching gives
a cleaner method and a measurable compute benefit.

## 8. Required Ablations

At minimum, compare:

| Variant | Purpose |
| --- | --- |
| Fixed 2-step | Low-cost baseline and expected drift failure |
| Fixed 4-step | High-cost quality baseline |
| Adaptive 2/4-step | Tests uncertainty-driven compute allocation |
| Adaptive without M2 | Tests whether memory health is necessary |
| Adaptive without role uncertainty | Tests generic motion/confidence scheduling |
| Adaptive without re-anchor | Isolates event-triggered recovery |
| Adaptive without hysteresis | Measures schedule oscillation and instability |
| Fixed owner extent | Tests the interaction with adaptive role extent |
| Read/write memory without abstention | Measures memory contamination |

The compute comparison must match either average NFE or wall-clock budget. A
quality-only comparison between methods with different compute is insufficient.

## 9. Evaluation Metrics

Report metrics in five groups:

1. **Editing:** target edit score, source removal, and edit localization.
2. **Preservation:** background LPIPS/PSNR or feature distance outside the owner
   region.
3. **Temporal identity:** object embedding consistency, temporal warping error,
   and long-horizon drift.
4. **Memory health:** false-write rate, owner purity, recovery time, and
   contamination persistence.
5. **Efficiency:** average NFE, fraction of 2-step/4-step/recovery blocks, peak
   memory, and end-to-end wall-clock time.

Report metrics against video duration to show whether errors remain bounded or
grow over time. Also report the delay between an uncertainty event and recovery.

## 10. Necessary Evaluation Cases

The benchmark set should contain:

- a held tool with no contact;
- a tool contacting rigid and deformable objects;
- fast motion and motion blur;
- temporary full or partial hand/object occlusion;
- object release and re-grasp;
- multiple candidate objects near the hand;
- weak motion or a stationary hand;
- different object sizes and interaction durations;
- multiple edits and prompts;
- long videos with repeated interaction cycles.

The cook example is a useful qualitative case but cannot by itself establish
generalization or an ICLR-level contribution.

## 11. Claim Boundary and Success Criteria

The intended claim is not merely that extra steps improve quality. The claim is
that the role/memory belief identifies when extra computation is useful and
when memory must be protected or repaired.

The contribution is successful only if experiments show all of the following:

- risk is predictive of future editing or identity failure;
- adaptive compute improves the quality-efficiency Pareto frontier over fixed
  schedules;
- recovery reduces persistent drift and false memory writes;
- the controller generalizes across scenes without case-specific thresholds;
- long-video error growth is lower than fixed 2-step and competitive with fixed
  4-step inference.

If adaptive scheduling only reproduces fixed 4-step quality at nearly 4-step
cost, it should be reported as an engineering mechanism rather than a major
novelty.

## 12. Candidate Paper Wording

Short contribution statement:

> We introduce a belief-aware causal compute controller that uses token-role
> uncertainty and memory consistency to dynamically switch between fast and
> corrective inference and to trigger read-only re-anchoring, preventing local
> errors from accumulating during long-horizon streaming edits.

Possible method names:

- Closed-Loop Belief-Controlled StreamEdit
- Role-Adaptive Causal StreamEdit
- Belief-Aware Causal Editing
- Interaction-Aware Adaptive Causal Control
