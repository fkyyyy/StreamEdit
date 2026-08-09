# Debug Session: reference-source-regression
- **Status**: [OPEN]
- **Issue**: Customized reference editing still regresses to the source object after a few blocks, with no visible improvement after commit b9162fe.
- **Debug Server**: http://10.254.206.67:7777/event
- **Log File**: .dbg/trae-debug-log-reference-source-regression.ndjson

## Reproduction Steps
1. Run `hand_role_bayes_flow_customized_kv` with the aligned Coca-Cola first-frame reference.
2. Observe that the first blocks contain the target can.
3. Observe that later blocks regress to the white source bottle and blue cap.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Expected Evidence |
|----|------------|------------|--------|-------------------|
| H1 | Commitment updates belief after the Bayes action was already computed | High | Low | Post-commit belief changes while final Bayes action remains equal to the pre-commit action |
| H2 | Velocity routing consumes a stale pre-commit preserve map | High | Low | Final velocity preserve coefficient matches pre-commit rather than post-commit belief |
| H3 | Reference transport does not spatially cover the target object tokens | Medium | Low | Transport support is small or has weak overlap with target attention |
| H4 | The post-commit action changes correctly but the source residual magnitude dominates the target field | Medium | Medium | Low preserve coefficient but source residual contribution remains larger than target velocity |

## Log Evidence
Instrumentation added for:
- H3: reference support size, write strength, and hand contact.
- H1/H3: preserve action before and after commitment on the transported region.
- H2/H4: final preserve action and velocity contribution on the same region.

Collector connectivity verified locally and the log was cleared before the
pre-fix run. Runtime evidence is pending.

## Verification Conclusion
Pending.
