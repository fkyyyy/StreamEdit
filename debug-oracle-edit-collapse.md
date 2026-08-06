# Debug Session: oracle-edit-collapse
- **Status**: [OPEN]
- **Issue**: Oracle role-flow editing is visible in early frames but collapses in later causal blocks.
- **Debug Server**: `http://10.74.55.101:7777/event`
- **Log File**: `.dbg/trae-debug-log-oracle-edit-collapse.ndjson`

## Reproduction Steps
1. Check out `roleflow-oracle`.
2. Run `STEP=15 bash Self-Forcing_StreamEdit/run_907_oracle_role_flow.sh`.
3. Inspect the generated video, role NPZ files, and `run.log`.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | Oracle object/edit roles shrink too early because of mask alignment or thresholding | High | Low | Rejected: block 3/4 retain 4.36%/4.50% edit coverage after edit gain collapses |
| B | Generated target KV writes progressively erase target identity | High | Medium | Pending: role remains stable while target/source velocity or attention gap falls |
| C | Original broad binary KV mask conflicts with oracle role flow and source-blends the target branch | Medium | Medium | Pending: compare effective KV mask against oracle edit role |
| D | Mid-denoising target-mask union causes a coverage or routing discontinuity | Medium | Medium | Pending: compare mask statistics before and after target injection |

## Log Evidence
- Commit `df79986` contains all 7 role blocks and the generated 81-frame video.
- Role partitions are exact and sum to one in every block.
- Relative to the successful oracle target, edit gain drops from about 30% in block 0 to about 2% in block 3, while block 3 edit coverage is 4.36%.
- Source-blue chroma retention rises to roughly 0.69-0.85 in later visible frames.

## Verification Conclusion
Pending.
