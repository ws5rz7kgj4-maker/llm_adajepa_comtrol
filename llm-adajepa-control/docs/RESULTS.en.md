# Results, evidence and limits

[中文](RESULTS.zh-CN.md) · [Home](../README.md)

## What the numbers represent

The archived audit covers five methods, two tasks, four conditions, 25 scenes per task–condition cell, and three training seeds:

$$5\times2\times4\times25\times3=3000\text{ episodes}.$$

There are 200 distinct scene configurations, shared across methods and training seeds. These are not 3,000 independent environments. Success comes from the simulator task-success check.

| Method | Successes / 600 | Rate | Invalid / proposal responses |
|---|---:|---:|---:|
| adaptive_only | 599 | 99.83% | No proposals |
| error | 504 | 84.00% | 348/991 (35.12%) |
| joint | 262 | 43.67% | 694/2239 (31.00%) |
| joint_async | 462 | 77.00% | 364/1834 (19.85%) |
| joint_rule | 396 | 66.00% | 0/2196 (rule outputs) |

The 7,260 proposal records include **5,064 VLM responses and 2,196 rule responses**. All 1,406 invalid VLM responses were rejected by the three-skill requirement. That is a schema failure category, not proof of malformed JSON or incorrect semantic reasoning.

## The aggregate hides a large task difference

| Method | Lift successes / 300 | PickPlaceCan successes / 300 |
|---|---:|---:|
| adaptive_only | 300 | 299 |
| error | 269 | 235 |
| joint | 260 | **2** |
| joint_async | 278 | **184** |
| joint_rule | 256 | 140 |

Synchronous joint control is particularly weak on PickPlaceCan. Asynchrony restores much of that performance but does not reach the adaptive-only baseline.

In matched scene–seed comparisons, async succeeds where sync fails in 206 cells; sync succeeds where async fails in 6. Async never succeeds on a cell where the adaptive-only baseline fails; adaptive-only alone succeeds in 137 cells. These are descriptive paired counts, not an independence-adjusted statistical test.

Recompute all tables:

```bash
python3 scripts/reproduce_results.py
```

See [condition breakdown](../results/by_condition.csv), [seed breakdown](../results/by_seed.csv), and [paired counts](../results/paired_descriptive_counts.csv).

## What is established, and what remains a hypothesis

| Evidence | Supported interpretation | Not established |
|---|---|---|
| 43.67% synchronous vs 77.00% asynchronous | The asynchronous setting performs better in these archived runs | Latency alone causes the entire difference |
| 99.83% adaptive-only | The low-level baseline is highly capable on these tasks | Broad real-world robustness |
| Invalid proposal records | Structured-output reliability is a real integration issue | Constrained decoding would recover a specific success rate |
| Grasp dwell/action mismatches | State-machine bookkeeping deserves investigation | A measured causal contribution from this bug |
| Five audited methods | A complete five-method audit subset | Completed evidence for all 21 protocol arms |

During grasp dwell, open-gripper commands appear in 1 adaptive-only, 70 error-triggered, 53 synchronous, 307 asynchronous and 320 rule-controlled episodes. These counts do not track total failure monotonically. They should not be used as a one-variable explanation.

## Reproducibility boundary

- This release rechecks the packaged evidence hashes, full episode grid, response records and parser behavior.
- Source configuration fingerprint matches the archived audit protocol. This alone does not prove every executed source byte was identical.
- Original NPZ SHA256 receipts are preserved from the historical audit; their source archives were not downloaded or independently reverified for this release.
- Training trajectories, checkpoints, full evaluation NPZs and RGB frames are not bundled.
- The full 501 MB step trace is replaced by a documented 10-episode illustration: nominal scene 000, seed 11, both tasks, all five methods. Selection is deterministic and is not used for aggregate statistics.
- No new simulation or model training was run during packaging. These are simulator results, not physical robot results.
- Simple binomial confidence intervals would ignore shared scenes and training seeds. This release reports descriptive counts; stronger inference needs a justified paired/hierarchical design and more independent runs.

## Next controlled experiments

Keep the archived release immutable. For a new protocol, separately test matched artificial delay, executed-skill-aware dwell counters, constrained decoding, and bounded skill handoffs. Compare each change to adaptive-only and the unchanged synchronous/asynchronous baselines. Report success, response validity, stale responses, execution timing and actual disturbance exposure.

Changing several mechanisms together may improve the score but will not identify which change helped.
