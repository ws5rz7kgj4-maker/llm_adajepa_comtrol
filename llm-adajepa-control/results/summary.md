# Historical audit results (offline recalculation)

3,000 episodes; 600 per method; the same 200 scenes × 3 training seeds. Descriptive historical statistics, not a new simulation run. Rule replies are not VLM calls.

| Method | Successful episodes | Success rate | Invalid / responses |
|---|---:|---:|---:|
| `adaptive_only` | 599/600 | 99.83% | — |
| `error` | 504/600 | 84.00% | 348/991 |
| `joint` | 262/600 | 43.67% | 694/2239 |
| `joint_async` | 462/600 | 77.00% | 364/1834 |
| `joint_rule` | 396/600 | 66.00% | 0/2196 |
