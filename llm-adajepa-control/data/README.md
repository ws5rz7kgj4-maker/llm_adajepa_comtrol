# Evidence data / 证据数据

| File | Contents / 内容 |
|---|---|
| audit/episodes.jsonl | 3,000 historical episode summaries / 历史逐回合摘要 |
| audit/calls.jsonl | 7,260 proposal records; 5,064 VLM + 2,196 rules / 建议与原始回复 |
| audit/report.json | Original historical audit report / 原审计汇总 |
| audit/file_receipts.json | Inherited SHA256 receipts for 3,000 original NPZs / 原 NPZ 哈希回执 |
| audit/trace_examples.jsonl | 1,557 steps from ten deterministic illustrative episodes / 十回合逐步示例 |
| audit/trace_selection.json | Selection rule and full trace hash / 选择规则与完整轨迹哈希 |
| manifest.json | Hashes of bundled evidence and preserved source / 包内证据与源码哈希 |

Each line is a JSON object. `success` is the episode outcome; `applied` indicates whether a returned proposal was accepted. `stale` and `episode_ended` are separate conditions. `recorded_invalid` and `reparsed_invalid` retain the historical interface failure reason.

每行是一个 JSON 对象。`success` 是回合结果，`applied` 表示建议是否被采用；`stale` 与 `episode_ended` 是不同字段。`recorded_invalid` 和 `reparsed_invalid` 保留历史接口拒绝原因。

Trace fields include state, next state, action, executed skill, pending request and reconstructed policy phase/dwell. Reconstructed phase is an audit interpretation, not an additional sensor. No RGB frames or model weights are in this release.

逐步示例包含状态、下一状态、动作、执行技能、待返回请求，以及重建的技能阶段与停留计数。重建阶段是审计推导，不是额外传感器。本包没有 RGB 帧或模型权重。

The ten illustrative episodes are nominal scene 000, seed 11, both tasks and all five methods. They are not used for aggregate rates. All 3,000 episode summaries are used for the statistics.

十个示例按确定规则选择：正常条件场景 000、种子 11、两个任务和五种方法。总体统计使用完整 3000 回合摘要，不使用这十个示例估计成功率。
