# Preserved research source / 保留的研究源码

The `experiment/*.py` files, selected historical tests, `requirements.txt` and `cloud_entry.sh` are copied byte-for-byte from the v5 snapshot. The original archive manifest was checked before copying. See [provenance](../docs/provenance.json).

`experiment/*.py`、部分历史测试、`requirements.txt` 与 `cloud_entry.sh` 按字节保留自 v5 快照。复制前已检查原包的文件清单哈希，详见[来源记录](../docs/provenance.json)。

Use [the reproduction guide](../docs/REPRODUCIBILITY.en.md) / [复现指南](../docs/REPRODUCIBILITY.zh-CN.md). The offline entry points are under `scripts/`, not this directory.

| Module | Purpose / 功能 |
|---|---|
| spec.py | Fixed protocol and scene manifests / 固定协议与场景 |
| env.py | Panda tasks, disturbances and skill controller / 任务、扰动、技能控制 |
| model.py | Latent predictor ensemble, adaptation and MPC / 动力学集成、适应与规划 |
| train.py | Collection, validation and training / 数据采集、验证与训练 |
| proposer.py | Frozen Qwen VLM and rule proposer / Qwen 与规则建议器 |
| core.py | Parser, trigger and asynchronous bus / 解析、触发与异步 |
| evaluate.py | Episode loop and result capture / 回合执行与日志 |
| storage.py | Atomic local writes and checkpoint publishing / 原子保存与检查点 |
| driver.py | Original multi-worker orchestration / 原多进程编排 |
| recovery.py | Source and data recovery checks / 来源与数据恢复检查 |
| rendering.py, preflight.py | Renderer and model/environment checks / 渲染与预检 |
| analysis.py, check_cloud.py | Original full-protocol analysis and checks / 原完整协议分析与检查 |

Only five arms have bundled audit results. Other arms remain code/configuration, not demonstrated results in this release. Original paid-job submission scripts and account configuration are omitted; the retained bootstrap is archival and assumes its original environment.

本包仅有五组审计证据，其余实验组只表示源码中有定义。个人付费云任务提交脚本与账户配置未打包；保留的启动脚本仅用于历史源码检查，依赖原环境。
