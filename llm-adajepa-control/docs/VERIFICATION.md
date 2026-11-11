# Release verification / 本次验证记录

Prepared: **2026-11-11**. This page distinguishes local software/evidence checks from a new robot experiment.

整理日期：**2026-11-11**。以下区分本次软件与证据检查，以及尚未重跑的机器人实验。

| Check / 检查 | Observed result / 实际结果 |
|---|---|
| Original v5 archive file manifest / 原包文件清单 | 36 listed files matched SHA256 / 36 个列出文件哈希一致 |
| Preserved source / 保留源码 | 23 files copied byte-for-byte, including selected tests and historical requirements/bootstrap / 23 个文件按字节保留 |
| Bundled source and evidence / 包内源码与证据 | 30 SHA256 entries verified / 30 项哈希核验 |
| Audit coverage / 审计覆盖 | 3,000 of 3,000 expected episode identities; no duplicates / 完整覆盖且无重复 |
| Proposal records / 建议记录 | 7,260: 5,064 VLM + 2,196 rules / 数量与逐回合记录一致 |
| Historical parser replay / 历史解析器重放 | 5,064 VLM replies, zero mismatches / 5064 条，无不一致 |
| New offline regressions / 新离线回归测试 | **18 passed** |
| Preserved CPU tests / 保留的 CPU 测试 | **67 passed** |
| Historical tables / 历史结果表 | Recomputed from all episode records / 从完整回合数据复算 |
| Plot / 结果图 | Generated from verified records and visually checked / 根据核验数据生成并目视检查 |
| GPU stages / GPU 流程 | Not rerun / 本次未重跑 |
| Real robot / 真机 | Not tested / 未测试 |
| Remote GitHub CI / 远端 CI | Workflow supplied; not executed on GitHub during packaging / 已提供配置，本次未在 GitHub 执行 |

## Commands for the checks / 检查命令

```bash
python3 scripts/verify_data.py
python3 scripts/check_repository.py
python3 -m unittest discover -s tests -v
python3 scripts/reproduce_results.py
python3 scripts/demo.py --lang en
python3 scripts/demo.py --lang zh
PYTHONPATH=research python -m pytest -q research/tests/test_protocol.py research/tests/test_batch_storage.py research/tests/test_rendering.py research/tests/test_data_recovery.py research/tests/test_checkpoint_driver.py
```

The CPU suite includes parser/trigger behavior, thread delivery, stale-response handling, atomic/batched storage, mocked renderer selection, recovery checks and checkpoint orchestration. Model and collection tests requiring PyTorch were preserved but not run in this packaging environment. Local verification used Python 3.12; the workflow also targets Python 3.11, whose remote execution is not claimed.

CPU 测试涵盖解析／触发、线程返回、过期处理、原子与批量存储、模拟渲染选择、恢复检查和检查点编排。需要 PyTorch 的模型与采集测试已保留，但未在本次环境运行。本地验证使用 Python 3.12；工作流另配置 Python 3.11，但不声称其远端运行已通过。

Hash validation checks consistency of the provided files. It does not independently prove the archived simulation occurred or validate the original NPZ files that are not distributed here. The source configuration agrees with the historical protocol; exact executed source identity remains a provenance limitation.

哈希检查证明本包提供文件的一致性，不会独立证明历史仿真发生过，也不等于核验未打包的原始 NPZ。配置与历史协议一致，但实际执行源码的完整身份仍是来源边界。
