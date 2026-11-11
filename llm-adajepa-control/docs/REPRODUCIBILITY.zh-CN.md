# 复现指南

[English](REPRODUCIBILITY.en.md) · [首页](../README.zh-CN.md)

## A. 在普通电脑上复算现有证据

在仓库根目录运行，要求 Python 3.10 及以上：

```bash
python3 scripts/verify_data.py
python3 scripts/reproduce_results.py
python3 scripts/demo.py --lang zh
python3 -m unittest discover -s tests -v
```

无需安装依赖或填写密钥。脚本检查 3000 个回合身份、7260 条建议记录，将 5064 条 VLM 回复重新交给原始解析器，并生成结果表。这是离线审计复算，不会重新训练、调用模型或独立核验原始 NPZ 压缩包。

需要把输出放到独立目录时：

```bash
python3 scripts/reproduce_results.py --out results/local
```

Linux／macOS 可选运行保留的 CPU 测试：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
PYTHONPATH=research python -m pytest -q research/tests/test_protocol.py research/tests/test_batch_storage.py research/tests/test_rendering.py research/tests/test_data_recovery.py research/tests/test_checkpoint_driver.py
```

其中渲染和 Hub 测试使用模拟对象，通过不代表本机已经实际完成 GPU 渲染或远端上传。模型测试和数据采集测试还需要 PyTorch，不属于最小离线演示。

## B. 重新开展仿真实验

原研究实现面向 Linux／CUDA，需要 PyTorch、torchvision、MuJoCo、robosuite 1.5.1 和 Qwen 权重，**不是 Apple MPS 移植版**。本包不含训练轨迹和模型权重；首次初始化会下载 ResNet、Qwen 的预训练权重。

历史启动代码检查的环境如下：

| 组件 | 源码中的预期值 |
|---|---|
| Python | 3.12 |
| PyTorch／torchvision | 2.14.1／0.29.1 |
| CUDA | 13.0 |
| 原云任务配置 | 四张 GPU |
| robosuite | 1.5.1 |
| Qwen revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` |

这些是原始文件中的设置，不表示本次已安装或重新验证该环境。[requirements.txt](../research/requirements.txt) 保留历史依赖。[cloud_entry.sh](../research/cloud_entry.sh) 为来源与恢复测试而保留，它依赖原云环境，不是普通电脑快速入口。

在已经配置好兼容 CUDA／PyTorch 的环境中安装其余依赖：

```bash
python -m pip install -r research/requirements.txt
python -m pip install --no-deps robosuite==1.5.1
```

系统渲染依赖参考 [robosuite 官方安装文档](https://robosuite.ai/docs/installation.html)。如果历史版本无法解析，应记录新环境并重新验证，不要悄悄替换后仍称为“完全相同的复现”。本次没有确认完整 GPU 环境的当前可安装性，也不承诺换硬件后得到相同分数。

只查看协议，不加载重型库：

```bash
python3 scripts/run_simulation.py --stage inspect
```

在渲染环境可用后，以下命令仅在本地写出结果，不提交云任务或上传 Hub：

```bash
export MUJOCO_GL=egl
python scripts/run_simulation.py --stage preflight --root runs/local
python scripts/run_simulation.py --stage collect --root runs/local
python scripts/run_simulation.py --stage check-data --root runs/local
python scripts/run_simulation.py --stage train --seed 11 --root runs/local
python scripts/run_simulation.py --stage evaluate --seed 11 --method adaptive_only --task Lift --condition nominal --scene-index 0 --root runs/local
python scripts/run_simulation.py --stage evaluate --seed 11 --method joint_async --task Lift --condition nominal --scene-index 0 --root runs/local
```

- `collect`：采集 240 条训练轨迹和 40 条验证轨迹。
- `train`：训练一个种子下的三个集成成员，检查验证门槛。
- `evaluate`：仅运行指定的**一个回合**，需要对应训练权重；已有同名回合时拒绝覆盖。

该本地入口是这次新增的整理代码，其 GPU 分支尚未在本次环境运行。若扩大实验，另行训练 29、47 两个种子，并完整覆盖计划场景。两个演示回合不构成新基准结果。

原 `experiment.evaluate` 默认会安排 21 组 × 200 场景 × 3 种子，即 12600 个回合，属于独立的大规模实验，不是离线快速演示。

## C. 保持证据可追溯

原始实验模块按字节保留；来源压缩包和模块哈希见 [provenance.json](provenance.json)，包内证据与源码哈希见 [manifest.json](../data/manifest.json)。

新实验应记录 Git commit、环境、模型版本、渲染器、配置、场景种子和产物哈希。即使旧配置指纹没变，只要修改触发器、控制器或输出约束，也应视为新的实验。

历史审计保留在 `data/audit`，新运行输出放入被 Git 忽略的 `runs/`。本包的原始 NPZ 回执继承自历史审计；独立核验需要另取原始压缩包。

原分析代码要求完整 21 组数据，不能拿本包五组审计去冒充完整实验。
