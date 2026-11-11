# Reproduction guide

[中文](REPRODUCIBILITY.zh-CN.md) · [Home](../README.md)

## A. Reproduce the bundled audit on a laptop

From the repository root, with Python 3.10+:

```bash
python3 scripts/verify_data.py
python3 scripts/reproduce_results.py
python3 scripts/demo.py --lang en
python3 -m unittest discover -s tests -v
```

No installation or credentials are needed. This verifies all 3,000 episode identities, all 7,260 proposal records, replays 5,064 VLM replies through the original parser, and recomputes results. It does not recreate missing training data, run a model or authenticate the original NPZ archives independently.

To keep generated output separate:

```bash
python3 scripts/reproduce_results.py --out results/local
```

Optional historical CPU tests on Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
PYTHONPATH=research python -m pytest -q research/tests/test_protocol.py research/tests/test_batch_storage.py research/tests/test_rendering.py research/tests/test_data_recovery.py research/tests/test_checkpoint_driver.py
```

The rendering and Hub tests use mocks; passing them does not demonstrate live GPU rendering or network publication. The model and collection tests also require PyTorch; they are not part of the laptop quickstart.

## B. Start a new simulation experiment

The preserved research pipeline uses Linux/CUDA, PyTorch/torchvision, MuJoCo, robosuite 1.5.1 and Qwen weights. It is not an Apple MPS port. Model weights and trajectories are not included. ResNet and Qwen download their pretrained weights when first initialized.

The original bootstrap source checks this runtime:

| Component | Snapshot expectation |
|---|---|
| Python | 3.12 |
| PyTorch / torchvision | 2.14.1 / 0.29.1 |
| CUDA | 13.0 |
| Original cloud worker allocation | Four GPUs |
| robosuite | 1.5.1 |
| Qwen revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` |

These are values in the historical source, not a claim that this release installed or revalidated that environment. Dependency pins are preserved in [requirements.txt](../research/requirements.txt). The archived [cloud bootstrap](../research/cloud_entry.sh) is retained for source-identity/recovery tests; it assumes its original runtime and is **not the quickstart**.

For an already provisioned compatible CUDA environment, install the remaining dependencies:

```bash
python -m pip install -r research/requirements.txt
python -m pip install --no-deps robosuite==1.5.1
```

Use the [official robosuite installation guide](https://robosuite.ai/docs/installation.html) for system rendering requirements. If pins cannot be resolved, record a new environment and validate it rather than calling it an identical reproduction. This release does not certify current package availability or exact score reproduction under different hardware.

Inspect the protocol without loading heavy libraries:

```bash
python3 scripts/run_simulation.py --stage inspect
```

After configuring a working renderer, the following commands use **local output only**, do not submit a cloud job, and do not publish to a Hub:

```bash
export MUJOCO_GL=egl
python scripts/run_simulation.py --stage preflight --root runs/local
python scripts/run_simulation.py --stage collect --root runs/local
python scripts/run_simulation.py --stage check-data --root runs/local
python scripts/run_simulation.py --stage train --seed 11 --root runs/local
python scripts/run_simulation.py --stage evaluate --seed 11 --method adaptive_only --task Lift --condition nominal --scene-index 0 --root runs/local
python scripts/run_simulation.py --stage evaluate --seed 11 --method joint_async --task Lift --condition nominal --scene-index 0 --root runs/local
```

`collect` generates 240 training plus 40 validation trajectories. `train` trains one seed’s three-member ensemble and applies the validation gate. `evaluate` runs **one selected episode**, refuses to overwrite an existing episode, and requires its trained checkpoint. This local wrapper is new packaging code; its GPU stages have not been exercised for this release.

Train seeds 29 and 47 separately when expanding the experiment. Do not interpret two selected episodes as a new benchmark. The original `experiment.evaluate` module schedules all 21 arms × 200 scenes × 3 seeds (12,600 episodes); that is a large separate experiment, not the demo.

## C. Keep evidence traceable

Original experiment modules are preserved byte-for-byte. Their hashes and source archive hashes are in [provenance.json](provenance.json); bundled source/data hashes are in [manifest.json](../data/manifest.json).

For new experiments, record the Git commit, environment, model revision, renderer, configuration, scene seeds and artifact hashes. Changing triggers, controller code or output constraints creates a new experiment even if the old configuration fingerprint remains the same.

Current raw audit files remain under `data/audit`. New outputs go under ignored `runs/`. The original audit contains inherited receipts; independently checking them requires the original NPZ archives, which this ZIP does not distribute.

The original 21-arm analysis expects complete data and must not be used to imply that the five-arm audit contains the entire experiment.
