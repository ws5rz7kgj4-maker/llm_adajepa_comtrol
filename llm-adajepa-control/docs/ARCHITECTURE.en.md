# Architecture and mathematical model

[中文](ARCHITECTURE.zh-CN.md) · [Home](../README.md)

This document describes the preserved 3D source, not an idealized future system.

## 1. Inputs and actions

Source: [env.py](../research/experiment/env.py), [spec.py](../research/experiment/spec.py).

| Quantity | Actual implementation |
|---|---|
| Robot/tasks | Panda; Lift and PickPlaceCan; fixed wrist orientation |
| Image | 128×128 RGB from agentview, vertically flipped |
| State `s_t` | 12 values: end-effector xyz, object xyz, two gripper joints, goal xyz, task bit |
| Action `a_t` | 4 values: normalized Cartesian delta xyz and gripper command |
| Control target | 20 Hz; 240-step cap per episode |
| Test shifts | nominal; mass/inertia ×4; sliding friction ×0.15; 2 N world-x force at steps 50–54 |

Cartesian action values are normalized controller commands, not direct meter displacements. Object coordinates are simulator observations. The 20 Hz target is not a hard real-time guarantee or the physics integrator frequency.

## 2. Representation and transition prediction

Source: [model.py](../research/experiment/model.py).

The frozen ImageNet ResNet-18 returns `f(I_t)∈R^512`. The concatenated representation is standardized using training statistics:

$$z_t=\operatorname{Normalize}([f(I_t),s_t])\in\mathbb{R}^{524}.$$

Each of three residual predictors has input dimension `524+4+8=536` and hidden widths 256, 256 with SiLU activation:

$$\hat z_{t+1}^{(m)}=z_t+M\odot g_{\theta_m}([z_t,a_t,v_t]).$$

`M` is 1 for the first 520 entries and 0 for the last 4: goal xyz and task identity remain fixed. `v_t` contains eight normalized-state velocities for temporal variants; it is zero for the five audited methods. The predictors learn transitions; the visual encoder is not updated.

For one transition:

$$
\ell(\hat z,z')=
\frac{1}{512}\sum_{j=1}^{512}(\hat z_j-z'_j)^2+
\frac{1}{8}\sum_{j=513}^{520}(\hat z_j-z'_j)^2.
$$

The two groups are averaged separately so the many visual dimensions do not simply outweigh the dynamic state dimensions. There is no pixel reconstruction decoder, language-model training loss, or official AdaJEPA transformer in this implementation.

## 3. Offline training and online adaptation

Source: [train.py](../research/experiment/train.py), [model.py](../research/experiment/model.py).

Offline training uses 240 nominal training trajectories and 40 nominal validation trajectories. For each outer seed (11, 29, 47), three predictors use bootstrap samples, AdamW at `3×10^-4`, batch size 256, up to 30 epochs, and gradient norm clipping at 1. Temporal features are randomly masked during training to support temporal and non-temporal variants.

The validation gate requires both total loss and physical-state loss to be more than 5% below a persistence predictor (`\hat z_{t+1}=z_t`), both with and without temporal inputs. This is a code-level gate; model checkpoints and validation loss files are not bundled here.

Online, a replay buffer holds up to 80 transitions. The latest 5 are held out from the current gradient update. At least 20 earlier transitions are needed; one Adam step is enabled at each five-step segment boundary:

$$\theta^+=\operatorname{AdamStep}(\theta,\nabla_\theta L_{\rm earlier};\eta=10^{-4}).$$

The diagnostic adaptation gain is:

$$g_{\rm adapt}=\frac{L_{\rm before}-L_{\rm after}}{\max(L_{\rm before},10^{-8})}.$$

Positive gain means this update reduced loss on the latest five transitions. It does **not** establish improved task success or generalization. Sliding windows overlap over time. Model weights and optimizer state reset between formal episodes.

## 4. Skill selection and model-predictive control

Available skills: `approach, descend, grasp, lift, carry, place, release, regrasp, retreat`.

The VLM proposes three alternatives. The world model compares them using a shared, stage-dependent heuristic target. It does not execute all three in sequence. In comparison-enabled arms, the model’s score, not the VLM’s `preferred` index, selects the skill.

For a selected skill, sample 32 action sequences of horizon `H=5` around its hand-coded controller. For predictor `m`, the terminal cost is:

$$
J_m(U)=\|\hat e_{t+H}^{(m)}-g_{\rm skill}\|_2^2+
\frac{0.001}{4H}\sum_{h=0}^{H-1}\|a_{t+h}\|_2^2.
$$

Here `\hat e` is the predicted **end-effector** position after denormalization. This is not a learned reward or an object-to-goal distance cost.

$$
U^*=\arg\min_U\left[\operatorname{mean}_mJ_m(U)+
\lambda\operatorname{std}_mJ_m(U)\right].
$$

Execute only the first action and replan. All five audited arms use `\lambda=0`; the source also defines risk variants with `\lambda=1`, but their results are not included. Ensemble disagreement is an uncertainty proxy, not calibrated safety probability.

## 5. The intervention rule

Source: [core.py](../research/experiment/core.py), [evaluate.py](../research/experiment/evaluate.py).

After an initial step-0 request, the joint trigger uses the last four five-step feedback segments:

$$
S=\left[\sum_{k=1}^{4}\Delta d_k<0.005\right],\quad
H=\left[\frac{1}{4}\sum_{k=1}^{4}e_k>\tau_{0.95}\right].
$$

`Δd` is progress in object-to-goal distance; `e` is prediction error. `τ` comes from validation errors. Let `A` mean that at least three valid adaptation gains exist and their mean is at most 0.02:

$$\operatorname{trigger}=S\land(A\lor\neg H).$$

Interpretation: intervene when progress stalls and either adaptation is ineffective or prediction error is already low. This heuristic does not prove which subsystem caused a failure.

Requests are limited to four per episode, with a 30-step cooldown and no second request while one is pending. The error-only arm triggers on high prediction error; fixed-trigger variants are defined separately.

## 6. Model interface and asynchronous execution

Source: [proposer.py](../research/experiment/proposer.py), [core.py](../research/experiment/core.py).

The frozen Qwen2.5-VL-3B receives a resized RGB image, task, state, recent history and current skill. Generation is greedy, with at most 192 new tokens and a pinned model revision. No language-model fine-tuning occurs.

```json
{"skills":["descend","grasp","lift"],"preferred":0}
```

The parser requires exactly three known skill entries and an integer index 0–2, with no extra keys. It tolerates surrounding prose/code fences but rejects multiple proposal objects. Repeated skill names are allowed by the historical parser; keep that behavior when replaying evidence.

A one-worker thread handles inference; the main thread owns physics and world-model updates. In synchronous mode the robot holds position using its last gripper command while physics advances. In asynchronous mode the bottom controller continues while a request is pending.

A returned response is stale if the episode differs, its step age exceeds 60, object displacement exceeds 0.08 m, or wall age exceeds 15 s. Valid replies are scored on the **current** state/model and override the skill for up to 20 steps. End-of-episode replies are drained and never transferred to a new episode.

The thread cannot forcibly cancel an in-progress CUDA operation. The 15-second freshness threshold is not a guaranteed inference termination deadline. The old aggregate `stale_calls` field excludes late replies drained after episode termination; inspect individual records for diagnosis.

## 7. Known diagnostic issues

The common `SkillPolicy.skill` method can advance dwell counters even when an override or hold action is executed. The audit records phase/action discrepancies and open-gripper commands during grasp dwell. These are diagnostic observations, not an isolated causal estimate. A new experiment should gate counters by executed behavior and compare against the unchanged baseline.

The workspace monitor checks finite state and simple end-effector bounds; it does not certify collision avoidance or contact forces. Some episodes succeed before the scheduled impulse, so an “impulse” label does not prove every episode experienced that force.
