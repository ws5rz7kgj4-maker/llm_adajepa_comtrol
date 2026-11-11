# Interview guide: LLMs and adaptive world models

[中文](INTERVIEW.zh-CN.md) · [Home](../README.md) · [Equations](ARCHITECTURE.en.md)

Use the 3D Panda version consistently. Do not mix its numbers with an earlier 2D pushing experiment or a separate dual-arm project.

## One-minute introduction

> My project investigates how a vision-language planner interacts with an adaptive controller in Panda manipulation simulation. A frozen Qwen2.5-VL model proposes three candidate skills from an image, state and feedback. A learned dynamics ensemble compares those candidates, performs short-horizon planning and adapts to recent transitions.
>
> I focused on integration reliability: when to call the model, whether its output meets the interface contract, and whether the answer is still relevant when it arrives. In the 3,000-episode audit, asynchronous execution reached 77% success compared with 43.67% synchronous execution, but the adaptive-only baseline reached 99.83%.
>
> This negative result makes task-level evaluation essential. I used AI assistance in development and preparation, and I explain the mechanisms through source code, archived outputs and reproducible analysis.

Adjust personal contribution statements to your actual involvement. The repository supports technical claims, not an assumption that every line was written unaided.

## Five-minute structure

1. **Problem, 40 seconds:** local control and higher-level strategy operate at different speeds; the combination needs measurement.
2. **Architecture, 70 seconds:** image/state → representation → dynamics/MPC; feedback → VLM → validation → current-state candidate comparison.
3. **Protocol, 50 seconds:** two tasks, four conditions, 200 scenes, three training seeds and five audited methods.
4. **Results, 60 seconds:** start with the adaptive-only baseline; explain synchronous/asynchronous differences and the PickPlaceCan split.
5. **Diagnosis, 60 seconds:** show a two-skill response and a stale response; distinguish observed problems from causal hypotheses.

## Three-minute live walkthrough

```bash
python3 scripts/demo.py --lang en
python3 scripts/verify_data.py
python3 scripts/reproduce_results.py
```

Say explicitly that this is archived evidence replay, not live inference. Show an accepted response, a schema failure and a structurally valid but stale response. Open `results/by_task.csv`, then `research/experiment/core.py` to connect the evidence to the parser, trigger and freshness checks.

## Likely questions

**1. Is this a real LLM project?**  
It integrates an actual pretrained multimodal model into a simulated execution loop, with archived model outputs and outcomes. It is not LLM pretraining, a production deployment or a physical-robot demonstration. If the role requires a shipped product, explain that boundary.

**2. LLM or VLM?**  
The upper layer is Qwen2.5-VL and receives both image and text inputs. “Multimodal LLM” or “vision-language model” is more precise than “text-only LLM.”

**3. Is this official AdaJEPA?**  
No. It is inspired by adapting latent dynamics inside planning, using frozen ResNet features, observed state and residual MLP predictors. Do not claim official weights, benchmark reproduction or superiority over the paper.

**4. Why a world model?**  
It predicts action consequences so candidate skills and continuous actions can be compared before execution. The VLM suggests skills; the lower layer handles continuous control.

**5. What is trained?**  
The dynamics predictors are trained offline and adapted within each episode. Qwen and the visual encoder are frozen. There is no language-model fine-tuning, RLHF or DPO. Each of three outer training seeds has a three-member ensemble.

**6. Where does 524 come from?**  
512 visual features plus 12 state entries: end-effector xyz, object xyz, two gripper joints, goal xyz and one task bit.

**7. What is the loss?**  
Visual-feature MSE plus dynamic-state MSE, averaged separately. Goal coordinates and the task bit are fixed. It is latent transition prediction, not pixel reconstruction.

**8. Why online adaptation?**  
Training uses nominal dynamics; tests vary mass, friction and external force. Recent transitions reveal current dynamics. Better prediction still needs to be checked against actual task success.

**9. How is leakage controlled?**  
Scene seeds are separated by split, thresholds come from validation, and weights/optimizer reset per episode. Test-time adaptation uses observed past transitions. Overlapping rolling audit windows do not constitute independent long-term generalization evidence.

**10. What does MPC do here?**  
Evaluate 32 five-step candidate action sequences, execute the first action of the lowest-cost sequence, observe again and replan. The position term measures predicted end-effector distance to a skill target.

**11. Are the three skills a sequence?**  
No. They are alternatives for the next skill. In comparison-enabled arms the world model selects one, rather than directly following the VLM’s preferred index.

**12. Why not call the model every step?**  
Inference adds latency, resource use and interface failures. Calls are event-triggered, capped at four per episode and separated by a 30-step cooldown. This does not prove the chosen trigger is optimal.

**13. What changes between sync and async?**  
Synchronous execution holds the end effector while physics advances; asynchronous execution continues lower-level control. Returned proposals are checked for freshness and evaluated against the current state.

**14. How are stale replies rejected?**  
Check episode identity, step age, object displacement and wall age. Thresholds are 60 steps, 8 cm and 15 seconds. Late replies never cross episode boundaries. Valid syntax is insufficient for current relevance.

**15. What output failure did you find?**  
An actual example is `{"skills":["lift","carry"],"preferred":1}`. It can be valid JSON but violates the three-candidate contract. Constrained decoding is a proposed follow-up, not a demonstrated fix in the archived result.

**16. Is 77% a good result?**  
It recovers performance relative to 43.67% synchronous execution, while remaining below 99.83% adaptive-only. The improvement is 33.33 **percentage points**, not a claim of beating the original baseline.

**17. Why does the model hurt performance?**  
The audit establishes performance differences, invalid replies and execution/bookkeeping discrepancies. Delay, unsuitable overrides and handoff logic are plausible contributors; their individual causal effects are not isolated.

**18. What would you change next?**  
Test executed-action-aware dwell counters, matched artificial delay, constrained outputs and bounded handoffs separately. Keep an adaptive-only baseline and record new evidence under a new protocol.

**19. How do you interpret the statistics?**  
There are 600 evaluations per method but shared scenes and only three training seeds. Report grouped and paired counts first; a rigorous interval or hypothesis test must respect the dependence structure.

**20. Is the system safe?**  
It has finite-state/workspace checks and stale-response rejection. It does not provide certified collision avoidance, force safety or physical-robot guarantees. Ensemble disagreement is not calibrated safety probability.

**21. What is end-to-end here?**  
The research workflow connects data collection, representation, training, model I/O, scheduling, execution, logging and evaluation. It is not joint gradient training of all components or a deployed frontend/backend product.

**22. How does this transfer to a marketing-AI/agent role?**  
Schema contracts, asynchronous work, state versions, stale outputs, logging and evaluation are transferable. For example, a research-agent report may become stale after requirements change. Crawling, SEO/GEO and marketing workflow deployment are not implemented in this repository.

**23. Why not use an agent framework?**  
Explicit control makes timing and evidence easier to inspect. A framework can simplify business orchestration, but its value should be judged by observability and constraints rather than its presence alone.

**24. What did you personally contribute?**  
Separate problem framing, design choices, coding, AI assistance, experiment operation and analysis. Be ready to explain two concrete code paths and reproduce one result. Do not say “everything independently” unless that is accurate.

**25. What if asked to modify code live?**  
Make an isolated change, such as exposing a rejection reason or testing an additional response-version check. Explain expected behavior and validation. Never imply a new mechanism produced the archived 77% result.

## Contribution map

| Component | Accurate framing |
|---|---|
| Qwen, ResNet and robosuite | Third-party foundations |
| Triggering, scheduling, comparison, adaptation and evaluation | Project implementation; describe personal participation factually |
| Five-method historical audit | Existing evidence, not new training for this release |
| Bilingual docs, replay scripts, demo and CI | AI-assisted packaging and verification added for this release |
| Vision-only control, real robots, marketing platform and online service | Extensions not implemented or validated here |

## CV wording

> Investigated VLM-guided adaptive robot control in an AI-assisted Panda simulation project, integrating frozen Qwen2.5-VL skill proposals with latent dynamics, online adaptation and MPC. Audited 3,000 episodes: asynchronous execution reached 77.00% success versus 43.67% synchronous and 99.83% adaptive-only, with traceable schema checks and stale-response handling.

Before the interview, be able to trace one observation-to-action cycle, locate the loss and online update, calculate 462/600, distinguish syntax/schema/semantic/freshness failures, and state the project’s real boundaries.
