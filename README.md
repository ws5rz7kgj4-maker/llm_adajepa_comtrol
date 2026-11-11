LLM + AdaJEPA-inspired Control
A research project on when a vision-language planner helps—or disrupts—an adaptive robot controller.

中文说明 · Architecture & mathematics · Interview guide · Reproduction

This repository packages a state-assisted Panda manipulation experiment and its historical audit. A frozen Qwen2.5-VL-3B proposes candidate skills from an RGB image and structured feedback. A learned latent dynamics ensemble evaluates those skills and selects low-level actions, adapting online to recent transitions.

The central engineering question is how to connect a slow, fallible model to a continuously changing control loop: when to ask, how to validate the answer, and whether it is still relevant when it arrives.

This is an AdaJEPA-inspired implementation, not the official AdaJEPA model, checkpoint, or benchmark. The visual encoder is a frozen ImageNet ResNet-18, and simulator object coordinates are available. It is not a vision-only or real-robot result.

Try it in one minute
Python 3.10+; no GPU, model downloads, API key, or third-party package is needed for the offline demo.

python3 scripts/demo.py --lang en
python3 scripts/verify_data.py
python3 scripts/reproduce_results.py
python3 -m unittest discover -s tests -v
The demo replays archived model replies through the preserved parser. It does not call a live model or simulate a new robot episode. It shows an accepted reply, a schema failure, and a stale reply. Chinese output: python3 scripts/demo.py --lang zh.

Main finding
The bundled audit covers 3,000 episodes: 5 methods × 200 scenes × 3 training seeds. Each method has the same 600 scene–seed evaluations.

Method	Successful episodes	Success rate
Adaptive controller, no VLM (adaptive_only)	599/600	99.83%
Prediction-error trigger (error)	504/600	84.00%
Joint trigger, synchronous (joint)	262/600	43.67%
Joint trigger, asynchronous (joint_async)	462/600	77.00%
Joint trigger, rule proposals (joint_rule)	396/600	66.00%
Historical success rates across five methods

Asynchronous execution is 33.33 percentage points above synchronous execution, but 22.83 points below the adaptive-only baseline. This is a useful negative result: integrating a language model does not automatically improve a capable controller. The evidence supports a scheduling comparison on this benchmark, not a universal claim that language models help or harm robotics.

The code defines a 21-arm protocol; this release includes evidence for five arms only. See results and limitations, per-task tables, and provenance.

How the system works

Perception: 512 image features + 12 observed state values.
Learning: three residual predictors per training seed; recent-transition adaptation.
Planning: three candidate skills from the VLM, then model-based comparison; 32 action sequences over a five-step horizon.
Execution: synchronous hold versus asynchronous continuation; one outstanding request, bounded call budget, stale-response checks.
Evaluation: fixed train/validation/test scene seeds, archived raw responses, per-episode records, and integrity checks.
“Three skills” means three alternatives for the next skill, not a three-step plan.

Repository map
Location	Purpose
research/experiment	Preserved simulation, training, planning, orchestration and logging code
scripts	Lightweight replay, evidence checks, table generation and local simulation entry point
data/audit	All 3,000 episode records; 7,260 proposal records including 2,196 rule replies
results	Recomputed aggregate, task, condition, seed and paired descriptive tables
docs/INTERVIEW.en.md	Pitch, technical questions, honest CV wording and application-engineering connections
docs/ARCHITECTURE.en.md	Code-linked architecture and equations
docs/REPRODUCIBILITY.en.md	Offline audit versus a new GPU experiment
docs/UPLOAD.md	GitHub upload instructions in English and Chinese
docs/VERIFICATION.md	Checks actually run for this release
Scope and next steps
This is an end-to-end research workflow, from data collection to evaluation and diagnosis. It is not a deployed web product. No real-hardware validation, calibrated safety guarantee, or superiority over the adaptive-only baseline is claimed.

Next experiments should isolate delay, skill-handoff bookkeeping and output constraints with controlled ablations. Keep the existing audit unchanged and record new runs separately.

References and attribution
AdaJEPA official repository — inspiration for adaptation inside planning; this implementation differs.
Qwen2.5-VL-3B-Instruct — pretrained multimodal proposer.
robosuite — manipulation tasks and simulation interface.
ResNet-18 model documentation — visual backbone.
Project owner: Songyuan Li. This portfolio package was organized with AI assistance. Personal contribution wording should reflect the work the owner can explain and substantiate; see the interview guide. Third-party weights are not redistributed. See NOTICE.md.
