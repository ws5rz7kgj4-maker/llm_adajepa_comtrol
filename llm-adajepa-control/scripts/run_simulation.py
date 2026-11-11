"""Explicit local entry point for the preserved GPU research pipeline."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from experiment.spec import ARMS, CONDITIONS, CONFIG, MODEL_SEEDS, TASKS, fingerprint, scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inspect", "preflight", "collect", "check-data", "train", "evaluate"),
                        default="inspect")
    parser.add_argument("--root", type=Path, default=ROOT / "runs" / "local")
    parser.add_argument("--seed", type=int, choices=MODEL_SEEDS, default=11)
    parser.add_argument("--task", choices=TASKS, default="Lift")
    parser.add_argument("--condition", choices=CONDITIONS, default="nominal")
    parser.add_argument("--scene-index", type=int, choices=range(25), default=0)
    parser.add_argument("--method", choices=[a.name for a in ARMS], default="adaptive_only")
    args = parser.parse_args()
    if args.stage == "inspect":
        print(json.dumps({
            "protocol": fingerprint(), "train_scenes": len(scenes("train")),
            "validation_scenes": len(scenes("val")), "test_scenes": len(scenes()),
            "full_protocol_episodes": len(scenes()) * len(ARMS) * len(MODEL_SEEDS),
            "bundled_audit_episodes": 3000, "config": CONFIG,
        }, indent=2))
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this preserved research pipeline. Use scripts/demo.py for CPU-only replay.")
    torch.set_num_threads(2)
    from experiment.storage import Store
    store = Store(args.root)
    if args.stage == "preflight":
        from experiment.preflight import check
        check(args.root, store)
    elif args.stage in ("collect", "check-data", "train"):
        from experiment.train import collect, check_data, train_seed
        if args.stage == "collect":
            collect(args.root, store)
        elif args.stage == "check-data":
            check_data(args.root, store)
        else:
            check_data(args.root, store)
            train_seed(args.root, args.seed, store)
    else:
        from experiment.model import VisualEncoder, WorldModel
        from experiment.evaluate import episode
        from experiment.proposer import Proposer
        arm = next(a for a in ARMS if a.name == args.method)
        ident = f"test-{args.task}-{args.condition}-{args.scene_index:03}"
        scene = next(s for s in scenes() if s["id"] == ident)
        episode_id = f"{ident}__seed{args.seed}__{arm.name}"
        if (args.root / "rows" / f"{episode_id}.json").exists() or (
            args.root / "episodes" / f"{episode_id}.npz"
        ).exists():
            raise SystemExit("Episode already exists; choose a fresh --root to avoid overwriting evidence.")
        model = WorldModel(args.root / "models" / f"model-{args.seed}.pt")
        proposer = Proposer() if arm.trigger != "none" and arm.proposer == "vlm" else None
        episode(scene, arm, model, VisualEncoder(), proposer, args.seed, args.root, store)


if __name__ == "__main__":
    main()
