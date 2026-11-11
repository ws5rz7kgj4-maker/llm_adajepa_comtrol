"""Interview demo: replay archived outputs through the actual proposal parser."""
import argparse
import json
from audit import load_verified, markdown_table, parse_proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    args = parser.parse_args()
    zh = args.lang == "zh"
    _, calls, summary = load_verified()
    print("历史记录离线回放：没有实时模型调用或新仿真。\n" if zh else
          "Offline replay of historical records: no live model call or new simulation.\n")
    print(markdown_table(summary["methods"], zh))
    selections = [
        ("有效回复 / Valid response", next(r for r in calls if r["method"] == "joint" and r["applied"])),
        ("结构不符合要求 / Schema failure", next(r for r in calls if r["recorded_invalid"])),
        ("过期回复 / Stale response", next(r for r in calls if r["stale"] and r["proposal"] is not None and r["method"] != "joint_rule" and not r["episode_ended"])),
    ]
    for title, record in selections:
        print(f"\n--- {title} ---")
        print(f"{record['episode']} / call {record['call']}")
        print("Raw:", record["raw"])
        try:
            print("Parsed:", json.dumps(parse_proposal(record["raw"]), ensure_ascii=False))
        except (ValueError, TypeError) as exc:
            print("Rejected:", str(exc))
        print(json.dumps({k: record[k] for k in (
            "requested_step", "returned_step", "age_steps", "drift_m",
            "stale", "episode_ended", "applied", "selected"
        )}, indent=2))
    print("\n43.67% → 77.00%: +33.33 percentage points; adaptive-only: 99.83%.")
    print("三个技能是候选项，不是依次执行的计划；过期/无效回复不会被采纳。"
          if zh else "Three skills are alternatives, not a sequential plan; invalid/stale replies are not applied.")


if __name__ == "__main__":
    main()
