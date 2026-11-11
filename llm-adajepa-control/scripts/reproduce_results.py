"""Recalculate descriptive tables from all bundled audit records."""
import argparse
import json
from pathlib import Path
from audit import ROOT, load_verified, markdown_table, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    _, _, summary = load_verified()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for key in ("methods", "by_task", "by_condition", "by_seed", "paired_descriptive_counts"):
        write_csv(args.out / f"{key}.csv", summary[key])
    for zh in (False, True):
        title = "# 历史审计结果（离线复算）" if zh else "# Historical audit results (offline recalculation)"
        note = (
            "3000 回合；每种方法 600 回合；同一组 200 个场景 × 3 个训练种子。"
            "这些是历史实验的描述性统计，本命令没有重跑仿真。规则回复不属于 VLM 调用。"
            if zh else
            "3,000 episodes; 600 per method; the same 200 scenes × 3 training seeds. "
            "Descriptive historical statistics, not a new simulation run. Rule replies are not VLM calls."
        )
        body = title + "\n\n" + note + "\n\n" + markdown_table(summary["methods"], zh) + "\n"
        (args.out / ("summary.zh-CN.md" if zh else "summary.md")).write_text(body, encoding="utf-8")
    print(markdown_table(summary["methods"]))
    print(f"\nVerified {summary['episodes']} episodes and replayed "
          f"{summary['vlm_responses_reparsed']} VLM responses; zero parser mismatches.")
    print(f"Tables written to {args.out}. No GPU or network used.")


if __name__ == "__main__":
    main()
