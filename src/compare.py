"""Print a side-by-side comparison of two or more completed runs.

Usage:
    python -m src.compare runs/imagenet_seed42 runs/benthicnet_seed42 runs/random_seed42
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args()

    rows = []
    for run_dir in args.run_dirs:
        with open(os.path.join(run_dir, "results.json")) as f:
            data = json.load(f)
        rows.append(
            {
                "run": os.path.basename(run_dir),
                "pretrain": data["args"]["pretrain"],
                "macro_f1": data["test_metrics"]["macro_f1"],
                "balanced_accuracy": data["test_metrics"]["balanced_accuracy"],
            }
        )

    rows.sort(key=lambda r: r["macro_f1"], reverse=True)

    header = f"{'run':<24}{'pretrain':<14}{'macro_f1':<12}{'balanced_acc':<12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['run']:<24}{r['pretrain']:<14}{r['macro_f1']:<12.4f}{r['balanced_accuracy']:<12.4f}")


if __name__ == "__main__":
    main()
