"""Label-budget sweep: does BenthicNet's SSL pretraining advantage only show
up once labels get scarce, as the paper's own claims would predict?

For each backbone, the val/test splits are held fixed; only the number of
training examples per class is varied. The backbone (and any checkpoint
load) is built once per pretrain type and reused across budgets -- only the
linear head is retrained each time.

Usage:
    python -m src.sweep --budgets 5,10,20,50,full
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import PatchDataset, load_patch_table, make_splits
from src.models import LinearProbe, build_backbone
from src.train import make_transform, run_epoch


def subsample_per_class(df, n, seed):
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), n), random_state=seed))
        .reset_index(drop=True)
    )


def parse_budgets(spec):
    budgets = []
    for token in spec.split(","):
        token = token.strip()
        budgets.append(None if token == "full" else int(token))
    return budgets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrains", default="imagenet,benthicnet,random")
    parser.add_argument("--budgets", default="5,10,20,50,full")
    parser.add_argument("--benthicnet-ckpt", default="data/checkpoints/bt-400e_rn50_epoch=399.ckpt")
    parser.add_argument(
        "--data-dir",
        default="data/CCR_benthic_analyses/image_patch_aggregation/data/standard_patches",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-class", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="runs/sweep")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    budgets = parse_budgets(args.budgets)
    pretrains = args.pretrains.split(",")

    full_df = load_patch_table(args.data_dir, min_per_class=args.min_per_class)
    classes = sorted(full_df["label"].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    train_full, val_df, test_df = make_splits(full_df, seed=args.seed)
    print(f"Classes ({len(classes)}); train_full={len(train_full)} val={len(val_df)} test={len(test_df)}")

    results_csv = os.path.join(args.output_dir, "sweep_results.csv")
    rows = []
    done = set()
    if os.path.exists(results_csv):
        existing = pd.read_csv(results_csv)
        rows = existing.to_dict("records")
        done = {(r["pretrain"], str(r["budget"])) for r in rows}
        print(f"Resuming: {len(done)} (pretrain, budget) combos already done -- {sorted(done)}")
    for pretrain in pretrains:
        print(f"\n=== backbone: {pretrain} ===")
        backbone, feature_dim, mean, std = build_backbone(pretrain, args.benthicnet_ckpt)
        backbone = backbone.to(device)
        transform = make_transform(mean, std)

        val_loader = DataLoader(
            PatchDataset(val_df, label_to_idx, transform),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            PatchDataset(test_df, label_to_idx, transform),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )

        for budget in budgets:
            label = "full" if budget is None else str(budget)
            if (pretrain, label) in done:
                print(f"[{pretrain}] budget={label} already done, skipping")
                continue

            train_sub = train_full if budget is None else subsample_per_class(train_full, budget, args.seed)
            train_loader = DataLoader(
                PatchDataset(train_sub, label_to_idx, transform),
                batch_size=min(args.batch_size, len(train_sub)), shuffle=True, num_workers=args.num_workers,
            )

            model = LinearProbe(backbone, feature_dim, len(classes)).to(device)
            optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)

            best_val_f1, best_state = -1.0, None
            for epoch in range(1, args.epochs + 1):
                run_epoch(model, train_loader, device, optimizer)
                val_metrics, _, _ = run_epoch(model, val_loader, device)
                if val_metrics["macro_f1"] > best_val_f1:
                    best_val_f1 = val_metrics["macro_f1"]
                    best_state = {k: v.clone() for k, v in model.head.state_dict().items()}

            model.head.load_state_dict(best_state)
            test_metrics, _, _ = run_epoch(model, test_loader, device)
            print(
                f"[{pretrain}] budget={label} (n_train={len(train_sub)}) "
                f"test_macro_f1={test_metrics['macro_f1']:.3f} test_bal_acc={test_metrics['balanced_accuracy']:.3f}"
            )
            rows.append(
                {
                    "pretrain": pretrain,
                    "budget": label,
                    "n_train": len(train_sub),
                    "macro_f1": test_metrics["macro_f1"],
                    "balanced_accuracy": test_metrics["balanced_accuracy"],
                }
            )

            # Write after every run, not just at the end, so a kill mid-sweep
            # doesn't lose completed results.
            results = pd.DataFrame(rows)
            results.to_csv(os.path.join(args.output_dir, "sweep_results.csv"), index=False)
            with open(os.path.join(args.output_dir, "sweep_results.json"), "w") as f:
                json.dump(rows, f, indent=2)
            if len(results["pretrain"].unique()) > 1 or len(rows) > 1:
                plot_sweep(results, os.path.join(args.output_dir, "sweep_plot.png"))

    print(f"\nSaved results + plot to {args.output_dir}/")


def plot_sweep(results, out_path):
    order = sorted(results["budget"].unique(), key=lambda b: (b == "full", 0 if b == "full" else int(b)))
    fig, ax = plt.subplots(figsize=(7, 5))
    for pretrain, group in results.groupby("pretrain"):
        group = group.set_index("budget").loc[order].reset_index()
        ax.plot(group["budget"], group["macro_f1"], marker="o", label=pretrain)
    ax.set_xlabel("Training examples per class")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Linear-probe performance vs. label budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
