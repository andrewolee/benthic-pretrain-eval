"""Train and evaluate a linear probe on top of a frozen ResNet-50 backbone.

Usage:
    python -m src.train --pretrain imagenet
    python -m src.train --pretrain benthicnet --benthicnet-ckpt data/checkpoints/bt-400e_rn50_epoch=399.ckpt
    python -m src.train --pretrain random   # untrained-backbone control
"""
import argparse
import json
import os

import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data import PatchDataset, load_patch_table, make_splits
from src.models import LinearProbe, build_backbone


def make_transform(mean, std):
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def run_epoch(model, loader, device, optimizer=None, class_weights=None):
    training = optimizer is not None
    model.head.train(training)

    all_preds, all_targets = [], []
    total_loss = 0.0
    for images, targets in tqdm(loader, leave=False):
        images, targets = images.to(device), targets.to(device)

        if training:
            optimizer.zero_grad()
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, targets, weight=class_weights)
        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(targets)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    metrics = {
        "loss": total_loss / len(all_targets),
        "balanced_accuracy": balanced_accuracy_score(all_targets, all_preds),
        "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
    }
    return metrics, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", choices=["imagenet", "benthicnet", "random"], required=True)
    parser.add_argument("--benthicnet-ckpt", default="data/checkpoints/bt-400e_rn50_epoch=399.ckpt")
    parser.add_argument(
        "--data-dir",
        default="data/CCR_benthic_analyses/image_patch_aggregation/data/standard_patches",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-class", type=int, default=20)
    parser.add_argument("--class-weighted", action="store_true", help="weight loss inverse to class frequency")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_patch_table(args.data_dir, min_per_class=args.min_per_class)
    classes = sorted(df["label"].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    train_df, val_df, test_df = make_splits(df, seed=args.seed)
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Split sizes -- train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    backbone, feature_dim, mean, std = build_backbone(args.pretrain, args.benthicnet_ckpt)
    transform = make_transform(mean, std)

    train_ds = PatchDataset(train_df, label_to_idx, transform)
    val_ds = PatchDataset(val_df, label_to_idx, transform)
    test_ds = PatchDataset(test_df, label_to_idx, transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = LinearProbe(backbone, feature_dim, len(classes)).to(device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)

    class_weights = None
    if args.class_weighted:
        counts = train_df["label"].map(label_to_idx).value_counts().sort_index()
        weights = (1.0 / counts).values
        weights = weights / weights.sum() * len(classes)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    run_name = f"{args.pretrain}_seed{args.seed}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    best_val_f1 = -1.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, device, optimizer, class_weights)
        val_metrics, _, _ = run_epoch(model, val_loader, device)
        print(
            f"[{run_name}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.3f} "
            f"val_macro_f1={val_metrics['macro_f1']:.3f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.3f}"
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.head.state_dict().items()}

    model.head.load_state_dict(best_state)
    test_metrics, test_preds, test_targets = run_epoch(model, test_loader, device)
    print(f"[{run_name}] TEST macro_f1={test_metrics['macro_f1']:.3f} bal_acc={test_metrics['balanced_accuracy']:.3f}")

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "classes": classes,
                "history": history,
                "test_metrics": test_metrics,
                "test_preds": test_preds,
                "test_targets": test_targets,
            },
            f,
            indent=2,
        )
    print(f"Saved results to {run_dir}/results.json")


if __name__ == "__main__":
    main()
