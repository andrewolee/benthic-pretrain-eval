"""Loading and splitting the CCR benthic image patches (ImageFolder layout)."""
import os

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Not a real taxonomic class -- a catch-all for patches the annotators
# couldn't confidently assign. Excluded by default.
DEFAULT_EXCLUDE = {"unknown"}


def load_patch_table(root_dir, exclude=DEFAULT_EXCLUDE, min_per_class=20):
    """Walk an ImageFolder-style directory and return a DataFrame of
    (path, label), dropping excluded classes and classes too small to
    stratify into train/val/test.
    """
    rows = []
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir) or class_name in exclude:
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({"path": os.path.join(class_dir, fname), "label": class_name})

    df = pd.DataFrame(rows)
    counts = df["label"].value_counts()
    keep = counts[counts >= min_per_class].index
    dropped = sorted(set(counts.index) - set(keep))
    if dropped:
        print(f"Dropping classes with < {min_per_class} patches: {dropped}")
    return df[df["label"].isin(keep)].reset_index(drop=True)


def make_splits(df, test_size=0.3, val_size=0.15, seed=42):
    """Stratified train/val/test split by label."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    val_frac_of_train_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_frac_of_train_val,
        stratify=train_val["label"],
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


class PatchDataset(Dataset):
    def __init__(self, df, label_to_idx, transform):
        self.paths = df["path"].tolist()
        self.labels = [label_to_idx[l] for l in df["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]
