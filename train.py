"""DermaVision AI - train and evaluate on HAM10000.

Kaggle example:
    python train.py --data_dir /kaggle/input/skin-cancer-mnist-ham10000 --out_dir /kaggle/working/models

Writes to --out_dir: dermavision_best.pt, metrics.json, history.json,
confusion_matrix.png, training_curves.png and samples/ (a few test images for demos).
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dermavision import CLASS_CODES, IMAGENET_MEAN, IMAGENET_STD, DermaNet, eval_transform

LABELS = list(range(len(CLASS_CODES)))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ data
def load_metadata(data_dir):
    """Read HAM10000_metadata.csv and attach the file path of every image."""
    meta = next(data_dir.rglob("HAM10000_metadata*"), None)
    if meta is None:
        raise FileNotFoundError(f"HAM10000_metadata.csv not found under {data_dir}")
    first_line = meta.read_text(errors="ignore").splitlines()[0]
    df = pd.read_csv(meta, sep="\t" if "\t" in first_line else ",")  # Kaggle: CSV, Dataverse: tab-separated
    paths = {p.stem: str(p) for p in data_dir.rglob("*.jpg")}
    df["path"] = df["image_id"].map(paths)
    missing = int(df["path"].isna().sum())
    if missing:
        print(f"Warning: {missing} metadata rows have no image file and were dropped")
        df = df.dropna(subset=["path"])
    df["label"] = df["dx"].map({c: i for i, c in enumerate(CLASS_CODES)})
    return df.reset_index(drop=True)


def split_by_lesion(df, seed):
    """70/15/15 split by LESION (not by image), stratified by diagnosis.

    HAM10000 contains several photos of the same lesion. Splitting by image would
    put near-duplicates in both train and test and inflate the scores (data leakage).
    """
    lesions = df.drop_duplicates("lesion_id")[["lesion_id", "dx"]]
    train_l, rest = train_test_split(lesions, test_size=0.30, stratify=lesions["dx"], random_state=seed)
    val_l, test_l = train_test_split(rest, test_size=0.50, stratify=rest["dx"], random_state=seed)
    return [df[df["lesion_id"].isin(part["lesion_id"])].reset_index(drop=True)
            for part in (train_l, val_l, test_l)]


class HAMDataset(Dataset):
    def __init__(self, frame, transform):
        self.paths = frame["path"].tolist()
        self.labels = frame["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.transform(image), self.labels[i]


def train_transform(img_size):
    # Dermoscopic images have no canonical orientation, so flips and rotations are safe.
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# -------------------------------------------------------------- training
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    loss_sum, preds, truths = 0.0, [], []
    for x, y in tqdm(loader, leave=False, desc="train" if training else "val"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_sum += loss.item() * len(y)
        preds.append(logits.argmax(1).cpu())
        truths.append(y.cpu())
    y_true, y_pred = torch.cat(truths).numpy(), torch.cat(preds).numpy()
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    return loss_sum / len(y_true), macro_f1, balanced_accuracy_score(y_true, y_pred)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in tqdm(loader, leave=False, desc="test"):
        probs.append(model(x.to(device)).softmax(dim=1).cpu())
        labels.append(y)
    probs = torch.cat(probs).numpy().astype(np.float64)
    return probs / probs.sum(axis=1, keepdims=True), torch.cat(labels).numpy()


# --------------------------------------------------------------- outputs
def plot_confusion(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(CLASS_CODES)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), CLASS_CODES)
    ax.set_yticks(range(n), CLASS_CODES)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (test set; % of each true class)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ("loss", "f1"), ("Loss", "Macro F1")):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], label="validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_samples(test_df, dest, per_class=2):
    """Copy a few held-out test images so the app can offer demo samples."""
    dest.mkdir(parents=True, exist_ok=True)
    for code, group in test_df.groupby("dx"):
        for _, row in group.head(per_class).iterrows():
            shutil.copy(row["path"], dest / f"{code}_{row['image_id']}.jpg")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="folder containing HAM10000_metadata.csv and the images")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--patience", type=int, default=6, help="stop after this many epochs without improvement")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pretrained", action="store_true", help="skip ImageNet weights (offline debugging)")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # mixed precision: faster and lighter on GPU
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    # 1) data: lesion-level split
    df = load_metadata(Path(args.data_dir))
    train_df, val_df, test_df = split_by_lesion(df, args.seed)
    print(f"Images: train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")
    print("Training class counts:\n" + train_df["dx"].value_counts().to_string())

    def make_loader(frame, transform, train):
        return DataLoader(HAMDataset(frame, transform), batch_size=args.batch_size, shuffle=train,
                          drop_last=train, num_workers=args.workers, pin_memory=use_amp,
                          persistent_workers=args.workers > 0)

    train_loader = make_loader(train_df, train_transform(args.img_size), True)
    val_loader = make_loader(val_df, eval_transform(args.img_size), False)
    test_loader = make_loader(test_df, eval_transform(args.img_size), False)

    # 2) model, class-weighted loss (counters the ~67% nevus imbalance), optimiser
    counts = np.bincount(train_df["label"], minlength=len(CLASS_CODES))
    class_weights = torch.tensor((counts.sum() / (len(counts) * counts)) ** 0.5, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    model = DermaNet(len(CLASS_CODES), pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 3) train; keep the epoch with the best validation macro-F1 (not accuracy)
    ckpt_path = out / "dermavision_best.pt"
    best_f1, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        va_loss, va_f1, va_bacc = run_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_f1": tr_f1,
                        "val_loss": va_loss, "val_f1": va_f1, "val_bacc": va_bacc})
        print(f"epoch {epoch:02d} | train loss {tr_loss:.3f} F1 {tr_f1:.3f} | "
              f"val loss {va_loss:.3f} F1 {va_f1:.3f} bal.acc {va_bacc:.3f} | {time.time() - t0:.0f}s")
        if va_f1 > best_f1:
            best_f1, bad_epochs = va_f1, 0
            torch.save({"state_dict": model.state_dict(), "class_names": CLASS_CODES,
                        "img_size": args.img_size, "arch": "efficientnet_b0"}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    # 4) evaluate the best checkpoint once on the untouched test split
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    probs, y_true = predict_probs(model, test_loader, device)
    y_pred = probs.argmax(axis=1)
    print(classification_report(y_true, y_pred, labels=LABELS, target_names=CLASS_CODES, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=LABELS))
    except ValueError:
        auc = float("nan")
    report = classification_report(y_true, y_pred, labels=LABELS, target_names=CLASS_CODES,
                                   output_dict=True, zero_division=0)
    metrics = {
        "n_test_images": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
        "macro_auc_ovr": auc,
        "per_class": {c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                      for c in CLASS_CODES},
    }
    print({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    plot_confusion(confusion_matrix(y_true, y_pred, labels=LABELS), out / "confusion_matrix.png")
    plot_curves(history, out / "training_curves.png")
    export_samples(test_df, out / "samples")
    print(f"\nDone. Copy the '{out}' folder next to app.py, then run: streamlit run app.py")


if __name__ == "__main__":
    main()
