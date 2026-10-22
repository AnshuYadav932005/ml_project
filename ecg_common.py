"""
ecg_common.py
=============
Shared code used by all four algorithm folders (CNN, LSTM, GRU, Transformer).

Keeping data loading, the training loop and the evaluation/plotting in ONE place
guarantees that every model is trained and tested on exactly the same split with
exactly the same procedure, so the final comparison is fair (as promised on
slide 12: "trained on the same train/test split").

Dataset: MIT-BIH Arrhythmia (Kaggle "ECG Heartbeat Categorization Dataset")
    data/mitbih_train.csv  -> 87,554 beats
    data/mitbih_test.csv   -> 21,892 beats
    each row = 187 signal samples (125 Hz) + 1 label column (0..4)
"""

import os
import json
import time
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # render to files, no GUI window needed
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
)

# --------------------------------------------------------------------------
# Global constants
# --------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)",
               "Fusion (F)", "Unknown (Q)"]
SHORT_NAMES = ["N", "S", "V", "F", "Q"]
NUM_CLASSES = 5
SEQ_LEN = 187
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int = SEED):
    """Fix every random number generator so results can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_data(val_size: float = 0.1):
    """
    Read the Kaggle CSV files and return numpy arrays.

    - The official test file is kept completely untouched for final scoring.
    - 10 % of the training file is split off (stratified, so every class keeps
      its proportion) as a validation set used to pick the best epoch.
    Signals are already normalised to [0, 1] by the dataset authors.
    """
    train_df = pd.read_csv(os.path.join(DATA_DIR, "mitbih_train.csv"), header=None)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "mitbih_test.csv"), header=None)

    X_full = train_df.iloc[:, :SEQ_LEN].values.astype(np.float32)
    y_full = train_df.iloc[:, SEQ_LEN].values.astype(np.int64)
    X_test = test_df.iloc[:, :SEQ_LEN].values.astype(np.float32)
    y_test = test_df.iloc[:, SEQ_LEN].values.astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=val_size, stratify=y_full, random_state=SEED)

    return X_train, y_train, X_val, y_val, X_test, y_test


def beat_lengths(x):
    """
    True length of each beat, ignoring the trailing zero padding.

    In this dataset every beat is zero-padded at the END to 187 samples (a short
    V beat can be ~50 real samples + ~137 zeros). Recurrent models (LSTM/GRU)
    classify from the final hidden state, so they would otherwise have to carry
    the information through 100+ padding steps. The lengths are passed to
    torch.nn.utils.rnn.pack_padded_sequence so the RNN stops at the real end.
    x: (batch, 187, 1) -> lengths: (batch,) on CPU (required by pack_padded_sequence)
    """
    nonzero = (x[..., 0] > 0).cpu()
    pos = torch.arange(1, x.shape[1] + 1).expand_as(nonzero)
    return (nonzero * pos).max(dim=1).values.clamp(min=1)


def make_loader(X, y, batch_size=256, shuffle=False):
    """
    Wrap numpy arrays into a PyTorch DataLoader.
    Signals are reshaped to (batch, 187, 1): 187 time-steps, 1 feature each.
    """
    ds = TensorDataset(torch.from_numpy(X).unsqueeze(-1), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model, loader):
    """Run the model on a loader and return (class probabilities, true labels)."""
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        logits = model(xb.to(DEVICE))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3,
                patience=5, results_dir=None):
    """
    Standard supervised training loop shared by all four models.

    - Loss      : Cross-entropy (matches the softmax output on the slides)
    - Optimiser : Adam
    - Scheduler : ReduceLROnPlateau halves the LR when validation loss stalls
    - Selection : the weights with the best validation macro-F1 are kept
                  (macro-F1 treats the rare classes S and F as equally important
                  as Normal, which accuracy alone would hide)
    - Early stop: training stops if macro-F1 hasn't improved for `patience` epochs
    """
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "train_acc": [],
               "val_acc": [], "val_f1": [], "lr": []}
    best_f1, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        # ---- 1. training pass: forward -> loss -> backward -> update ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilises RNN training
            optimizer.step()
            running_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)

        # ---- 2. validation pass: measure generalisation, no gradient ----
        val_probs, val_y = predict_proba(model, val_loader)
        val_loss = nn.functional.nll_loss(
            torch.log(torch.from_numpy(val_probs) + 1e-9), torch.from_numpy(val_y)).item()
        val_pred = val_probs.argmax(1)
        val_acc = accuracy_score(val_y, val_pred)
        val_f1 = precision_recall_fscore_support(val_y, val_pred, average="macro",
                                                 zero_division=0)[2]

        # ---- 3. bookkeeping ----
        history["train_loss"].append(running_loss / total)
        history["train_acc"].append(correct / total)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}/{epochs} | train_loss {running_loss/total:.4f} "
              f"acc {correct/total:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f} "
              f"macroF1 {val_f1:.4f} | {time.time()-start:.0f}s", flush=True)

        # ---- 4. keep the best checkpoint / early stopping ----
        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    train_time = time.time() - start
    model.load_state_dict(best_state)
    if results_dir:
        torch.save(best_state, os.path.join(results_dir, "best_model.pt"))
    history.update(best_epoch=best_epoch, train_time_sec=train_time)
    return model, history


# --------------------------------------------------------------------------
# Plots (all built from sklearn / matplotlib library display helpers)
# --------------------------------------------------------------------------
def plot_training_curves(history, model_name, out_dir):
    """Loss, accuracy and macro-F1 per epoch, with the selected epoch marked."""
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(ep, history["train_loss"], "o-", label="train")
    axes[0].plot(ep, history["val_loss"], "o-", label="validation")
    axes[0].set_title("Cross-entropy loss")
    axes[1].plot(ep, history["train_acc"], "o-", label="train")
    axes[1].plot(ep, history["val_acc"], "o-", label="validation")
    axes[1].set_title("Accuracy")
    axes[2].plot(ep, history["val_f1"], "o-", color="tab:green", label="validation macro-F1")
    axes[2].set_title("Validation macro-F1")
    for ax in axes:
        ax.axvline(history["best_epoch"], color="grey", ls="--", lw=1, label="best epoch")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{model_name} - training curves")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=130)
    plt.close(fig)


def plot_confusion_matrices(y_true, y_pred, model_name, out_dir):
    """sklearn ConfusionMatrixDisplay: raw counts and row-normalised (= recall)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=SHORT_NAMES, cmap="Blues",
        values_format="d", ax=axes[0], colorbar=False)
    axes[0].set_title("Counts")
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=SHORT_NAMES, cmap="Blues",
        normalize="true", values_format=".3f", ax=axes[1], colorbar=False)
    axes[1].set_title("Normalised by true class (diagonal = recall)")
    fig.suptitle(f"{model_name} - confusion matrix on the test set")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=130)
    plt.close(fig)


def plot_roc_pr_curves(y_true, probs, model_name, out_dir):
    """One-vs-rest ROC and Precision-Recall curves per class (sklearn displays)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for c in range(NUM_CLASSES):
        RocCurveDisplay.from_predictions(
            (y_true == c).astype(int), probs[:, c], name=SHORT_NAMES[c], ax=axes[0])
        PrecisionRecallDisplay.from_predictions(
            (y_true == c).astype(int), probs[:, c], name=SHORT_NAMES[c], ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("ROC curves (one-vs-rest)")
    axes[1].set_title("Precision-Recall curves (one-vs-rest)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"{model_name} - ROC and PR curves on the test set")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_pr_curves.png"), dpi=130)
    plt.close(fig)


def plot_per_class_metrics(report_dict, model_name, out_dir):
    """Grouped bar chart of precision / recall / F1 for each heartbeat class."""
    df = pd.DataFrame({SHORT_NAMES[i]: report_dict[CLASS_NAMES[i]]
                       for i in range(NUM_CLASSES)}).T[["precision", "recall", "f1-score"]]
    ax = df.plot.bar(figsize=(10, 5), rot=0, ylim=(0, 1.05), width=0.8)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=8)
    ax.set_title(f"{model_name} - per-class metrics on the test set")
    ax.set_ylabel("Score")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_class_metrics.png"), dpi=130)
    plt.close()


def plot_misclassified_examples(X_test, y_true, y_pred, model_name, out_dir):
    """Show one wrongly-predicted beat per true class, so errors can be inspected."""
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(20, 3.5), sharey=True)
    t = np.arange(SEQ_LEN) / 125.0  # seconds (125 Hz sampling)
    rng = np.random.default_rng(SEED)
    for c, ax in enumerate(axes):
        wrong = np.where((y_true == c) & (y_pred != c))[0]
        if len(wrong) == 0:
            ax.set_title(f"True {SHORT_NAMES[c]}: no errors")
            continue
        i = rng.choice(wrong)
        ax.plot(t, X_test[i], color="tab:red")
        ax.set_title(f"True {SHORT_NAMES[c]} -> predicted {SHORT_NAMES[y_pred[i]]}")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{model_name} - example misclassified beats")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "misclassified_examples.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Evaluation + written metrics analysis
# --------------------------------------------------------------------------
def evaluate_and_report(model, model_name, history, test_loader, X_test, out_dir,
                        extra_notes=""):
    """
    Score the final model on the held-out test set, save every figure, a
    metrics.json (read later by compare_models.py) and a metrics_analysis.md.
    """
    # ---- predictions + inference speed ----
    t0 = time.time()
    probs, y_true = predict_proba(model, test_loader)
    infer_ms = (time.time() - t0) / len(y_true) * 1000
    y_pred = probs.argmax(1)

    # ---- metrics ----
    acc = accuracy_score(y_true, y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   digits=4, output_dict=True, zero_division=0)
    report_txt = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                       digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics = {
        "model": model_name,
        "parameters": n_params,
        "epochs_trained": len(history["train_loss"]),
        "best_epoch": history["best_epoch"],
        "train_time_sec": round(history["train_time_sec"], 1),
        "inference_ms_per_beat": round(infer_ms, 4),
        "device": str(DEVICE),
        "accuracy": acc,
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
        "roc_auc_macro_ovr": auc,
        "per_class": {SHORT_NAMES[i]: {k: report[CLASS_NAMES[i]][k]
                                       for k in ("precision", "recall", "f1-score", "support")}
                      for i in range(NUM_CLASSES)},
        "confusion_matrix": cm.tolist(),
        "history": {k: v for k, v in history.items() if isinstance(v, list)},
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report_txt)

    # ---- figures ----
    plot_training_curves(history, model_name, out_dir)
    plot_confusion_matrices(y_true, y_pred, model_name, out_dir)
    plot_roc_pr_curves(y_true, probs, model_name, out_dir)
    plot_per_class_metrics(report, model_name, out_dir)
    plot_misclassified_examples(X_test, y_true, y_pred, model_name, out_dir)

    # ---- written analysis ----
    write_analysis(metrics, report_txt, cm, history, out_dir, extra_notes)
    print(report_txt)
    print(f"Accuracy {acc:.4f} | macro-F1 {f_m:.4f} | ROC-AUC {auc:.4f}")
    return metrics


def write_analysis(m, report_txt, cm, history, out_dir, extra_notes):
    """Generate metrics_analysis.md with the numbers and a data-driven discussion."""
    pc = m["per_class"]
    by_f1 = sorted(pc, key=lambda c: pc[c]["f1-score"])
    weakest, strongest = by_f1[0], by_f1[-1]

    # most common confusion (largest off-diagonal cell)
    off = np.array(cm, dtype=float)
    np.fill_diagonal(off, 0)
    ti, pi = np.unravel_index(off.argmax(), off.shape)
    top_conf = int(off[ti, pi])

    # train vs validation gap at the chosen epoch -> over-fitting indicator
    be = m["best_epoch"] - 1
    gap = history["train_acc"][be] - history["val_acc"][be]
    fit_note = ("negligible - the model generalises well"
                if gap < 0.005 else "small - mild over-fitting"
                if gap < 0.015 else "noticeable - the model is over-fitting")

    # minority (S + F) vs majority recall
    minority_recall = np.mean([pc["S"]["recall"], pc["F"]["recall"]])

    figs = [("training_curves.png", "Training curves"),
            ("confusion_matrix.png", "Confusion matrix"),
            ("roc_pr_curves.png", "ROC & Precision-Recall curves"),
            ("per_class_metrics.png", "Per-class precision / recall / F1"),
            ("misclassified_examples.png", "Example misclassified beats")]
    extra_figs = [f for f in sorted(os.listdir(out_dir))
                  if f.endswith(".png") and f not in {x[0] for x in figs}]

    lines = [
        f"# {m['model']} - Metrics Analysis",
        "",
        "Test set: official MIT-BIH test split (21,892 beats), never seen during training.",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | **{m['accuracy']:.4f}** |",
        f"| Precision (macro) | {m['precision_macro']:.4f} |",
        f"| Recall (macro) | {m['recall_macro']:.4f} |",
        f"| F1-score (macro) | **{m['f1_macro']:.4f}** |",
        f"| Precision (weighted) | {m['precision_weighted']:.4f} |",
        f"| Recall (weighted) | {m['recall_weighted']:.4f} |",
        f"| F1-score (weighted) | {m['f1_weighted']:.4f} |",
        f"| ROC-AUC (macro, one-vs-rest) | {m['roc_auc_macro_ovr']:.4f} |",
        f"| Trainable parameters | {m['parameters']:,} |",
        f"| Epochs trained (best epoch) | {m['epochs_trained']} ({m['best_epoch']}) |",
        f"| Training time ({m['device']}) | {m['train_time_sec']/60:.1f} min |",
        f"| Inference time per beat | {m['inference_ms_per_beat']:.3f} ms |",
        "",
        "## Per-class report",
        "",
        "```",
        report_txt.rstrip(),
        "```",
        "",
        "## Analysis",
        "",
        f"- **Accuracy vs macro-F1:** accuracy is {m['accuracy']:.2%}, but ~83% of test beats are "
        f"Normal, so accuracy is dominated by that class. Macro-F1 ({m['f1_macro']:.4f}) weights all "
        f"five classes equally and is the fairer single number for this imbalanced problem.",
        f"- **Strongest class:** {strongest} (F1 = {pc[strongest]['f1-score']:.4f}). "
        f"**Weakest class:** {weakest} (F1 = {pc[weakest]['f1-score']:.4f}, "
        f"support = {int(pc[weakest]['support'])} beats).",
        f"- **Rare classes:** the mean recall on the two rarest classes, S and F, is "
        f"{minority_recall:.4f}. Each has under 3% of the training data, so these "
        f"are the hardest classes to learn.",
        f"- **Most frequent error:** {top_conf} true **{SHORT_NAMES[ti]}** beats were predicted "
        f"as **{SHORT_NAMES[pi]}**. "
        + ("S beats look very similar to Normal beats in a single-beat window, since the "
           "difference is mainly early timing, so this confusion is expected."
           if (ti, pi) in [(1, 0), (0, 1)] else
           "Fusion beats are by definition a mix of Normal and Ventricular morphology, "
           "so this confusion is expected." if 3 in (ti, pi) else ""),
        f"- **Over-fitting check:** at the selected epoch, train accuracy minus validation accuracy "
        f"is {gap:+.4f} ({fit_note}).",
        f"- **Separability:** a macro ROC-AUC of {m['roc_auc_macro_ovr']:.4f} means the predicted "
        f"probabilities rank the classes almost perfectly. A lot of the remaining F1 loss comes from "
        f"the default arg-max decision threshold on the rare classes.",
    ]
    if extra_notes:
        lines += ["", "## Model-specific notes", "", extra_notes]
    lines += ["", "## Figures", ""]
    for f, title in figs:
        lines += [f"### {title}", f"![{title}]({f})", ""]
    for f in extra_figs:
        lines += [f"### {f[:-4].replace('_', ' ').title()}", f"![{f}]({f})", ""]

    with open(os.path.join(out_dir, "metrics_analysis.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
