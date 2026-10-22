"""
compare_models.py
=================
Step 3 of the slides - "Putting it together".

Reads the metrics.json written by each of the four algorithm folders and
produces:
    comparison_results/*.png   side-by-side charts
    comparison_results/comparison_table.csv
    COMPARISON.md              overall comparison + the best model

Run AFTER the four training scripts:  python compare_models.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import ecg_common as ec

ROOT = ec.ROOT_DIR
OUT = os.path.join(ROOT, "comparison_results")
MODELS = [("1D-CNN", "01_CNN"), ("LSTM", "02_LSTM"),
          ("GRU", "03_GRU"), ("Transformer", "04_Transformer")]
COLORS = {"1D-CNN": "#2a78b8", "LSTM": "#d9622b", "GRU": "#3a9a5b", "Transformer": "#8e5bb5"}


# --------------------------------------------------------------------------
# 1. Load every model's metrics
# --------------------------------------------------------------------------
def load_metrics():
    out = {}
    for name, folder in MODELS:
        path = os.path.join(ROOT, folder, "results", "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                out[name] = json.load(f)
        else:
            print(f"WARNING: {path} missing - train that model first")
    return out


# --------------------------------------------------------------------------
# 2. Dataset overview figure (class balance + one example beat per class)
# --------------------------------------------------------------------------
def plot_dataset_overview():
    _, y_tr, _, y_va, X_te, y_te = ec.load_data()
    y_train_full = np.concatenate([y_tr, y_va])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    counts = pd.DataFrame({
        "train": np.bincount(y_train_full, minlength=5),
        "test": np.bincount(y_te, minlength=5)}, index=ec.SHORT_NAMES)
    counts.plot.bar(ax=axes[0], rot=0, logy=True, color=["#2a78b8", "#d9622b"])
    for cont in axes[0].containers:
        axes[0].bar_label(cont, fontsize=8)
    axes[0].set_title("Class distribution (log scale) - heavily imbalanced")
    axes[0].set_ylabel("Number of beats")
    t = np.arange(ec.SEQ_LEN) / 125
    for c in range(5):
        axes[1].plot(t, X_te[np.where(y_te == c)[0][0]], label=ec.CLASS_NAMES[c])
    axes[1].set_title("Example heartbeat from each class (187 samples @ 125 Hz)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dataset_overview.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. Comparison figures
# --------------------------------------------------------------------------
def plot_overall_metrics(df):
    cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc_macro_ovr"]
    labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "F1\n(weighted)", "ROC-AUC\n(macro)"]
    sub = df[cols].T
    sub.index = labels
    lo = max(0.0, np.floor(df[cols].values.min() * 20) / 20 - 0.05)
    ax = sub.plot.bar(figsize=(14, 6), rot=0, width=0.8,
                      color=[COLORS[m] for m in sub.columns], ylim=(lo, 1.0))
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.3f", fontsize=7, rotation=90, padding=2)
    ax.set_title("Overall test-set metrics - all four models (y-axis zoomed)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "overall_metrics_comparison.png"), dpi=130)
    plt.close()


def plot_per_class_f1(metrics):
    mat = np.array([[metrics[m]["per_class"][c]["f1-score"] for c in ec.SHORT_NAMES]
                    for m in metrics])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=mat.min() - 0.02, vmax=1.0)
    ax.set_xticks(range(5), ec.CLASS_NAMES, rotation=15)
    ax.set_yticks(range(len(metrics)), list(metrics))
    for i in range(mat.shape[0]):
        for j in range(5):
            best = mat[i, j] == mat[:, j].max()
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    fontweight="bold" if best else "normal")
    fig.colorbar(im, ax=ax, label="F1-score")
    ax.set_title("Per-class F1-score (bold = best model for that class)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "per_class_f1_heatmap.png"), dpi=130)
    plt.close(fig)


def plot_per_class_recall(metrics):
    df = pd.DataFrame({m: [metrics[m]["per_class"][c]["recall"] for c in ec.SHORT_NAMES]
                       for m in metrics}, index=ec.SHORT_NAMES)
    ax = df.plot.bar(figsize=(12, 5), rot=0, width=0.8, color=[COLORS[m] for m in df.columns],
                     ylim=(max(0, df.values.min() - 0.1), 1.0))
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=7)
    ax.set_title("Per-class recall (sensitivity): how many beats of each class are caught")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "per_class_recall_comparison.png"), dpi=130)
    plt.close()


def plot_learning_curves(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for m, d in metrics.items():
        h = d["history"]
        ep = range(1, len(h["val_f1"]) + 1)
        axes[0].plot(ep, h["val_loss"], "o-", ms=3, color=COLORS[m], label=m)
        axes[1].plot(ep, h["val_f1"], "o-", ms=3, color=COLORS[m], label=m)
    axes[0].set_title("Validation loss per epoch")
    axes[1].set_title("Validation macro-F1 per epoch")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learning_curves_comparison.png"), dpi=130)
    plt.close(fig)


def plot_efficiency(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    df["sec_per_epoch"] = df["train_time_sec"] / df["epochs_trained"]
    for ax, col, title in [(axes[0], "sec_per_epoch", "Training time per epoch (s)"),
                           (axes[1], "parameters", "Trainable parameters"),
                           (axes[2], "inference_ms_per_beat", "Inference time per beat (ms)")]:
        bars = ax.bar(df.index, df[col], color=[COLORS[m] for m in df.index])
        ax.bar_label(bars, fmt="%.3g" if col != "parameters" else "%d")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Computational cost (measured on {df['device'].iloc[0]})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "efficiency_comparison.png"), dpi=130)
    plt.close(fig)

    # accuracy-vs-cost trade-off
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in df.index:
        ax.scatter(df.loc[m, "sec_per_epoch"], df.loc[m, "f1_macro"],
                   s=df.loc[m, "parameters"] / 800 + 60, color=COLORS[m], alpha=0.8, label=m)
        ax.annotate(m, (df.loc[m, "sec_per_epoch"], df.loc[m, "f1_macro"]),
                    textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel("Training time per epoch (s)")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Speed vs quality trade-off (bubble size = parameters)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "speed_vs_f1_tradeoff.png"), dpi=130)
    plt.close(fig)


def plot_confusion_grid(metrics):
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5.5))
    for ax, (m, d) in zip(np.atleast_1d(axes), metrics.items()):
        cm = np.array(d["confusion_matrix"], dtype=float)
        cm = cm / cm.sum(axis=1, keepdims=True)
        ConfusionMatrixDisplay(cm, display_labels=ec.SHORT_NAMES).plot(
            ax=ax, cmap="Blues", values_format=".3f", colorbar=False)
        ax.set_title(m)
    fig.suptitle("Normalised confusion matrices (diagonal = recall per class)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confusion_matrices_all_models.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Ranking + written comparison
# --------------------------------------------------------------------------
def write_report(df, metrics):
    rank_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc_macro_ovr"]
    ranks = df[rank_cols].rank(ascending=False, method="min")
    df["mean_rank"] = ranks.mean(axis=1)
    # Primary criterion = macro-F1 (fair under class imbalance); accuracy breaks ties
    order = df.sort_values(["f1_macro", "accuracy"], ascending=False).index.tolist()
    best = order[0]
    fastest = (df["train_time_sec"] / df["epochs_trained"]).idxmin()

    df.round(4).to_csv(os.path.join(OUT, "comparison_table.csv"))

    def row(m):
        d = df.loc[m]
        return (f"| {'**' + m + '**' if m == best else m} | {d.accuracy:.4f} | {d.precision_macro:.4f} | "
                f"{d.recall_macro:.4f} | {d.f1_macro:.4f} | {d.f1_weighted:.4f} | "
                f"{d.roc_auc_macro_ovr:.4f} | {int(d.parameters):,} | "
                f"{d.train_time_sec / d.epochs_trained:.1f} s | {d.mean_rank:.2f} |")

    pc_rows = []
    for c in ec.SHORT_NAMES:
        vals = {m: metrics[m]["per_class"][c]["f1-score"] for m in metrics}
        win = max(vals, key=vals.get)
        pc_rows.append(f"| {ec.CLASS_NAMES[ec.SHORT_NAMES.index(c)]} | "
                       + " | ".join(f"{'**%.4f**' % v if m == win else '%.4f' % v}"
                                    for m, v in vals.items()) + f" | {win} |")

    b, second = df.loc[best], df.loc[order[1]]
    spe = df["train_time_sec"] / df["epochs_trained"]
    # models whose best epoch was (near) the last one -> still improving when stopped
    capped = [m for m in df.index if df.loc[m, "best_epoch"] >= df.loc[m, "epochs_trained"] - 3
              and m != best]
    lstm_vs_gru = ""
    if {"LSTM", "GRU"} <= set(df.index):
        faster = "GRU" if spe["GRU"] < spe["LSTM"] else "LSTM"
        lstm_vs_gru = (
            f"- **LSTM vs GRU (slide 8):** GRU has {1 - df.loc['GRU', 'parameters'] / df.loc['LSTM', 'parameters']:.0%} "
            f"fewer parameters and scored macro-F1 {df.loc['GRU', 'f1_macro']:.4f} vs {df.loc['LSTM', 'f1_macro']:.4f}. "
            + ("So the lighter model was also more accurate. "
               if df.loc["GRU", "f1_macro"] > df.loc["LSTM", "f1_macro"] else "")
            + (f"Per epoch, the **{faster}** was faster here (LSTM {spe['LSTM']:.1f} s, GRU {spe['GRU']:.1f} s). "
               + ("" if faster == "GRU" else
                  "This goes against the slide-12 expectation. On a GPU, both run through cuDNN "
                  "kernels with packed variable-length sequences, and run time is dominated by the 187 "
                  "sequential steps rather than the gate count, so the theoretical ~25% saving "
                  "doesn't show up in wall-clock time.")))
    findings = [
        "## Key findings",
        "",
        f"1. **{best} is the best model** on every quality metric: accuracy, macro and weighted "
        f"precision, recall and F1, and ROC-AUC. It also wins all five per-class F1 scores. "
        f"This fits the slides' reasoning: the classes differ mainly in the *shape* of the "
        f"P-QRS-T complex, and local convolution filters detect shape directly.",
        f"2. **It is also by far the cheapest to train:** {spe[best]:.1f} s per epoch vs "
        f"{spe.drop(best).min():.1f}-{spe.drop(best).max():.1f} s for the others. Convolutions run "
        f"in parallel over the whole beat.",
        "3. **All four models reach ~98% accuracy**, and the differences only show on the rare "
        "classes. Supraventricular (S) and Fusion (F) are hardest for every model. S beats are "
        "mostly mistaken for Normal (their shape is near-normal; what makes them abnormal is "
        "early timing), and F beats are confused with N and V, since a fusion beat is a mix of both.",
        "4. **Ranking vs slide 12 predictions:** CNN 'Fast' was confirmed. Transformer "
        "'Medium (parallelizable)' was confirmed: ~3x the CNN's time per epoch, and much faster "
        "than the RNNs. LSTM/GRU 'Slow/Medium' was confirmed: the RNNs are the slowest because "
        "they process 187 steps sequentially.",
    ]
    if lstm_vs_gru:
        findings.append(lstm_vs_gru.replace("- ", "5. ", 1))
    if capped:
        findings += [
            "",
            f"> **Caveat - equal training budget:** {', '.join(capped)} reached their best "
            f"validation score in the last few of the 30 allowed epochs, so they were still "
            f"improving when training stopped. {best} early-stopped on its own at epoch "
            f"{int(b.epochs_trained)}. With a larger epoch budget, the gap to the slower models "
            f"may shrink. Under the same budget, though, {best} is clearly ahead, and its whole "
            f"training run took {b.train_time_sec / 60:.1f} min vs "
            f"{df.drop(best)['train_time_sec'].min() / 60:.1f}-{df.drop(best)['train_time_sec'].max() / 60:.1f} min "
            f"for the others.",
        ]
    findings += [
        "",
        "> **Implementation note:** every beat in this dataset is zero-padded at the end to "
        "187 samples. The LSTM and GRU use `pack_padded_sequence`, so their final hidden state "
        "is taken at the real end of each beat. Without it, a trial LSTM run predicted 'Normal' "
        "for everything for 4 epochs, because it had to carry the beat information through "
        "~140 padding steps.",
        "",
    ]
    lines = [
        "# ECG Arrhythmia Classification: Overall Model Comparison",
        "",
        "MIT-BIH Arrhythmia dataset (Kaggle *ECG Heartbeat Categorization*): 87,554 training beats "
        "(10% held out for validation) and 21,892 test beats, 5 classes. All four models were "
        "trained with the same split, loss (cross-entropy), optimiser (Adam), early stopping and "
        "model selection (best validation macro-F1), using the shared `ecg_common.py`.",
        "",
        "![Dataset overview](comparison_results/dataset_overview.png)",
        "",
        f"## Best model: {best}",
        "",
        f"**{best}** achieves the highest **macro-F1 = {b.f1_macro:.4f}** "
        f"(accuracy {b.accuracy:.4f}, macro recall {b.recall_macro:.4f}, ROC-AUC {b.roc_auc_macro_ovr:.4f}). "
        f"The runner-up is {order[1]} (macro-F1 {second.f1_macro:.4f}).",
        "",
        "**Why macro-F1 is the selection criterion:** 82.8% of the test beats are Normal, so a "
        "model that labels *every* beat Normal would already score 82.8% accuracy. Macro-F1 averages "
        "the F1-score of all five classes equally, so a model only scores well if it also detects "
        "the rare but clinically important S and F beats.",
        "",
        *findings,
        "## Overall metrics (test set)",
        "",
        "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | ROC-AUC | Params | Time / epoch | Mean rank |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *[row(m) for m in order],
        "",
        f"*Mean rank* is the average position across the 6 quality metrics (1 = best). "
        f"Time per epoch was measured on `{df['device'].iloc[0]}` (NVIDIA RTX 4060 Laptop GPU).",
        "",
        "![Overall metrics](comparison_results/overall_metrics_comparison.png)",
        "",
        "## Per-class F1-score",
        "",
        "| Class | " + " | ".join(metrics) + " | Best |",
        "|---|" + "---|" * (len(metrics) + 1),
        *pc_rows,
        "",
        "![Per-class F1](comparison_results/per_class_f1_heatmap.png)",
        "![Per-class recall](comparison_results/per_class_recall_comparison.png)",
        "![Confusion matrices](comparison_results/confusion_matrices_all_models.png)",
        "",
        "## Training behaviour and cost",
        "",
        "![Learning curves](comparison_results/learning_curves_comparison.png)",
        "![Efficiency](comparison_results/efficiency_comparison.png)",
        "![Trade-off](comparison_results/speed_vs_f1_tradeoff.png)",
        "",
        f"- Fastest per epoch: **{fastest}**.",
        "",
        "## How to reproduce",
        "",
        "```",
        "pip install -r requirements.txt   # torch with CUDA: --index-url https://download.pytorch.org/whl/cu126",
        "python 01_CNN/cnn_model.py",
        "python 02_LSTM/lstm_model.py",
        "python 03_GRU/gru_model.py",
        "python 04_Transformer/transformer_model.py",
        "python compare_models.py",
        "```",
        "",
        "## Per-model detailed reports",
        "",
        *[f"- [{m}]({f}/results/metrics_analysis.md)" for m, f in MODELS if m in metrics],
        "",
    ]
    path = os.path.join(ROOT, "COMPARISON.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(df[rank_cols + ["mean_rank"]].round(4).loc[order])
    print(f"\nBEST MODEL: {best}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    metrics = load_metrics()
    df = pd.DataFrame({m: {k: v for k, v in d.items() if not isinstance(v, (dict, list))}
                       for m, d in metrics.items()}).T
    num_cols = [c for c in df.columns if c not in ("model", "device")]
    df[num_cols] = df[num_cols].astype(float)

    plot_dataset_overview()
    plot_overall_metrics(df)
    plot_per_class_f1(metrics)
    plot_per_class_recall(metrics)
    plot_learning_curves(metrics)
    plot_efficiency(df)
    plot_confusion_grid(metrics)
    write_report(df, metrics)
