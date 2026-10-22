"""
Algorithm 4 of 4 - Transformer (Self-Attention Network)
=======================================================
Idea (slide 10): instead of reading the beat step-by-step, every time-step looks
at every other time-step at once, so long-range relationships (e.g. between the
P wave and the T wave) don't fade the way they can in a recurrent model.

Equations implemented (slide 11) - computed inside torch.nn.MultiheadAttention:
    Q = X W_Q,  K = X W_K,  V = X W_V           project each time-step
    A = softmax( Q K^T / sqrt(d_k) )            attention weights (187 x 187)
    Z = A V                                     weighted mix of values
The encoder block is written out explicitly (instead of using
nn.TransformerEncoderLayer) so the attention matrix A can be returned and
plotted.

Run:  python 04_Transformer/transformer_model.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# make the shared helper module in the project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ecg_common as ec

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODEL_NAME = "Transformer"


# --------------------------------------------------------------------------
# 1. Model definition
# --------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    """One Transformer encoder layer: self-attention + feed-forward, each with
    a residual connection and LayerNorm (pre-norm variant, more stable)."""

    def __init__(self, d_model=64, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        # Multi-head self-attention: Q, K, V all come from the same sequence X
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        # Position-wise feed-forward network applied to every time-step independently
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, need_weights=False):
        h = self.norm1(x)
        z, attn = self.attn(h, h, h, need_weights=need_weights,
                            average_attn_weights=True)       # Z = softmax(QK^T/sqrt(d)) V
        x = x + self.drop(z)                                  # residual connection
        x = x + self.drop(self.ff(self.norm2(x)))             # feed-forward + residual
        return x, attn


class ECG_Transformer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=2, d_ff=128, num_classes=5):
        super().__init__()
        # Embedding: each scalar sample -> d_model vector (one token per time-step)
        self.embed = nn.Linear(1, d_model)
        # Learned positional embedding: attention itself has no notion of order,
        # so the position of each time-step must be added explicitly
        self.pos = nn.Parameter(torch.zeros(1, ec.SEQ_LEN, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        # Classifier on the average of all time-step representations
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, x, return_attention=False):
        # x: (batch, 187, 1) -> (batch, 187, d_model)
        x = self.embed(x) + self.pos
        attn_maps = []
        for blk in self.blocks:
            x, a = blk(x, need_weights=return_attention)
            attn_maps.append(a)
        x = self.norm(x).mean(dim=1)          # global average pooling over time
        logits = self.classifier(x)
        return (logits, attn_maps) if return_attention else logits


# --------------------------------------------------------------------------
# 2. Model-specific visualisation: attention maps
# --------------------------------------------------------------------------
def plot_attention(model, X_test, y_test, out_dir):
    """
    For one beat of each class:
      top    - the beat, coloured by how much attention each time-step RECEIVES
               (i.e. which parts of the waveform the model finds important)
      bottom - the full 187x187 attention matrix A of the last layer
               (row = query time-step, column = key time-step it looks at)
    """
    model.eval()
    fig, axes = plt.subplots(2, ec.NUM_CLASSES, figsize=(20, 8),
                             gridspec_kw={"height_ratios": [1, 2]})
    for c in range(ec.NUM_CLASSES):
        i = np.where(y_test == c)[0][0]
        x = torch.from_numpy(X_test[i]).view(1, 187, 1).to(ec.DEVICE)
        with torch.no_grad():
            _, maps = model(x, return_attention=True)
        A = maps[-1][0].cpu().numpy()                 # (187, 187)
        importance = A.mean(axis=0)                   # attention received per time-step
        sc = axes[0, c].scatter(np.arange(187), X_test[i], c=importance, cmap="inferno", s=12)
        axes[0, c].plot(X_test[i], color="grey", lw=0.6, zorder=0)
        axes[0, c].set_title(f"Class {ec.SHORT_NAMES[c]} - attention received")
        axes[0, c].set_xlim(0, 186)
        axes[1, c].imshow(A, cmap="viridis", aspect="auto")
        axes[1, c].set_xlabel("Key time-step")
        axes[1, c].set_ylabel("Query time-step")
    fig.colorbar(sc, ax=axes[0, :].tolist(), shrink=0.8, label="mean attention")
    fig.suptitle(f"{MODEL_NAME} - self-attention of the last encoder layer")
    fig.savefig(os.path.join(out_dir, "attention_maps.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. Main: load data -> train -> evaluate -> save results
# --------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ec.set_seed()

    X_train, y_train, X_val, y_val, X_test, y_test = ec.load_data()
    train_loader = ec.make_loader(X_train, y_train, shuffle=True)
    val_loader = ec.make_loader(X_val, y_val)
    test_loader = ec.make_loader(X_test, y_test)

    model = ECG_Transformer()
    print(model)
    with open(os.path.join(RESULTS_DIR, "model_architecture.txt"), "w") as f:
        f.write(str(model))

    model, history = ec.train_model(model, train_loader, val_loader,
                                    epochs=30, lr=1e-3, patience=6,
                                    results_dir=RESULTS_DIR)

    plot_attention(model, X_test, y_test, RESULTS_DIR)
    ec.evaluate_and_report(
        model, MODEL_NAME, history, test_loader, X_test, RESULTS_DIR,
        extra_notes="- Every one of the 187 time-steps attends to all others "
                    "(a 187x187 matrix per head per layer), so all positions are processed "
                    "**in parallel**. This is the 'Medium (parallelizable)' speed on slide 12. "
                    "The cost grows with the square of the sequence length.\n"
                    "- `attention_maps.png` shows which parts of each waveform the model "
                    "relies on. Bright vertical bands in the matrix are time-steps (usually "
                    "the R peak and T wave) that every other position looks at.")
