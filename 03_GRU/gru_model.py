"""
Algorithm 3 of 4 - Gated Recurrent Unit (GRU)
=============================================
Idea (slide 8): a streamlined LSTM. The forget and input gates are merged into a
single "update gate" and there is no separate cell state - same memory idea,
fewer parameters, faster training.

Equations implemented (slide 9) - all computed inside torch.nn.GRU:
    z_t  = sigma(W_z [h_{t-1}, x_t])                 update gate
    r_t  = sigma(W_r [h_{t-1}, x_t])                 reset gate
    h~_t = tanh (W [r_t * h_{t-1}, x_t])             candidate hidden state
    h_t  = (1 - z_t) * h_{t-1} + z_t * h~_t          new hidden state
The final hidden state h_187 is passed to a dense layer -> softmax over 5 classes.

To make the LSTM-vs-GRU comparison fair, this model uses exactly the same
sizes as the LSTM (2 layers, 64 hidden units, same classifier head).

Run:  python 03_GRU/gru_model.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt

# make the shared helper module in the project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ecg_common as ec

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODEL_NAME = "GRU"


# --------------------------------------------------------------------------
# 1. Model definition
# --------------------------------------------------------------------------
class ECG_GRU(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, num_classes=5):
        super().__init__()
        # Recurrent part: 2 stacked GRU layers (3 gate blocks each instead of the LSTM's 4,
        # so ~25% fewer recurrent weights)
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=0.2)
        # Classifier on top of the final hidden state (identical to the LSTM's)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, return_sequence=False):
        # x: (batch, 187, 1). Same packing as the LSTM: the GRU stops at each
        # beat's real end instead of reading the trailing zero padding.
        packed = pack_padded_sequence(x, ec.beat_lengths(x), batch_first=True,
                                      enforce_sorted=False)
        outputs, h_n = self.gru(packed)      # no cell state, only the hidden state
        if return_sequence:                  # h_t for every t (used for the plot below)
            return pad_packed_sequence(outputs, batch_first=True,
                                       total_length=ec.SEQ_LEN)[0]
        return self.classifier(h_n[-1])      # final hidden state of the top layer


# --------------------------------------------------------------------------
# 2. Model-specific visualisation: the hidden state over time
# --------------------------------------------------------------------------
def plot_hidden_states(model, X_test, y_test, out_dir):
    """
    For one beat of each class, show how the 64 hidden units of the top GRU
    layer evolve across the 187 time-steps.
    """
    model.eval()
    fig, axes = plt.subplots(2, ec.NUM_CLASSES, figsize=(20, 6),
                             gridspec_kw={"height_ratios": [1, 2]})
    for c in range(ec.NUM_CLASSES):
        i = np.where(y_test == c)[0][0]
        x = torch.from_numpy(X_test[i]).view(1, 187, 1).to(ec.DEVICE)
        with torch.no_grad():
            h = model(x, return_sequence=True)[0].cpu().numpy().T   # (64, 187)
        axes[0, c].plot(X_test[i], color="black")
        axes[0, c].set_xlim(0, 186)
        axes[0, c].set_title(f"Class {ec.SHORT_NAMES[c]} - input beat")
        axes[1, c].imshow(h, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        axes[1, c].set_xlabel("Time-step")
        axes[1, c].set_ylabel("Hidden unit")
    fig.suptitle(f"{MODEL_NAME} - hidden state h_t of the top layer over time")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hidden_states.png"), dpi=130)
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

    model = ECG_GRU()
    print(model)
    with open(os.path.join(RESULTS_DIR, "model_architecture.txt"), "w") as f:
        f.write(str(model))

    model, history = ec.train_model(model, train_loader, val_loader,
                                    epochs=30, lr=2e-3, patience=6,
                                    results_dir=RESULTS_DIR)

    plot_hidden_states(model, X_test, y_test, RESULTS_DIR)
    ec.evaluate_and_report(
        model, MODEL_NAME, history, test_loader, X_test, RESULTS_DIR,
        extra_notes="- The GRU has the same layer sizes as the LSTM but ~25% fewer recurrent "
                    "weights (3 gate blocks instead of 4, and no separate cell state). "
                    "Each step is therefore cheaper, which is the speed-vs-accuracy "
                    "trade-off slide 8 set out to test.\n"
                    "- Like the LSTM, it is still sequential over the 187 time-steps. "
                    "`hidden_states.png` shows how its memory changes over the beat.")
