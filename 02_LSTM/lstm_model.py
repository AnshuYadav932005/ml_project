"""
Algorithm 2 of 4 - Long Short-Term Memory (LSTM)
================================================
Idea (slide 6): read the ECG beat one time-step at a time while carrying a
"memory" (cell state) forward. Gates decide what to keep, forget and output, so
the build-up of rhythm across the sequence can be captured.

Equations implemented (slide 7) - all computed inside torch.nn.LSTM:
    f_t  = sigma(W_f [h_{t-1}, x_t] + b_f)        forget gate
    i_t  = sigma(W_i [h_{t-1}, x_t] + b_i)        input gate
    c~_t = tanh (W_c [h_{t-1}, x_t] + b_c)        candidate memory
    c_t  = f_t * c_{t-1} + i_t * c~_t             cell state update
    h_t  = o_t * tanh(c_t)                        hidden state (o_t = output gate)
The final hidden state h_187 is passed to a dense layer -> softmax over 5 classes.

Run:  python 02_LSTM/lstm_model.py
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
MODEL_NAME = "LSTM"


# --------------------------------------------------------------------------
# 1. Model definition
# --------------------------------------------------------------------------
class ECG_LSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, num_classes=5):
        super().__init__()
        # Recurrent part: 2 stacked LSTM layers; each reads 1 value per time-step.
        # The 2nd layer reads the hidden states of the 1st -> more abstract rhythm features.
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2)
        # Classifier on top of the final hidden state
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, return_sequence=False):
        # x: (batch, 187, 1). Beats are zero-padded at the end, so tell the LSTM
        # each beat's real length: it then stops updating at the true end of the
        # signal and h_n is the memory right after the last real sample.
        packed = pack_padded_sequence(x, ec.beat_lengths(x), batch_first=True,
                                      enforce_sorted=False)
        outputs, (h_n, c_n) = self.lstm(packed)
        if return_sequence:                  # h_t for every t (used for the plot below)
            return pad_packed_sequence(outputs, batch_first=True,
                                       total_length=ec.SEQ_LEN)[0]
        return self.classifier(h_n[-1])      # final hidden state of the top layer


# --------------------------------------------------------------------------
# 2. Model-specific visualisation: the hidden state ("memory") over time
# --------------------------------------------------------------------------
def plot_hidden_states(model, X_test, y_test, out_dir):
    """
    For one beat of each class, show how the 64 hidden units of the top LSTM
    layer evolve across the 187 time-steps. Changes in the pattern show where
    the network updates its memory, e.g. around the QRS complex.
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

    model = ECG_LSTM()
    print(model)
    with open(os.path.join(RESULTS_DIR, "model_architecture.txt"), "w") as f:
        f.write(str(model))

    model, history = ec.train_model(model, train_loader, val_loader,
                                    epochs=30, lr=2e-3, patience=6,
                                    results_dir=RESULTS_DIR)

    plot_hidden_states(model, X_test, y_test, RESULTS_DIR)
    ec.evaluate_and_report(
        model, MODEL_NAME, history, test_loader, X_test, RESULTS_DIR,
        extra_notes="- The LSTM has to process the 187 time-steps **one after another**, "
                    "so it cannot be parallelised over time. That makes it the slowest model "
                    "to train, as predicted on slide 12.\n"
                    "- It classifies from the final hidden state alone, so information "
                    "from early in the beat (P wave) has to survive every later gate update "
                    "to influence the decision. Beats are packed with "
                    "`pack_padded_sequence`, so the LSTM stops at the real end of each beat. "
                    "Without this, in a first trial run, it spent 4 epochs predicting "
                    "'Normal' for everything and only reached macro-F1 0.54 after 8 epochs, "
                    "because it had to remember the beat through ~140 zero-padding steps.\n"
                    "- `hidden_states.png` shows how the memory changes over the beat "
                    "(the flat region after the real end is padding the LSTM never processes).")
