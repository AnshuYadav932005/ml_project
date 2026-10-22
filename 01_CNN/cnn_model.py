"""
Algorithm 1 of 4 - 1D Convolutional Neural Network (1D-CNN)
===========================================================
Idea (slide 4): slide small filters across the ECG beat to detect local wave
shapes (P wave, QRS complex, T wave). Stacking Conv -> ReLU -> Pool layers builds
increasingly abstract shape detectors; a final dense layer maps them to 5 classes.

Equations implemented (slide 5):
    y_i  = sum_k( x_{i+k} * w_k ) + b          -> nn.Conv1d
    a_i  = max(0, y_i)                          -> nn.ReLU
    p_j  = max(a_{2j}, a_{2j+1})                -> nn.MaxPool1d(2)
    y^   = softmax(W * flatten(p) + b)          -> nn.Flatten + nn.Linear
                                                   (softmax is applied inside the
                                                    cross-entropy loss / at prediction)
Run:  python 01_CNN/cnn_model.py
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
MODEL_NAME = "1D-CNN"


# --------------------------------------------------------------------------
# 1. Model definition
# --------------------------------------------------------------------------
class ECG_CNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # Feature extractor: 3 blocks of Conv -> BatchNorm -> ReLU -> MaxPool.
        # Each block doubles the number of filters and halves the length:
        #   (1, 187) -> (32, 93) -> (64, 46) -> (128, 23)
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=5, padding=2),  # convolution  y = x*w + b
                nn.BatchNorm1d(c_out),                             # keeps activations well scaled
                nn.ReLU(),                                         # a = max(0, y)
                nn.MaxPool1d(2),                                   # p = max of each pair
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))

        # Classifier: flatten the pooled feature maps and map them to 5 classes
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # (128, 23) -> 2944 vector
            nn.Linear(128 * 23, 128),
            nn.ReLU(),
            nn.Dropout(0.3),                    # regularisation against over-fitting
            nn.Linear(128, num_classes),        # logits; softmax gives the probabilities
        )

    def forward(self, x):
        # x arrives as (batch, 187, 1); Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------
# 2. Model-specific visualisation: what the first-layer filters respond to
# --------------------------------------------------------------------------
def plot_feature_maps(model, X_test, y_test, out_dir):
    """
    Pass one beat of each class through the first conv block and show the
    activations of a few filters. Bright regions show where a filter
    'detected' its shape, e.g. many filters fire on the QRS spike.
    """
    model.eval()
    fig, axes = plt.subplots(2, ec.NUM_CLASSES, figsize=(20, 6),
                             gridspec_kw={"height_ratios": [1, 2]})
    for c in range(ec.NUM_CLASSES):
        i = np.where(y_test == c)[0][0]
        x = torch.from_numpy(X_test[i]).view(1, 187, 1).to(ec.DEVICE)
        with torch.no_grad():
            # first block without the pooling -> one activation per time-step
            act = model.features[0][:3](x.transpose(1, 2))[0, :16].cpu().numpy()
        axes[0, c].plot(X_test[i], color="black")
        axes[0, c].set_title(f"Class {ec.SHORT_NAMES[c]} - input beat")
        axes[0, c].set_xlim(0, 186)
        axes[1, c].imshow(act, aspect="auto", cmap="viridis")
        axes[1, c].set_xlabel("Time-step")
        axes[1, c].set_ylabel("Filter #")
    fig.suptitle(f"{MODEL_NAME} - first-layer feature maps (16 of 32 filters after ReLU)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_maps.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. Main: load data -> train -> evaluate -> save results
# --------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ec.set_seed()

    # Data: same split for every model
    X_train, y_train, X_val, y_val, X_test, y_test = ec.load_data()
    train_loader = ec.make_loader(X_train, y_train, shuffle=True)
    val_loader = ec.make_loader(X_val, y_val)
    test_loader = ec.make_loader(X_test, y_test)

    # Build and summarise the model
    model = ECG_CNN()
    print(model)
    with open(os.path.join(RESULTS_DIR, "model_architecture.txt"), "w") as f:
        f.write(str(model))

    # Train (best validation macro-F1 weights are restored at the end)
    model, history = ec.train_model(model, train_loader, val_loader,
                                    epochs=30, lr=1e-3, patience=6,
                                    results_dir=RESULTS_DIR)

    # Model-specific figure + standard evaluation report
    plot_feature_maps(model, X_test, y_test, RESULTS_DIR)
    ec.evaluate_and_report(
        model, MODEL_NAME, history, test_loader, X_test, RESULTS_DIR,
        extra_notes="- The CNN only looks at local windows. Its receptive field after "
                    "3 blocks is about 36 samples (~0.3 s), which covers a full QRS "
                    "complex. That is why it is good at spotting abnormal beat **shapes** "
                    "(V, Q).\n"
                    "- Convolutions run in parallel over the whole beat, so it is the "
                    "fastest model to train and to run (see `feature_maps.png` for what "
                    "the first-layer filters respond to).")
