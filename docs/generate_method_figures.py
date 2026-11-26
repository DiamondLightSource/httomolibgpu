"""Generate conceptual method illustrations (NumPy/Matplotlib; no GPU needed).

These are deterministic synthetic examples, not library validation results.
Run from any directory; figures are written beside the method documentation.
"""
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "source/_static/figures/method_guides"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    }
)


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


x = np.linspace(0, 1, 256)
transmission = 0.85 - 0.4 * np.exp(-(((x - 0.5) / 0.16) ** 2))
dark = 60 + 10 * x
flat = 1200 * (1 + 0.22 * np.cos(2 * np.pi * x))
raw = dark + (flat - dark) * transmission
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
for y, label in [
    (raw, "Raw projection I"),
    (flat, "Mean flat F"),
    (dark, "Mean dark D"),
]:
    axes[0].plot(x, y, label=label)
axes[0].set(
    xlabel="Detector position (normalised)",
    ylabel="Detector counts",
    title="Uneven illumination and offset",
)
axes[0].legend(fontsize=9)
axes[1].plot(
    x, (raw - dark) / (flat - dark), color="#7030a0", label="Corrected transmission"
)
axes[1].plot(x, transmission, "--", color="#009e73", label="Synthetic ground truth")
axes[1].set(
    xlabel="Detector position (normalised)",
    ylabel="Transmission",
    title="Dark / flat correction",
)
axes[1].legend(fontsize=9)
save(fig, "normalisation.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
w = np.linspace(0, 1, 100)
left = 0.6 + 0.03 * np.sin(w * 8)
right = left + 0.12
axes[0].plot(w, 1 - w, label="Left weight")
axes[0].plot(w, w, label="Right weight")
axes[0].set(
    xlabel="Position within overlap",
    ylabel="Weight",
    title="Complementary linear ramps",
)
axes[0].legend()
axes[1].plot(
    w, np.where(w < 0.5, left, right), "--", label="Hard join", color="#d55e00"
)
axes[1].plot(w, (1 - w) * left + w * right, label="Blended overlap", color="#7030a0")
axes[1].set(
    xlabel="Position within overlap",
    ylabel="Intensity (arbitrary units)",
    title="Smoother transition, mismatch remains",
)
axes[1].legend()
save(fig, "seam_blending.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
frames = np.arange(12)
values = np.array([0.7, 0.8, 0.6, 0.9, 0.8, 0.7, 0.65, 0.75, 0.85, 0.8, 0.7, 0.9])
axes[0].plot(frames, values, "o-", label="Input frames", color="#0072b2")
for start in range(0, 12, 3):
    axes[0].hlines(
        values[start : start + 3].mean(),
        start - 0.3,
        start + 2.3,
        color="#d55e00",
        linewidth=3,
    )
axes[0].plot([], [], color="#d55e00", linewidth=3, label="Mean of each group of 3")
axes[0].set(
    xlabel="Consecutive frame index",
    ylabel="Intensity (arbitrary units)",
    title="Group averaging",
)
axes[0].legend(fontsize=9)
k = np.arange(1, 17)
axes[1].plot(k, 1 / np.sqrt(k), "o-", color="#7030a0")
axes[1].set(
    xlabel="Independent frames averaged",
    ylabel="Relative noise standard deviation",
    title="Ideal independent-noise model",
)
save(fig, "frame_averaging.png")

fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
k = np.linspace(0, 4, 400)
for a in [0.1, 1, 4]:
    ax.plot(k, 1 / (1 + a * k * k), label=f"α = {a}")
ax.set(
    xlabel="Spatial angular frequency (normalised)",
    ylabel="Transfer function H(k)",
    title="Paganin low-pass filtering: stronger α smooths more",
)
ax.legend()
save(fig, "paganin_transfer.png")
