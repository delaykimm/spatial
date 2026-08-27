"""Compares axis_pipeline.py's axis vectors + axis_steering.py's steering results across
the 4 datasets (What'sUp/SpatialTunnel/Aug1/Aug2), at LAYER=25:

  1. cross_dataset_cosine() + draw_heatmap(): cosine similarity per dataset pair and axis
     -- does "horizontal" from What'sUp's real photos point the same way as "horizontal"
     from Aug2's synthetic renders?
  2. load_steering_results() + draw_steering_comparison(): all 4 datasets' steering curves
     together (one panel per axis) -- does injecting the axis flip the answer everywhere?

Run axis_pipeline.py and axis_steering.py for all 4 datasets first.
"""
import numpy as np

import axis_pipeline as ap

RESULTS = ap.RESULTS  # both already computed by axis_pipeline.py (which this file imports anyway)
PLOTS = ap.PLOTS
DATASETS = ["whatsup", "spatialtunnel", "aug1", "aug2"]
DATASET_LABELS = {"whatsup": "What'sUp", "spatialtunnel": "SpatialTunnel",
                   "aug1": "Aug1", "aug2": "Aug2"}
AXES3 = ["horizontal", "vertical", "closefar"]
# whatsup/spatialtunnel/aug1's multilayer_axis_vectors.npz key their axes x/y/z; only
# aug2's already uses horizontal/vertical/closefar directly (see axis_pipeline.py's
# DATASET_CONFIGS). This translates the former to the latter so the rest of this file can
# use horizontal/vertical/closefar uniformly, matching axis_steering.py's own keys.
AXIS_KEY = {"horizontal": "x", "vertical": "y", "closefar": "z"}
LAYER = ap.CFG["cross_dataset_analysis"]["layer"]

unit = ap.unit  # canonical L2-normalize now lives in axis_pipeline.py


def load_axes():
    """Returns {dataset: {'horizontal':(4096,),'vertical':...,'closefar':...}} at LAYER,
    using each dataset's '+' axis."""
    wu = np.load(f"{RESULTS}/axis_vectors/whatsup_multilayer_axis_vectors.npz")
    st = np.load(f"{RESULTS}/axis_vectors/spatialtunnel_multilayer_axis_vectors.npz")
    a1 = np.load(f"{RESULTS}/axis_vectors/aug1_multilayer_axis_vectors.npz")
    a2 = np.load(f"{RESULTS}/axis_vectors/aug2_multilayer_axis_vectors.npz")
    return {
        "whatsup":       {ax: wu[f"{AXIS_KEY[ax]}_right_axis"][LAYER] for ax in AXES3},
        "spatialtunnel": {ax: st[f"{AXIS_KEY[ax]}_right_axis"][LAYER] for ax in AXES3},
        "aug1":          {ax: a1[f"{AXIS_KEY[ax]}_right_axis"][LAYER] for ax in AXES3},
        "aug2":          {ax: a2[f"{ax}_plus_axis"][LAYER] for ax in AXES3},
    }


def cross_dataset_cosine():
    """Returns ({pair_label: {axis: cosine}}, pairs) at LAYER, for all 6 dataset pairs."""
    raw = load_axes()
    pairs = [(DATASETS[i], DATASETS[j]) for i in range(4) for j in range(i + 1, 4)]
    out = {}
    for d1, d2 in pairs:
        out[f"{d1}-{d2}"] = {ax: float(np.dot(unit(raw[d1][ax]), unit(raw[d2][ax]))) for ax in AXES3}
    return out, pairs


def draw_heatmap(cross_cos, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import FancyBboxPatch

    CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f0efec", "#1c5cab", "#0a2f5c"])
    NORM = Normalize(vmin=0.0, vmax=1.0)
    SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
    pair_labels = [f"{DATASET_LABELS[d1]}-{DATASET_LABELS[d2]}" for d1, d2 in pairs]
    M = np.array([[cross_cos[f"{d1}-{d2}"][ax] for ax in AXES3] for d1, d2 in pairs])  # (6,3)
    n_rows, n_cols = M.shape

    # square cells with a visible gap and rounded corners -- imshow can't do per-cell
    # gaps/rounding, so each cell is its own FancyBboxPatch instead.
    GAP = 0.06
    fig, ax = plt.subplots(figsize=(5.2, 9), dpi=140, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for r in range(n_rows):
        for c in range(n_cols):
            v = M[r, c]
            y = n_rows - 1 - r  # row 0 at the top
            ax.add_patch(FancyBboxPatch(
                (c + GAP / 2, y + GAP / 2), 1 - GAP, 1 - GAP,
                boxstyle="round,pad=0,rounding_size=0.09",
                linewidth=0, facecolor=CMAP(NORM(v))))
            txt_color = "#fcfcfb" if v > 0.55 else INK
            ax.text(c + 0.5, y + 0.5, f"{v:+.3f}", ha="center", va="center",
                     fontsize=10, color=txt_color, zorder=3)

    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.set_xticks([c + 0.5 for c in range(n_cols)]); ax.set_xticklabels(AXES3, color=INK, fontsize=11)
    ax.set_yticks([n_rows - 1 - r + 0.5 for r in range(n_rows)]); ax.set_yticklabels(pair_labels, color=INK, fontsize=10.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=INK, length=0)

    sm = ScalarMappable(cmap=CMAP, norm=NORM)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.05, label="cosine similarity")
    cbar.ax.yaxis.label.set_color(INK)
    cbar.ax.tick_params(colors=MUTED)
    cbar.outline.set_edgecolor(MUTED)

    fig.suptitle(f"Cross-dataset axis-vector cosine similarity (layer {LAYER})",
                 color=INK, fontsize=13.5, y=0.97)
    out_path = f"{PLOTS}/axis_vectors/cross_dataset_axis_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


# colors match cross_axis_alignment.py's draw_datasetaxis_comparison_plot, so the same
# dataset reads as the same color across every plot in this pipeline.
STEERING_DATASETS = [
    ("whatsup", "What'sUp", "#1baf7a"),
    ("spatialtunnel", "SpatialTunnel", "#eda100"),
    ("aug1", "Aug1", "#e87ba4"),
    ("aug2", "Aug2", "#008300"),
]


def load_steering_results():
    """Returns {dataset: npz} from axis_steering.py's output -- run that script for all 4
    datasets first (python axis_steering.py --dataset {name})."""
    return {name: np.load(f"{RESULTS}/steering/{name}_global_axis_steering.npz") for name, _, _ in STEERING_DATASETS}


def draw_steering_comparison(steering_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, SECONDARY, MUTED, GRID = "#ffffff", "#1a1a19", "#52514e", "#898781", "#e5e4df"

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), dpi=150, facecolor=SURFACE, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("Global (pooled) axis steering: 4-dataset comparison",
                 color=INK, fontsize=14.5, fontweight="bold", x=0.045, ha="left", y=0.99)

    for i, axis_name in enumerate(AXES3):
        ax = axes[i]
        ax.set_facecolor(SURFACE)
        for name, label, color in STEERING_DATASETS:
            d = steering_data[name]
            fr = d[f"{axis_name}_alpha_fractions"]
            acc = d[f"{axis_name}_acc"] * 100
            vf = d[f"{axis_name}_valid_frac"]
            ax.plot(fr, acc, color=color, linewidth=2.1, label=label, zorder=3, solid_capstyle="round")
            ax.scatter(fr, acc, s=12 + 70 * vf, color=color, zorder=4, alpha=0.85,
                        edgecolors=SURFACE, linewidths=1)
        ax.axhline(50, color=MUTED, linewidth=1, linestyle=(0, (2, 2)), alpha=0.6, zorder=1)
        ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (2, 2)), alpha=0.6, zorder=1)
        ax.set_title(axis_name, color=INK, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-3, 103)
        ax.set_xlabel("steering alpha (x reference hidden norm)", color=INK, fontsize=10)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=SECONDARY, length=0, labelsize=9.5)
        if i == 0:
            ax.set_ylabel("P(answer = true '+' object) (%)", color=INK, fontsize=10.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.985), frameon=False,
               fontsize=9.5, labelcolor=SECONDARY, handlelength=1.6, handletextpad=0.5, ncols=2)

    fig.subplots_adjust(wspace=0.06, top=0.83, bottom=0.12, left=0.055, right=0.98)
    out_path = f"{PLOTS}/steering/global_axis_steering_4datasets.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    cross_cos, pairs = cross_dataset_cosine()
    draw_heatmap(cross_cos, pairs)

    steering_data = load_steering_results()
    draw_steering_comparison(steering_data)
