"""Ad-hoc alignment check on N totally random CLEVR object pairs (any two objects in any
scene, no purity/chain structure at all): extracts r_AB from a fresh forward pass, projects
it onto the (horizontal, closefar) axis units, and compares that 2D "latent coordinate" to
the pair's real (horizontal, closefar) displacement via cosine similarity, per layer.

Axis units are recomputed (CPU only) from the already-extracted clevr triplet shard
(triplet_pipeline.build_own_axes_signed, both cross-val groups pooled -- these random
pairs are independent of that shard, so no leakage concern).

Usage:
    python analysis/random_pair_alignment.py --clevr-val-dir /path/to/CLEVR_v1.0/images/val --n 500 --model qwen3vl
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, one level above this file's subfolder
sys.path[:0] = [os.path.join(_ROOT, d) for d in ("core", "analysis", "steering", "chain_hop", "multihop_referring")]
import argparse
import json
import random

import numpy as np
from PIL import Image

import axis_pipeline as ap
import triplet_pipeline as cxp
import cross_axis_analysis as caa


def build_axis_units():
    triplets = caa.load_all_triplets("clevr")
    groups = caa.group_split(len(triplets))
    real_pos = cxp.real_positions("clevr")
    own_axes = cxp.build_own_axes_signed(triplets, groups, real_pos)
    return {ax: ap.unit(own_axes[1][ax] + own_axes[2][ax]) for ax in ["horizontal", "closefar"]}


def sample_pairs(n, seed):
    with open(cxp.CLEVR_SCENES_PATH) as f:
        scenes = json.load(f)["scenes"]
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < n:
        s = rng.choice(scenes)
        if len(s["objects"]) < 2:
            continue
        i, j = rng.sample(range(len(s["objects"])), 2)
        objA, objB = s["objects"][i], s["objects"][j]
        if cxp.clevr_desc(objA) == cxp.clevr_desc(objB):
            continue
        pairs.append((s, objA, objB))
    return pairs


def real_displacement(s, objA, objB):
    right_dir, behind_dir = s["directions"]["right"], s["directions"]["behind"]
    return np.array([
        cxp.dot(objB["3d_coords"], right_dir) - cxp.dot(objA["3d_coords"], right_dir),
        cxp.dot(objB["3d_coords"], behind_dir) - cxp.dot(objA["3d_coords"], behind_dir),
    ])


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--clevr-val-dir", required=True, help="path to CLEVR_v1.0/images/val/")
    cli.add_argument("--n", type=int, default=500)
    cli.add_argument("--model", choices=ap.MODELS, default=ap.DEFAULT_MODEL)
    cli.add_argument("--seed", type=int, default=42)
    args = cli.parse_args()
    ap.set_model(args.model)

    print("building axis units from cached clevr triplet shard (CPU only)...")
    axis_units = build_axis_units()

    print(f"sampling {args.n} random object pairs...")
    pairs = sample_pairs(args.n, args.seed)

    model, proc = ap.load_model()
    num_l = ap.NUM_HIDDEN_STATES
    all_cos = np.zeros((args.n, num_l))
    img_cache = {}
    for k, (s, objA, objB) in enumerate(pairs):
        path = os.path.join(args.clevr_val_dir, s["image_filename"])
        if path not in img_cache:
            img_cache[path] = Image.open(path).convert("RGB")
            if len(img_cache) > 4:
                img_cache.pop(next(iter(img_cache)))
        img = img_cache[path]
        A_desc, B_desc = cxp.clevr_desc(objA), cxp.clevr_desc(objB)
        hA, hB = ap.object_nodes_alllayers(model, proc, img, A_desc, B_desc)
        r_AB = hB - hA

        latent_coord = np.stack([np.einsum("ld,ld->l", r_AB, axis_units[ax])
                                  for ax in ["horizontal", "closefar"]], axis=-1)  # (num_l, 2)
        real_disp = real_displacement(s, objA, objB)
        cos = latent_coord @ real_disp / (np.linalg.norm(latent_coord, axis=-1) * np.linalg.norm(real_disp) + 1e-9)
        all_cos[k] = cos

        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{args.n} pairs done")

    out_path = f"{ap.RESULTS}/readout_vqa/random_pair_alignment_{args.model}.npz"
    np.savez(out_path, cos=all_cos)
    print(f"saved -> {out_path}")

    mean_cos = all_cos.mean(axis=0)
    std_cos = all_cos.std(axis=0)
    layers = ap.report_layers(num_l)
    print(f"\n=== [{args.model}] mean cos(latent coord, real displacement) over {args.n} random pairs ===")
    for L in layers:
        print(f"  layer {L:2d}: mean={mean_cos[L]:.3f}  std={std_cos[L]:.3f}")
    print(f"best layer: {mean_cos.argmax()}  mean_cos={mean_cos.max():.3f}")

    draw_plot(mean_cos, std_cos, args.n, args.model, num_l)


def draw_plot(mean_cos, std_cos, n, model, num_l):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, SECONDARY, MUTED, GRID = "#ffffff", "#1a1a19", "#52514e", "#898781", "#e5e4df"
    COLOR = "#2a78d6"

    layers = np.arange(num_l)
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    se = std_cos / np.sqrt(n)
    ax.fill_between(layers, mean_cos - se, mean_cos + se, color=COLOR, alpha=0.2, zorder=1)
    ax.plot(layers, mean_cos, color=COLOR, linewidth=2.2, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (2, 2)), alpha=0.6, zorder=1)

    ax.set_xlabel("layer", color=INK, fontsize=11.5)
    ax.set_ylabel("mean cos(latent coordinate, real displacement)", color=INK, fontsize=11)
    ax.set_title(f"[{model}] {n} random CLEVR object pairs: latent-vs-real alignment by layer\n"
                 f"(shaded band = +/- standard error)",
                 color=INK, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=SECONDARY, length=0)

    fig.subplots_adjust(top=0.85, bottom=0.11, left=0.11, right=0.97)
    out_path = f"{ap.PLOTS}/readout_vqa/random_pair_alignment_{model}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
