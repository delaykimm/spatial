"""Different-image steering: inject one image's axis vector while asking about a
DIFFERENT image's SAME relationship -- does it still causally steer the answer? Ports
ours/binding's triplet3ax_steering_crossimage.py.

Complements diffpair_steering.py (fixes image, varies pair); here the pair is fixed and
the image varies:

  --dataset sameaxis: r_AC represents "the axis" regardless of scene -- inject a donor
      scene's r_AC into the target scene's A-vs-B/B-vs-C question.
  --dataset {clevr,triplet3ax}: no single vector stands for "the axis" here, so inject a
      DIFFERENT triplet's r_AB into this triplet's A-vs-B question (same for r_BC/B-vs-C)
      -- same axis, same pair type, different image.

Still steering => the vector is a generic, transferable axis direction. Collapsing =>
same-image success depended on the exact image being asked about.

Donor assignment: deterministic offset pairing within same-(axis,pair) pools
(donor = pool[(i + DONOR_OFFSET) % len(pool)]) -- every target gets a different real
image's vector, not one fixed donor.

Usage:
    python steering/diffimage_steering.py --dataset sameaxis --model qwen3vl
    python steering/diffimage_steering.py --dataset clevr --model qwen3vl
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, one level above this file's subfolder
sys.path[:0] = [os.path.join(_ROOT, d) for d in ("core", "analysis", "steering", "chain_hop", "multihop_referring")]
import argparse
import random
from collections import defaultdict

import numpy as np

import triplet_pipeline as cxp
import cross_axis_analysis as caa
import diffpair_steering as ss
import axis_steering as steer
import axis_pipeline as ap

DATASETS = ["sameaxis", "clevr", "triplet3ax"]
DONOR_OFFSET = ap.CFG["diffimage_steering"]["donor_offset"]


def sameaxis_diffimage_items():
    """{axis_name: [items]} -- same items as diffpair_steering.sameaxis_items(), each
    additionally carrying 'donor_idx' (a different scene, same axis)."""
    manifest = cxp.sameaxis_load_or_build_manifest()
    out = {}
    for axis_name in ss.AXES3:
        n_scenes = len(manifest[axis_name])
        items = ss.adjacent_pairs_indexed(manifest[axis_name])
        for it in items:
            it["axis"] = axis_name
            it["donor_idx"] = (it["scene_idx"] + DONOR_OFFSET) % n_scenes
            assert it["donor_idx"] != it["scene_idx"]
        out[axis_name] = items
    return out


def crossaxis_diffimage_items(dataset):
    """{axis_name: [items]} -- every triplet gives an AB item (axis=ax1) and a BC item
    (axis=ax2), minus/plus_obj from real-world ground truth (no guaranteed ordering).
    Each item also carries 'donor_config'/'donor_idx': a different triplet from the SAME
    (axis, pair) pool, so its r_AB/r_BC is a real vector for that exact axis."""
    cfg = cxp.DATASET_CFG[dataset]
    triplets_by_config = cfg["load_fn"]()
    axis_pairs = caa.CLEVR_AXIS_PAIRS if dataset == "clevr" else cxp.AXIS_PAIRS
    real_pos = cxp.real_positions(dataset)

    items_by_axis = defaultdict(list)
    for config, triplets in triplets_by_config.items():
        ax1, ax2 = axis_pairs[config]
        for idx, t in enumerate(triplets):
            pos = real_pos[(config, idx)]
            for pair, e1, e2, axis_name in [("AB", "A", "B", ax1), ("BC", "B", "C", ax2)]:
                real_disp = pos[e2][axis_name] - pos[e1][axis_name]
                minus_obj, plus_obj = (t[e1], t[e2]) if real_disp > 0 else (t[e2], t[e1])
                items_by_axis[axis_name].append({"image_path": t["image_path"], "minus_obj": minus_obj,
                                                   "plus_obj": plus_obj, "config": config, "idx": idx,
                                                   "pair": pair, "axis": axis_name})

    for axis_name, items in items_by_axis.items():
        by_pair = defaultdict(list)
        for it in items:
            by_pair[it["pair"]].append(it)
        for pair, plist in by_pair.items():
            n = len(plist)
            for i, it in enumerate(plist):
                donor = plist[(i + DONOR_OFFSET) % n]
                it["donor_config"], it["donor_idx"] = donor["config"], donor["idx"]
    return dict(items_by_axis)


def run(dataset, model):
    ap.set_model(model)
    m, proc = ap.load_model()
    steer_module = steer.find_decoder_layers(m)[ss.LAYER - 1]
    rng = random.Random(ss.SEED)
    out = {}

    if dataset == "sameaxis":
        items_by_axis = sameaxis_diffimage_items()
        shard, img_dir = ss.sameaxis_shard_and_imgdir()
        own_vec_fn = lambda it: shard[f"{it['axis']}|{it['scene_idx']}|r_AC"]
        diffimage_vec_fn = lambda it: shard[f"{it['axis']}|{it['donor_idx']}|r_AC"]
        own_header = "this scene's OWN r_AC (baseline, matches diffpair_steering.py --dataset sameaxis)"
        cross_header = f"a DIFFERENT scene's r_AC (same axis, donor from offset={DONOR_OFFSET} pairing)"
    else:
        items_by_axis = crossaxis_diffimage_items(dataset)
        shard, img_dir = ss.crossaxis_shard_and_imgdir(dataset)
        own_vec_fn = lambda it: shard[f"{it['config']}|{it['idx']}|r_{it['pair']}"]
        diffimage_vec_fn = lambda it: shard[f"{it['donor_config']}|{it['donor_idx']}|r_{it['pair']}"]
        own_header = "this triplet's OWN r_AB/r_BC (baseline, matches diffpair_steering.py's own-leg logic)"
        cross_header = (f"a DIFFERENT triplet's r_AB/r_BC (same axis, same pair type, "
                         f"donor from offset={DONOR_OFFSET} pairing within that pool)")

    for axis_name, pool in items_by_axis.items():
        items = rng.sample(pool, min(ss.N_ITEMS, len(pool)))
        for mode, vec_fn, header in [("own", own_vec_fn, own_header), ("diffimage", diffimage_vec_fn, cross_header)]:
            print(f"\n{'='*90}\n[{model}][{dataset}][{axis_name}][{mode}] CROSS-IMAGE ({header})\n"
                  f"N={len(items)} items, layer={ss.LAYER}\n{'='*90}")
            alphas, acc, vf, ref_norm = ss.steer_sweep(m, proc, steer_module, items, img_dir,
                                                         vec_fn, ss.POS_PHRASE[axis_name])
            out[f"{axis_name}_{mode}_alphas"] = np.array(alphas)
            out[f"{axis_name}_{mode}_alpha_fractions"] = np.array(ss.ALPHA_FRACTIONS)
            out[f"{axis_name}_{mode}_acc"] = np.array(acc)
            out[f"{axis_name}_{mode}_valid_frac"] = np.array(vf)
            out[f"{axis_name}_{mode}_ref_norm"] = ref_norm
    return out


def draw_plot(dataset, out, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, SECONDARY, MUTED, GRID = "#ffffff", "#1a1a19", "#52514e", "#898781", "#e5e4df"
    MODE_COLORS = {"own": "#c0392b", "diffimage": "#2a78d6"}
    if dataset == "sameaxis":
        MODE_LABELS = {"own": "own r_AC (same scene)", "diffimage": "donor scene's r_AC (different image)"}
        title = f"[{model}] sameaxis: does a DIFFERENT scene's r_AC still steer this scene's A-vs-B/B-vs-C?"
    else:
        MODE_LABELS = {"own": "own r_AB/r_BC (same triplet)", "diffimage": "donor triplet's r_AB/r_BC (different image)"}
        title = f"[{model}] {dataset}: does a DIFFERENT triplet's r_AB/r_BC still steer this triplet's own question?"

    axes_order = [a for a in (ss.AXES3 if dataset == "sameaxis" else cxp.AXES_BY_DATASET[dataset])
                  if f"{a}_own_acc" in out]
    fig, axes_plt = plt.subplots(1, len(axes_order), figsize=(5.3 * len(axes_order), 6), dpi=150,
                                  facecolor=SURFACE, sharey=True, squeeze=False)
    axes_plt = axes_plt[0]
    fig.suptitle(title, color=INK, fontsize=13.5, fontweight="bold", x=0.045, ha="left", y=0.99)

    for i, axis_name in enumerate(axes_order):
        ax = axes_plt[i]
        ax.set_facecolor(SURFACE)
        for mode in ["own", "diffimage"]:
            prefix = f"{axis_name}_{mode}"
            fr = out[f"{prefix}_alpha_fractions"]
            acc = out[f"{prefix}_acc"] * 100
            vf = out[f"{prefix}_valid_frac"]
            ax.plot(fr, acc, color=MODE_COLORS[mode], linewidth=2.2, zorder=3, solid_capstyle="round",
                     label=MODE_LABELS[mode])
            ax.scatter(fr, acc, s=15 + 90 * vf, color=MODE_COLORS[mode], zorder=4, alpha=0.85,
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
            ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=INK)

    fig.subplots_adjust(wspace=0.06, top=0.85, bottom=0.12, left=0.06, right=0.98)
    out_path = f"{ap.PLOTS}/steering/{dataset}_diffimage_steering_{model}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset", choices=DATASETS, default="sameaxis")
    cli.add_argument("--model", choices=ap.MODELS, default=ap.DEFAULT_MODEL)
    args = cli.parse_args()

    out = run(args.dataset, args.model)
    out_path = f"{ap.RESULTS}/steering/{args.dataset}_diffimage_steering_{args.model}.npz"
    np.savez(out_path, **out)
    print(f"\nsaved -> {out_path}")
    draw_plot(args.dataset, out, args.model)
