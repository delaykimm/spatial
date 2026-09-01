"""Same-image, different-pair steering: inject a DIFFERENT pair's vector while asking
about THIS scene's judgment on that pair -- does it still causally move the answer?

  (A) --dataset sameaxis: A,B,C sorted along ONE axis per scene. Inject this scene's r_AC
      (a genuine single-axis span, same axis as the A-vs-B/B-vs-C question).
  (B) --dataset {clevr, triplet3ax}: cross-axis L-shaped triplets (A->B pure axis1, B->C
      pure axis2), so AC is the only pair with real signal on both axes: inject r_AB into
      AC's axis1 reading, r_BC into AC's axis2 reading. No guaranteed A<B<C ordering here,
      so the true "+" side is looked up fresh per item from real-world ground truth.

Both share axis_steering.py's steering machinery and output schema, so results are
directly comparable across dataset families.

Usage:
    python steering/diffpair_steering.py --dataset sameaxis --model qwen3vl
    python steering/diffpair_steering.py --dataset clevr --model qwen3vl
    python steering/diffpair_steering.py --dataset triplet3ax --model llava
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, one level above this file's subfolder
sys.path[:0] = [os.path.join(_ROOT, d) for d in ("core", "analysis", "steering", "chain_hop", "multihop_referring")]
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

import triplet_pipeline as cxp
import cross_axis_analysis as caa
import axis_steering as steer
import axis_pipeline as ap

_CFG = ap.CFG["diffpair_steering"]
LAYER = _CFG["layer"]
N_ITEMS = _CFG["n_items"]
SEED = _CFG["seed"]
ALPHA_FRACTIONS = [round(x, 1) for x in np.arange(-1.2, 1.21, 0.2)]
AXES3 = ["horizontal", "vertical", "closefar"]
POS_PHRASE = _CFG["pos_phrase"]
DATASETS = ["sameaxis", "clevr", "triplet3ax"]

unit = ap.unit  # canonical L2-normalize now lives in axis_pipeline.py


# =============================================================================
# (A) sameaxis: r_AC is a genuine single-axis span (A,B,C sorted along ONE axis)
# =============================================================================
def adjacent_pairs_indexed(scenes):
    """Every scene contributes 2 items: the A-vs-B judgment and the B-vs-C judgment (both
    steered with THIS scene's r_AC, a different pair's vector). A/B/C sorted by
    construction, so minus_obj=A-or-B/plus_obj=B-or-C needs no real-world lookup."""
    out = []
    for idx, s in enumerate(scenes):
        out.append({"image_path": s["image_path"], "minus_obj": s["A"], "plus_obj": s["B"],
                     "scene_idx": idx, "pair": "AB"})
        out.append({"image_path": s["image_path"], "minus_obj": s["B"], "plus_obj": s["C"],
                     "scene_idx": idx, "pair": "BC"})
    return out


def sameaxis_items():
    manifest = cxp.sameaxis_load_or_build_manifest()
    return {axis_name: adjacent_pairs_indexed(manifest[axis_name]) for axis_name in AXES3}


def sameaxis_shard_and_imgdir():
    return np.load(cxp.shard_out_path("sameaxis"), allow_pickle=True), cxp.SAMEAXIS_IMAGES_DIR


# =============================================================================
# (B) cross-axis {clevr, triplet3ax}: AC is the only pair with real signal on BOTH axes
# =============================================================================
def crossaxis_items(dataset):
    """{axis_name: [items]} -- AC is the only pair with a "different pair, same axis"
    vector available from this same scene (AB is only ever axis1-real, BC axis2-real), so
    every item asks about AC, steered with this scene's own r_AB or r_BC. minus/plus_obj
    from real-world ground truth (no guaranteed ordering)."""
    cfg = cxp.DATASET_CFG[dataset]
    triplets_by_config = cfg["load_fn"]()
    axis_pairs = caa.CLEVR_AXIS_PAIRS if dataset == "clevr" else cxp.AXIS_PAIRS
    real_pos = cxp.real_positions(dataset)

    items_by_axis = defaultdict(list)
    for config, triplets in triplets_by_config.items():
        ax1, ax2 = axis_pairs[config]
        for idx, t in enumerate(triplets):
            pos = real_pos[(config, idx)]
            for axis_name, src_field in [(ax1, "r_AB"), (ax2, "r_BC")]:
                real_disp = pos["C"][axis_name] - pos["A"][axis_name]
                minus_obj, plus_obj = (t["A"], t["C"]) if real_disp > 0 else (t["C"], t["A"])
                items_by_axis[axis_name].append({"image_path": t["image_path"], "minus_obj": minus_obj,
                                                   "plus_obj": plus_obj, "config": config, "idx": idx,
                                                   "pair": "AC", "axis": axis_name, "src_field": src_field})
    return dict(items_by_axis)


def crossaxis_shard_and_imgdir(dataset):
    cfg = cxp.DATASET_CFG[dataset]
    return np.load(cxp.shard_out_path(dataset), allow_pickle=True), cfg["images_dir"]


# =============================================================================
# shared steering sweep
# =============================================================================
def steer_sweep(m, proc, steer_module, items, img_dir, item_vec_fn, pos_phrase):
    """item_vec_fn(it) -> (num_l,d) raw direction vector to inject for this item (only
    the LAYER-th row is actually used)."""
    ref_items = [dict(it, image_path=os.path.join(img_dir, it["image_path"])) for it in items]
    ref_norm = steer.reference_hidden_norm(m, proc, ref_items, LAYER)
    alphas = [f * ref_norm for f in ALPHA_FRACTIONS]

    acc_by_alpha, valid_frac_by_alpha = [], []
    for alpha, frac in zip(alphas, ALPHA_FRACTIONS):
        n_correct, n_total = 0, 0
        for it in items:
            item_vec_np = unit(item_vec_fn(it)[LAYER])
            item_vec = torch.tensor(item_vec_np, dtype=torch.float16, device="cuda") * alpha
            img = Image.open(os.path.join(img_dir, it["image_path"])).convert("RGB")
            with steer.Steerer(steer_module, item_vec):
                ans = steer.ask_2choice(m, proc, img, it["minus_obj"], it["plus_obj"], pos_phrase)
            if ans == "?":
                continue
            n_total += 1
            n_correct += (ans == "B")  # ask_2choice's 2nd arg (B) is always plus_obj here
        acc = n_correct / max(n_total, 1)
        valid_frac = n_total / len(items)
        acc_by_alpha.append(acc)
        valid_frac_by_alpha.append(valid_frac)
        print(f"  alpha_frac={frac:+.1f}  P(answer=true '+')={acc:.3f}  valid={valid_frac:.2f}  "
              f"(n={n_total}/{len(items)})")
    return alphas, acc_by_alpha, valid_frac_by_alpha, ref_norm


def run(dataset, model, axes_filter=None):
    """axes_filter: optional list of axis names to restrict to (for splitting work
    across GPUs via --axis; see __main__'s --merge mode for combining the results back
    into one file)."""
    ap.set_model(model)
    m, proc = ap.load_model()
    steer_module = steer.find_decoder_layers(m)[LAYER - 1]
    rng = random.Random(SEED)
    out = {}

    if dataset == "sameaxis":
        items_by_axis = sameaxis_items()
        if axes_filter:
            items_by_axis = {a: v for a, v in items_by_axis.items() if a in axes_filter}
        shard, img_dir = sameaxis_shard_and_imgdir()
        header = "this scene's own r_AC (single-axis span) injected into a DIFFERENT pair's question"

        for axis_name, pool in items_by_axis.items():
            items = rng.sample(pool, min(N_ITEMS, len(pool)))
            print(f"\n{'='*90}\n[{model}][{dataset}][{axis_name}] CROSS-LEG ({header})\n"
                  f"N={len(items)} items, layer={LAYER}\n{'='*90}")
            vec_fn = lambda it, ax=axis_name: shard[f"{ax}|{it['scene_idx']}|r_AC"]
            alphas, acc, vf, ref_norm = steer_sweep(m, proc, steer_module, items, img_dir,
                                                      vec_fn, POS_PHRASE[axis_name])
            out[f"{axis_name}_alphas"] = np.array(alphas)
            out[f"{axis_name}_alpha_fractions"] = np.array(ALPHA_FRACTIONS)
            out[f"{axis_name}_acc"] = np.array(acc)
            out[f"{axis_name}_valid_frac"] = np.array(vf)
            out[f"{axis_name}_ref_norm"] = ref_norm
        return out

    # cross-axis {clevr, triplet3ax}: AC's per-axis question, steered with this SAME
    # scene's other-pair pure leg (r_AB for the axis1 reading, r_BC for axis2) -- see
    # crossaxis_items' docstring for why AC is the only pair this test can be run on.
    items_by_axis = crossaxis_items(dataset)
    if axes_filter:
        items_by_axis = {a: v for a, v in items_by_axis.items() if a in axes_filter}
    shard, img_dir = crossaxis_shard_and_imgdir(dataset)
    header = "AC's per-axis question, steered with this SAME scene's other-pair pure leg (r_AB/r_BC)"

    def vec_fn(it):
        return shard[f"{it['config']}|{it['idx']}|{it['src_field']}"]

    for axis_name, pool in items_by_axis.items():
        items = rng.sample(pool, min(N_ITEMS, len(pool)))
        print(f"\n{'='*90}\n[{model}][{dataset}][{axis_name}] CROSS-LEG ({header})\n"
              f"N={len(items)} items, layer={LAYER}\n{'='*90}")
        alphas, acc, vf, ref_norm = steer_sweep(m, proc, steer_module, items, img_dir,
                                                  vec_fn, POS_PHRASE[axis_name])
        out[f"{axis_name}_alphas"] = np.array(alphas)
        out[f"{axis_name}_alpha_fractions"] = np.array(ALPHA_FRACTIONS)
        out[f"{axis_name}_acc"] = np.array(acc)
        out[f"{axis_name}_valid_frac"] = np.array(vf)
        out[f"{axis_name}_ref_norm"] = ref_norm
    return out


def draw_diffpair_plot(out, dataset, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, SECONDARY, MUTED, GRID = "#ffffff", "#1a1a19", "#52514e", "#898781", "#e5e4df"

    axes_order = [a for a in (AXES3 if dataset == "sameaxis" else cxp.AXES_BY_DATASET[dataset])
                  if f"{a}_acc" in out]
    if dataset == "sameaxis":
        title = f"[{model}] {dataset}: this scene's own r_AC injected into a DIFFERENT pair's question"
    else:
        title = f"[{model}] {dataset}: AC's question, steered with this scene's own r_AB/r_BC (other pair)"

    fig, axes_plt = plt.subplots(1, len(axes_order), figsize=(5.3 * len(axes_order), 6), dpi=150,
                                  facecolor=SURFACE, sharey=True, squeeze=False)
    axes_plt = axes_plt[0]
    fig.suptitle(title, color=INK, fontsize=14, fontweight="bold", x=0.045, ha="left", y=0.99)

    for i, axis_name in enumerate(axes_order):
        ax = axes_plt[i]
        ax.set_facecolor(SURFACE)
        fr = out[f"{axis_name}_alpha_fractions"]
        acc = out[f"{axis_name}_acc"] * 100
        vf = out[f"{axis_name}_valid_frac"]
        ax.plot(fr, acc, color="#c0392b", linewidth=2.2, zorder=3, solid_capstyle="round")
        ax.scatter(fr, acc, s=15 + 90 * vf, color="#c0392b", zorder=4, alpha=0.85,
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

    fig.subplots_adjust(wspace=0.06, top=0.85, bottom=0.12, left=0.06, right=0.98)
    out_path = f"{ap.PLOTS}/steering/{dataset}_diffpair_steering_{model}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


def _partial_path(dataset, model, axis_name):
    return f"{ap.RESULTS}/steering/diffpair_steering_{dataset}_{model}_partial_{axis_name}.npz"


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset", required=True, choices=DATASETS)
    cli.add_argument("--model", choices=ap.MODELS, default=ap.DEFAULT_MODEL)
    cli.add_argument("--axis", action="append", choices=AXES3,
                      help="restrict to this axis (repeatable); saves a partial file instead of "
                           "the merged output -- run once per axis (e.g. across separate GPUs), "
                           "then once more with --merge to combine")
    cli.add_argument("--merge", action="store_true",
                      help="combine all --axis partial files on disk into the final merged "
                           "npz + plot, instead of computing anything")
    args = cli.parse_args()

    all_axes = AXES3 if args.dataset == "sameaxis" else cxp.AXES_BY_DATASET[args.dataset]

    if args.merge:
        out = {}
        for axis_name in all_axes:
            p = _partial_path(args.dataset, args.model, axis_name)
            if not os.path.exists(p):
                print(f"[merge] missing {p}, skipping {axis_name}")
                continue
            out.update(np.load(p, allow_pickle=True))
        out_path = f"{ap.RESULTS}/steering/diffpair_steering_{args.dataset}_{args.model}.npz"
        np.savez(out_path, **out)
        print(f"saved -> {out_path}")
        draw_diffpair_plot(out, args.dataset, args.model)
        raise SystemExit

    out = run(args.dataset, args.model, axes_filter=args.axis)

    if args.axis:
        # partial run (one or more --axis given): save one file per requested axis so
        # separate GPU processes never clobber each other's output
        for axis_name in args.axis:
            axis_out = {k: v for k, v in out.items() if k.startswith(f"{axis_name}_")}
            p = _partial_path(args.dataset, args.model, axis_name)
            np.savez(p, **axis_out)
            print(f"saved -> {p}")
    else:
        out_path = f"{ap.RESULTS}/steering/diffpair_steering_{args.dataset}_{args.model}.npz"
        np.savez(out_path, **out)
        print(f"\nsaved -> {out_path}")
        draw_diffpair_plot(out, args.dataset, args.model)
