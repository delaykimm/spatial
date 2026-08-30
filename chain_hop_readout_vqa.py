"""hop-count sweep: latent computation (hat_c_AZ = c_AB + c_BC + ... + c_Y,Z, summed
hidden-state relation vectors, sign read out via own-axis projection) vs MLLM generation
(direct VQA on the endpoints, free-form) -- same "is Z more {right/up/close} than A?"
question, answered two different ways, at hop = 2..6.

- READOUT (latent computation): no new GPU work, reuses chain_hop_pipeline.py's cached
  shard. own-axis unit vector is calibrated group-disjoint (half the chains, sign-
  corrected by real per-leg displacement, pooled across ALL hop counts since a leg is the
  same underlying quantity regardless of chain length) and applied to the OTHER half, so
  there's no train/eval leakage. Predicted sign = sign(hat_c_AZ . own_axis) at each layer.
- VQA (MLLM generation): fresh forward passes, free-form generation + parsing, via
  axis_pipeline.ask_2choice -- "which object is {more to the right / higher up / farther},
  the {first} or the {last}?" on the original image.

Two interchangeable backends for where the chains come from (--source):
  - synthetic (chain_hop_pipeline.py): rendered, random-walk, all 3 axes
  - clevr (clevr_chain_pipeline.py): mined from real CLEVR photos, random N objects/random
    order, only 2 axes (horizontal/closefar -- CLEVR has no vertical position variation)
Both expose the same interface (AXES, HOPS, CHAIN_IMAGES_DIR, chain_hop_load_or_build_manifest,
chain_shard_out_path, extract_chain), so every function below is backend-agnostic.

Run first:
    python chain_hop_pipeline.py --model {model}            # or: python clevr_chain_pipeline.py ...

Usage:
    python chain_hop_readout_vqa.py --model qwen3vl
    python chain_hop_readout_vqa.py --model qwen3vl --source clevr
    python chain_hop_readout_vqa.py --model qwen3vl --vqa-limit 5   (quick correctness check)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json

import numpy as np
from PIL import Image

import cross_axis_analysis as caa
import axis_pipeline as ap

_CFG = ap.CFG["chain_hop"]
DEFAULT_LAYER = _CFG["default_layer"]
AXIS_Q = ap.CFG["cross_axis_readout_vqa"]["axis_question_phrase"]  # same wording as cross_axis_readout_vqa.py

# set by __main__ based on --source; module-level so every function below (which reads
# these as globals) works unchanged for whichever backend was selected
chp = None
AXES = None
HOPS = None

unit = ap.unit
report_layers = ap.report_layers


# =============================================================================
# READOUT: sign-match accuracy of the summed legs, own-axis projection, no new GPU work
# =============================================================================
def group_split_by_axis(manifest, axis, seed=caa.SEED):
    """{(hops, idx): 1|2} disjoint half-split over every chain of this axis, pooled across
    all hop counts -- reuses cross_axis_analysis.group_split (same SEED convention)."""
    items = [(hops, idx) for hops in HOPS for idx in range(len(manifest[axis][str(hops)]))]
    groups = caa.group_split(len(items), seed)
    return {item: g for item, g in zip(items, groups)}


def build_axis_units(manifest, shard, axis, groups_by_item):
    """{1|2: (NUM_L,d) sign-corrected mean unit vector} -- pools every individual leg
    (across all hop counts) in that group, flipped by its own real-world step sign before
    pooling (same reasoning as triplet_pipeline.build_own_axes_signed: unsigned pooling
    lets opposite-signed legs cancel)."""
    pools = {1: [], 2: []}
    for hops in HOPS:
        scenes = manifest[axis][str(hops)]
        for idx, s in enumerate(scenes):
            g = groups_by_item[(hops, idx)]
            objs, labels = s["objects"], s["labels"]
            for i in range(hops):
                a, b = labels[i], labels[i + 1]
                leg = shard[f"{axis}|{hops}|{idx}|leg{i}"]
                real_step = objs[b]["scalar"] - objs[a]["scalar"]
                pools[g].append(leg * (1.0 if real_step > 0 else -1.0))
    return {g: unit(np.mean(pools[g], axis=0)) for g in [1, 2]}


def readout_accuracy(manifest, shard, axis, groups_by_item, axis_units):
    """{hops: (NUM_L,) sign-match accuracy of hat_c_AZ = sum(legs), projected onto the
    OTHER group's own-axis calibration (cross-validated, no leakage)."""
    num_l = shard[f"{axis}|{HOPS[0]}|0|leg0"].shape[0]
    out = {}
    for hops in HOPS:
        scenes = manifest[axis][str(hops)]
        N = len(scenes)
        correct = np.zeros((N, num_l))
        for idx, s in enumerate(scenes):
            g = groups_by_item[(hops, idx)]
            axis_unit = axis_units[2 if g == 1 else 1]
            legs = np.stack([shard[f"{axis}|{hops}|{idx}|leg{i}"] for i in range(hops)])
            hat = legs.sum(axis=0)                          # (NUM_L, d)
            proj = np.einsum("ld,ld->l", hat, axis_unit)    # (NUM_L,)
            objs, labels = s["objects"], s["labels"]
            real_net = objs[labels[-1]]["scalar"] - objs[labels[0]]["scalar"]
            correct[idx] = (np.sign(proj) == np.sign(real_net)).astype(float)
        out[hops] = correct.mean(axis=0)
    return out, num_l


# =============================================================================
# VQA: fresh forward passes, free-form generation + parsing (endpoints only, A vs Z)
# =============================================================================
def vqa_accuracy(manifest, model, proc, axis, limit=None):
    """{hops: dict(n, n_valid, n_correct, acc_all, acc_valid)} + flat list of per-item
    records, for one axis."""
    phrase = AXIS_Q[axis]
    out = {}
    records = []
    for hops in HOPS:
        scenes = manifest[axis][str(hops)]
        if limit:
            scenes = scenes[:limit]
        n, n_valid, n_correct = 0, 0, 0
        for idx, s in enumerate(scenes):
            objs, labels = s["objects"], s["labels"]
            first, last = labels[0], labels[-1]
            real_net = objs[last]["scalar"] - objs[first]["scalar"]
            correct_letter = "B" if real_net > 0 else "A"  # A=first(term_a), B=last(term_b)

            img = Image.open(os.path.join(chp.CHAIN_IMAGES_DIR, s["image_path"])).convert("RGB")
            pred = ap.ask_2choice(model, proc, img, objs[first]["desc"], objs[last]["desc"], phrase)

            n += 1
            n_valid += int(pred != "?")
            n_correct += int(pred == correct_letter)
            records.append(dict(axis=axis, hops=hops, idx=idx, pred=pred, correct=correct_letter))
        out[hops] = dict(n=n, n_valid=n_valid, n_correct=n_correct,
                          acc_all=n_correct / n, acc_valid=n_correct / n_valid if n_valid else float("nan"))
        print(f"[{axis}|hop{hops}] VQA n={n} valid={n_valid} acc_all={out[hops]['acc_all']:.3f}")
    return out, records


# =============================================================================
# plot -- THE plot: x=hops, y=accuracy, latent computation vs MLLM generation
# =============================================================================
def draw_hop_accuracy_plot(readout_by_axis, vqa_by_axis, model, layer, num_l, source="synthetic"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, SECONDARY, MUTED, GRID = "#ffffff", "#1a1a19", "#52514e", "#898781", "#e5e4df"
    LATENT_COLOR, VQA_COLOR = "#2a78d6", "#eb6834"

    hops_arr = np.array(HOPS)
    # mean over axes, band = min/max over axes -- one number per hop count per method
    latent_by_axis = np.array([[readout_by_axis[ax][h][layer] * 100 for h in HOPS] for ax in AXES])
    vqa_arr_by_axis = np.array([[vqa_by_axis[ax][h]["acc_all"] * 100 for h in HOPS] for ax in AXES])
    latent_mean, latent_lo, latent_hi = latent_by_axis.mean(0), latent_by_axis.min(0), latent_by_axis.max(0)
    vqa_mean, vqa_lo, vqa_hi = vqa_arr_by_axis.mean(0), vqa_arr_by_axis.min(0), vqa_arr_by_axis.max(0)

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.fill_between(hops_arr, latent_lo, latent_hi, color=LATENT_COLOR, alpha=0.15, zorder=1)
    ax.plot(hops_arr, latent_mean, color=LATENT_COLOR, linewidth=2.4, marker="o",
             label=f"latent computation (readout, layer {layer})", zorder=3)
    ax.fill_between(hops_arr, vqa_lo, vqa_hi, color=VQA_COLOR, alpha=0.15, zorder=1)
    ax.plot(hops_arr, vqa_mean, color=VQA_COLOR, linewidth=2.4, marker="o",
             label="MLLM generation (VQA)", zorder=3)

    ax.axhline(50, color=MUTED, linewidth=1.2, linestyle=(0, (2, 2)), alpha=0.8, zorder=1)
    ax.text(hops_arr[-1], 51.5, "chance (50%)", color=MUTED, fontsize=9.5, ha="right")

    ax.set_xlabel("reasoning hops", color=INK, fontsize=11.5)
    ax.set_ylabel("accuracy (%)", color=INK, fontsize=11.5)
    ax.set_xticks(hops_arr)
    ax.set_ylim(35, 105)
    ax.set_title(f"[{model}] latent computation vs MLLM generation, by reasoning hops ({source})\n"
                 f"(shaded band = min/max across {'/'.join(AXES)} axes)",
                 color=INK, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=SECONDARY, length=0)
    ax.legend(loc="lower left", frameon=False, fontsize=10, labelcolor=INK)

    fig.subplots_adjust(top=0.85, bottom=0.11, left=0.10, right=0.97)
    prefix = "" if source == "synthetic" else f"{source}_"
    out_path = f"{ap.PLOTS}/readout_vqa/{prefix}chain_hop_accuracy_by_hops_{model}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


def draw_hop_accuracy_by_axis_plot(readout_by_axis, vqa_by_axis, model, layer, source="synthetic"):
    """Diagnostic breakdown: one panel per axis, same 2 lines -- so an aggregate trend in
    the main plot can be checked against each axis individually."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, SECONDARY, MUTED, GRID = "#ffffff", "#1a1a19", "#52514e", "#898781", "#e5e4df"
    LATENT_COLOR, VQA_COLOR = "#2a78d6", "#eb6834"
    hops_arr = np.array(HOPS)

    fig, axes_plt = plt.subplots(1, len(AXES), figsize=(6.0 * len(AXES), 5.6), dpi=150,
                                  facecolor=SURFACE, sharey=True)
    for col, axis_name in enumerate(AXES):
        a = axes_plt[col]
        a.set_facecolor(SURFACE)
        latent = [readout_by_axis[axis_name][h][layer] * 100 for h in HOPS]
        vqa = [vqa_by_axis[axis_name][h]["acc_all"] * 100 for h in HOPS]
        a.plot(hops_arr, latent, color=LATENT_COLOR, linewidth=2.2, marker="o", label="latent computation")
        a.plot(hops_arr, vqa, color=VQA_COLOR, linewidth=2.2, marker="o", label="MLLM generation")
        a.axhline(50, color=MUTED, linewidth=1.1, linestyle=(0, (2, 2)), alpha=0.8, zorder=1)
        a.set_title(axis_name, color=INK, fontsize=12, fontweight="bold")
        a.set_xlabel("reasoning hops", color=INK, fontsize=10.5)
        a.set_xticks(hops_arr)
        a.set_ylim(35, 105)
        a.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right"]:
            a.spines[spine].set_visible(False)
        a.tick_params(colors=SECONDARY, length=0)
        if col == 0:
            a.set_ylabel("accuracy (%)", color=INK, fontsize=11)
        if col == len(AXES) - 1:
            a.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK)

    fig.suptitle(f"[{model}] latent computation vs MLLM generation, by axis ({source}, readout layer {layer})",
                 color=INK, fontsize=14, fontweight="bold", y=1.02)
    fig.subplots_adjust(top=0.86, bottom=0.13, left=0.06, right=0.98, wspace=0.08)
    prefix = "" if source == "synthetic" else f"{source}_"
    out_path = f"{ap.PLOTS}/readout_vqa/{prefix}chain_hop_accuracy_by_hops_by_axis_{model}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--model", choices=ap.MODELS, default=ap.DEFAULT_MODEL)
    cli.add_argument("--source", choices=["synthetic", "clevr"], default="synthetic",
                      help="synthetic = chain_hop_pipeline.py (rendered, 3 axes); "
                           "clevr = clevr_chain_pipeline.py (mined from real CLEVR photos, 2 axes)")
    cli.add_argument("--layer", type=int, default=DEFAULT_LAYER, help="fixed layer used for the latent-computation line")
    cli.add_argument("--vqa-limit", type=int, default=None, help="only run VQA on the first N chains per hop count (debug)")
    args = cli.parse_args()
    ap.set_model(args.model)

    if args.source == "clevr":
        import clevr_chain_pipeline as chp
    else:
        import chain_hop_pipeline as chp
    AXES, HOPS = chp.AXES, chp.HOPS

    manifest = chp.chain_hop_load_or_build_manifest()
    shard = np.load(chp.chain_shard_out_path(), allow_pickle=True)

    print(f"=== [{args.model}] READOUT (latent computation, sign-match, own axis) ===")
    readout_by_axis = {}
    num_l = None
    for axis in AXES:
        groups_by_item = group_split_by_axis(manifest, axis)
        axis_units = build_axis_units(manifest, shard, axis, groups_by_item)
        readout_by_axis[axis], num_l = readout_accuracy(manifest, shard, axis, groups_by_item, axis_units)
        for hops in HOPS:
            acc = readout_by_axis[axis][hops]
            print(f"[{axis}|hop{hops}] layer{args.layer}={acc[args.layer]*100:5.1f}%  best={acc.max()*100:5.1f}%@L{acc.argmax()}")

    print(f"\n=== [{args.model}] VQA (MLLM generation, free-form) ===")
    model, proc = ap.load_model()
    vqa_by_axis, records = {}, []
    for axis in AXES:
        vqa_by_axis[axis], recs = vqa_accuracy(manifest, model, proc, axis, limit=args.vqa_limit)
        records.extend(recs)

    prefix = "" if args.source == "synthetic" else f"{args.source}_"
    tag = "" if args.vqa_limit is None else f"_debuglimit{args.vqa_limit}"
    readout_flat = {f"{axis}_{hops}": readout_by_axis[axis][hops] for axis in AXES for hops in HOPS}
    np.savez(f"{ap.RESULTS}/readout_vqa/{prefix}chain_hop_readout_{args.model}{tag}.npz", **readout_flat)
    with open(f"{ap.RESULTS}/readout_vqa/{prefix}chain_hop_vqa_{args.model}{tag}.json", "w") as f:
        json.dump(dict(vqa={axis: vqa_by_axis[axis] for axis in AXES}, records=records), f, indent=2)
    print(f"\nsaved -> {ap.RESULTS}/readout_vqa/{prefix}chain_hop_readout_{args.model}{tag}.npz")
    print(f"saved -> {ap.RESULTS}/readout_vqa/{prefix}chain_hop_vqa_{args.model}{tag}.json")

    if args.vqa_limit is None:
        draw_hop_accuracy_plot(readout_by_axis, vqa_by_axis, args.model, args.layer, num_l, source=args.source)
        draw_hop_accuracy_by_axis_plot(readout_by_axis, vqa_by_axis, args.model, args.layer, source=args.source)
    else:
        print(f"\n[debug --vqa-limit {args.vqa_limit}] skipping plots (partial data)")
