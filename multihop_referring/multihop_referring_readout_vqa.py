"""hop-count sweep for the GENUINE multi-hop comparison:

- latent computation: identical method to chain_hop_readout_vqa.py -- sum consecutive leg
  vectors, project onto a cross-validated own-axis unit vector, sign-match against the real
  net displacement (Z's scalar - A's scalar, walking the SAME ranks the chain visited).
- MLLM generation: fresh forward passes -- the model is given the nested referring phrase
  ("the object right of the object left of ... A") and must say whether it points at Z (the
  correct end of the chain) or Y (one hop short). This is the actual multi-hop task: a
  model that stops one hop early gets a concrete, scoreable wrong answer, unlike
  chain_hop_readout_vqa.py's direct "is Z more right than A?" (answerable without ever
  using the chain, see the multi-hop-QA discussion this was redesigned from).

Two interchangeable backends (--source): synthetic (multihop_referring_pipeline.py,
rendered) / clevr (clevr_multihop_referring_pipeline.py, mined from real CLEVR photos).

Run first:
    python multihop_referring/multihop_referring_pipeline.py --model {model}
    # or: python multihop_referring/clevr_multihop_referring_pipeline.py --model {model}

Usage:
    python multihop_referring/multihop_referring_readout_vqa.py --model qwen3vl
    python multihop_referring/multihop_referring_readout_vqa.py --model qwen3vl --source clevr
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, one level above this file's subfolder
sys.path[:0] = [os.path.join(_ROOT, d) for d in ("core", "analysis", "steering", "chain_hop", "multihop_referring")]
import argparse
import json
import random
import re

import numpy as np
import torch
from PIL import Image

import cross_axis_analysis as caa
import axis_pipeline as ap

_CFG = ap.CFG["multihop_referring"]
DEFAULT_LAYER = _CFG["default_layer"]
SEED_VQA = _CFG["seed_vqa"]

# set by __main__ based on --source
chp = None
AXES = None
HOPS = None
IMAGES_DIR = None

unit = ap.unit


# =============================================================================
# READOUT: sign-match accuracy of the summed legs, own-axis projection, no new GPU work
# (identical method to chain_hop_readout_vqa.py, adapted to this manifest's walk_ranks/
# objects_by_rank schema instead of labels/objects)
# =============================================================================
def group_split_by_axis(manifest, axis, seed=caa.SEED):
    items = [(hops, idx) for hops in HOPS for idx in range(len(manifest[axis][str(hops)]))]
    groups = caa.group_split(len(items), seed)
    return {item: g for item, g in zip(items, groups)}


def build_axis_units(manifest, shard, axis, groups_by_item):
    pools = {1: [], 2: []}
    for hops in HOPS:
        scenes = manifest[axis][str(hops)]
        for idx, s in enumerate(scenes):
            g = groups_by_item[(hops, idx)]
            walk_ranks, objs = s["walk_ranks"], s["objects_by_rank"]
            for i in range(hops):
                a_rank, b_rank = walk_ranks[i], walk_ranks[i + 1]
                leg = shard[f"{axis}|{hops}|{idx}|leg{i}"]
                real_step = objs[b_rank]["scalar"] - objs[a_rank]["scalar"]
                pools[g].append(leg * (1.0 if real_step > 0 else -1.0))
    return {g: unit(np.mean(pools[g], axis=0)) for g in [1, 2]}


def readout_accuracy(manifest, shard, axis, groups_by_item, axis_units):
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
            hat = legs.sum(axis=0)
            proj = np.einsum("ld,ld->l", hat, axis_unit)
            walk_ranks, objs = s["walk_ranks"], s["objects_by_rank"]
            real_net = objs[walk_ranks[-1]]["scalar"] - objs[walk_ranks[0]]["scalar"]
            correct[idx] = (np.sign(proj) == np.sign(real_net)).astype(float)
        out[hops] = correct.mean(axis=0)
    return out, num_l


# =============================================================================
# VQA: fresh forward passes -- resolve the nested referring phrase, discriminate Z vs Y
# =============================================================================
@torch.inference_mode()
def ask_referring_2choice(model, proc, image, phrase, term_a, term_b):
    q = (f"In the image, consider {phrase}. Is that the {term_a} or the {term_b}? "
         f"Answer with {term_a} or {term_b}, only.")
    inp = ap.build_inputs(proc, image, q)
    n_tok = max(len(proc.tokenizer(term_a, add_special_tokens=False)["input_ids"]),
                len(proc.tokenizer(term_b, add_special_tokens=False)["input_ids"])) + 3
    g = model.generate(**inp, max_new_tokens=n_tok, do_sample=False)
    t = proc.tokenizer.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    a_hit = bool(re.search(rf"\b{re.escape(term_a.lower())}\b", t.lower()))
    b_hit = bool(re.search(rf"\b{re.escape(term_b.lower())}\b", t.lower()))
    if a_hit and not b_hit:
        return "A"
    if b_hit and not a_hit:
        return "B"
    return "?"


def vqa_accuracy(manifest, model, proc, axis, limit=None):
    """{hops: dict(...)} + records. Which slot (A/B) holds the true Z is randomized per
    item so the model can't win by a fixed positional bias."""
    rng = random.Random(SEED_VQA)
    out = {}
    records = []
    for hops in HOPS:
        scenes = manifest[axis][str(hops)]
        if limit:
            scenes = scenes[:limit]
        n, n_valid, n_correct = 0, 0, 0
        for idx, s in enumerate(scenes):
            z_desc, y_desc = s["z_desc"], s["y_desc"]
            flip = rng.random() < 0.5
            term_a, term_b = (y_desc, z_desc) if flip else (z_desc, y_desc)
            correct_letter = "B" if flip else "A"

            img = Image.open(os.path.join(IMAGES_DIR, s["image_path"])).convert("RGB")
            pred = ask_referring_2choice(model, proc, img, s["phrase"], term_a, term_b)

            n += 1
            n_valid += int(pred != "?")
            n_correct += int(pred == correct_letter)
            records.append(dict(axis=axis, hops=hops, idx=idx, phrase=s["phrase"],
                                 z_desc=z_desc, y_desc=y_desc, pred=pred, correct=correct_letter))
        out[hops] = dict(n=n, n_valid=n_valid, n_correct=n_correct,
                          acc_all=n_correct / n, acc_valid=n_correct / n_valid if n_valid else float("nan"))
        print(f"[{axis}|hop{hops}] VQA n={n} valid={n_valid} acc_all={out[hops]['acc_all']:.3f}")
    return out, records


# =============================================================================
# plot -- x=hops, y=accuracy, latent computation vs genuine multi-hop MLLM generation
# =============================================================================
def draw_hop_accuracy_plot(readout_by_axis, vqa_by_axis, model, layer, source="synthetic"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, SECONDARY, MUTED, GRID = "#ffffff", "#1a1a19", "#52514e", "#898781", "#e5e4df"
    LATENT_COLOR, VQA_COLOR = "#2a78d6", "#eb6834"

    hops_arr = np.array(HOPS)
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
             label="MLLM generation (genuine multi-hop referring)", zorder=3)

    ax.axhline(50, color=MUTED, linewidth=1.2, linestyle=(0, (2, 2)), alpha=0.8, zorder=1)
    ax.text(hops_arr[-1], 51.5, "chance (50%)", color=MUTED, fontsize=9.5, ha="right")

    ax.set_xlabel("reasoning hops", color=INK, fontsize=11.5)
    ax.set_ylabel("accuracy (%)", color=INK, fontsize=11.5)
    ax.set_xticks(hops_arr)
    ax.set_ylim(35, 105)
    ax.set_title(f"[{model}] latent computation vs genuine multi-hop MLLM generation ({source})\n"
                 f"(shaded band = min/max across {'/'.join(AXES)} axes)",
                 color=INK, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=SECONDARY, length=0)
    ax.legend(loc="lower left", frameon=False, fontsize=10, labelcolor=INK)

    fig.subplots_adjust(top=0.85, bottom=0.11, left=0.10, right=0.97)
    prefix = "" if source == "synthetic" else f"{source}_"
    out_path = f"{ap.PLOTS}/readout_vqa/{prefix}multihop_referring_accuracy_by_hops_{model}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--model", choices=ap.MODELS, default=ap.DEFAULT_MODEL)
    cli.add_argument("--source", choices=["synthetic", "clevr"], default="synthetic")
    cli.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    cli.add_argument("--vqa-limit", type=int, default=None)
    args = cli.parse_args()
    ap.set_model(args.model)

    if args.source == "clevr":
        import clevr_multihop_referring_pipeline as chp
    else:
        import multihop_referring_pipeline as chp
    AXES, HOPS, IMAGES_DIR = chp.AXES, chp.HOPS, chp.CHAIN_IMAGES_DIR

    manifest = chp.chain_hop_load_or_build_manifest()
    shard = np.load(chp.chain_shard_out_path(), allow_pickle=True)

    print(f"=== [{args.model}] READOUT (latent computation, sign-match, own axis) ===")
    readout_by_axis = {}
    for axis in AXES:
        groups_by_item = group_split_by_axis(manifest, axis)
        axis_units = build_axis_units(manifest, shard, axis, groups_by_item)
        readout_by_axis[axis], num_l = readout_accuracy(manifest, shard, axis, groups_by_item, axis_units)
        for hops in HOPS:
            acc = readout_by_axis[axis][hops]
            print(f"[{axis}|hop{hops}] layer{args.layer}={acc[args.layer]*100:5.1f}%  best={acc.max()*100:5.1f}%@L{acc.argmax()}")

    print(f"\n=== [{args.model}] VQA (genuine multi-hop referring) ===")
    model, proc = ap.load_model()
    vqa_by_axis, records = {}, []
    for axis in AXES:
        vqa_by_axis[axis], recs = vqa_accuracy(manifest, model, proc, axis, limit=args.vqa_limit)
        records.extend(recs)

    prefix = "" if args.source == "synthetic" else f"{args.source}_"
    tag = "" if args.vqa_limit is None else f"_debuglimit{args.vqa_limit}"
    readout_flat = {f"{axis}_{hops}": readout_by_axis[axis][hops] for axis in AXES for hops in HOPS}
    np.savez(f"{ap.RESULTS}/readout_vqa/{prefix}multihop_referring_readout_{args.model}{tag}.npz", **readout_flat)
    with open(f"{ap.RESULTS}/readout_vqa/{prefix}multihop_referring_vqa_{args.model}{tag}.json", "w") as f:
        json.dump(dict(vqa={axis: vqa_by_axis[axis] for axis in AXES}, records=records), f, indent=2)
    print(f"\nsaved -> {ap.RESULTS}/readout_vqa/{prefix}multihop_referring_readout_{args.model}{tag}.npz")
    print(f"saved -> {ap.RESULTS}/readout_vqa/{prefix}multihop_referring_vqa_{args.model}{tag}.json")

    if args.vqa_limit is None:
        draw_hop_accuracy_plot(readout_by_axis, vqa_by_axis, args.model, args.layer, source=args.source)
    else:
        print(f"\n[debug --vqa-limit {args.vqa_limit}] skipping plot (partial data)")
