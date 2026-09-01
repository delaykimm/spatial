"""4-way (A/B/C/D) readout vs VQA for sameaxis -- extends the 2-way pairwise sign-match
(cross_axis_readout_vqa.py) to "which of all 4 objects is the extremum (closest/
farthest/leftmost/...)?". D is forced outside [A,C] by construction, so it's a real
candidate extremum, not a free win for A/C.

- READOUT: put all 4 objects on one shared per-layer scalar line, anchored at A (pos_A=0),
  via pairwise vectors already in the shard (r_AB/r_AC/r_AD) projected onto a
  group-disjoint own-axis calibration (no sign correction needed -- sameaxis triplets are
  sorted A<B<C by construction). Predicted extremum = argmax/argmin per layer, vs the true
  extremum from real coordinates.
- VQA (fresh generation): 4-option lettered MCQ with role<->letter shuffled per item, so
  the model can't learn a positional shortcut.

Usage:
    python sameaxis_4way_readout_vqa.py --model qwen3vl
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import random
import re

import numpy as np
import torch
from PIL import Image

import triplet_pipeline as cxp
import axis_pipeline as ap

SEED = ap.CFG["sameaxis_4way"]["seed"]
REAL_IDX = {"horizontal": 0, "vertical": 2, "closefar": 1}  # index into saved (x,y,z) center
QUESTION_TYPES = {
    "horizontal": [("leftmost", "min"), ("rightmost", "max")],
    "vertical": [("lowest", "min"), ("highest", "max")],
    "closefar": [("closest", "min"), ("farthest", "max")],
}
PHRASE = {
    "leftmost": "farthest to the left", "rightmost": "farthest to the right",
    "closest": "closest to the camera", "farthest": "farthest from the camera",
    "highest": "positioned highest", "lowest": "positioned lowest",
}
ROLES = ["A", "B", "C", "D"]

unit = ap.unit  # canonical L2-normalize now lives in axis_pipeline.py


# =============================================================================
# own-axis calibration -- group-disjoint, NO sign correction needed (sameaxis triplets
# are sorted A<B<C by construction, so r_AB/r_BC/r_AC are all already '+' directions)
# =============================================================================
def sameaxis_own_axis_vectors(shard, axes):
    out = {}
    for ax_i, axis_name in enumerate(axes):
        keys = [str(k) for k in shard["keys"] if str(k).startswith(f"{axis_name}|")]
        n = len(keys)
        rng = random.Random(SEED + ax_i)  # deterministic per-axis seed -- NOT hash(axis_name),
                                           # which is randomized per-process (PYTHONHASHSEED)
                                           # and would make this group split irreproducible
        idxs = list(range(n))
        rng.shuffle(idxs)
        half = n // 2
        group1 = set(idxs[:half])

        diffs1, diffs2 = [], []
        for i, k in enumerate(keys):
            for pair_key in ["r_AB", "r_BC", "r_AC"]:
                (diffs1 if i in group1 else diffs2).append(shard[f"{k}|{pair_key}"])
        # both groups already point the SAME '+' direction (sameaxis triplets are sorted
        # A<B<C by construction, so r_AB/r_BC/r_AC are all already '+') -- add, don't subtract.
        out[axis_name] = unit(np.mean(diffs1, axis=0) + np.mean(diffs2, axis=0))
    return out


# =============================================================================
# READOUT
# =============================================================================
def readout_test(model):
    ap.set_model(model)
    shard = np.load(cxp.shard_out_path("sameaxis"), allow_pickle=True)
    ad_shard = np.load(cxp.sameaxis_ad_shard_out_path(), allow_pickle=True)
    manifest = cxp.sameaxis_load_or_build_manifest()
    axes = list(QUESTION_TYPES.keys())
    axis_units = sameaxis_own_axis_vectors(shard, axes)
    num_l = shard[f"{axes[0]}|0|r_AB"].shape[0]

    out = {}
    for axis_name in axes:
        axis_unit = axis_units[axis_name]
        scenes = manifest[axis_name]
        N = len(scenes)
        pos = np.zeros((N, num_l, 4))
        real = np.zeros((N, 4))
        ridx = REAL_IDX[axis_name]
        for idx, s in enumerate(scenes):
            r_AB = shard[f"{axis_name}|{idx}|r_AB"]
            r_AC = shard[f"{axis_name}|{idx}|r_AC"]
            r_AD = ad_shard[f"{axis_name}|{idx}|r_AD"]
            pos[idx, :, 1] = np.einsum("ld,ld->l", r_AB, axis_unit)
            pos[idx, :, 2] = np.einsum("ld,ld->l", r_AC, axis_unit)
            pos[idx, :, 3] = np.einsum("ld,ld->l", r_AD, axis_unit)
            for j, role in enumerate(ROLES):
                real[idx, j] = s["objects"][role]["center"][ridx]

        true_min = np.argmin(real, axis=1)
        true_max = np.argmax(real, axis=1)
        pred_min = np.argmin(pos, axis=2)
        pred_max = np.argmax(pos, axis=2)

        for qtype, kind in QUESTION_TYPES[axis_name]:
            true_role = true_min if kind == "min" else true_max
            pred_role = pred_min if kind == "min" else pred_max
            acc = (pred_role == true_role[:, None]).mean(axis=0)
            out[f"{axis_name}_{qtype}"] = acc
            print(f"[{axis_name}|{qtype}] layer1={acc[1]*100:5.1f}%  "
                  f"mid={acc[num_l//2]*100:5.1f}%  last={acc[-1]*100:5.1f}%  "
                  f"best={acc.max()*100:5.1f}%@L{acc.argmax()}")
    return out, num_l


# =============================================================================
# VQA
# =============================================================================
@torch.inference_mode()
def ask_4way(model, proc, image, letter_to_role, scene, qtype):
    opt_text = " ".join(f"{letter}) {scene['objects'][role]['desc']}" for letter, role in letter_to_role.items())
    q = f"In the image, which object is {PHRASE[qtype]}? {opt_text} Answer with the letter only (A, B, C, or D)."
    inp = ap.build_inputs(proc, image, q)
    g = model.generate(**inp, max_new_tokens=4, do_sample=False)
    t = proc.tokenizer.decode(g[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    hits = re.findall(r"\b([ABCD])\b", t.upper())
    if len(set(hits)) == 1:
        return hits[0]
    return "?"


def vqa_test(m, proc):
    manifest = cxp.sameaxis_load_or_build_manifest()
    rng = random.Random(SEED)

    out = {}
    records = []
    for axis_name in QUESTION_TYPES:
        scenes = manifest[axis_name]
        ridx = REAL_IDX[axis_name]
        for qtype, kind in QUESTION_TYPES[axis_name]:
            n, n_valid, n_correct = 0, 0, 0
            for s in scenes:
                real_vals = [s["objects"][role]["center"][ridx] for role in ROLES]
                true_role = ROLES[int(np.argmin(real_vals) if kind == "min" else np.argmax(real_vals))]

                shuffled_roles = ROLES[:]
                rng.shuffle(shuffled_roles)
                letter_to_role = dict(zip("ABCD", shuffled_roles))

                img = Image.open(os.path.join(cxp.SAMEAXIS_IMAGES_DIR, s["image_path"])).convert("RGB")
                pred_letter = ask_4way(m, proc, img, letter_to_role, s, qtype)
                pred_role = letter_to_role.get(pred_letter)

                n += 1
                n_valid += int(pred_letter != "?")
                n_correct += int(pred_role == true_role)
                records.append(dict(axis=axis_name, qtype=qtype, true_role=true_role,
                                     letter_to_role=letter_to_role, pred_letter=pred_letter, pred_role=pred_role))
            out[f"{axis_name}_{qtype}"] = dict(n=n, n_valid=n_valid, n_correct=n_correct,
                                                acc_all=n_correct / n,
                                                acc_valid=n_correct / n_valid if n_valid else float("nan"))
            print(f"[{axis_name}|{qtype}] n={n} valid={n_valid} acc_all={out[f'{axis_name}_{qtype}']['acc_all']:.3f} "
                  f"acc_valid={out[f'{axis_name}_{qtype}']['acc_valid']:.3f}")
    return out, records


# =============================================================================
# plot -- mirrors ours/binding's triplet3ax_plot_4way_readout_vqa.py
# =============================================================================
def draw_4way_plot(readout, vqa, model, num_l):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, SECONDARY, MUTED, GRID = "#ffffff", "#1a1a19", "#52514e", "#898781", "#e5e4df"
    QTYPES = [
        ("horizontal_leftmost", "leftmost", "#2a78d6"), ("horizontal_rightmost", "rightmost", "#7ab4f0"),
        ("vertical_lowest", "lowest", "#eb6834"), ("vertical_highest", "highest", "#f5ab84"),
        ("closefar_closest", "closest", "#1baf7a"), ("closefar_farthest", "farthest", "#7fdcb8"),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    layer = np.arange(num_l)
    for key, label, color in QTYPES:
        acc = readout[key] * 100
        ax.plot(layer, acc, color=color, linewidth=2.1, label=f"{label} (readout)",
                 solid_capstyle="round", zorder=3)
        vqa_acc = vqa[key]["acc_all"] * 100
        ax.axhline(vqa_acc, color=color, linewidth=1.1, linestyle=(0, (4, 2)), alpha=0.55, zorder=2)

    ax.axhline(25, color=MUTED, linewidth=1.2, linestyle=(0, (2, 2)), alpha=0.8, zorder=1)
    ax.text(num_l - 1, 26.5, "chance (25%, 4-way)", color=MUTED, fontsize=9.5, ha="right")

    fig.suptitle(f"[{model}] sameaxis: 4-way (A/B/C/D) extremum readout accuracy by layer, vs VQA accuracy",
                 color=INK, fontsize=14.5, fontweight="bold", x=0.06, ha="left", y=0.975)

    ax.set_xlim(0, num_l - 1)
    ax.set_ylim(0, 105)
    ax.set_xlabel("layer index", color=INK, fontsize=11)
    ax.set_ylabel("accuracy (%)", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=SECONDARY, length=0, labelsize=10)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=SECONDARY, ncols=2,
              handlelength=1.8, handletextpad=0.6)

    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.07, right=0.97)
    out_path = f"{ap.PLOTS}/readout_vqa/sameaxis_4way_readout_vqa_{model}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--model", choices=ap.MODELS, default=ap.DEFAULT_MODEL)
    args = cli.parse_args()
    ap.set_model(args.model)

    print(f"=== [{args.model}] 4-WAY READOUT accuracy (own axis projection, anchored at A) ===")
    readout, num_l = readout_test(args.model)

    print(f"\n=== [{args.model}] 4-WAY VQA accuracy (lettered MCQ, shuffled letter<->object mapping) ===")
    m, proc = ap.load_model()
    vqa, records = vqa_test(m, proc)

    np.savez(f"{ap.RESULTS}/readout_vqa/sameaxis_4way_readout_{args.model}.npz", **readout)
    with open(f"{ap.RESULTS}/readout_vqa/sameaxis_4way_vqa_{args.model}.json", "w") as f:
        json.dump(dict(vqa=vqa, records=records), f, indent=2)
    print(f"\nsaved -> {ap.RESULTS}/readout_vqa/sameaxis_4way_readout_{args.model}.npz")
    print(f"saved -> {ap.RESULTS}/readout_vqa/sameaxis_4way_vqa_{args.model}.json")

    draw_4way_plot(readout, vqa, args.model, num_l)
