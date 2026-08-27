"""Sign-match READOUT vs VQA accuracy, per axis and per leg pair (AB/BC/AC).

- READOUT (no new GPU work, reuses cross_axis_alignment.py's cached
  {dataset}_alignment_{model}.npz -- run that first): sign(model's own-axis component) ==
  sign(real displacement along that axis)? Per layer.
- VQA (fresh forward passes, free-form generation + parsing): "which object is {more to
  the right / higher up / farther}, the X or the Y?" on the original image. Legs have no
  guaranteed A<B<C ordering, so the correct answer is looked up fresh per triplet from
  real-world ground truth.

Usage:
    python cross_axis_readout_vqa.py --dataset clevr
    python cross_axis_readout_vqa.py --dataset triplet3ax --model llava
    python cross_axis_readout_vqa.py --dataset clevr --vqa-limit 10   (quick correctness check)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
from collections import defaultdict

import numpy as np
from PIL import Image

import triplet_pipeline as cxp
import cross_axis_analysis as caa
import axis_pipeline as ap

PAIRS = cxp.PAIRS
AXES_BY_DATASET = cxp.AXES_BY_DATASET
DEFAULT_LAYER = ap.CFG["cross_axis_readout_vqa"]["default_layer"]
AXIS_Q = ap.CFG["cross_axis_readout_vqa"]["axis_question_phrase"]

report_layers = ap.report_layers  # canonical version now lives in axis_pipeline.py


# =============================================================================
# READOUT: sign-match accuracy, reusing cross_axis_alignment.py's cached npz
# =============================================================================
def relevant_mask(triplets, pair, axis_name):
    """Which triplets count toward axis_name's sign-match for this pair. AB is only tested
    on its own pure axis (ax1), BC on ax2 -- testing the other axis would just be near-zero
    jitter noise, diluting accuracy toward chance. AC is the genuine diagonal leg, so it's
    tested against both axes."""
    if pair == "AB":
        return np.array([t["ax1"] == axis_name for t in triplets])
    if pair == "BC":
        return np.array([t["ax2"] == axis_name for t in triplets])
    return np.array([axis_name in (t["ax1"], t["ax2"]) for t in triplets])  # AC


def readout_accuracy(dataset, model):
    path = f"{ap.RESULTS}/alignment/{dataset}_alignment_{model}.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run cross_axis_alignment.py --dataset {dataset} "
                                 f"--model {model} first (readout reuses its cached model/real components)")
    d = np.load(path, allow_pickle=True)
    axes = AXES_BY_DATASET[dataset]
    triplets = caa.load_all_triplets(dataset)  # same shard-key order as the alignment npz's rows
    out = {}
    for pair in PAIRS:
        model_allaxes = d[f"{pair}_model_allaxes"]  # (N,num_l,k)
        real_allaxes = d[f"{pair}_real_allaxes"]     # (N,k)
        for ai, axis_name in enumerate(axes):
            mask = relevant_mask(triplets, pair, axis_name)
            if not mask.any():
                continue
            model_ax = model_allaxes[mask][:, :, ai]  # (n_relevant,num_l)
            real_ax = real_allaxes[mask][:, ai]         # (n_relevant,)
            sign_match = (np.sign(model_ax) == np.sign(real_ax)[:, None]).astype(float)
            out[f"{axis_name}_{pair}"] = sign_match.mean(axis=0)  # (num_l,)
    return out


def print_readout_summary(readout, axes, dataset, model, num_l):
    print(f"\n=== [{model}][{dataset}] READOUT accuracy (sign-match, own axis, no new GPU work) ===")
    layers = report_layers(num_l)
    for axis_name in axes:
        for pair in PAIRS:
            r = readout.get(f"{axis_name}_{pair}")
            if r is None:
                continue  # e.g. no triplet's leg is pure along this axis for this pair (can happen with --vqa-limit)
            vals = "  ".join(f"L{L}={r[L]*100:5.1f}%" for L in layers)
            print(f"[{axis_name}|{pair}] {vals}  best={r.max()*100:5.1f}%@L{r.argmax()}")


# =============================================================================
# VQA: fresh forward passes, free-form generation + parsing
# =============================================================================
def ask_pair(model, proc, image, X, Y, axis_name):
    """Thin wrapper around axis_pipeline.ask_2choice (shared generate+parse logic) that
    relabels its generic 'A'/'B' answer back to this file's 'X'/'Y' convention."""
    ans = ap.ask_2choice(model, proc, image, X, Y, AXIS_Q[axis_name])
    return {"A": "X", "B": "Y", "?": "?"}[ans]


def pair_axes_to_ask(config, axis_pairs):
    """{pair: [axis_name, ...]} -- which axis questions are worth asking for this config's
    AB/BC/AC legs (same purity restriction as relevant_mask, same reason: a near-zero-
    jitter leg has no well-defined correct answer)."""
    ax1, ax2 = axis_pairs[config]
    return {"AB": [ax1], "BC": [ax2], "AC": [ax1, ax2]}


def _vqa_jobs(triplets_by_config, limit=None):
    jobs = [(config, idx, t) for config, triplets in triplets_by_config.items()
            for idx, t in enumerate(triplets)]
    return jobs[:limit] if limit else jobs


def _cached_image(img_cache, path, max_size=4):
    """LRU-ish cache of at most `max_size` open images -- triplets sharing a scene ask
    several questions in a row, so this avoids re-opening/re-decoding the same file."""
    if path not in img_cache:
        img_cache[path] = Image.open(path).convert("RGB")
        if len(img_cache) > max_size:
            img_cache.pop(next(iter(img_cache)))
    return img_cache[path]


def _ask_triplet_all_axes(m, proc, img, t, config, idx, real_pos, axis_pairs):
    """Every axis-relevant question for one triplet (pair_axes_to_ask's purity-filtered
    set), as a list of (key, correct, valid, record) -- key is f"{axis_name}_{pair}",
    matching readout_accuracy's per-(axis,pair) grouping so the two can be compared 1:1."""
    pos = real_pos[(config, idx)]
    axes_by_pair = pair_axes_to_ask(config, axis_pairs)
    out = []
    for pair in PAIRS:
        e1, e2 = pair[0], pair[1]
        X, Y = t[e1], t[e2]
        for axis_name in axes_by_pair[pair]:
            real_disp = pos[e2][axis_name] - pos[e1][axis_name]
            real = "Y" if real_disp > 0 else "X"
            pred = ask_pair(m, proc, img, X, Y, axis_name)
            record = dict(config=config, idx=idx, pair=pair, axis=axis_name, X=X, Y=Y, pred=pred, real=real)
            out.append((f"{axis_name}_{pair}", pred == real, pred != "?", record))
    return out


def vqa_accuracy(dataset, model, limit=None):
    ap.set_model(model)
    cfg = cxp.DATASET_CFG[dataset]
    triplets_by_config = cfg["load_fn"]()
    axis_pairs = caa.CLEVR_AXIS_PAIRS if dataset == "clevr" else cxp.AXIS_PAIRS
    real_pos = cxp.real_positions(dataset)

    jobs = _vqa_jobs(triplets_by_config, limit)
    n_questions = sum(len(axs) for axs in pair_axes_to_ask(list(triplets_by_config.keys())[0], axis_pairs).values())
    print(f"[{model}][{dataset}] VQA: {len(jobs)} triplets, ~{n_questions} axis-relevant questions each "
          f"= ~{len(jobs) * n_questions} questions")

    m, proc = ap.load_model()
    tallies = defaultdict(lambda: dict(n=0, n_valid=0, n_correct=0))
    records = []
    img_cache = {}
    for i, (config, idx, t) in enumerate(jobs):
        img = _cached_image(img_cache, os.path.join(cfg["images_dir"], t["image_path"]))
        for key, correct, valid, record in _ask_triplet_all_axes(m, proc, img, t, config, idx, real_pos, axis_pairs):
            tallies[key]["n"] += 1
            tallies[key]["n_valid"] += int(valid)
            tallies[key]["n_correct"] += int(correct)
            records.append(record)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(jobs)} triplets done")

    out = {}
    for key, v in tallies.items():
        out[key] = dict(n=v["n"], n_valid=v["n_valid"], n_correct=v["n_correct"],
                         acc_all=v["n_correct"] / v["n"],
                         acc_valid=v["n_correct"] / v["n_valid"] if v["n_valid"] else float("nan"))
    return out, records


def print_vqa_summary(vqa, axes, dataset, model):
    print(f"\n=== [{model}][{dataset}] VQA accuracy (direct question, free-form generation) ===")
    for axis_name in axes:
        for pair in PAIRS:
            v = vqa.get(f"{axis_name}_{pair}")
            if v is None:
                continue  # e.g. no triplet's leg is pure along this axis for this pair (can happen with --vqa-limit)
            print(f"[{axis_name}|{pair}] n={v['n']} valid={v['n_valid']} "
                  f"acc_all={v['acc_all']:.3f} acc_valid={v['acc_valid']:.3f}")


# =============================================================================
# plots -- mirror ours/binding's triplet3ax_plot_signmatch_readout_vqa[.py/_bar.py]
# =============================================================================
PAIR_COLORS = [("AB", "#2a78d6"), ("BC", "#eb6834"), ("AC", "#1baf7a")]


def draw_readout_vqa_plot(dataset):
    """Mirrors ours/binding's plot_signmatch_by_layer_both_axes.py: one panel per axis
    (2 for clevr, 3 for triplet3ax), 3 lines = 3 models, AC-leg sign-match accuracy vs
    relative depth. Reads only cached {dataset}_signmatch_readout_{model}.npz files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, SECONDARY, MUTED, GRID = "#ffffff", "#1a1a19", "#52514e", "#898781", "#e5e4df"
    COLORS = {"qwen3vl": "#2a78d6", "qwen2": "#eb6834", "llava": "#1baf7a"}
    MODEL_DISPLAY = ap.CFG["cross_axis_alignment"]["model_display"]

    axes_list = AXES_BY_DATASET[dataset]
    per_model = {}
    for model in ap.MODELS:
        path = f"{ap.RESULTS}/readout_vqa/{dataset}_signmatch_readout_{model}.npz"
        if os.path.exists(path):
            per_model[model] = np.load(path, allow_pickle=True)
    if not per_model:
        print(f"[{dataset}] no cached signmatch readout npz found for any model, skipping plot")
        return

    fig, axes_plt = plt.subplots(1, len(axes_list), figsize=(6.5 * len(axes_list), 6.0), dpi=150,
                                  facecolor=SURFACE, sharey=True, squeeze=False)
    axes_plt = axes_plt[0]
    fig.suptitle(f"{dataset}: direction (sign) grounding accuracy by layer -- by axis",
                 color=INK, fontsize=15.5, fontweight="bold", x=0.045, ha="left", y=1.01)

    for col, axis_name in enumerate(axes_list):
        ax = axes_plt[col]
        ax.set_facecolor(SURFACE)
        for model, d in per_model.items():
            key = f"{axis_name}_AC"
            if key not in d:
                continue
            match = d[key]
            num_l = len(match)
            depth = np.arange(num_l) / (num_l - 1)
            ax.plot(depth, match * 100, color=COLORS.get(model, "#000000"), linewidth=2.3,
                     label=MODEL_DISPLAY.get(model, model), solid_capstyle="round", zorder=3)
        ax.axhline(50, color=MUTED, linewidth=1.2, linestyle=(0, (2, 2)), alpha=0.8, zorder=1)
        ax.text(1.0, 51.5, "chance (50%)", color=MUTED, fontsize=9.5, ha="right")
        ax.set_title(axis_name, color=INK, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(40, 102)
        ax.set_xlabel("relative depth (layer / max layer)", color=INK, fontsize=10.5)
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x*100:.0f}%")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=SECONDARY, length=0, labelsize=10)
        if col == 0:
            ax.set_ylabel("sign match accuracy (%)", color=INK, fontsize=11)
        else:
            ax.tick_params(labelleft=False)

    handles = [plt.Line2D([0], [0], color=COLORS.get(m, "#000000"), linewidth=2.3) for m in per_model]
    fig.legend(handles, [MODEL_DISPLAY.get(m, m) for m in per_model], loc="lower center",
               ncols=len(per_model), frameon=False, fontsize=10.5, labelcolor=SECONDARY,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.8, handletextpad=0.6, columnspacing=1.8)

    fig.subplots_adjust(wspace=0.05, top=0.85, bottom=0.20, left=0.06, right=0.98)
    out_path = f"{ap.PLOTS}/readout_vqa/{dataset}_signmatch_readout_vqa.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


def draw_readout_vqa_bar(readout, vqa, axes, dataset, model, layer=DEFAULT_LAYER):
    """AB/BC/AC averaged together per axis -- readout at `layer` vs VQA baseline, as a
    grouped bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, SECONDARY, MUTED, GRID = "#ffffff", "#1a1a19", "#52514e", "#898781", "#e5e4df"
    READOUT_COLOR, VQA_COLOR = "#2a78d6", "#eb6834"

    readout_avg, vqa_avg = [], []
    for axis_name in axes:
        r_vals = [readout[f"{axis_name}_{p}"][layer] * 100 for p in PAIRS]
        v_vals = [vqa[f"{axis_name}_{p}"]["acc_all"] * 100 for p in PAIRS]
        readout_avg.append(np.mean(r_vals))
        vqa_avg.append(np.mean(v_vals))
        print(f"[{axis_name}] readout(L{layer})={np.mean(r_vals):.1f}%  VQA={np.mean(v_vals):.1f}%  "
              f"(AB/BC/AC readout={r_vals}, VQA={v_vals})")

    fig, ax = plt.subplots(figsize=(2.6 * len(axes) + 3, 6.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.arange(len(axes))
    w = 0.32
    b1 = ax.bar(x - w / 2, readout_avg, width=w, color=READOUT_COLOR, zorder=3,
                label=f"readout (layer {layer})", edgecolor=SURFACE, linewidth=1)
    b2 = ax.bar(x + w / 2, vqa_avg, width=w, color=VQA_COLOR, zorder=3,
                label="VQA baseline", edgecolor=SURFACE, linewidth=1)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.1f}",
                    ha="center", va="bottom", color=INK, fontsize=10)

    fig.suptitle(f"[{model}] {dataset}: pairwise readout vs VQA, AB/BC/AC averaged",
                 color=INK, fontsize=14.5, fontweight="bold", x=0.06, ha="left", y=0.99)

    ax.set_xticks(x)
    ax.set_xticklabels(axes, color=INK, fontsize=12)
    ax.set_ylabel("accuracy (%)", color=INK, fontsize=11)
    ax.set_ylim(0, 112)
    ax.axhline(50, color=MUTED, linewidth=1, linestyle=(0, (2, 2)), alpha=0.5, zorder=1)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=SECONDARY, length=0, labelsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=SECONDARY,
              handlelength=1.6, handletextpad=0.6)

    fig.subplots_adjust(top=0.86, bottom=0.09, left=0.09, right=0.97)
    out_path = f"{ap.PLOTS}/readout_vqa/{dataset}_signmatch_readout_vqa_bar_{model}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset", required=True, choices=cxp.DATASETS)
    cli.add_argument("--model", choices=ap.MODELS, default=ap.DEFAULT_MODEL)
    cli.add_argument("--layer", type=int, default=DEFAULT_LAYER, help="layer used by the bar-chart summary")
    cli.add_argument("--vqa-limit", type=int, default=None, help="only run VQA on the first N triplets (debug)")
    args = cli.parse_args()
    ap.set_model(args.model)
    axes = AXES_BY_DATASET[args.dataset]

    readout = readout_accuracy(args.dataset, args.model)
    num_l = len(next(iter(readout.values())))
    print_readout_summary(readout, axes, args.dataset, args.model, num_l)

    vqa, records = vqa_accuracy(args.dataset, args.model, limit=args.vqa_limit)
    print_vqa_summary(vqa, axes, args.dataset, args.model)

    tag = "" if args.vqa_limit is None else f"_debuglimit{args.vqa_limit}"
    np.savez(f"{ap.RESULTS}/readout_vqa/{args.dataset}_signmatch_readout_{args.model}{tag}.npz", **readout)
    with open(f"{ap.RESULTS}/readout_vqa/{args.dataset}_vqa_baseline_{args.model}{tag}.json", "w") as f:
        json.dump(dict(vqa=vqa, records=records), f, indent=2)
    print(f"\nsaved -> {ap.RESULTS}/readout_vqa/{args.dataset}_signmatch_readout_{args.model}{tag}.npz")
    print(f"saved -> {ap.RESULTS}/readout_vqa/{args.dataset}_vqa_baseline_{args.model}{tag}.json")

    if args.vqa_limit is None:
        draw_readout_vqa_plot(args.dataset)
        draw_readout_vqa_bar(readout, vqa, axes, args.dataset, args.model, layer=args.layer)
    else:
        print(f"\n[debug --vqa-limit {args.vqa_limit}] skipping plots (partial/uneven axis-pair coverage)")
