"""Extracts per-layer direction vectors (left/right, up/down, close/far) for 4 spatial
datasets from a VLM's hidden states, + a within-dataset 6x6 cosine sanity-check heatmap.

- Method: matched pairs (both polarities photographed) -> split into 2 disjoint groups ->
  diff-of-means direction per layer, per group. Groups should agree (cosine ~ -1).
- Needs spatial/data/: whatsup_a/b, spatialtunnel_images+metadata.csv, aug1_images,
  aug2_images+render_manifest_disjoint.json

Usage:
    python axis_pipeline.py --dataset whatsup                          # extract+merge+heatmap
    python axis_pipeline.py --dataset whatsup --shard 0 --num-shards 8 # shard N of 8 (last one auto-merges)
    python axis_pipeline.py --heatmap-only                             # redraw from what's already merged
"""
import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "data")        # static input data (images, manifests)
RESULTS = os.path.join(_HERE, "results")  # pipeline outputs (shards, axis vectors, npz/json)
PLOTS = os.path.join(RESULTS, "plot")     # every analysis PNG plot lives here, separate from
                                           # the data files (npz/json) alongside it in RESULTS
os.makedirs(PLOTS, exist_ok=True)

# Experiment-tuning constants (models, seeds, thresholds, phrases, ...) all live in
# config.yaml. CFG is the loaded dict; every module that imports axis_pipeline reads its
# own section into plain module-level constants, so call sites (e.g. cxp.PURITY_THRESH)
# are unaffected by where the value actually comes from.
with open(f"{_HERE}/config.yaml") as _f:
    CFG = yaml.safe_load(_f)


# =============================================================================
# model loading + hidden-state extraction engine
# =============================================================================
# suffix is appended to every output filename so different models' extractions don't
# clobber each other. qwen3vl keeps an empty suffix (its files predate --model).
MODEL_CONFIGS = CFG["models"]["configs"]
MODELS = list(MODEL_CONFIGS.keys())
DEFAULT_MODEL = CFG["models"]["default"]

PROMPT = CFG["models"]["prompt"]

# active-model state, set via set_model() -- the rest of this file reads these globals
# instead of threading a model argument through every function (one model per process).
MODEL_KEY = DEFAULT_MODEL
MODEL_ID = MODEL_CONFIGS[MODEL_KEY]["model_id"]
NUM_HIDDEN_STATES = MODEL_CONFIGS[MODEL_KEY]["num_hidden_states"]
D = MODEL_CONFIGS[MODEL_KEY]["d"]


def set_model(model_key):
    """Switch the active model. Must be called (if not using the qwen3vl default) before
    load_model() or any extraction/path function below."""
    global MODEL_KEY, MODEL_ID, NUM_HIDDEN_STATES, D
    MODEL_KEY = model_key
    cfg = MODEL_CONFIGS[model_key]
    MODEL_ID, NUM_HIDDEN_STATES, D = cfg["model_id"], cfg["num_hidden_states"], cfg["d"]


def model_suffix():
    return MODEL_CONFIGS[MODEL_KEY]["suffix"]


def suffixed(path):
    """Inserts the active model's suffix before a path's extension, or before the
    directory's trailing part if `path` has no extension (e.g. a shard_dir)."""
    base, ext = os.path.splitext(path)
    return f"{base}{model_suffix()}{ext}"


def axis_out_path(dataset):
    return suffixed(DATASET_CONFIGS[dataset]["out_path"])


def axis_shard_dir(dataset):
    return suffixed(DATASET_CONFIGS[dataset]["shard_dir"])


def pick_free_gpu():
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"]
        ).decode()
        best_idx, best_free = None, -1
        for line in out.strip().splitlines():
            idx, used, total = [int(x) for x in line.split(",")]
            free = total - used
            if free > best_free:
                best_idx, best_free = idx, free
        return best_idx
    except Exception:
        return None


def load_model():
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpu = pick_free_gpu()
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[gpu] CUDA_VISIBLE_DEVICES not set -> auto-picked cuda:{gpu} (most free memory)")
    import transformers
    proc = transformers.AutoProcessor.from_pretrained(MODEL_ID)
    model_class = getattr(transformers, MODEL_CONFIGS[MODEL_KEY]["model_class"])
    model = model_class.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda").eval()
    return model, proc


def build_inputs(proc, image, text, device="cuda"):
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    prompt = proc.apply_chat_template(conv, add_generation_prompt=True)
    inp = proc(images=image, text=prompt, return_tensors="pt")
    return {k: (v.to(device, torch.float16) if torch.is_floating_point(v) else v.to(device))
            for k, v in inp.items() if torch.is_tensor(v)}


@torch.inference_mode()
def ask_2choice(model, proc, image, term_a, term_b, phrase, device="cuda"):
    """Generate + parse: "which object is {phrase}, the {term_a} or the {term_b}?" ->
    'A' (term_a), 'B' (term_b), or '?' (neither/both matched). Shared by axis_steering.py
    and cross_axis_readout_vqa.py, whose 2-choice VQA question is otherwise identical."""
    q = f"In the image, which object is {phrase}, the {term_a} or the {term_b}? Answer with {term_a} or {term_b}, only."
    inp = build_inputs(proc, image, q, device)
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


@torch.inference_mode()
def single_object_node_alllayers(model, proc, image, obj, other, device="cuda"):
    """Last-token hidden state at EVERY layer from one forward pass, shape
    (NUM_HIDDEN_STATES, 4096) -- output_hidden_states already computes all layers in a
    single call, so this costs no extra compute over extracting just one layer."""
    inp = build_inputs(proc, image, PROMPT.format(obj=obj, other=other), device)
    H = model(**inp, output_hidden_states=True, use_cache=False, return_dict=True).hidden_states
    return np.stack([H[l][0, -1].float().cpu().numpy() for l in range(NUM_HIDDEN_STATES)], axis=0)


def object_nodes_alllayers(model, proc, image, A, B, device="cuda"):
    """Asks the model about A relative to B, then B relative to A (each its own forward
    pass, since the prompt names one object as 'obj' and the other as 'other')."""
    hA = single_object_node_alllayers(model, proc, image, A, B, device)
    hB = single_object_node_alllayers(model, proc, image, B, A, device)
    return hA, hB  # each (NUM_HIDDEN_STATES, 4096)


# =============================================================================
# What'sUp: real product photos, matched object-pairs with both polarities photographed
# =============================================================================
WHATSUP_DIRS = [f"{DATA}/whatsup_a", f"{DATA}/whatsup_b"]
# rel string -> (axis, is_positive_phrasing). is_positive_phrasing=True means the relation
# string itself describes the '+' direction (x+ = right, y+ = up/on, z+ = front/closer).
WU_REL = {"left": ("x", False), "right": ("x", True), "on": ("y", True),
          "under": ("y", False), "front": ("z", True), "behind": ("z", False)}


def _clean_obj_name(s):
    s = re.sub(r"^\d+_", "", s)   # drop leading index prefix, e.g. "000_beer-bottle" -> "beer-bottle"
    s = re.sub(r"_\d+$", "", s)   # drop trailing index suffix, just in case
    return s.replace("-", " ").replace("_", " ").strip()


def whatsup_items():
    """Every What'sUp photo as (path, A, B, axis, is_A_plus), parsed from filenames like
    '000_beer-bottle_left_of_001_can.jpg'."""
    items = []
    for d in WHATSUP_DIRS:
        for f in sorted(Path(d).glob("*.jp*g")):
            stem = f.stem
            for rel in ["left", "right", "front", "behind", "on", "under"]:
                tok = f"_{rel}_"
                if tok in stem:
                    A, B = stem.split(tok, 1)
                    ax, ap = WU_REL[rel]
                    items.append((str(f), _clean_obj_name(A), _clean_obj_name(B), ax, ap))
                    break
    return items


def build_matched_pairs():
    """One entry per object-pair that has BOTH polarities photographed (e.g. both
    'A_left_of_B' and 'A_right_of_B' exist) -- pairs missing one side are dropped."""
    items = whatsup_items()
    pairs = []
    for axis in ["x", "y", "z"]:
        groups = defaultdict(dict)
        for path, A, B, ax, ap in items:
            if ax != axis:
                continue
            groups[frozenset({A, B})][ap] = (path, A, B)
        for key, v in groups.items():
            if True not in v or False not in v:
                continue
            path_t, A, B = v[True]
            path_f, _, _ = v[False]
            pairs.append({"axis": axis, "A": A, "B": B,
                          "path_true": path_t, "path_false": path_f})
    return pairs


# =============================================================================
# SpatialTunnel: synthetic CLEVR-style renders (cube/sphere + color, plain tunnel bg)
# =============================================================================
SPATIALTUNNEL_METADATA = f"{DATA}/spatialtunnel_metadata.csv"  # index/category/A/B/answer,
# extracted once from contrastive-probing's contrastive_probing.tsv (image blobs dropped
# since all 1200 images are already decoded into SPATIALTUNNEL_IMG_DIR below)
SPATIALTUNNEL_IMG_DIR = f"{DATA}/spatialtunnel_images"
# left/right/above/below rows have A,B = the two object names directly (no regex needed)
# and answer = the category itself (confirms A {answer} B).
ST_REL = {"left": ("x", False), "right": ("x", True), "below": ("y", False), "above": ("y", True)}


def spatialtunnel_items():
    """Every SpatialTunnel row as (path, A, B, axis, is_A_plus). close/far rows have A,B =
    two candidate objects; answer='A'/'B' picks which one is the close/far one relative to
    the OTHER (reference)."""
    df = pd.read_csv(SPATIALTUNNEL_METADATA)
    items = []
    for _, row in df.iterrows():
        cat = row["category"]
        img_path = os.path.join(SPATIALTUNNEL_IMG_DIR, f"{row['index']}.jpg")
        if cat in ST_REL:
            axis, ap_of_A = ST_REL[cat]
            items.append((img_path, row["A"], row["B"], axis, ap_of_A))
        elif cat in ("close", "far"):
            target = row["A"] if row["answer"] == "A" else row["B"]
            other = row["B"] if row["answer"] == "A" else row["A"]
            items.append((img_path, target, other, "z", cat == "close"))
    return items


# =============================================================================
# Aug1 (mplot3d): synthetic 3D-rendered primitives, pre-rendered to DATA/aug1_images
# =============================================================================
AUG1_SHAPES = CFG["axis_pipeline"]["aug1_shapes"]
AUG1_COLORS = CFG["axis_pipeline"]["aug1_colors"]
AUG1_SEED = CFG["axis_pipeline"]["aug1_seed"]
AUG1_N_SCENES = CFG["axis_pipeline"]["aug1_n_scenes"]


def aug1_desc(obj):
    return f"{obj['color']} {obj['shape']}"


def build_aug1_scene(scene_id, axis, rng):
    """Randomly assigns shape+color+slot to two probe objects A/B for one scene. Pure
    metadata -- matches the scene actually rendered (once, ahead of time) to
    DATA/aug1_images/{axis}_{scene_id}_{orig,swap}.png."""
    shape_a, shape_b = rng.sample(AUG1_SHAPES, 2)
    color_a, color_b = rng.sample(AUG1_COLORS, 2)
    a_slot = rng.choice(["plus", "minus"])
    b_slot = "minus" if a_slot == "plus" else "plus"
    return {
        "scene_id": scene_id, "axis": axis,
        "A": {"shape": shape_a, "color": color_a, "slot": a_slot},
        "B": {"shape": shape_b, "color": color_b, "slot": b_slot},
    }


# =============================================================================
# per-dataset job-list construction: each returns [(axis, group, path, A, B, a_is_plus), ...]
# (a_is_plus = is A the '+' member; B is then '-'). Group-split RNG per dataset:
# whatsup shares one random.Random(42) across x/y/z; the rest use a fresh seed per axis.
# =============================================================================
AXES = ["x", "y", "z"]
DATASETS = ["whatsup", "spatialtunnel", "aug1", "aug2"]


def _jobs_whatsup():
    pairs = build_matched_pairs()
    rng = random.Random(42)  # shared instance advanced across all 3 axes, matches original
    group_of = {}
    for axis in AXES:
        ax_pairs = [p for p in pairs if p["axis"] == axis]
        shuffled = ax_pairs[:]
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        for p in shuffled[:half]:
            group_of[id(p)] = 1
        for p in shuffled[half:]:
            group_of[id(p)] = 2
    jobs = []
    for p in pairs:
        g = group_of[id(p)]
        jobs.append((p["axis"], g, p["path_true"], p["A"], p["B"], True))
        jobs.append((p["axis"], g, p["path_false"], p["A"], p["B"], False))
    return jobs


def _jobs_spatialtunnel():
    items = spatialtunnel_items()  # (path, A, B, axis, ap)
    jobs = []
    for axis in AXES:
        ax_items = [it for it in items if it[3] == axis]
        combo_items = defaultdict(list)
        for it in ax_items:
            combo_items[frozenset({it[1], it[2]})].append(it)
        combos = list(combo_items.keys())
        rng = random.Random(42)  # fresh instance per axis
        rng.shuffle(combos)
        half = len(combos) // 2
        for g, combo_slice in [(1, combos[:half]), (2, combos[half:])]:
            for c in combo_slice:
                for path, A, B, ax, ap in combo_items[c]:
                    jobs.append((axis, g, path, A, B, ap))
    return jobs


def _jobs_aug1():
    axis_key = {"horizontal": "x", "vertical": "y", "closefar": "z"}
    img_dir = Path(f"{DATA}/aug1_images")
    jobs = []
    for axis in ["horizontal", "vertical", "closefar"]:
        rng = random.Random(AUG1_SEED[axis])
        scenes = [build_aug1_scene(f"scene{i:03d}", axis, rng) for i in range(AUG1_N_SCENES)]
        shuffled = scenes[:]
        random.Random(AUG1_SEED[axis] + 1).shuffle(shuffled)
        half = len(shuffled) // 2
        group1_ids = {s["scene_id"] for s in shuffled[:half]}
        for scene in scenes:
            g = 1 if scene["scene_id"] in group1_ids else 2
            for tag in ["orig", "swap"]:
                swap = tag == "swap"
                A, B = scene["A"], scene["B"]
                a_slot = ("minus" if A["slot"] == "plus" else "plus") if swap else A["slot"]
                b_slot = ("minus" if B["slot"] == "plus" else "plus") if swap else B["slot"]
                assert a_slot != b_slot, "A/B expected to always be opposite-slot in this dataset"
                path = str(img_dir / f"{axis}_{scene['scene_id']}_{tag}.png")
                jobs.append((axis_key[axis], g, path, aug1_desc(A), aug1_desc(B), a_slot == "plus"))
    return jobs


def _jobs_aug2(manifest_path=f"{DATA}/aug2_render_manifest_disjoint.json"):
    axis_key = {"horizontal": "x", "vertical": "y", "closefar": "z"}
    seed = {"horizontal": 11042, "vertical": 12042, "closefar": 13042}
    with open(manifest_path) as f:
        manifest = json.load(f)
    jobs = []
    for axis in ["horizontal", "vertical", "closefar"]:
        entries = manifest[axis]
        rng = random.Random(seed[axis])  # fresh instance per axis
        shuffled = entries[:]
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        group1_pairs = {(e["obj1"], e["obj2"]) for e in shuffled[:half]}
        for e in entries:
            g = 1 if (e["obj1"], e["obj2"]) in group1_pairs else 2
            # plus_image/minus_image are stored relative to DATA. plus_image: obj1='-',
            # obj2='+'.  minus_image: obj1='+', obj2='-' (reversed).
            jobs.append((axis_key[axis], g, os.path.join(DATA, e["plus_image"]), e["obj1"], e["obj2"], False))
            jobs.append((axis_key[axis], g, os.path.join(DATA, e["minus_image"]), e["obj1"], e["obj2"], True))
    return jobs


DATASET_CONFIGS = {
    "whatsup": dict(
        jobs_fn=_jobs_whatsup,
        shard_dir=f"{RESULTS}/shards/whatsup_multilayer_shards",
        out_path=f"{RESULTS}/axis_vectors/whatsup_multilayer_axis_vectors.npz",
        axis_names={"x": "x", "y": "y", "z": "z"},
        plus_key="right_axis", minus_key="left_axis",
    ),
    "spatialtunnel": dict(
        jobs_fn=_jobs_spatialtunnel,
        shard_dir=f"{RESULTS}/shards/spatialtunnel_multilayer_shards",
        out_path=f"{RESULTS}/axis_vectors/spatialtunnel_multilayer_axis_vectors.npz",
        axis_names={"x": "x", "y": "y", "z": "z"},
        plus_key="right_axis", minus_key="left_axis",
    ),
    "aug1": dict(
        jobs_fn=_jobs_aug1,
        shard_dir=f"{RESULTS}/shards/aug1_multilayer_shards",
        out_path=f"{RESULTS}/axis_vectors/aug1_multilayer_axis_vectors.npz",
        axis_names={"x": "x", "y": "y", "z": "z"},
        plus_key="right_axis", minus_key="left_axis",
    ),
    "aug2": dict(
        jobs_fn=_jobs_aug2,
        shard_dir=f"{RESULTS}/shards/aug2_multilayer_shards",
        out_path=f"{RESULTS}/axis_vectors/aug2_multilayer_axis_vectors.npz",
        axis_names={"x": "horizontal", "y": "vertical", "z": "closefar"},
        plus_key="plus_axis", minus_key="minus_axis",
    ),
}


# =============================================================================
# extract (GPU) -> merge (CPU) -> within-dataset cosine heatmap
# =============================================================================
HEATMAP_L = CFG["axis_pipeline"]["heatmap_layer"]
NAMES6 = ["x+", "x-", "y+", "y-", "z+", "z-"]


def unit(v, axis=-1):
    """Canonical L2-normalize, used by every module in this pipeline (import axis_pipeline
    as ap and call ap.unit(...) instead of redefining this locally)."""
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + 1e-9)


def report_layers(num_l):
    """8 evenly-spaced layer indices (linspace(1, num_l-1, 8), deduplicated) for compact
    per-layer console summaries -- identical logic previously duplicated in
    cross_axis_analysis.py and cross_axis_readout_vqa.py."""
    idx = np.unique(np.round(np.linspace(1, num_l - 1, 8)).astype(int))
    return idx.tolist()


def report_layers_frac(num_l, fracs=None):
    """Like report_layers, but from explicit fractions-of-depth (including 0.0) instead
    of linspace(1, num_l-1, 8) -- cross_axis_alignment.py's own, deliberately different
    layer choice (starts at layer 0, not 1). `fracs` defaults to config.yaml's
    cross_axis_alignment.report_layers_frac."""
    fracs = fracs if fracs is not None else CFG["cross_axis_alignment"]["report_layers_frac"]
    return sorted(set(int(round(f * (num_l - 1))) for f in fracs))


def cos_perlayer(a, b):
    return np.sum(unit(a) * unit(b), axis=-1)


@torch.inference_mode()
def run_extract(dataset, shard, num_shards):
    cfg = DATASET_CONFIGS[dataset]
    shard_dir = axis_shard_dir(dataset)
    os.makedirs(shard_dir, exist_ok=True)
    jobs = cfg["jobs_fn"]()
    my_jobs = jobs[shard::num_shards]
    print(f"[{dataset}] shard {shard}/{num_shards}: {len(my_jobs)} of {len(jobs)} image-reads")

    model, proc = load_model()
    sums = {ax: {1: {"plus": np.zeros((NUM_HIDDEN_STATES, D)), "minus": np.zeros((NUM_HIDDEN_STATES, D))},
                 2: {"plus": np.zeros((NUM_HIDDEN_STATES, D)), "minus": np.zeros((NUM_HIDDEN_STATES, D))}} for ax in AXES}
    counts = {ax: {1: {"plus": 0, "minus": 0}, 2: {"plus": 0, "minus": 0}} for ax in AXES}

    for i, (ax, g, path, A, B, a_is_plus) in enumerate(my_jobs):
        img = Image.open(path).convert("RGB")
        hA, hB = object_nodes_alllayers(model, proc, img, A, B)
        h_plus, h_minus = (hA, hB) if a_is_plus else (hB, hA)
        sums[ax][g]["plus"] += h_plus; counts[ax][g]["plus"] += 1
        sums[ax][g]["minus"] += h_minus; counts[ax][g]["minus"] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(my_jobs)} done")

    out_path = os.path.join(shard_dir, f"shard_{shard}.npz")
    flat = {}
    for ax in AXES:
        for g in [1, 2]:
            for sign in ["plus", "minus"]:
                flat[f"{ax}_{g}_{sign}_sum"] = sums[ax][g][sign]
                flat[f"{ax}_{g}_{sign}_count"] = counts[ax][g][sign]
    np.savez(out_path, **flat)
    print(f"saved -> {out_path}")


def run_merge(dataset):
    cfg = DATASET_CONFIGS[dataset]
    shard_files = sorted(glob.glob(f"{axis_shard_dir(dataset)}/shard_*.npz"))
    print(f"[{dataset}] merging {len(shard_files)} shards")

    sums = {ax: {1: {"plus": np.zeros((NUM_HIDDEN_STATES, D)), "minus": np.zeros((NUM_HIDDEN_STATES, D))},
                 2: {"plus": np.zeros((NUM_HIDDEN_STATES, D)), "minus": np.zeros((NUM_HIDDEN_STATES, D))}} for ax in AXES}
    counts = {ax: {1: {"plus": 0, "minus": 0}, 2: {"plus": 0, "minus": 0}} for ax in AXES}
    for sf in shard_files:
        d = np.load(sf)
        for ax in AXES:
            for g in [1, 2]:
                for sign in ["plus", "minus"]:
                    sums[ax][g][sign] += d[f"{ax}_{g}_{sign}_sum"]
                    counts[ax][g][sign] += int(d[f"{ax}_{g}_{sign}_count"])

    out = {}
    for ax in AXES:
        p1 = sums[ax][1]["plus"] / counts[ax][1]["plus"]
        m1 = sums[ax][1]["minus"] / counts[ax][1]["minus"]
        p2 = sums[ax][2]["plus"] / counts[ax][2]["plus"]
        m2 = sums[ax][2]["minus"] / counts[ax][2]["minus"]
        right_axis = p1 - m1
        left_axis = m2 - p2
        c = cos_perlayer(right_axis, left_axis)
        out_name = cfg["axis_names"][ax]
        print(f"[{dataset}][{out_name}] n(g1+)={counts[ax][1]['plus']} n(g1-)={counts[ax][1]['minus']} "
              f"n(g2+)={counts[ax][2]['plus']} n(g2-)={counts[ax][2]['minus']}")
        print(f"[{dataset}][{out_name}] within_cos per layer: min={c.min():.3f} max={c.max():.3f} "
              f"@layer{HEATMAP_L}={c[HEATMAP_L]:.3f}")
        out[f"{out_name}_{cfg['plus_key']}"] = right_axis
        out[f"{out_name}_{cfg['minus_key']}"] = left_axis
        out[f"{out_name}_within_cos_perlayer"] = c

    out_path = axis_out_path(dataset)
    np.savez(out_path, **out)
    print(f"[{dataset}] saved -> {out_path}")
    return out


def _direction_vectors(dataset, merged=None):
    """{'x+':(4096,), 'x-':..., ...} at HEATMAP_L, for the within-dataset 6x6 cosine matrix."""
    cfg = DATASET_CONFIGS[dataset]
    d = merged if merged is not None else np.load(axis_out_path(dataset))
    vecs = {}
    for ax in AXES:
        out_name = cfg["axis_names"][ax]
        vecs[f"{ax}+"] = d[f"{out_name}_{cfg['plus_key']}"][HEATMAP_L]
        vecs[f"{ax}-"] = d[f"{out_name}_{cfg['minus_key']}"][HEATMAP_L]
    return vecs


def within_dataset_cosine_matrix(dataset, merged=None):
    vecs = _direction_vectors(dataset, merged)
    U = {k: unit(v) for k, v in vecs.items()}
    return np.array([[np.dot(U[a], U[b]) for b in NAMES6] for a in NAMES6])


def draw_heatmap():
    """(Re)draws the 2x2-panel within-dataset 6x6 cosine heatmap from whichever of the 4
    datasets have already been merged (dataviz-skill diverging blue/red palette)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    LABELS = {"whatsup": "What'sUp", "spatialtunnel": "spatialTunnel",
              "aug1": "Aug1", "aug2": "Aug2"}
    available = [d for d in DATASETS if os.path.exists(axis_out_path(d))]
    if not available:
        print("no merged axis-vector files found yet, skipping heatmap")
        return
    matrices = {LABELS[d]: within_dataset_cosine_matrix(d) for d in available}

    CMAP = LinearSegmentedColormap.from_list("div_blue_red", ["#1c5cab", "#f0efec", "#e34948"])
    SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"

    fig, axes = plt.subplots(2, 2, figsize=(11, 11), dpi=140, facecolor=SURFACE)
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")

    im = None
    for ax, (dname, M) in zip(axes, matrices.items()):
        ax.axis("on")
        ax.set_facecolor(SURFACE)
        im = ax.imshow(M, cmap=CMAP, vmin=-1, vmax=1)
        ax.set_xticks(range(6)); ax.set_xticklabels(NAMES6, color=INK, fontsize=10)
        ax.set_yticks(range(6)); ax.set_yticklabels(NAMES6, color=INK, fontsize=10)
        ax.set_title(dname, color=INK, fontsize=13, pad=10, fontweight="bold")
        for i in range(6):
            for j in range(6):
                v = M[i, j]
                txt_color = "#fcfcfb" if abs(v) > 0.55 else INK
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8.5, color=txt_color)
        for spine in ax.spines.values():
            spine.set_color(MUTED); spine.set_linewidth(0.6)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.03, label="cosine similarity")
        cbar.ax.yaxis.label.set_color(INK)
        cbar.ax.tick_params(colors=MUTED)
        cbar.outline.set_edgecolor(MUTED)

    fig.suptitle(f"Within-dataset axis-vector cosine similarity (6x6, layer {HEATMAP_L}, "
                 f"model={MODEL_KEY})", color=INK, fontsize=15, y=0.97)
    out_path = f"{PLOTS}/axis_vectors/axis_cosine_heatmap_{MODEL_KEY}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--heatmap-only", action="store_true")
    args = ap.parse_args()
    set_model(args.model)

    if args.heatmap_only:
        draw_heatmap()
    else:
        if not args.dataset:
            ap.error("--dataset is required for the extract phase")
        run_extract(args.dataset, args.shard, args.num_shards)
        # auto-merge once every shard for this dataset is on disk, so a multi-GPU run
        # never needs a manual merge step -- whichever shard process happens to finish
        # last triggers it. Harmless if two shards finish at nearly the same time and
        # both trigger it (merge just re-reads the same shard files).
        n_done = len(glob.glob(f"{axis_shard_dir(args.dataset)}/shard_*.npz"))
        if n_done >= args.num_shards:
            print(f"[{args.dataset}] all {args.num_shards} shard(s) present -> auto-merging")
            run_merge(args.dataset)
            draw_heatmap()
        else:
            print(f"[{args.dataset}] {n_done}/{args.num_shards} shards done, "
                  f"waiting on the rest before merging")
