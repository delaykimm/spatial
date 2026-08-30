"""Genuine multi-hop MLLM generation test (rendered): unlike chain_hop_pipeline.py's direct
"is Z more right than A?" (answerable by comparing 2 named endpoints, no chain-traversal
needed -- see the multi-hop-QA literature discussion this followed from), here the model
must RESOLVE a nested referring expression to find out which object Z even is:

    "the object to the {dir_hop} of [the object to the {dir_hop-1} of [ ... the object to
    the {dir_1} of {A's description} ... ]]"

then discriminate it from Y (the object ONE HOP SHORT of Z) -- so stopping the chain early
gives the wrong answer. This only works if "the object to the right of X" is UNAMBIGUOUS,
which requires X's neighbors to be well-ordered: objects are placed at N SORTED, gapped
scalars along one axis (rank 0..N-1), and the chain is a random walk on ranks (each step
+/-1 in rank, i.e., "next object in that direction" -- guaranteed to exist and be unique by
construction, unlike chain_hop's free random walk).

Reuses chain_hop_pipeline's off-axis jitter/rendering helpers and triplet_pipeline's
renderer -- only the placement (sorted ranks, not a free walk) and the traversal (walk on
RANKS, not on world-coordinate steps) differ.

Usage:
    python multihop_referring_pipeline.py --model qwen3vl
    python multihop_referring_pipeline.py --render-only
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import random
import string
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import triplet_pipeline as cxp
import axis_pipeline as ap
import chain_hop_pipeline as chp  # reuse off-axis jitter helpers + extract_chain

_CFG = ap.CFG["multihop_referring"]
AXES = ["horizontal", "closefar"]
HOPS = _CFG["hops"]
N_SCENES_PER_HOP = _CFG["n_scenes_per_hop"]
SEED = _CFG["seed"]
MIN_GAP = _CFG["min_gap"]           # min gap between adjacent-rank scalars
OFF_JITTER = _CFG["off_jitter"]
OBJ_SIZE = _CFG["obj_size"]
MIN_DIST_3D = _CFG["obj_min_dist_3d"]
DIR_PHRASE = _CFG["dir_phrase"]     # {axis: {"+": "...", "-": "..."}}

cxp.OBJ_SIZE = OBJ_SIZE  # _render_scene draws at triplet_pipeline's own module-level OBJ_SIZE

CHAIN_IMAGES_DIR = f"{ap.DATA}/multihop_referring_images"
CHAIN_MANIFEST_PATH = f"{ap.DATA}/multihop_referring_manifest.json"
CHAIN_SHARD_BASE = f"{ap.RESULTS}/shards/multihop_referring_shard.npz"


def chain_shard_out_path():
    return ap.suffixed(CHAIN_SHARD_BASE)


# =============================================================================
# scene construction: N objects at SORTED, gapped scalars (rank 0..N-1) -- so "next object
# to the {direction}" is always unambiguous -- then a random walk ON RANKS for the chain
# =============================================================================
def sample_sorted_scalars(axis, n, min_gap, rng):
    """n scalars, sorted, each >= min_gap apart -- direct construction (not rejection
    sampling: for n=7 the naive "sample n uniforms, reject unless all gaps >= min_gap" has
    an acceptance probability of ~1e-5 here, since (n-1)*min_gap eats most of the range).
    Standard trick: sample the n "slack" positions in the reduced free range, sort them,
    then add back i*min_gap to the i-th one."""
    lo, hi = cxp._bounds(axis)
    free = (hi - lo) - (n - 1) * min_gap
    if free < 0:
        raise RuntimeError(f"n={n} objects with min_gap={min_gap} don't fit in axis={axis}'s range")
    base = sorted(rng.uniform(0, free) for _ in range(n))
    return [lo + b + i * min_gap for i, b in enumerate(base)]


def rank_walk(n, hops, rng):
    """Random walk on ranks 0..n-1, each step +/-1, retried near a boundary. Returns the
    list of ranks visited (length hops+1) -- may revisit a rank non-adjacently."""
    r = rng.randrange(n)
    ranks = [r]
    for _ in range(hops):
        placed = False
        for _try in range(50):
            step = rng.choice([-1, 1])
            nxt = r + step
            if 0 <= nxt <= n - 1:
                ranks.append(nxt)
                r = nxt
                placed = True
                break
        if not placed:
            return None
    return ranks


def build_scene(axis, hops, rng):
    """N=hops+1 objects at sorted scalars, random walk on ranks for the chain order.
    Returns (objs_by_rank, walk_ranks) or raises RuntimeError."""
    n = hops + 1
    for _attempt in range(2000):
        walk_ranks = rank_walk(n, hops, rng)
        # Z landing back on A's exact rank is a real, frequent case for a +/-1 walk this
        # short (e.g. starting mid-range at hops=2 makes it the ONLY possible outcome) --
        # reject it, since "is Z more right than A" has no defined answer when Z==A.
        if walk_ranks is None or walk_ranks[-1] == walk_ranks[0]:
            continue
        scalars = sample_sorted_scalars(axis, n, MIN_GAP[axis], rng)
        combos = rng.sample(cxp.ALL_COMBOS, n)
        x_base, y_base = chp.chain_off_axis_bases(axis, rng)
        objs_by_rank = []
        for rank, scalar in enumerate(scalars):
            x, y, z_base = chp.chain_world_pos(axis, scalar, x_base, y_base, rng)
            objs_by_rank.append({"rank": rank, "shape": combos[rank][0], "color": combos[rank][1],
                                  "x": x, "y": y, "z_base": z_base, "scalar": scalar})
        centers = [cxp.sameaxis_object_center(o["x"], o["y"], o["z_base"], OBJ_SIZE) for o in objs_by_rank]
        overlap = any(cxp.sameaxis_dist3(centers[i], centers[j]) < MIN_DIST_3D
                      for i in range(n) for j in range(i + 1, n))
        if overlap:
            continue
        for o, c in zip(objs_by_rank, centers):
            o["center"] = c
            o["desc"] = cxp.sameaxis_desc(o["shape"], o["color"])
        return objs_by_rank, walk_ranks
    raise RuntimeError(f"could not place referring scene axis={axis} hops={hops}")


def referring_phrase(axis, walk_ranks, objs_by_rank):
    """Nested referring expression for the LAST object in the walk, e.g. "the object to
    the right of the object to the left of ... {A's description}". Innermost = first hop
    (nearest A), outermost = last hop (= Z)."""
    phrase = objs_by_rank[walk_ranks[0]]["desc"]
    for prev_rank, cur_rank in zip(walk_ranks[:-1], walk_ranks[1:]):
        sign = "+" if cur_rank > prev_rank else "-"
        phrase = f"the object {DIR_PHRASE[axis][sign]} {phrase}"
    return phrase


def chain_hop_build_manifest():
    img_dir = Path(CHAIN_IMAGES_DIR)
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for axis in AXES:
        rng = random.Random(SEED[axis])
        per_hop = {}
        for hops in HOPS:
            scenes_out = []
            for idx in range(N_SCENES_PER_HOP):
                objs_by_rank, walk_ranks = build_scene(axis, hops, rng)
                fname = cxp._render_scene(f"ref_{axis}_hop{hops}", idx, objs_by_rank, img_dir)
                phrase = referring_phrase(axis, walk_ranks, objs_by_rank)
                z_rank, y_rank = walk_ranks[-1], walk_ranks[-2]
                scenes_out.append({
                    "image_path": fname, "axis": axis, "hops": hops,
                    "walk_ranks": walk_ranks, "phrase": phrase,
                    "z_desc": objs_by_rank[z_rank]["desc"], "y_desc": objs_by_rank[y_rank]["desc"],
                    "objects_by_rank": [{"desc": o["desc"], "scalar": o["scalar"]} for o in objs_by_rank],
                })
            per_hop[str(hops)] = scenes_out
            print(f"[multihop_referring|{axis}|hop{hops}] {len(scenes_out)} scenes rendered "
                  f"({hops + 1} objects each)")
        manifest[axis] = per_hop
    with open(CHAIN_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"saved -> {CHAIN_MANIFEST_PATH}")
    return manifest


def chain_hop_load_or_build_manifest():
    return cxp.load_or_build_json(CHAIN_MANIFEST_PATH, chain_hop_build_manifest, label="multihop_referring")


# =============================================================================
# extraction (GPU) -- reuses chain_hop_pipeline.extract_chain, over the WALK's consecutive
# descriptions (labels are just the walk order, not the sorted ranks)
# =============================================================================
def build_shard(model_key):
    ap.set_model(model_key)
    manifest = chain_hop_load_or_build_manifest()
    model, proc = ap.load_model()

    keys, flat = [], {}
    n_total = sum(len(manifest[axis][str(h)]) for axis in AXES for h in HOPS)
    n_done = 0
    for axis in AXES:
        for hops in HOPS:
            scenes = manifest[axis][str(hops)]
            for idx, s in enumerate(scenes):
                img = Image.open(os.path.join(CHAIN_IMAGES_DIR, s["image_path"])).convert("RGB")
                descs = [s["objects_by_rank"][r]["desc"] for r in s["walk_ranks"]]
                legs = chp.extract_chain(model, proc, img, descs)
                key = f"{axis}|{hops}|{idx}"
                keys.append(key)
                for i, leg in enumerate(legs):
                    flat[f"{key}|leg{i}"] = leg
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  {n_done}/{n_total} chains extracted")

    out_path = chain_shard_out_path()
    np.savez(out_path, keys=np.array(keys, dtype=object), **flat)
    print(f"saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--model", choices=ap.MODELS, default=ap.DEFAULT_MODEL)
    cli.add_argument("--render-only", action="store_true")
    args = cli.parse_args()

    manifest = chain_hop_load_or_build_manifest()
    if not args.render_only:
        build_shard(args.model)
