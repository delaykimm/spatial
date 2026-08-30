"""N-hop random-walk chains on ONE axis: A->B->C->...->Z, each leg's sign (+/-) picked at
random (NOT sorted, unlike sameaxis) -- so "is Z more {right/up/close} than A?" has a real
~50/50 answer that only the accumulated hops determine, at every hop count. This is what
makes the hop-count sweep meaningful instead of a trivial always-same-answer question.

Tests: does latent computation (sum consecutive hidden-state relation vectors
hat_c_AZ = c_AB + c_BC + ... + c_Y,Z, then read out its sign) hold up as hops increase,
compared to the MLLM directly answering the same A-vs-Z question by looking at the image?
See chain_hop_readout_vqa.py for that comparison + the accuracy-vs-hops plot.

Reuses triplet_pipeline.py's sameaxis rendering primitives (same object palette, same
world-space bounds/camera, same overlap guard) -- only the placement logic (random walk
instead of sorted-and-forced-distractor) and the object count (variable, not fixed at 4)
differ.

Usage:
    python chain_hop_pipeline.py                       # render (if needed) + extract, all axes/hops
    python chain_hop_pipeline.py --model llava
    python chain_hop_pipeline.py --render-only          # just (re)build images + manifest, no GPU
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

_CFG = ap.CFG["chain_hop"]
AXES = ["horizontal", "closefar"]  # vertical's world range is too narrow for hop=6, see config.yaml
HOPS = _CFG["hops"]
N_SCENES_PER_HOP = _CFG["n_scenes_per_hop"]
SEED = _CFG["seed"]
STEP_RANGE = _CFG["step_range"]
OFF_JITTER = _CFG["off_jitter"]
OBJ_SIZE = _CFG["obj_size"]
MIN_DIST_3D = _CFG["obj_min_dist_3d"]
MIN_NET_DISP_FRAC = _CFG["min_net_disp_frac"]

# triplet_pipeline._render_scene() draws every object at ITS OWN module-level OBJ_SIZE
# (0.8, not a parameter) -- override it so the actual rendering matches the smaller size
# our own overlap check (MIN_DIST_3D above) assumes. Safe: this process only ever renders
# chain_hop scenes, never triplet3ax/sameaxis ones that'd want the original 0.8.
cxp.OBJ_SIZE = OBJ_SIZE

CHAIN_IMAGES_DIR = f"{ap.DATA}/chain_hop_images"
CHAIN_MANIFEST_PATH = f"{ap.DATA}/chain_hop_manifest.json"
CHAIN_SHARD_BASE = f"{ap.RESULTS}/shards/chain_hop_shard.npz"


def chain_shard_out_path():
    return ap.suffixed(CHAIN_SHARD_BASE)


# =============================================================================
# scene construction -- random walk along one axis (not sorted), everything else
# (object palette, world bounds, camera) borrowed from sameaxis. Off-axis dims use this
# module's own small OFF_JITTER -- NOT sameaxis_world_pos's wide independent-per-object
# sampling, which would swamp the (smaller) on-axis hop signal with irrelevant
# depth/position noise: sameaxis's own tests don't care about single-leg purity (only
# 4-way extremum ranking), but ours does. OBJ_SIZE/MIN_DIST_3D are also this module's own
# (smaller than triplet_pipeline's), needed to fit hop=6 inside horizontal/closefar's
# world bounds -- see config.yaml's chain_hop section for the numbers.
# =============================================================================
def chain_off_axis_bases(axis, rng):
    """One shared (x_base, y_base) for the WHOLE chain -- every object gets only a small
    +/-OFF_JITTER wobble around these, instead of each object independently resampling
    from sameaxis's wide position ranges."""
    x_base = rng.uniform(-cxp.SAMEAXIS_X_JITTER, cxp.SAMEAXIS_X_JITTER)
    y_base = rng.uniform(cxp.SAMEAXIS_Y_MID_LO, cxp.SAMEAXIS_Y_MID_HI)
    return x_base, y_base


def chain_world_pos(axis, scalar, x_base, y_base, rng):
    j = OFF_JITTER
    if axis == "horizontal":
        return scalar, y_base + rng.uniform(-j, j), 0.0
    if axis == "closefar":
        return x_base + rng.uniform(-j, j), scalar, 0.0
    raise ValueError(axis)


def chain_build_scene(axis, n_objects, rng, min_step, max_step, min_net_disp):
    """Random walk of n_objects-1 signed steps along `axis` (each step's sign is an
    independent coin flip -- this is what keeps "is Z more {axis-direction} than A?"
    non-trivial at every hop count, unlike sameaxis's sorted A<B<C). Retries on
    out-of-bounds walks, near-zero net displacement (endpoint relation would be a coin
    flip regardless of hops), or 3D overlap -- same retry-loop convention as
    triplet_pipeline.sameaxis_build_scene/t3ax_build_scene."""
    lo, hi = cxp._bounds(axis)
    labels = string.ascii_uppercase[:n_objects]
    for _attempt in range(5000):
        scalars = [rng.uniform(lo, hi)]
        ok = True
        for _ in range(n_objects - 1):
            placed = False
            for _step_try in range(50):  # retry just this step (not the whole walk) near a boundary
                step = rng.uniform(min_step, max_step) * rng.choice([-1.0, 1.0])
                nxt = scalars[-1] + step
                if lo <= nxt <= hi:
                    scalars.append(nxt)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok or abs(scalars[-1] - scalars[0]) < min_net_disp:
            continue

        combos = rng.sample(cxp.ALL_COMBOS, n_objects)
        x_base, y_base = chain_off_axis_bases(axis, rng)
        objs = []
        for i, (label, scalar) in enumerate(zip(labels, scalars)):
            x, y, z_base = chain_world_pos(axis, scalar, x_base, y_base, rng)
            objs.append({"label": label, "shape": combos[i][0], "color": combos[i][1],
                         "x": x, "y": y, "z_base": z_base, "scalar": scalar})

        centers = [cxp.sameaxis_object_center(o["x"], o["y"], o["z_base"], OBJ_SIZE) for o in objs]
        overlap = any(cxp.sameaxis_dist3(centers[i], centers[j]) < MIN_DIST_3D
                      for i in range(n_objects) for j in range(i + 1, n_objects))
        if overlap:
            continue
        for o, c in zip(objs, centers):
            o["center"] = c
            o["desc"] = cxp.sameaxis_desc(o["shape"], o["color"])
        return objs
    raise RuntimeError(f"could not place chain scene axis={axis} n_objects={n_objects}")


def chain_hop_build_manifest():
    img_dir = Path(CHAIN_IMAGES_DIR)
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for axis in AXES:
        rng = random.Random(SEED[axis])
        min_step, max_step = STEP_RANGE[axis]
        per_hop = {}
        for hops in HOPS:
            n_objects = hops + 1
            min_net_disp = min_step * MIN_NET_DISP_FRAC
            scenes_out = []
            for idx in range(N_SCENES_PER_HOP):
                objs = chain_build_scene(axis, n_objects, rng, min_step, max_step, min_net_disp)
                fname = cxp._render_scene(f"{axis}_hop{hops}", idx, objs, img_dir)
                scenes_out.append({
                    "image_path": fname, "axis": axis, "hops": hops,
                    "labels": [o["label"] for o in objs],
                    "objects": {o["label"]: {"desc": o["desc"], "scalar": o["scalar"],
                                              "center": list(o["center"])} for o in objs},
                })
            per_hop[str(hops)] = scenes_out
            print(f"[chain_hop|{axis}|hop{hops}] {len(scenes_out)} scenes rendered "
                  f"({n_objects} objects each)")
        manifest[axis] = per_hop
    with open(CHAIN_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"saved -> {CHAIN_MANIFEST_PATH}")
    return manifest


def chain_hop_load_or_build_manifest():
    return cxp.load_or_build_json(CHAIN_MANIFEST_PATH, chain_hop_build_manifest, label="chain_hop")


# =============================================================================
# extraction (GPU) -- same diff-of-hidden-states method as triplet_pipeline.extract_triplet,
# generalized to N-1 consecutive legs instead of a fixed AB/BC/AC triple
# =============================================================================
@torch.inference_mode()
def extract_chain(model, proc, img, descs):
    """r_{i,i+1} = h_{i+1}(rel i) - h_i(rel i+1) for every consecutive pair along the
    chain. `descs` must be the objects' natural-language descriptions (e.g. "red sphere"),
    IN CHAIN ORDER -- NOT the manifest's letter labels (A/B/C/...), which mean nothing to
    the model and would silently turn every query into "where is A relative to B?".
    Returns a list of (NUM_HIDDEN_STATES, d) arrays, length len(descs)-1."""
    legs = []
    for a, b in zip(descs[:-1], descs[1:]):
        hA, hB = ap.object_nodes_alllayers(model, proc, img, a, b)
        legs.append(hB - hA)
    return legs


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
                descs = [s["objects"][lab]["desc"] for lab in s["labels"]]
                legs = extract_chain(model, proc, img, descs)
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
    cli.add_argument("--render-only", action="store_true", help="only (re)build images + manifest, no GPU/model")
    args = cli.parse_args()

    manifest = chain_hop_load_or_build_manifest()
    if not args.render_only:
        build_shard(args.model)
