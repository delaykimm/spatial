"""N-hop chains MINED FROM REAL CLEVR PHOTOS (not rendered) -- data/clevr_val_scenes.json's
full 15,000-scene metadata, instead of the synthetic renderer in chain_hop_pipeline.py.

No collinearity/purity requirement: A->B->C->...->Z is just N random objects from one
scene, in random order. The additive-composition question (hat_c_AZ = sum of consecutive
hidden-state relation vectors) doesn't need each individual leg to move purely along one
axis -- it only needs the FINAL A-vs-Z relation along the tested axis (horizontal="right"
or closefar="behind"; CLEVR has no vertical position variation, same 2-axis restriction as
triplet_pipeline.py's clevr triplets) to be non-trivial, which is checked post-hoc
(min_diag_len) regardless of how the random path zigzags in between.

Exposes the same interface as chain_hop_pipeline.py (AXES, HOPS, CHAIN_IMAGES_DIR,
chain_hop_load_or_build_manifest, chain_shard_out_path) plus reuses its extract_chain
(dataset-agnostic diff-of-hidden-states over consecutive labels), so
chain_hop_readout_vqa.py can analyze either backend via --source.

Needs data/clevr_val_scenes.json (already in this repo) + a local CLEVR v1.0 val image
folder to copy referenced images from the first time (same requirement as
reproduce_datasets/fetch_clevr_images.py).

Usage:
    python chain_hop/clevr_chain_pipeline.py --clevr-val-dir /path/to/CLEVR_v1.0/images/val --model qwen3vl
    python chain_hop/clevr_chain_pipeline.py --model qwen3vl              # once images are already copied
    python chain_hop/clevr_chain_pipeline.py --clevr-val-dir ... --render-only   # search + copy only, no GPU
"""
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, one level above this file's subfolder
sys.path[:0] = [os.path.join(_ROOT, d) for d in ("core", "analysis", "steering", "chain_hop", "multihop_referring")]
import argparse
import json
import random
import shutil
import string
from pathlib import Path

import numpy as np
from PIL import Image

import triplet_pipeline as cxp
import axis_pipeline as ap
import chain_hop_pipeline as chp  # reuse extract_chain (dataset-agnostic)

_CFG = ap.CFG["clevr_chain_hop"]
AXES = ["horizontal", "closefar"]  # CLEVR has no vertical position variation
AXIS_DIRS = {"horizontal": "right", "closefar": "behind"}
HOPS = _CFG["hops"]
N_SCENES_PER_HOP = _CFG["n_scenes_per_hop"]
SEED = _CFG["seed"]
MIN_DIAG_LEN = _CFG["min_diag_len"]

CHAIN_IMAGES_DIR = f"{ap.DATA}/clevr_chain_hop_images"
CHAIN_MANIFEST_PATH = f"{ap.DATA}/clevr_chain_hop_manifest.json"
CHAIN_SHARD_BASE = f"{ap.RESULTS}/shards/clevr_chain_hop_shard.npz"


def chain_shard_out_path():
    return ap.suffixed(CHAIN_SHARD_BASE)


# =============================================================================
# search: N random objects from the scene, random order -- no purity/collinearity filter,
# just enough net A-vs-Z displacement along the tested axis to be non-trivial
# =============================================================================
def pick_random_chain(objs, axis_dir, n_objects, rng, min_diag_len):
    """N distinct random objects, random order. Returns (labels_unused, chain_idx,
    descs, scalars) or None if too few objects, duplicate descriptions (can't phrase a
    VQA question), or |scalar_last - scalar_first| is too small (near-trivial answer
    regardless of hop count)."""
    if len(objs) < n_objects:
        return None
    chain_idx = rng.sample(range(len(objs)), n_objects)
    rng.shuffle(chain_idx)
    descs = [cxp.clevr_desc(objs[i]) for i in chain_idx]
    if len(set(descs)) < n_objects:
        return None
    scalars = [cxp.dot(objs[i]["3d_coords"], axis_dir) for i in chain_idx]
    if abs(scalars[-1] - scalars[0]) < min_diag_len:
        return None
    return chain_idx, descs, scalars


def chain_hop_build_manifest():
    with open(cxp.CLEVR_SCENES_PATH) as f:
        scenes = json.load(f)["scenes"]

    manifest = {}
    for axis in AXES:
        axis_dir_name = AXIS_DIRS[axis]
        rng = random.Random(SEED[axis])
        scene_order = list(range(len(scenes)))
        rng.shuffle(scene_order)

        per_hop = {}
        si_ptr = 0
        for hops in HOPS:
            n_objects = hops + 1
            scenes_out = []
            while len(scenes_out) < N_SCENES_PER_HOP and si_ptr < len(scene_order):
                s = scenes[scene_order[si_ptr]]
                si_ptr += 1
                objs = s["objects"]
                axis_dir = s["directions"][axis_dir_name]
                picked = pick_random_chain(objs, axis_dir, n_objects, rng, MIN_DIAG_LEN)
                if picked is None:
                    continue
                _, descs, scalars = picked
                labels = list(string.ascii_uppercase[:n_objects])
                scenes_out.append({
                    "image_path": s["image_filename"], "axis": axis, "hops": hops,
                    "labels": labels,
                    "objects": {lab: {"desc": d, "scalar": sc} for lab, d, sc in zip(labels, descs, scalars)},
                })
            per_hop[str(hops)] = scenes_out
            print(f"[clevr_chain_hop|{axis}|hop{hops}] {len(scenes_out)}/{N_SCENES_PER_HOP} chains found "
                  f"(scanned {si_ptr}/{len(scene_order)} scenes so far)")
        manifest[axis] = per_hop
    with open(CHAIN_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"saved -> {CHAIN_MANIFEST_PATH}")
    return manifest


def chain_hop_load_or_build_manifest():
    return cxp.load_or_build_json(CHAIN_MANIFEST_PATH, chain_hop_build_manifest, label="clevr_chain_hop")


def copy_images(manifest, clevr_val_dir):
    Path(CHAIN_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    needed = sorted({s["image_path"] for axis in manifest for hops in manifest[axis] for s in manifest[axis][hops]})
    copied, already, missing = 0, 0, []
    for fname in needed:
        dst = Path(CHAIN_IMAGES_DIR) / fname
        if dst.exists():
            already += 1
            continue
        src = Path(clevr_val_dir) / fname
        if not src.exists():
            missing.append(fname)
            continue
        shutil.copy2(src, dst)
        copied += 1
    print(f"copied {copied} new images (+{already} already present) -> {CHAIN_IMAGES_DIR}")
    if missing:
        raise FileNotFoundError(f"{len(missing)} referenced images not found under {clevr_val_dir}, "
                                 f"e.g. {missing[:5]}")


# =============================================================================
# extraction (GPU) -- reuses chain_hop_pipeline.extract_chain, identical method
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
                descs = [s["objects"][lab]["desc"] for lab in s["labels"]]
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
    cli.add_argument("--clevr-val-dir", default=None,
                      help="path to CLEVR_v1.0/images/val/ -- needed only the first time, to copy "
                           "referenced images into data/clevr_chain_hop_images/")
    cli.add_argument("--render-only", action="store_true", help="only search + build manifest + copy images, no GPU/model")
    args = cli.parse_args()

    manifest = chain_hop_load_or_build_manifest()
    if args.clevr_val_dir:
        copy_images(manifest, args.clevr_val_dir)
    elif not os.path.exists(CHAIN_IMAGES_DIR):
        raise SystemExit("no --clevr-val-dir given and no cached images in data/clevr_chain_hop_images/ yet "
                          "-- pass --clevr-val-dir once")

    if not args.render_only:
        build_shard(args.model)
