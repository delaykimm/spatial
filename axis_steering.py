"""Global (pooled) axis steering: inject each dataset's shared axis vector
(axis_pipeline.py's {dataset}_multilayer_axis_vectors.npz, LAYER=25) into that dataset's
own images' 2-choice question, sweep injection strength (alpha), measure how often the
answer flips toward '+'. Every item uses the SAME pooled vector -- no own-direction arm.

Reuses axis_pipeline.py's model loading + per-dataset item loaders; only the
steering-specific bits (hook injection, 2-choice generation) live here.

Usage: python axis_steering.py --dataset {whatsup,spatialtunnel,aug1,aug2}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import axis_pipeline as ap

_CFG = ap.CFG["axis_steering"]
LAYER = _CFG["layer"]
N_ITEMS = _CFG["n_items"]
SEED = _CFG["seed"]
FINE_ALPHA_FRACTIONS = [round(x, 1) for x in np.arange(-1.2, 1.21, 0.1)]    # horizontal, closefar
COARSE_ALPHA_FRACTIONS = [round(x, 1) for x in np.arange(-1.2, 1.21, 0.2)]  # vertical (already flat)
ALPHA_FRACTIONS_BY_AXIS = {"horizontal": FINE_ALPHA_FRACTIONS, "vertical": COARSE_ALPHA_FRACTIONS,
                            "closefar": FINE_ALPHA_FRACTIONS}

AXES3 = ["horizontal", "vertical", "closefar"]
AXIS_KEY = _CFG["axis_key"]
POSWORD = _CFG["posword"]
DATASETS = ["whatsup", "spatialtunnel", "aug1", "aug2"]

unit = ap.unit  # canonical L2-normalize now lives in axis_pipeline.py


def find_decoder_layers(model):
    candidates = [(name, mod) for name, mod in model.named_modules()
                  if isinstance(mod, nn.ModuleList) and len(mod) == ap.NUM_HIDDEN_STATES - 1]
    assert len(candidates) == 1
    return candidates[0][1]


class Steerer:
    def __init__(self, layer_module, vec):
        self.layer_module = layer_module
        self.vec = vec
        self.handle = None

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0] + self.vec.to(output[0].dtype)
            return (hs,) + output[1:]
        return output + self.vec.to(output.dtype)

    def __enter__(self):
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        self.handle.remove()


ask_2choice = ap.ask_2choice  # generate+parse now lives in axis_pipeline.py (shared with
                               # cross_axis_readout_vqa.py's ask_pair)


@torch.inference_mode()
def reference_hidden_norm(model, proc, items, layer, device="cuda", n_probe=8):
    norms = []
    for it in items[:n_probe]:
        img = Image.open(it["image_path"]).convert("RGB")
        inp = ap.build_inputs(proc, img, f"Describe the {it['minus_obj']}.", device)
        H = model(**inp, output_hidden_states=True, use_cache=False, return_dict=True).hidden_states[layer][0]
        norms.append(H.float().norm(dim=-1).mean().item())
    return float(np.mean(norms))


# =============================================================================
# per-dataset item pool (same axis, ignoring the group-disjoint split -- steering just
# samples N_ITEMS randomly from everything available on that axis)
# =============================================================================
def dataset_pool(dataset, axis_name, aug2_manifest=None):
    axis_key = AXIS_KEY[axis_name]
    if dataset == "whatsup":
        return [{"image_path": path, "minus_obj": B if is_plus else A, "plus_obj": A if is_plus else B}
                for path, A, B, ax, is_plus in ap.whatsup_items() if ax == axis_key]
    if dataset == "spatialtunnel":
        return [{"image_path": path, "minus_obj": B if is_plus else A, "plus_obj": A if is_plus else B}
                for path, A, B, ax, is_plus in ap.spatialtunnel_items() if ax == axis_key]
    if dataset == "aug1":
        rng = random.Random(ap.AUG1_SEED[axis_name])
        scenes = [ap.build_aug1_scene(f"scene{i:03d}", axis_name, rng) for i in range(ap.AUG1_N_SCENES)]
        out = []
        for scene in scenes:
            for tag in ["orig", "swap"]:
                swap = tag == "swap"
                A, B = scene["A"], scene["B"]
                a_slot = ("minus" if A["slot"] == "plus" else "plus") if swap else A["slot"]
                path = f"{ap.DATA}/aug1_images/{axis_name}_{scene['scene_id']}_{tag}.png"
                A_desc, B_desc = ap.aug1_desc(A), ap.aug1_desc(B)
                if a_slot == "plus":
                    out.append({"image_path": path, "minus_obj": B_desc, "plus_obj": A_desc})
                else:
                    out.append({"image_path": path, "minus_obj": A_desc, "plus_obj": B_desc})
        return out
    if dataset == "aug2":
        # plus_image/minus_image are stored relative to DATA (see axis_pipeline._jobs_aug2)
        out = []
        for entry in aug2_manifest[axis_name]:
            obj1, obj2 = entry["obj1"], entry["obj2"]
            out.append({"image_path": os.path.join(ap.DATA, entry["plus_image"]), "minus_obj": obj1, "plus_obj": obj2})
            out.append({"image_path": os.path.join(ap.DATA, entry["minus_image"]), "minus_obj": obj2, "plus_obj": obj1})
        return out
    raise ValueError(dataset)


def load_axis_vector(dataset, axis_name):
    """unit(plus - minus) at LAYER, using axis_pipeline.DATASET_CONFIGS' per-dataset key
    convention (whatsup/spatialtunnel/aug1: x/y/z_right_axis|left_axis; aug2:
    horizontal/vertical/closefar_plus_axis|minus_axis)."""
    cfg = ap.DATASET_CONFIGS[dataset]
    out_name = cfg["axis_names"][AXIS_KEY[axis_name]]
    d = np.load(cfg["out_path"])
    plus = d[f"{out_name}_{cfg['plus_key']}"][LAYER]
    minus = d[f"{out_name}_{cfg['minus_key']}"][LAYER]
    return unit(plus - minus)


if __name__ == "__main__":
    ap_cli = argparse.ArgumentParser()
    ap_cli.add_argument("--dataset", required=True, choices=DATASETS)
    args = ap_cli.parse_args()

    aug2_manifest = None
    if args.dataset == "aug2":
        with open(f"{ap.DATA}/aug2_render_manifest_disjoint.json") as f:
            aug2_manifest = json.load(f)

    model, proc = ap.load_model()
    decoder_layers = find_decoder_layers(model)
    steer_module = decoder_layers[LAYER - 1]
    rng = random.Random(SEED)

    out = {}
    for axis_name in AXES3:
        pool = dataset_pool(args.dataset, axis_name, aug2_manifest)
        items = rng.sample(pool, min(N_ITEMS, len(pool)))
        pos_phrase = POSWORD[axis_name]
        print(f"\n[{args.dataset}/{axis_name}] GLOBAL AXIS pool={len(pool)}, using N={len(items)}, pos_phrase='{pos_phrase}'")

        shared_vec_np = load_axis_vector(args.dataset, axis_name)
        alpha_fractions = ALPHA_FRACTIONS_BY_AXIS[axis_name]
        ref_norm = reference_hidden_norm(model, proc, items, LAYER)
        alphas = [f * ref_norm for f in alpha_fractions]
        print(f"reference hidden norm={ref_norm:.2f}")

        acc_by_alpha, valid_frac_by_alpha = [], []
        for alpha, frac in zip(alphas, alpha_fractions):
            vec = torch.tensor(shared_vec_np, dtype=torch.float16, device="cuda") * alpha
            n_correct, n_total = 0, 0
            with Steerer(steer_module, vec):
                for it in items:
                    img = Image.open(it["image_path"]).convert("RGB")
                    ans = ask_2choice(model, proc, img, it["minus_obj"], it["plus_obj"], pos_phrase)
                    if ans == "?":
                        continue
                    n_total += 1
                    n_correct += (ans == "B")
            acc = n_correct / max(n_total, 1)
            valid_frac = n_total / len(items)
            acc_by_alpha.append(acc)
            valid_frac_by_alpha.append(valid_frac)
            print(f"  alpha_frac={frac:+.1f}  P(answer=true '+')={acc:.3f}  valid={valid_frac:.2f}  (n={n_total}/{len(items)})")

        out[f"{axis_name}_alphas"] = np.array(alphas)
        out[f"{axis_name}_alpha_fractions"] = np.array(alpha_fractions)
        out[f"{axis_name}_acc"] = np.array(acc_by_alpha)
        out[f"{axis_name}_valid_frac"] = np.array(valid_frac_by_alpha)
        out[f"{axis_name}_ref_norm"] = ref_norm

    out_path = f"{ap.RESULTS}/steering/{args.dataset}_global_axis_steering.npz"
    np.savez(out_path, **out)
    print(f"\nsaved -> {out_path}")
