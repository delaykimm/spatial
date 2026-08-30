"""Regenerates triplet3ax_cross_axis_images/ + triplet3ax_cross_axis_triplets.json: L-shaped
(A,B,C) triplets where A->B is pure on one axis and B->C is pure on a DIFFERENT axis (plus
an unrelated distractor D), so r_AC is a genuine 2-axis diagonal -- purity is guaranteed
BY CONSTRUCTION (each leg only ever moves along its assigned axis, plus a small
perpendicular jitter), not searched for like the real-photo CLEVR version.

6 configs = the 3 axis pairs (horizontal-closefar, horizontal-vertical, vertical-closefar)
x 2 leg orderings each (e.g. horiz_then_z vs z_then_horiz), for a symmetry check -- same
idea as CLEVR's 2 configs.

Usage: python generate_triplet3ax.py [--out-images DIR] [--out-json PATH]
"""
import argparse
import json
import random
from pathlib import Path

from render3d import COLORS, SHAPES, X_RANGE, Y_FAR, Y_NEAR, render_multiobject_scene

# 6 configs: 3 axis pairs x 2 leg orderings.
AXIS_PAIRS = {
    "horiz_then_z":    ("horizontal", "closefar"),
    "z_then_horiz":    ("closefar", "horizontal"),
    "horiz_then_vert": ("horizontal", "vertical"),
    "vert_then_horiz": ("vertical", "horizontal"),
    "vert_then_z":     ("vertical", "closefar"),
    "z_then_vert":     ("closefar", "vertical"),
}
CONFIGS = list(AXIS_PAIRS.keys())
SEED = {c: 6042 + 100 * i for i, c in enumerate(CONFIGS)}
N_PER_CONFIG = 60
PURITY_THRESH = 0.85  # |cos| >= 0.85 <=> leg within ~31.8 deg of the target axis (reporting only
                       # -- construction already guarantees it, this just confirms it in the printout)

OBJ_SIZE = 0.8
AXIS_DIM = {"horizontal": 0, "closefar": 1, "vertical": 2}  # index into (x, y, z_base)
# vertical's world range (2.3 units) is much narrower than horizontal's (5.0) or closefar's
# (7.7), so its leg needs to be shorter -- but still long enough (> MIN_DIST_3D) that the
# two endpoint objects don't overlap.
LEG_LEN = {"horizontal": (1.2, 2.0), "closefar": (1.2, 2.0), "vertical": (1.1, 1.6)}
OFF_JITTER = 0.22  # perpendicular jitter half-width -- small enough that even the shortest
                    # leg (vertical, 1.1) keeps purity well above 0.85: 1.1/sqrt(1.1^2+0.22^2)=0.981
MIN_DIST_3D = 1.05  # overlap guard
X_RANGE_SAFE = X_RANGE - 0.5
Y_LO_SAFE, Y_HI_SAFE = Y_NEAR + 0.4, Y_FAR - 0.4
Z_LO, Z_HI = 0.35, 2.65  # floating height range for the (unstacked) vertical axis here
ALL_COMBOS = [(shape, color) for shape in SHAPES for color in COLORS]


def purity(d_on, d_off):
    """d_on: displacement along the target axis. d_off: displacement along the other axis."""
    mag = (d_on ** 2 + d_off ** 2) ** 0.5
    return abs(d_on) / (mag + 1e-9), mag


def _bounds(axis):
    return {"horizontal": (-X_RANGE_SAFE, X_RANGE_SAFE), "closefar": (Y_LO_SAFE, Y_HI_SAFE),
            "vertical": (Z_LO, Z_HI)}[axis]


def _base_point(axes_used, rng):
    """axes_used: the 2-axis tuple tested by this config. The 3rd, UNUSED axis is held at
    a fixed, sensible constant for the whole scene -- isolating exactly the 2 tested axes."""
    x = rng.uniform(-X_RANGE_SAFE + 2.5, X_RANGE_SAFE - 2.5) if "horizontal" in axes_used else 0.0
    y = rng.uniform(Y_LO_SAFE + 1.5, Y_HI_SAFE - 1.5) if "closefar" in axes_used else (Y_LO_SAFE + Y_HI_SAFE) / 2
    z = rng.uniform(Z_LO + 1.0, Z_HI - 1.0) if "vertical" in axes_used else 0.0
    return [x, y, z]


def _step(pos, axis, other_axis, rng):
    """Moves `pos` along `axis` by a random signed leg length, with a small perpendicular
    jitter along `other_axis` (the 3rd, untested axis never moves)."""
    lo, hi = LEG_LEN[axis]
    length = rng.uniform(lo, hi) * rng.choice([-1, 1])
    off = rng.uniform(-OFF_JITTER, OFF_JITTER)
    new = list(pos)
    new[AXIS_DIM[axis]] += length
    new[AXIS_DIM[other_axis]] += off
    return new, length, off


def build_scene(config, rng):
    axis1, axis2 = AXIS_PAIRS[config]
    axes_used = (axis1, axis2)
    for _attempt in range(200):
        a_pos = _base_point(axes_used, rng)
        b_pos, ab_on, ab_off = _step(a_pos, axis1, axis2, rng)
        c_pos, bc_on, bc_off = _step(b_pos, axis2, axis1, rng)
        pts = [a_pos, b_pos, c_pos]
        if not all(_bounds(ax)[0] <= p[AXIS_DIM[ax]] <= _bounds(ax)[1] for p in pts for ax in axes_used):
            continue

        combos = rng.sample(ALL_COMBOS, 4)
        objs = [
            {"label": "A", "shape": combos[0][0], "color": combos[0][1], "x": a_pos[0], "y": a_pos[1], "z_base": a_pos[2]},
            {"label": "B", "shape": combos[1][0], "color": combos[1][1], "x": b_pos[0], "y": b_pos[1], "z_base": b_pos[2]},
            {"label": "C", "shape": combos[2][0], "color": combos[2][1], "x": c_pos[0], "y": c_pos[1], "z_base": c_pos[2]},
        ]
        d_pos = _base_point(axes_used, rng)  # same fixed-3rd-axis convention as A/B/C
        d_pos[AXIS_DIM[axis1]] = rng.uniform(*_bounds(axis1)) if axis1 in axes_used else d_pos[AXIS_DIM[axis1]]
        d_pos[AXIS_DIM[axis2]] = rng.uniform(*_bounds(axis2))
        objs.append({"label": "D", "shape": combos[3][0], "color": combos[3][1], "x": d_pos[0], "y": d_pos[1], "z_base": d_pos[2]})

        centers = [(o["x"], o["y"], o["z_base"] + OBJ_SIZE / 2) for o in objs]
        ok = all(sum((a - b) ** 2 for a, b in zip(centers[i], centers[j])) ** 0.5 >= MIN_DIST_3D
                 for i in range(4) for j in range(i + 1, 4))
        if not ok:
            continue

        p_ab, _ = purity(ab_on, ab_off)
        p_bc, _ = purity(bc_on, bc_off)
        diag_len = sum((c_pos[AXIS_DIM[ax]] - a_pos[AXIS_DIM[ax]]) ** 2 for ax in axes_used) ** 0.5
        for o in objs:
            o["desc"] = f"{o['color']} {o['shape']}"
        return objs, p_ab, p_bc, diag_len
    raise RuntimeError(f"could not place L-shaped scene config={config}")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out-images", default="../data/triplet3ax_cross_axis_images")
    ap_.add_argument("--out-json", default="../data/triplet3ax_cross_axis_triplets.json")
    args = ap_.parse_args()
    img_dir = Path(args.out_images)
    img_dir.mkdir(parents=True, exist_ok=True)

    out = {c: [] for c in CONFIGS}
    for config in CONFIGS:
        rng = random.Random(SEED[config])
        for idx in range(N_PER_CONFIG):
            objs, p_ab, p_bc, diag_len = build_scene(config, rng)
            fname = f"{config}_{idx:03d}.png"
            render_multiobject_scene(objs, OBJ_SIZE, img_dir / fname)
            by_label = {o["label"]: o for o in objs}
            out[config].append({
                "image_path": fname,
                "A": by_label["A"]["desc"], "B": by_label["B"]["desc"], "C": by_label["C"]["desc"],
                "purity_AB": p_ab, "purity_BC": p_bc, "diag_len": diag_len,
            })
            if (idx + 1) % 15 == 0:
                print(f"[{config}] {idx+1}/{N_PER_CONFIG} rendered")
        pabs = [t["purity_AB"] for t in out[config]]
        pbcs = [t["purity_BC"] for t in out[config]]
        mean_ab, mean_bc = sum(pabs) / len(pabs), sum(pbcs) / len(pbcs)
        print(f"[{config}] n={len(out[config])}  purity_AB mean={mean_ab:.3f}  "
              f"purity_BC mean={mean_bc:.3f}  (guaranteed >{PURITY_THRESH} by construction)")

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.out_json}")


if __name__ == "__main__":
    main()
