"""Triplet (A,B,C) datasets + hidden-state extraction, via --dataset:

  - clevr: L-shaped cross-axis, from real CLEVR photos. Searches for a "corner" B where
    A->B is pure horizontal and B->C is pure close/far, so r_AC is a genuine 2-axis
    diagonal -- tests whether r_AB + r_BC ~= r_AC. REQUIRES data/clevr_val_scenes.json +
    data/clevr_cross_axis_images/ (nothing here regenerates those, must ship with data/).
  - triplet3ax: same L-shape, but built with our own synthetic 3D renderer -- purity is
    guaranteed by construction instead of searched for.
  - sameaxis: different structure -- A/B/C sorted along ONE axis (+ distractor D, forced
    outside [A,C]), so r_AC is a single-axis span, not a diagonal. Feeds
    diffpair_steering.py's "sameaxis" mode.

clevr/triplet3ax share one manifest schema (config -> [{image_path,A,B,C,purity_AB,
purity_BC,diag_len}, ...]); sameaxis is keyed by axis name instead, but has the same
image_path/A/B/C fields -- all 3 datasets use the same shard format.

Extraction (same prompt/method as axis_pipeline.object_nodes_alllayers, all layers):
  r_AB = h_B(B rel A) - h_A(A rel B); r_BC = h_C(C rel B) - h_B(B rel C);
  r_AC = h_C(C rel A) - h_A(A rel C)
Each leg is its own forward pass, so r_AB + r_BC == r_AC is a real, falsifiable test.

Usage:
    python triplet_pipeline.py --dataset clevr
    python triplet_pipeline.py --dataset triplet3ax --model llava
    python triplet_pipeline.py --dataset sameaxis
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import random

import numpy as np
import torch
from PIL import Image

from collections import defaultdict

import axis_pipeline as ap

_CFG = ap.CFG["triplet_pipeline"]
DATASETS = ["clevr", "triplet3ax", "sameaxis"]
PAIRS = ["AB", "BC", "AC"]
AXES_BY_DATASET = {"clevr": ["horizontal", "closefar"], "triplet3ax": ["horizontal", "vertical", "closefar"]}
CLEVR_CONFIGS = ["horiz_then_z", "z_then_horiz"]  # CLEVR has no vertical (all objects on one ground plane)
PURITY_THRESH = _CFG["purity_thresh"]  # |cos| >= 0.85 <=> leg within ~31.8 deg of the target axis
MIN_DIAG_LEN = _CFG["min_diag_len"]    # minimum ||A->C|| in the (horizontal,closefar) plane, world units


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def purity(d_on, d_off):
    """d_on: displacement along the target axis. d_off: displacement along the other axis."""
    mag = (d_on ** 2 + d_off ** 2) ** 0.5
    return abs(d_on) / (mag + 1e-9), mag


def load_or_build_json(path, build_fn, label=""):
    """Shared cache-check pattern for all 3 datasets' manifests: reuse the on-disk JSON if
    it already exists, otherwise call build_fn() (which is responsible for writing `path`
    itself, e.g. clevr_sample_triplets/t3ax_build_triplets/sameaxis_build_manifest)."""
    if os.path.exists(path):
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}found existing manifest -> {path} (reusing)")
        with open(path) as f:
            return json.load(f)
    return build_fn()


# =============================================================================
# CLEVR: search real photos for near-axis-aligned corners
# =============================================================================
CLEVR_SCENES_PATH = f"{ap.DATA}/clevr_val_scenes.json"
CLEVR_IMAGES_DIR = f"{ap.DATA}/clevr_cross_axis_images"
CLEVR_TRIPLETS_PATH = f"{ap.DATA}/clevr_cross_axis_triplets.json"
CLEVR_N_PER_CONFIG = _CFG["clevr"]["n_per_config"]
CLEVR_SEED = _CFG["clevr"]["seed"]


def clevr_desc(obj):
    return f"{obj['size']} {obj['color']} {obj['material']} {obj['shape']}"


def clevr_best_triple_for_scene(objs, right_dir, behind_dir, config):
    """config: 'horiz_then_z' (AB horizontal-pure, BC closefar-pure) or 'z_then_horiz' (swapped).
    Returns (score, A_i, B_i, C_i, purity_AB, purity_BC, diag_len) or None."""
    n = len(objs)
    if n < 3:
        return None
    right = [dot(o["3d_coords"], right_dir) for o in objs]
    behind = [dot(o["3d_coords"], behind_dir) for o in objs]

    best = None
    for b in range(n):
        others = [i for i in range(n) if i != b]
        for ii in range(len(others)):
            for jj in range(ii + 1, len(others)):
                x, y = others[ii], others[jj]
                for a_i, c_i in [(x, y), (y, x)]:
                    d_ab_right = right[b] - right[a_i]
                    d_ab_behind = behind[b] - behind[a_i]
                    d_bc_right = right[c_i] - right[b]
                    d_bc_behind = behind[c_i] - behind[b]
                    if config == "horiz_then_z":
                        p_ab, mag_ab = purity(d_ab_right, d_ab_behind)
                        p_bc, mag_bc = purity(d_bc_behind, d_bc_right)
                    else:
                        p_ab, mag_ab = purity(d_ab_behind, d_ab_right)
                        p_bc, mag_bc = purity(d_bc_right, d_bc_behind)
                    if mag_ab < 1e-6 or mag_bc < 1e-6:
                        continue
                    d_ac_right = right[c_i] - right[a_i]
                    d_ac_behind = behind[c_i] - behind[a_i]
                    diag_len = (d_ac_right ** 2 + d_ac_behind ** 2) ** 0.5
                    score = min(p_ab, p_bc)
                    if best is None or score > best[0]:
                        best = (score, a_i, b, c_i, p_ab, p_bc, diag_len)
    return best


def clevr_sample_triplets():
    with open(CLEVR_SCENES_PATH) as f:
        scenes = json.load(f)["scenes"]
    rng = random.Random(CLEVR_SEED)

    out = {c: [] for c in CLEVR_CONFIGS}
    for config in CLEVR_CONFIGS:
        scene_order = list(range(len(scenes)))
        rng.shuffle(scene_order)
        n_checked = 0
        for si in scene_order:
            if len(out[config]) >= CLEVR_N_PER_CONFIG:
                break
            n_checked += 1
            s = scenes[si]
            objs = s["objects"]
            if len(objs) < 3:
                continue
            best = clevr_best_triple_for_scene(objs, s["directions"]["right"], s["directions"]["behind"], config)
            if best is None:
                continue
            score, a_i, b_i, c_i, p_ab, p_bc, diag_len = best
            if score < PURITY_THRESH or diag_len < MIN_DIAG_LEN:
                continue
            descs = [clevr_desc(objs[i]) for i in (a_i, b_i, c_i)]
            if len(set(descs)) < 3:
                continue
            out[config].append({
                "image_path": s["image_filename"], "scene_index": si,
                "A": descs[0], "B": descs[1], "C": descs[2],
                "purity_AB": p_ab, "purity_BC": p_bc, "diag_len": diag_len,
            })
        print(f"[clevr|{config}] scanned {n_checked}/{len(scenes)} scenes -> {len(out[config])} qualifying "
              f"(purity>={PURITY_THRESH}, diag_len>={MIN_DIAG_LEN})")

    with open(CLEVR_TRIPLETS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {CLEVR_TRIPLETS_PATH}")
    return out


def clevr_load_or_sample():
    return load_or_build_json(CLEVR_TRIPLETS_PATH, clevr_sample_triplets, label="clevr")


# =============================================================================
# triplet3ax: directly construct + render synthetic L-shaped scenes (purity guaranteed)
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

T3AX_IMAGES_DIR = f"{ap.DATA}/triplet3ax_cross_axis_images"
T3AX_TRIPLETS_PATH = f"{ap.DATA}/triplet3ax_cross_axis_triplets.json"
T3AX_N_PER_CONFIG = _CFG["triplet3ax"]["n_per_config"]
OBJ_SIZE = _CFG["triplet3ax"]["obj_size"]
Y_NEAR, Y_FAR, X_RANGE = 2.5, 11.0, 3.0
X_RANGE_SAFE = X_RANGE - 0.5
Y_LO_SAFE, Y_HI_SAFE = Y_NEAR + 0.4, Y_FAR - 0.4
Z_LO, Z_HI = 0.35, 2.65  # floating height range, matches triplet3ax_stimuli.py's vertical axis

# 6 configs: the 3 axis pairs (horizontal-closefar, horizontal-vertical, vertical-closefar)
# x 2 leg orderings each, for a symmetry check (same principle as CLEVR's 2 configs).
AXIS_PAIRS = {
    "horiz_then_z":    ("horizontal", "closefar"),
    "z_then_horiz":    ("closefar", "horizontal"),
    "horiz_then_vert": ("horizontal", "vertical"),
    "vert_then_horiz": ("vertical", "horizontal"),
    "vert_then_z":     ("vertical", "closefar"),
    "z_then_vert":     ("closefar", "vertical"),
}
T3AX_CONFIGS = list(AXIS_PAIRS.keys())
T3AX_SEED = {c: 6042 + 100 * i for i, c in enumerate(T3AX_CONFIGS)}

AXIS_DIM = {"horizontal": 0, "closefar": 1, "vertical": 2}  # index into (x, y, z_base)
# vertical's world range (2.3 units) is much narrower than horizontal's (5.0) or
# closefar's (7.7), so its leg needs to be shorter -- but still long enough (>MIN_DIST_3D)
# that the two endpoint objects don't overlap.
LEG_LEN = _CFG["triplet3ax"]["leg_len"]
OFF_JITTER = _CFG["triplet3ax"]["off_jitter"]  # perpendicular jitter half-width -- small enough that
                                                # even the shortest leg (vertical, 1.1) keeps purity
                                                # well above 0.85: 1.1/sqrt(1.1^2+0.22^2) = 0.981
MIN_DIST_3D = _CFG["triplet3ax"]["min_dist_3d"]  # overlap guard, matches triplet3ax_stimuli.py's convention
ALL_COMBOS = [(shape, color) for shape in ap.AUG1_SHAPES for color in ap.AUG1_COLORS]


# ---- rendering primitives (reused from synthetic3d_stimuli.py -- axis_pipeline.py only
# needed the SHAPES_3D/COLORS_3D constants since it reads pre-rendered images, but this
# pipeline renders brand-new triplet3ax scenes so the actual drawing code is needed here) ----
def _sphere_mesh(cx, cy, cz, r, n=14):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = cx + r * np.outer(np.cos(u), np.sin(v))
    y = cy + r * np.outer(np.sin(u), np.sin(v))
    z = cz + r * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _cube_faces(cx, cy, cz, s):
    r = s / 2
    x0, x1 = cx - r, cx + r
    y0, y1 = cy - r, cy + r
    z0, z1 = cz, cz + s
    faces = []
    Y, Z = np.meshgrid([y0, y1], [z0, z1])
    faces.append((np.full_like(Y, x1), Y, Z))
    X, Z2 = np.meshgrid([x0, x1], [z0, z1])
    faces.append((X, np.full_like(X, y0), Z2))
    X2, Y2 = np.meshgrid([x0, x1], [y0, y1])
    faces.append((X2, Y2, np.full_like(X2, z1)))
    return faces


def _cylinder_mesh(cx, cy, cz, r, h, n=20):
    theta = np.linspace(0, 2 * np.pi, n)
    z = np.linspace(cz, cz + h, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = cx + r * np.cos(theta_grid)
    y = cy + r * np.sin(theta_grid)
    return x, y, z_grid


def _cone_mesh(cx, cy, cz, r, h, n=20):
    theta = np.linspace(0, 2 * np.pi, n)
    z_levels = np.array([cz, cz + h])
    radii = np.array([r, 0.0])
    theta_grid, z_grid = np.meshgrid(theta, z_levels)
    r_grid = radii[:, None] * np.ones_like(theta_grid)
    x = cx + r_grid * np.cos(theta_grid)
    y = cy + r_grid * np.sin(theta_grid)
    return x, y, z_grid


def draw_shadow_and_dropline(ax, cx, cy, cz, size):
    gray = (0.55, 0.55, 0.55)
    r = size / 2.2
    theta = np.linspace(0, 2 * np.pi, 16)
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), np.zeros_like(theta),
            color=gray, linewidth=0.9)
    if cz > 0.05:
        ax.plot([cx, cx], [cy, cy], [0, cz], color=gray, linewidth=0.8, linestyle="--")


def draw_ground_grid(ax, exclude_objects=None, n=7, n_segments=28):
    exclude_objects = exclude_objects or []
    proj = ax.get_proj()

    def to2d(x, y, z):
        x2, y2, _ = proj3d.proj_transform(x, y, z, proj)
        return x2, y2

    obj_screen = []
    for cx, cy, cz, r in exclude_objects:
        ox, oy = to2d(cx, cy, cz)
        ex, ey = to2d(cx + r, cy, cz)
        screen_r = ((ex - ox) ** 2 + (ey - oy) ** 2) ** 0.5
        obj_screen.append((ox, oy, max(screen_r, 1e-6)))

    def excluded(mx, my, mz):
        sx, sy = to2d(mx, my, mz)
        return any((sx - ox) ** 2 + (sy - oy) ** 2 < sr ** 2 for ox, oy, sr in obj_screen)

    light_gray = (0.85, 0.85, 0.85)
    xs = np.linspace(-X_RANGE, X_RANGE, n)
    ys = np.linspace(Y_NEAR - 1, Y_FAR + 1, n)
    y_full = np.linspace(Y_NEAR - 1, Y_FAR + 1, n_segments + 1)
    x_full = np.linspace(-X_RANGE, X_RANGE, n_segments + 1)
    for x in xs:
        for y0, y1 in zip(y_full[:-1], y_full[1:]):
            if not excluded(x, (y0 + y1) / 2, 0.0):
                ax.plot([x, x], [y0, y1], [0, 0], color=light_gray, linewidth=0.6)
    for y in ys:
        for x0, x1 in zip(x_full[:-1], x_full[1:]):
            if not excluded((x0 + x1) / 2, y, 0.0):
                ax.plot([x0, x1], [y, y], [0, 0], color=light_gray, linewidth=0.6)


def draw_object(ax, shape, color_name, cx, cy, cz, size, anchor=True):
    color = _COLOR_RGB[color_name]
    r = size / 2
    if anchor:
        draw_shadow_and_dropline(ax, cx, cy, cz, size)
    if shape == "sphere":
        x, y, z = _sphere_mesh(cx, cy, cz + r, r)
        ax.plot_surface(x, y, z, color=color, shade=True, antialiased=True, linewidth=0)
    elif shape == "cube":
        for X, Y, Z in _cube_faces(cx, cy, cz, size):
            ax.plot_surface(X, Y, Z, color=color, shade=True, antialiased=True, linewidth=0)
    elif shape == "cylinder":
        x, y, z = _cylinder_mesh(cx, cy, cz, r, size)
        ax.plot_surface(x, y, z, color=color, shade=True, antialiased=True, linewidth=0)
        theta = np.linspace(0, 2 * np.pi, 20)
        cap_x, cap_y = cx + r * np.cos(theta), cy + r * np.sin(theta)
        for cz_cap in (cz, cz + size):
            verts = [list(zip(cap_x, cap_y, [cz_cap] * len(theta)))]
            ax.add_collection3d(Poly3DCollection(verts, facecolor=color, linewidths=0))
    elif shape == "cone":
        x, y, z = _cone_mesh(cx, cy, cz, r, size)
        ax.plot_surface(x, y, z, color=color, shade=True, antialiased=True, linewidth=0)
        theta = np.linspace(0, 2 * np.pi, 20)
        base_x, base_y = cx + r * np.cos(theta), cy + r * np.sin(theta)
        verts = [list(zip(base_x, base_y, [cz] * len(theta)))]
        ax.add_collection3d(Poly3DCollection(verts, facecolor=color, linewidths=0))
    else:
        raise ValueError(shape)


def new_scene_figure(exclude_objects=None):
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = True
    ax.set_proj_type("persp", focal_length=0.15)
    ax.set_xlim(-X_RANGE, X_RANGE)
    ax.set_ylim(Y_NEAR - 1, Y_FAR + 1)
    ax.set_zlim(0, 3.2)
    ax.view_init(elev=12, azim=-80)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.5))
    draw_ground_grid(ax, exclude_objects=exclude_objects)
    return fig, ax


def save_scene(fig, path, canvas_size=512, pad_frac=0.12):
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    non_white = np.any(arr < 250, axis=-1)
    if non_white.any():
        ys, xs = np.where(non_white)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        w, h = x1 - x0, y1 - y0
        pad = int(max(w, h) * pad_frac) + 1
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(arr.shape[1], x1 + pad), min(arr.shape[0], y1 + pad)
        side = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, x1 = max(0, cx - side // 2), min(arr.shape[1], cx + side // 2)
        y0, y1 = max(0, cy - side // 2), min(arr.shape[0], cy + side // 2)
        img = img.crop((x0, y0, x1, y1))
    img = img.resize((canvas_size, canvas_size), Image.LANCZOS)
    img.save(path)


_COLOR_RGB = {
    "red": (0.86, 0.20, 0.20), "blue": (0.20, 0.35, 0.86),
    "green": (0.20, 0.63, 0.27), "yellow": (0.90, 0.78, 0.16),
    "purple": (0.59, 0.24, 0.71), "orange": (0.94, 0.55, 0.12),
}


# ---- L-shaped scene construction (purity guaranteed by construction, not searched) ----
def _bounds(axis):
    return {"horizontal": (-X_RANGE_SAFE, X_RANGE_SAFE), "closefar": (Y_LO_SAFE, Y_HI_SAFE),
            "vertical": (Z_LO, Z_HI)}[axis]


def _base_point(axes_used, rng):
    """axes_used: the 2-axis tuple tested by this config. The 3rd, UNUSED axis is held at
    a fixed, sensible constant for the whole scene (0 for horizontal untouched, mid-depth
    for closefar untouched, ground z=0 for vertical untouched) -- isolating exactly the 2
    tested axes, same principle as the horizontal+closefar-only version."""
    x = rng.uniform(-X_RANGE_SAFE + 2.5, X_RANGE_SAFE - 2.5) if "horizontal" in axes_used else 0.0
    y = rng.uniform(Y_LO_SAFE + 1.5, Y_HI_SAFE - 1.5) if "closefar" in axes_used else (Y_LO_SAFE + Y_HI_SAFE) / 2
    z = rng.uniform(Z_LO + 1.0, Z_HI - 1.0) if "vertical" in axes_used else 0.0
    return [x, y, z]


def _step(pos, axis, other_axis, rng):
    """Moves `pos` along `axis` by a random signed leg length, with a small perpendicular
    jitter along `other_axis` (the axis pair's OTHER tested member -- the 3rd, untested
    axis never moves). Returns (new_pos, on, off)."""
    lo, hi = LEG_LEN[axis]
    length = rng.uniform(lo, hi) * rng.choice([-1, 1])
    off = rng.uniform(-OFF_JITTER, OFF_JITTER)
    new = list(pos)
    new[AXIS_DIM[axis]] += length
    new[AXIS_DIM[other_axis]] += off
    return new, length, off


def t3ax_build_scene(config, idx, rng):
    """config keys into AXIS_PAIRS, e.g. 'horiz_then_z' -> A->B horizontal-pure, B->C
    closefar-pure. A/B/C/D share a fixed value on whichever axis isn't tested (0, or a
    floating mid-height if vertical is one of the 2 tested axes)."""
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
    raise RuntimeError(f"could not place L-shaped scene config={config} idx={idx}")


def _render_scene(tag, idx, objs, out_dir):
    """Shared by t3ax/sameaxis: draw all objects into one new 3D scene and save it as
    '{tag}_{idx:03d}.png' (tag = config name for triplet3ax, axis name for sameaxis)."""
    exclude = [(o["x"], o["y"], 0.0, OBJ_SIZE / 1.6) for o in objs]
    fig, ax = new_scene_figure(exclude_objects=exclude)
    for o in objs:
        draw_object(ax, o["shape"], o["color"], o["x"], o["y"], o["z_base"], OBJ_SIZE)
    out_path = out_dir / f"{tag}_{idx:03d}.png"
    save_scene(fig, str(out_path))
    return out_path.name


def t3ax_build_triplets():
    from pathlib import Path
    img_dir = Path(T3AX_IMAGES_DIR)
    img_dir.mkdir(parents=True, exist_ok=True)

    out = {c: [] for c in T3AX_CONFIGS}
    for config in T3AX_CONFIGS:
        rng = random.Random(T3AX_SEED[config])
        for idx in range(T3AX_N_PER_CONFIG):
            objs, p_ab, p_bc, diag_len = t3ax_build_scene(config, idx, rng)
            fname = _render_scene(config, idx, objs, img_dir)
            by_label = {o["label"]: o for o in objs}
            out[config].append({
                "image_path": fname,
                "A": by_label["A"]["desc"], "B": by_label["B"]["desc"], "C": by_label["C"]["desc"],
                "purity_AB": p_ab, "purity_BC": p_bc, "diag_len": diag_len,
            })
            if (idx + 1) % 15 == 0:
                print(f"[triplet3ax|{config}] {idx+1}/{T3AX_N_PER_CONFIG} rendered")
        pabs = [t["purity_AB"] for t in out[config]]
        pbcs = [t["purity_BC"] for t in out[config]]
        print(f"[triplet3ax|{config}] n={len(out[config])}  purity_AB mean={np.mean(pabs):.3f}  "
              f"purity_BC mean={np.mean(pbcs):.3f}  (guaranteed >{PURITY_THRESH} by construction)")

    with open(T3AX_TRIPLETS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {T3AX_TRIPLETS_PATH}")
    return out


def t3ax_load_or_build():
    return load_or_build_json(T3AX_TRIPLETS_PATH, t3ax_build_triplets, label="triplet3ax")


# =============================================================================
# sameaxis: A, B, C sorted along ONE tested axis (+ distractor D, forced outside [A,C],
# never queried). Unlike clevr/triplet3ax above (A->B/B->C pure on DIFFERENT axes, r_AC a
# diagonal mix), here everything's on ONE axis -- r_AC is a genuine single-axis span,
# what diffpair_steering.py's "sameaxis" mode needs.
# =============================================================================
SAMEAXIS_N_PER_AXIS = _CFG["sameaxis"]["n_per_axis"]
SAMEAXIS_SEED = _CFG["sameaxis"]["seed"]
SAMEAXIS_IMAGES_DIR = f"{ap.DATA}/sameaxis_images"
SAMEAXIS_MANIFEST_PATH = f"{ap.DATA}/sameaxis_manifest.json"
SAMEAXIS_SHARD_BASE = f"{ap.RESULTS}/shards/sameaxis_shard.npz"

# world ranges used when a dimension is NOT the tested axis (jitter only) -- the tested
# axis itself reuses this file's own X_RANGE/Y_NEAR/Y_FAR/Z_LO/Z_HI/OBJ_SIZE/
# MIN_DIST_3D/ALL_COMBOS constants (already match triplet3ax_stimuli.py's values exactly
# -- same renderer, same camera).
SAMEAXIS_X_JITTER = _CFG["sameaxis"]["x_jitter"]
SAMEAXIS_X_JITTER_CLOSEFAR = _CFG["sameaxis"]["x_jitter_closefar"]  # wider -- closefar objects get
                                                                     # stratified X lanes, see sameaxis_closefar_lanes()
SAMEAXIS_Y_MID_LO, SAMEAXIS_Y_MID_HI = 4.3, 8.2
SAMEAXIS_MIN_GAP_AXIS = _CFG["sameaxis"]["min_gap_axis"]  # along tested axis
SAMEAXIS_D_GAP = _CFG["sameaxis"]["d_gap"]  # min gap from A/C when forcing the distractor outside [A,C]


def sameaxis_desc(shape, color):
    return f"{color} {shape}"


def sameaxis_dist3(p, q):
    return sum((a - b) ** 2 for a, b in zip(p, q)) ** 0.5


def sameaxis_sample_axis_scalars(axis, min_gap, rng):
    """3 sorted, sufficiently-separated scalars along the tested axis for A,B,C."""
    lo, hi = _bounds(axis)
    for _try in range(2000):
        vals = sorted(rng.uniform(lo, hi) for _ in range(3))
        if vals[1] - vals[0] >= min_gap and vals[2] - vals[1] >= min_gap:
            return vals
    raise RuntimeError(f"could not sample separated scalars for axis={axis}")


def sameaxis_world_pos(axis, scalar, rng):
    """Given the tested-axis scalar for one of A/B/C, sample the other 2 (jittered)
    world coords -> returns (x, y, z_base)."""
    if axis == "horizontal":
        return scalar, rng.uniform(SAMEAXIS_Y_MID_LO, SAMEAXIS_Y_MID_HI), 0.0
    if axis == "closefar":
        return rng.uniform(-SAMEAXIS_X_JITTER_CLOSEFAR, SAMEAXIS_X_JITTER_CLOSEFAR), scalar, 0.0
    if axis == "vertical":
        return rng.uniform(-SAMEAXIS_X_JITTER, SAMEAXIS_X_JITTER), rng.uniform(SAMEAXIS_Y_MID_LO, SAMEAXIS_Y_MID_HI), scalar
    raise ValueError(axis)


def sameaxis_sample_distractor(axis, rng, a_scalar, c_scalar):
    """Distractor's tested-axis coordinate is forced OUTSIDE [A,C] -- below A or above C,
    ~50/50 (falls back to whichever side has room)."""
    lo, hi = _bounds(axis)
    room_below = (a_scalar - SAMEAXIS_D_GAP) - lo
    room_above = hi - (c_scalar + SAMEAXIS_D_GAP)
    can_below, can_above = room_below > 0.05, room_above > 0.05
    if can_below and can_above:
        use_below = rng.random() < 0.5
    elif can_below:
        use_below = True
    elif can_above:
        use_below = False
    else:
        raise RuntimeError(f"no room to place distractor outside [A,C] for axis={axis}")
    tested = rng.uniform(lo, a_scalar - SAMEAXIS_D_GAP) if use_below else rng.uniform(c_scalar + SAMEAXIS_D_GAP, hi)

    if axis == "horizontal":
        return tested, rng.uniform(SAMEAXIS_Y_MID_LO, SAMEAXIS_Y_MID_HI), 0.0
    if axis == "closefar":
        return rng.uniform(-SAMEAXIS_X_JITTER_CLOSEFAR, SAMEAXIS_X_JITTER_CLOSEFAR), tested, 0.0
    if axis == "vertical":
        return rng.uniform(-SAMEAXIS_X_JITTER, SAMEAXIS_X_JITTER), rng.uniform(SAMEAXIS_Y_MID_LO, SAMEAXIS_Y_MID_HI), tested
    raise ValueError(axis)


def sameaxis_object_center(x, y, z_base, size):
    """draw_object's cz is the object's BASE, not center -- sphere center = cz+r,
    cube/cylinder/cone span [cz, cz+size]. size/2 approximates the center for all 4
    shapes closely enough for ground-truth purposes."""
    return (x, y, z_base + size / 2)


def sameaxis_closefar_lanes(rng):
    """4 well-separated X lanes + small jitter, shuffled across A/B/C/D -- prevents two
    objects from sharing nearly the same X (which perspective would stack visually)."""
    lanes = [-1.7, -1.7 + 3.4 / 3, -1.7 + 2 * 3.4 / 3, 1.7]
    rng.shuffle(lanes)
    return [lane + rng.uniform(-0.1, 0.1) for lane in lanes]


def sameaxis_build_scene(axis, idx, rng):
    min_gap = SAMEAXIS_MIN_GAP_AXIS[axis]
    for _attempt in range(200):
        scalars = sameaxis_sample_axis_scalars(axis, min_gap, rng)
        combos = rng.sample(ALL_COMBOS, 4)
        lanes = sameaxis_closefar_lanes(rng) if axis == "closefar" else None
        objs = []
        for i, (label, scalar) in enumerate(zip("ABC", scalars)):
            x, y, z_base = sameaxis_world_pos(axis, scalar, rng)
            if lanes is not None:
                x = lanes[i]
            objs.append({"label": label, "shape": combos[i][0], "color": combos[i][1],
                         "x": x, "y": y, "z_base": z_base, "scalar": scalar})
        try:
            dx, dy, dz_base = sameaxis_sample_distractor(axis, rng, scalars[0], scalars[2])
        except RuntimeError:
            continue
        if lanes is not None:
            dx = lanes[3]
        objs.append({"label": "D", "shape": combos[3][0], "color": combos[3][1],
                     "x": dx, "y": dy, "z_base": dz_base, "scalar": None})

        centers = [sameaxis_object_center(o["x"], o["y"], o["z_base"], OBJ_SIZE) for o in objs]
        ok = all(sameaxis_dist3(centers[i], centers[j]) >= MIN_DIST_3D
                 for i in range(4) for j in range(i + 1, 4))
        if ok:
            for o, c in zip(objs, centers):
                o["center"] = c
                o["desc"] = sameaxis_desc(o["shape"], o["color"])
            return objs
    raise RuntimeError(f"could not place non-overlapping scene axis={axis} idx={idx}")


def sameaxis_build_manifest():
    from pathlib import Path
    img_dir = Path(SAMEAXIS_IMAGES_DIR)
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for axis in AXES_BY_DATASET["triplet3ax"]:  # ["horizontal", "vertical", "closefar"]
        rng = random.Random(SAMEAXIS_SEED[axis])
        scenes_out = []
        for idx in range(SAMEAXIS_N_PER_AXIS):
            objs = sameaxis_build_scene(axis, idx, rng)
            fname = _render_scene(axis, idx, objs, img_dir)
            by_label = {o["label"]: o for o in objs}
            scenes_out.append({
                "image_path": fname, "axis": axis,
                "A": by_label["A"]["desc"], "B": by_label["B"]["desc"],
                "C": by_label["C"]["desc"], "D": by_label["D"]["desc"],
                "objects": {o["label"]: {"desc": o["desc"], "center": list(o["center"])} for o in objs},
            })
            if (idx + 1) % 15 == 0:
                print(f"[{axis}] {idx+1}/{SAMEAXIS_N_PER_AXIS} rendered")
        manifest[axis] = scenes_out
    with open(SAMEAXIS_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"saved -> {SAMEAXIS_MANIFEST_PATH}")
    return manifest


def sameaxis_load_or_build_manifest():
    return load_or_build_json(SAMEAXIS_MANIFEST_PATH, sameaxis_build_manifest)


# =============================================================================
# Real-world position ground truth + sign-corrected own-axis calibration -- shared by
# cross_axis_alignment.py and cross_axis_analysis.py, no new GPU work.
# SIGN MATTERS: _step() gives each leg a random +/- sign, so pooling without correcting
# lets opposite-signed legs cancel out (verified: sign-corrected mean vector's norm is
# 2-7x larger than uncorrected).
# =============================================================================
def clevr_real_positions():
    """{(config, idx): {"A": {"horizontal": val, "closefar": val}, "B": ..., "C": ...}} --
    camera-relative right/behind scalar position (CLEVR's 2 supported axes) for every
    cross-axis triplet, read directly from clevr_val_scenes.json."""
    with open(CLEVR_SCENES_PATH) as f:
        scenes = json.load(f)["scenes"]
    with open(CLEVR_TRIPLETS_PATH) as f:
        triplets_by_config = json.load(f)
    out = {}
    for config, triplets in triplets_by_config.items():
        for idx, t in enumerate(triplets):
            s = scenes[t["scene_index"]]
            objs = {clevr_desc(o): o for o in s["objects"]}
            right_dir, behind_dir = s["directions"]["right"], s["directions"]["behind"]
            pos = {}
            for label in "ABC":
                o = objs[t[label]]
                pos[label] = {"horizontal": dot(o["3d_coords"], right_dir),
                               "closefar": dot(o["3d_coords"], behind_dir)}
            out[(config, idx)] = pos
    return out


def t3ax_real_positions():
    """{(config, idx): {"A": {"horizontal": x, "closefar": y, "vertical": z_base}, "B":
    ..., "C": ...}} -- recomputed by replaying the same seeded RNG sequence
    t3ax_build_triplets() used at generation time (no image is re-rendered, only the
    coordinate arithmetic)."""
    out = {}
    for config in T3AX_CONFIGS:
        rng = random.Random(T3AX_SEED[config])
        for idx in range(T3AX_N_PER_CONFIG):
            objs, _, _, _ = t3ax_build_scene(config, idx, rng)
            by_label = {o["label"]: o for o in objs}
            out[(config, idx)] = {lab: {"horizontal": by_label[lab]["x"], "closefar": by_label[lab]["y"],
                                         "vertical": by_label[lab]["z_base"]} for lab in "ABC"}
    return out


def real_positions(dataset):
    return clevr_real_positions() if dataset == "clevr" else t3ax_real_positions()


def real_displacement_vec(pos, pair, axes):
    e1, e2 = pair[0], pair[1]
    return np.array([pos[e2][ax] - pos[e1][ax] for ax in axes])


def build_own_axes_signed(triplets, groups, real_pos):
    """Group-disjoint own-axis calibration, sign-corrected: flips each pure leg (r_AB for
    ax1, r_BC for ax2) by its real-world displacement sign before pooling, so
    opposite-signed legs reinforce instead of canceling out. Returns
    {group: {axis_name: (NUM_L,d) vector}}."""
    pools = {1: defaultdict(list), 2: defaultdict(list)}
    for t, g in zip(triplets, groups):
        pos = real_pos[(t["config"], t["idx"])]
        ab_disp = pos["B"][t["ax1"]] - pos["A"][t["ax1"]]
        bc_disp = pos["C"][t["ax2"]] - pos["B"][t["ax2"]]
        pools[g][t["ax1"]].append(t["r_AB"] * (1.0 if ab_disp > 0 else -1.0))
        pools[g][t["ax2"]].append(t["r_BC"] * (1.0 if bc_disp > 0 else -1.0))
    return {g: {ax: np.mean(vecs, axis=0) for ax, vecs in pools[g].items()} for g in [1, 2]}


# =============================================================================
# shared extraction (GPU) -- identical for both datasets
# =============================================================================
DATASET_CFG = {
    "clevr": dict(load_fn=clevr_load_or_sample, images_dir=CLEVR_IMAGES_DIR,
                  shard_out_base=f"{ap.RESULTS}/shards/clevr_cross_axis_shard.npz"),
    "triplet3ax": dict(load_fn=t3ax_load_or_build, images_dir=T3AX_IMAGES_DIR,
                        shard_out_base=f"{ap.RESULTS}/shards/triplet3ax_cross_axis_shard.npz"),
    "sameaxis": dict(load_fn=sameaxis_load_or_build_manifest, images_dir=SAMEAXIS_IMAGES_DIR,
                      shard_out_base=SAMEAXIS_SHARD_BASE),
}


def shard_out_path(dataset):
    """Model-suffixed shard path (see axis_pipeline.py's suffixed()) -- different models'
    extractions never clobber each other."""
    return ap.suffixed(DATASET_CFG[dataset]["shard_out_base"])


SAMEAXIS_AD_SHARD_BASE = f"{ap.RESULTS}/shards/sameaxis_ad_shard.npz"


def sameaxis_ad_shard_out_path():
    """r_AD (=h_D-h_A) shard -- the ONE extra pairwise vector sameaxis_4way_readout_vqa.py
    needs beyond the main shard's r_AB/r_AC (both already anchored at A), so all 4
    objects can be placed on one shared scalar line per layer: pos_A=0, pos_B=proj(r_AB),
    pos_C=proj(r_AC), pos_D=proj(r_AD)."""
    return ap.suffixed(SAMEAXIS_AD_SHARD_BASE)


@torch.inference_mode()
def extract_triplet(model, proc, img, A, B, C):
    hA_ab, hB_ab = ap.object_nodes_alllayers(model, proc, img, A, B)
    hB_bc, hC_bc = ap.object_nodes_alllayers(model, proc, img, B, C)
    hA_ac, hC_ac = ap.object_nodes_alllayers(model, proc, img, A, C)
    r_AB = hB_ab - hA_ab
    r_BC = hC_bc - hB_bc
    r_AC = hC_ac - hA_ac
    return r_AB, r_BC, r_AC


@torch.inference_mode()
def extract_ad(model, proc, img, A, D):
    hA, hD = ap.object_nodes_alllayers(model, proc, img, A, D)
    return hD - hA


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset", required=True, choices=DATASETS)
    cli.add_argument("--model", choices=ap.MODELS, default=ap.DEFAULT_MODEL)
    args = cli.parse_args()
    ap.set_model(args.model)

    cfg = DATASET_CFG[args.dataset]
    triplets = cfg["load_fn"]()

    all_jobs = []
    for config in triplets:  # dataset-agnostic: 2 configs for clevr, 6 for triplet3ax
        for idx, t in enumerate(triplets[config]):
            all_jobs.append((config, idx, t))
    print(f"[{args.dataset}] {len(all_jobs)} total triplets")

    model, proc = ap.load_model()

    results = {}
    img_cache = {}
    for i, (config, idx, t) in enumerate(all_jobs):
        path = os.path.join(cfg["images_dir"], t["image_path"])
        if path not in img_cache:
            img_cache[path] = Image.open(path).convert("RGB")
            if len(img_cache) > 4:
                img_cache.pop(next(iter(img_cache)))
        img = img_cache[path]
        r_AB, r_BC, r_AC = extract_triplet(model, proc, img, t["A"], t["B"], t["C"])
        results[f"{config}|{idx}"] = {"r_AB": r_AB, "r_BC": r_BC, "r_AC": r_AC}
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(all_jobs)} done")

    flat = {}
    for key, d in results.items():
        flat[f"{key}|r_AB"] = d["r_AB"]
        flat[f"{key}|r_BC"] = d["r_BC"]
        flat[f"{key}|r_AC"] = d["r_AC"]
    out_path = shard_out_path(args.dataset)
    np.savez(out_path, keys=list(results.keys()), **flat)
    print(f"saved -> {out_path}")

    if args.dataset == "sameaxis":
        # extra pass: r_AD, the one pairwise vector sameaxis_4way_readout_vqa.py needs
        # beyond the main shard (reuses the same loaded model, no reload needed)
        ad_results = {}
        for i, (config, idx, t) in enumerate(all_jobs):
            img = img_cache.get(os.path.join(cfg["images_dir"], t["image_path"]))
            if img is None:
                img = Image.open(os.path.join(cfg["images_dir"], t["image_path"])).convert("RGB")
            ad_results[f"{config}|{idx}"] = extract_ad(model, proc, img, t["A"], t["D"])
            if (i + 1) % 20 == 0:
                print(f"  [AD] {i+1}/{len(all_jobs)} done")
        ad_out_path = sameaxis_ad_shard_out_path()
        np.savez(ad_out_path, keys=list(ad_results.keys()),
                 **{f"{k}|r_AD": v for k, v in ad_results.items()})
        print(f"saved -> {ad_out_path}")
