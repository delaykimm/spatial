"""Compositionality: does r_AB + r_BC ~= r_AC hold in the model's hidden states? Two views:

  (A) per-config violin plot: cos(r_AB+r_BC, r_AC) at one fixed --layer, split by config.
  (B) cross-representation permutation test, all layers, projected onto 6 spaces: raw (no
      projection), own axis (this dataset's group-disjoint sign-corrected pure legs), and
      each of the 4 external datasets' axes. K=1000 derangement permtest per rep; the 5
      projected ones also get a K_RAND=500 random-2D-subspace control (projecting onto ANY
      2D subspace inflates cosine on its own -- z_rand is how far above that baseline the
      real axis sits, the honest "is this axis actually special" number).

Run first:
    python triplet_pipeline.py --dataset {dataset} --model {model}
    python axis_pipeline.py --dataset {whatsup,spatialtunnel,aug1,aug2} --model {model}  # needed for view B

Usage:
    python cross_axis_analysis.py --dataset clevr
    python cross_axis_analysis.py --dataset triplet3ax --model llava --layer 20
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
from collections import defaultdict

import numpy as np

import triplet_pipeline as cxp
import axis_pipeline as ap

_CFG = ap.CFG["cross_axis_analysis"]
DEFAULT_LAYER = _CFG["default_layer"]
K = _CFG["k_permtest"]
SEED = _CFG["seed"]
K_RAND = _CFG["k_rand"]
SEED_RAND = _CFG["seed_rand"]
AXIS_KEY = {"horizontal": "x", "vertical": "y", "closefar": "z"}  # axis_pipeline's internal axis keys
EXTERNAL_DATASETS = ["whatsup", "spatialtunnel", "aug1", "aug2"]
CLEVR_AXIS_PAIRS = {"horiz_then_z": ("horizontal", "closefar"), "z_then_horiz": ("closefar", "horizontal")}
ALL_AXES_T3AX = ["horizontal", "vertical", "closefar"]  # triplet3ax uniquely has all 3 (CLEVR has no vertical)
# config name -> "A->B axis THEN B->C axis" display label, for the per-config plot's x-axis
CONFIG_LABELS = {
    "horiz_then_z": "horizontal → close/far", "z_then_horiz": "close/far → horizontal",
    "horiz_then_vert": "horizontal → vertical", "vert_then_horiz": "vertical → horizontal",
    "vert_then_z": "vertical → close/far", "z_then_vert": "close/far → vertical",
}

unit = ap.unit             # canonical L2-normalize now lives in axis_pipeline.py
report_layers = ap.report_layers  # ditto for the 8-evenly-spaced-layers helper


def load_shard(dataset):
    return np.load(cxp.shard_out_path(dataset), allow_pickle=True)


# =============================================================================
# (A) per-config summary + violin plot
# =============================================================================
def composition_cosines(dataset, layer):
    """Returns {config: [cos(r_AB+r_BC, r_AC), ...]} at the given layer, one value per
    triplet, grouped by config (2 for clevr, 6 for triplet3ax)."""
    d = load_shard(dataset)
    by_config = {}
    for k in d["keys"]:
        k = str(k)
        config = k.split("|")[0]
        r_ab = d[f"{k}|r_AB"][layer]
        r_bc = d[f"{k}|r_BC"][layer]
        r_ac = d[f"{k}|r_AC"][layer]
        cos = float(np.dot(unit(r_ab + r_bc), unit(r_ac)))
        by_config.setdefault(config, []).append(cos)
    return by_config


def own_axis_config_cos(dataset, layer, triplets=None, groups=None, own_axes=None):
    """Group-disjoint own-axis cos(r_AB+r_BC, r_AC) per triplet at `layer`, by config.
    own_axis_2d = this config's 2 tested axes; own_axis_3d (triplet3ax only) = all 3.
    Returns {config: {'own_axis_2d': [...], 'own_axis_3d': [...]}}.
    triplets/groups/own_axes: pass through if the caller already has them (avoids
    recomputing the same group split twice -- see __main__)."""
    triplets = triplets if triplets is not None else load_all_triplets(dataset)
    groups = groups if groups is not None else group_split(len(triplets))
    own_axes = own_axes if own_axes is not None else build_own_axes(triplets, groups, dataset)
    num_l = triplets[0]["r_AB"].shape[0]

    by_bucket = defaultdict(list)
    for i, (t, g) in enumerate(zip(triplets, groups)):
        by_bucket[(t["config"], g)].append(i)

    out = defaultdict(dict)
    for (config, g), idxs in by_bucket.items():
        other = 2 if g == 1 else 1
        ax1, ax2 = triplets[idxs[0]]["ax1"], triplets[idxs[0]]["ax2"]
        pred = np.stack([(triplets[i]["r_AB"] + triplets[i]["r_BC"])[layer] for i in idxs])
        targ = np.stack([triplets[i]["r_AC"][layer] for i in idxs])

        Q2 = basis_2d(own_axes[other][ax1], own_axes[other][ax2], num_l)[layer]
        cos2 = np.sum(unit(pred @ Q2) * unit(targ @ Q2), axis=-1)
        out[config].setdefault("own_axis_2d", []).extend(cos2.tolist())

        if dataset == "triplet3ax":
            vecs = [own_axes[other][ax] for ax in ALL_AXES_T3AX]
            Q3 = basis_nd(vecs, num_l)[layer]
            cos3 = np.sum(unit(pred @ Q3) * unit(targ @ Q3), axis=-1)
            out[config].setdefault("own_axis_3d", []).extend(cos3.tolist())
    return out


def print_config_summary(by_config, own_axis_by_config, dataset, layer, model):
    print(f"\n=== [{model}] {dataset} cross-axis compositionality: cos(r_AB+r_BC, r_AC) @ layer {layer} ===")
    for config, vals in by_config.items():
        vals = np.array(vals)
        print(f"  {config:18s} [raw]          n={len(vals):3d}  mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")
        for rep in ["own_axis_2d", "own_axis_3d"]:
            if rep in own_axis_by_config.get(config, {}):
                v = np.array(own_axis_by_config[config][rep])
                print(f"  {config:18s} [{rep:11s}] n={len(v):3d}  mean={v.mean():.3f}  std={v.std():.3f}  "
                      f"min={v.min():.3f}  max={v.max():.3f}")


def draw_config_plot(by_config, own_axis_by_config, dataset, layer, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, SECONDARY, MUTED, GRID = "#ffffff", "#1a1a19", "#52514e", "#898781", "#e5e4df"
    REP_COLORS = {"raw": "#2a78d6", "own_axis_2d": "#d6852a", "own_axis_3d": "#2ab06a"}
    REP_LABELS = {"raw": "raw (full-dim)", "own_axis_2d": "own axis (2D)", "own_axis_3d": "own axis (3D)(+vertical)"}

    configs = list(by_config.keys())
    has_3d = any("own_axis_3d" in own_axis_by_config.get(c, {}) for c in configs)
    reps = ["raw", "own_axis_2d"] + (["own_axis_3d"] if has_3d else [])
    n_reps = len(reps)
    width = 0.8 / n_reps

    fig, ax = plt.subplots(figsize=(2.3 * len(configs) + 2, 6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    rng = np.random.default_rng(0)
    for ri, rep in enumerate(reps):
        offset = (ri - (n_reps - 1) / 2) * width
        positions = [i + offset for i in range(len(configs))]
        data = [np.array(by_config[c]) if rep == "raw" else np.array(own_axis_by_config[c][rep])
                for c in configs]
        color = REP_COLORS[rep]
        parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=True, widths=width * 0.9)
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_alpha(0.5); pc.set_edgecolor(color)
        for key in ["cbars", "cmins", "cmaxes", "cmeans"]:
            parts[key].set_color(SECONDARY); parts[key].set_linewidth(1.0)
        for pos, vals in zip(positions, data):
            jitter = rng.uniform(-width * 0.28, width * 0.28, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=6, color=color, alpha=0.3, zorder=2)

    ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (2, 2)), alpha=0.6, zorder=1)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs], color=INK, fontsize=9.5, rotation=20, ha="right")
    ax.set_ylabel("cos(r_AB + r_BC, r_AC)", color=INK, fontsize=10.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"[{model}] {dataset}: cross-axis compositionality per config (layer {layer})",
                 color=INK, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=SECONDARY, length=0)

    handles = [plt.Line2D([0], [0], color=REP_COLORS[r], lw=6, alpha=0.6, label=REP_LABELS[r]) for r in reps]
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=False, labelcolor=INK)

    fig.subplots_adjust(bottom=0.22, top=0.9, left=0.09, right=0.97)
    out_path = f"{ap.PLOTS}/alignment/{dataset}_compositionality_by_config_{model}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


# =============================================================================
# (B) cross-representation permutation test + random-2D-subspace control
# =============================================================================
def load_all_triplets(dataset):
    """Returns list of dicts: {config, idx, ax1, ax2, r_AB, r_BC, r_AC (each (NUM_L,d))} --
    ax1 is the axis r_AB's leg is pure along, ax2 is r_BC's. (config, idx) lets callers
    look up ground truth via triplet_pipeline.real_positions()."""
    d = load_shard(dataset)
    axis_pairs = CLEVR_AXIS_PAIRS if dataset == "clevr" else cxp.AXIS_PAIRS
    triplets = []
    for k in d["keys"]:
        k = str(k)
        config, idx = k.split("|")
        ax1, ax2 = axis_pairs[config]
        triplets.append(dict(config=config, idx=int(idx), ax1=ax1, ax2=ax2,
                              r_AB=d[f"{k}|r_AB"], r_BC=d[f"{k}|r_BC"], r_AC=d[f"{k}|r_AC"]))
    return triplets


def group_split(n, seed=SEED):
    """1-indexed group (1 or 2) per triplet, disjoint halves."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    half = n // 2
    g1 = set(order[:half].tolist())
    return [1 if i in g1 else 2 for i in range(n)]


def build_own_axes(triplets, groups, dataset):
    """{group: {axis_name: (NUM_L,d) mean-pure-leg vector}} -- sign-corrected pooling of
    each group's pure legs (see triplet_pipeline.build_own_axes_signed for why the sign
    correction matters)."""
    real_pos = cxp.real_positions(dataset)
    return cxp.build_own_axes_signed(triplets, groups, real_pos)


def basis_nd(vecs, num_l):
    """vecs: list of k (NUM_L,d) raw (non-unit) direction vectors -> (NUM_L,d,k) per-layer
    QR-orthonormalized basis. Generalizes basis_2d to any number of axes (k=3 for
    triplet3ax's own_axis_3d, which pools all 3 known spatial axes at once)."""
    k = len(vecs)
    Q = np.zeros((num_l, vecs[0].shape[-1], k))
    for L in range(num_l):
        A = np.stack([unit(v[L]) for v in vecs], axis=1)
        Qd, Rd = np.linalg.qr(A)
        signs = np.sign(np.diag(Rd)); signs[signs == 0] = 1.0
        Q[L] = Qd * signs
    return Q


def basis_2d(axis_a_vec, axis_b_vec, num_l):
    """axis_a_vec/axis_b_vec: (NUM_L,d) raw (non-unit) direction vectors -> (NUM_L,d,2)
    per-layer QR-orthonormalized basis."""
    return basis_nd([axis_a_vec, axis_b_vec], num_l)


def project_batch(V, Q):
    """V: (N,NUM_L,d); Q: (NUM_L,d,2) -> (N,NUM_L,2)."""
    return np.einsum("nld,ldk->nlk", V, Q)


def norm_resid(S, target):
    return np.linalg.norm(S - target, axis=-1) / (np.linalg.norm(target, axis=-1) + 1e-9)


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def cosine_permtest(model_unit, real_unit, rng, K):
    """model_unit (N,num_l,k) vs real_unit (N,k) -> per-layer cos + K-derangement permtest
    z-score/p-value. Shared by cross_axis_alignment.py's run_alignment (own-axis) and
    run_alignment_datasetaxis (external axes) -- same test, different axis source."""
    N, num_l, _ = model_unit.shape
    cos_obs = np.einsum("nlk,nk->nl", model_unit, real_unit)
    obs_mean = cos_obs.mean(axis=0)

    null = np.zeros((K, num_l))
    for kk in range(K):
        perm = derangement(N, rng)
        null[kk] = np.einsum("nlk,nk->nl", model_unit, real_unit[perm]).mean(axis=0)
    z = (obs_mean - null.mean(0)) / (null.std(0) + 1e-12)
    p = (np.sum(null >= obs_mean[None, :], axis=0) + 1) / (K + 1)
    return dict(cos_obs=cos_obs, obs_mean=obs_mean, z=z, p=p)


def composition_battery(pred, target, rng, num_l):
    """pred, target: (N,NUM_L,d) for any d. Returns cos/resid arrays + K-derangement
    permtest z-score and empirical p-value per layer."""
    N = pred.shape[0]
    U, Tu = unit(pred), unit(target)
    cos = np.sum(U * Tu, axis=-1)          # (N,NUM_L)
    resid = norm_resid(pred, target)        # (N,NUM_L)

    null_cos = np.zeros((K, num_l))
    null_resid = np.zeros((K, num_l))
    for k in range(K):
        perm = derangement(N, rng)
        null_cos[k] = np.sum(U * Tu[perm], axis=-1).mean(axis=0)
        null_resid[k] = norm_resid(pred, target[perm]).mean(axis=0)

    obs_cos = cos.mean(axis=0)
    z_cos = (obs_cos - null_cos.mean(0)) / (null_cos.std(0) + 1e-12)
    p_cos = (np.sum(null_cos >= obs_cos[None, :], axis=0) + 1) / (K + 1)

    obs_resid = resid.mean(axis=0)
    z_resid = (null_resid.mean(0) - obs_resid) / (null_resid.std(0) + 1e-12)
    p_resid = (np.sum(null_resid <= obs_resid[None, :], axis=0) + 1) / (K + 1)

    return dict(cos=cos, resid=resid, z_cos=z_cos, p_cos=p_cos, z_resid=z_resid, p_resid=p_resid)


def random_2d_control(pred, target, rng, num_l, k=K_RAND):
    """pred, target: (N,NUM_L,d). Random 2D subspace per layer per trial -- how much
    cosine does projecting onto a totally arbitrary 2D direction give, just from the
    fewer-degrees-of-freedom effect? Returns (mean, std) each (NUM_L,)."""
    N, _, D = pred.shape
    trials = np.zeros((k, num_l))
    for t in range(k):
        for L in range(num_l):
            G = rng.standard_normal((D, 2))
            Q, _ = np.linalg.qr(G)
            p2 = pred[:, L, :] @ Q
            t2 = target[:, L, :] @ Q
            trials[t, L] = np.sum(unit(p2) * unit(t2), axis=-1).mean()
    return trials.mean(axis=0), trials.std(axis=0)


def add_random_control(rep_result, rand_mean, rand_std):
    """Mutates rep_result in place, adding rand_mean/rand_std/z_rand (how much of this
    2D-projected representation's cosine is above the random-2D-subspace baseline, i.e.
    genuine axis-specific signal rather than the generic low-dimensionality inflation)."""
    obs = rep_result["cos"].mean(axis=0)
    rep_result["rand_mean"] = rand_mean
    rep_result["rand_std"] = rand_std
    rep_result["z_rand"] = (obs - rand_mean) / (rand_std + 1e-12)


def external_axis_vector(ext_ds, axis_name):
    """(NUM_L,d) plus-minus diff for axis_name from an external spatial dataset
    (whatsup/spatialtunnel/aug1/aug2), via axis_pipeline.py's DATASET_CONFIGS -- for the
    model currently active (ap.set_model()) so it matches the triplet shard's model."""
    cfg = ap.DATASET_CONFIGS[ext_ds]
    out_name = cfg["axis_names"][AXIS_KEY[axis_name]]
    d = np.load(ap.axis_out_path(ext_ds))
    return d[f"{out_name}_{cfg['plus_key']}"] - d[f"{out_name}_{cfg['minus_key']}"]


def run_permtest_battery(dataset, model, triplets=None, groups=None, own_axes=None):
    """triplets/groups/own_axes: pass these through if the caller already has them (see
    own_axis_config_cos's docstring -- same reasoning, same SEED)."""
    triplets = triplets if triplets is not None else load_all_triplets(dataset)
    N = len(triplets)
    num_l = triplets[0]["r_AB"].shape[0]
    groups = groups if groups is not None else group_split(N)
    own_axes = own_axes if own_axes is not None else build_own_axes(triplets, groups, dataset)
    print(f"[{model}][{dataset}] N={N} cross-axis triplets, group sizes: "
          f"{groups.count(1)}/{groups.count(2)}, num_layers={num_l}")

    rng = np.random.default_rng(SEED)
    pred_raw = np.stack([t["r_AB"] + t["r_BC"] for t in triplets])
    targ_raw = np.stack([t["r_AC"] for t in triplets])

    results = {}
    print("\n[1/4] raw (full-dim)...")
    results["raw"] = composition_battery(pred_raw, targ_raw, rng, num_l)

    print("[2/4] own axis (group-disjoint)...")
    pred_own = np.zeros((N, num_l, 2)); targ_own = np.zeros((N, num_l, 2))
    by_bucket = defaultdict(list)
    for i, (t, g) in enumerate(zip(triplets, groups)):
        by_bucket[(t["config"], g)].append(i)
    for (config, g), idxs in by_bucket.items():
        ax1, ax2 = triplets[idxs[0]]["ax1"], triplets[idxs[0]]["ax2"]
        other = 2 if g == 1 else 1
        Q = basis_2d(own_axes[other][ax1], own_axes[other][ax2], num_l)
        pred_own[idxs] = project_batch(pred_raw[idxs], Q)
        targ_own[idxs] = project_batch(targ_raw[idxs], Q)
    results["own_axis"] = composition_battery(pred_own, targ_own, rng, num_l)

    print("[3/4] 4 external dataset axes...")
    by_config = defaultdict(list)
    for i, t in enumerate(triplets):
        by_config[t["config"]].append(i)
    for ext_ds in EXTERNAL_DATASETS:
        pred_ext = np.zeros((N, num_l, 2)); targ_ext = np.zeros((N, num_l, 2))
        for config, idxs in by_config.items():
            ax1, ax2 = triplets[idxs[0]]["ax1"], triplets[idxs[0]]["ax2"]
            Q = basis_2d(external_axis_vector(ext_ds, ax1), external_axis_vector(ext_ds, ax2), num_l)
            pred_ext[idxs] = project_batch(pred_raw[idxs], Q)
            targ_ext[idxs] = project_batch(targ_raw[idxs], Q)
        results[ext_ds] = composition_battery(pred_ext, targ_ext, rng, num_l)

    print("\n[4/4] random 2D-subspace control (is the real axis better than an arbitrary one?)...")
    rand_rng = np.random.default_rng(SEED_RAND)
    rand_mean, rand_std = random_2d_control(pred_raw, targ_raw, rand_rng, num_l)
    add_random_control(results["own_axis"], rand_mean, rand_std)
    for ext_ds in EXTERNAL_DATASETS:
        add_random_control(results[ext_ds], rand_mean, rand_std)

    return results, num_l


def print_permtest_summary(results, dataset, model, num_l):
    rep_names = ["raw", "own_axis"] + EXTERNAL_DATASETS
    rand_rep_names = ["own_axis"] + EXTERNAL_DATASETS
    layers = report_layers(num_l)
    header = f"{'layer':>5} " + " ".join(f"{r:>12}" for r in rep_names)
    rand_header = f"{'layer':>5} " + " ".join(f"{r:>12}" for r in rand_rep_names)

    print(f"\n{'='*100}\n[{model}][{dataset}] SUMMARY: mean cos(r_AB+r_BC, r_AC), by representation and layer\n{'='*100}")
    print(header)
    for L in layers:
        print(f"{L:5d} " + " ".join(f"{results[r]['cos'][:, L].mean():12.3f}" for r in rep_names))

    print(f"\n{'='*100}\n[{model}][{dataset}] SUMMARY: permutation-test z-score (cos), by representation and layer\n{'='*100}")
    print(header)
    for L in layers:
        print(f"{L:5d} " + " ".join(f"{results[r]['z_cos'][L]:12.2f}" for r in rep_names))

    print(f"\n{'='*100}\n[{model}][{dataset}] SUMMARY: random-2D-subspace-control z-score (z_rand), by representation and layer\n"
          f"(is this axis's cosine above what an arbitrary random 2D direction gives? raw has no 2D "
          f"projection so it's excluded here)\n{'='*100}")
    print(rand_header)
    for L in layers:
        print(f"{L:5d} " + " ".join(f"{results[r]['z_rand'][L]:12.2f}" for r in rand_rep_names))


def draw_comparison_plot(results, dataset, model, num_l):
    """Two-panel line plot across all layers: top = mean cos(r_AB+r_BC, r_AC) per
    representation (raw/own_axis/4 external datasets); bottom = z_rand (how far above the
    random-2D-subspace baseline each 2D-projected representation is -- raw excluded, it has
    no 2D projection). Saved as {dataset}_compositionality_by_layer_{model}.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, SECONDARY, MUTED, GRID = "#ffffff", "#1a1a19", "#52514e", "#898781", "#e5e4df"
    # per-dataset colors match cross_axis_alignment.py's draw_datasetaxis_comparison_plot,
    # so the same dataset reads as the same color across every plot in this pipeline.
    # "raw" matches ours/binding's plot_cross_axis_model_comparison.py.
    COLORS = {"raw": "#2a78d6", "own_axis": "#eb6834", "whatsup": "#1baf7a",
              "spatialtunnel": "#eda100", "aug1": "#e87ba4", "aug2": "#008300"}

    rep_names = ["raw", "own_axis"] + EXTERNAL_DATASETS
    rand_rep_names = ["own_axis"] + EXTERNAL_DATASETS
    layers = np.arange(num_l)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), dpi=150, facecolor=SURFACE, sharex=True)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=SECONDARY, length=0)

    for r in rep_names:
        cos_per_layer = results[r]["cos"].mean(axis=0)
        ax1.plot(layers, cos_per_layer, color=COLORS[r], linewidth=2, label=r, zorder=3)
    ax1.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (2, 2)), alpha=0.6, zorder=1)
    ax1.set_ylabel("mean cos(r_AB+r_BC, r_AC)", color=INK, fontsize=10.5)
    ax1.set_ylim(-0.1, 1.05)
    ax1.set_title(f"[{model}] {dataset}: cross-axis compositionality by representation, across layers",
                  color=INK, fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=INK)

    for r in rand_rep_names:
        ax2.plot(layers, results[r]["z_rand"], color=COLORS[r], linewidth=2, label=r, zorder=3)
    ax2.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (2, 2)), alpha=0.6, zorder=1)
    ax2.set_xlabel("layer", color=INK, fontsize=10.5)
    ax2.set_ylabel("z_rand (vs. random 2D subspace)", color=INK, fontsize=10.5)
    ax2.set_title("genuine axis-specific signal above the random-2D-projection baseline",
                  color=INK, fontsize=11, pad=8)

    fig.subplots_adjust(bottom=0.07, top=0.93, left=0.09, right=0.97, hspace=0.18)
    out_path = f"{ap.PLOTS}/alignment/{dataset}_compositionality_by_layer_{model}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset", required=True, choices=cxp.DATASETS)
    cli.add_argument("--model", choices=ap.MODELS, default=ap.DEFAULT_MODEL)
    cli.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                      help="fixed layer for view (A)'s per-config violin plot (ignored for view B, which sweeps every layer)")
    args = cli.parse_args()
    ap.set_model(args.model)

    # shared by both views below -- same SEED, so compute the group split + own-axis
    # calibration once instead of view A and view B each deriving their own
    triplets = load_all_triplets(args.dataset)
    groups = group_split(len(triplets))
    own_axes = build_own_axes(triplets, groups, args.dataset)

    # (A) per-config summary + violin plot
    by_config = composition_cosines(args.dataset, args.layer)
    own_axis_by_config = own_axis_config_cos(args.dataset, args.layer, triplets, groups, own_axes)
    print_config_summary(by_config, own_axis_by_config, args.dataset, args.layer, args.model)
    draw_config_plot(by_config, own_axis_by_config, args.dataset, args.layer, args.model)

    # (B) cross-representation permutation test + random-2D-subspace control
    results, num_l = run_permtest_battery(args.dataset, args.model, triplets, groups, own_axes)
    print_permtest_summary(results, args.dataset, args.model, num_l)

    out_path = ap.suffixed(f"{ap.RESULTS}/alignment/{args.dataset}_cross_axis_datasetaxis_permtest.npz")
    flat = {f"{r}_{k}": v for r, m in results.items() for k, v in m.items()}
    np.savez(out_path, **flat)
    print(f"\nsaved -> {out_path}")

    draw_comparison_plot(results, args.dataset, args.model, num_l)
