"""Shared matplotlib mplot3d renderer for every reproducible dataset in this folder
(aug1, triplet3ax, sameaxis). One camera, one ground grid, one set of primitives -- each
generate_*.py just picks object positions/shapes/colors and calls draw_object() +
save_scene().

No dependency outside matplotlib/numpy/pillow -- deliberately standalone (doesn't import
anything else from spatial/) so this folder can be read and run on its own.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from PIL import Image

# World extent: depth range objects can occupy, fixed axis limits so size in the
# rendered image is a genuine function of depth (not renormalized per-scene). Camera is
# a low, near-forward-facing angle looking along +Y.
Y_NEAR, Y_FAR = 2.5, 11.0
X_RANGE = 3.0
CAMERA_ELEV, CAMERA_AZIM = 12, -80

SHAPES = ["sphere", "cube", "cylinder", "cone"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]  # order matters: consumed by
                                                                  # rng.sample(COLORS, ...) below
_COLOR_RGB = {
    "red": (0.86, 0.20, 0.20), "blue": (0.20, 0.35, 0.86),
    "green": (0.20, 0.63, 0.27), "yellow": (0.90, 0.78, 0.16),
    "purple": (0.59, 0.24, 0.71), "orange": (0.94, 0.55, 0.12),
}


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
    """exclude_objects: [(cx, cy, cz, radius), ...] -- grid segments that would visually
    cross one of these (2D screen-projected, so it correctly covers elevated/tall objects
    too) are skipped, so the grid never draws over an object."""
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
    """(cx, cy, cz) is the object's BASE (ground-contact point), not its center."""
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


def new_scene_figure(exclude_objects=None, computed_zorder=True):
    """computed_zorder=False switches to manual draw order (later draw_object() calls
    paint over earlier ones unconditionally) instead of automatic per-artist depth
    sorting -- needed for aug1's zero-gap touching vertical stack, where automatic
    sorting can clip the wrong object regardless of gap size."""
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = computed_zorder
    ax.set_proj_type("persp", focal_length=0.15)
    ax.set_xlim(-X_RANGE, X_RANGE)
    ax.set_ylim(Y_NEAR - 1, Y_FAR + 1)
    ax.set_zlim(0, 3.2)
    ax.view_init(elev=CAMERA_ELEV, azim=CAMERA_AZIM)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.5))
    draw_ground_grid(ax, exclude_objects=exclude_objects)
    return fig, ax


def save_scene(fig, path, canvas_size=512, pad_frac=0.12):
    """Renders, then tightly crops to content (with padding) and resizes to a fixed
    square -- keeps every scene the same canvas size regardless of how much of the
    mplot3d figure the objects actually occupy."""
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


def render_multiobject_scene(objs, obj_size, path):
    """Shared by triplet3ax/sameaxis: draw every object in `objs` (each a dict with
    shape/color/x/y/z_base) into one new scene and save it to `path`. Ground-grid
    exclusion radius (obj_size/1.6) is a plain footprint circle, looser than aug1's
    per-object exclusion since these scenes never have a touching stack."""
    exclude = [(o["x"], o["y"], 0.0, obj_size / 1.6) for o in objs]
    fig, ax = new_scene_figure(exclude_objects=exclude)
    for o in objs:
        draw_object(ax, o["shape"], o["color"], o["x"], o["y"], o["z_base"], obj_size)
    save_scene(fig, str(path))
