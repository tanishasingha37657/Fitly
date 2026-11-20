"""
Measurement function v2 (Stage 1).

Idea: define each circumference ONCE on a template SMPL mesh (mean shape, same
pose as the data), as a closed loop of mesh edges (each loop point sits on an
edge (a, b) at fraction t). For any other body with the same topology, the
loop is re-evaluated from that body's vertices and its length is the sum of
point-to-point distances -- a vertex-path curve length like BodyM's, immune to
arms and to thighs touching, and fast (no per-mesh slicing).

Landmark rules (all found on the template, none hard-coded fractions of height
except the small search windows below):
  crotch_y : lowest level where one loop spans both legs (x-range contains 0)
  axilla_y : highest level (below the neck) where arms form separate loops
  chest    : max girth of the torso loop just below the axilla
  hips     : max girth of the torso loop shortly above the crotch
  waist    : min girth of the torso loop between hips and chest
  thigh    : left-thigh loop just below the crotch (ONE thigh)
  biacromial (PROXY): straight distance between two shoulder-cap vertices, each
             the highest vertex within SHOULDER_RADIUS of the SMPL shoulder joint
             (needs joints). BodyM's shoulder-breadth definition is not published.
Heights/weights: height = head-to-heel extent; weight = volume * density.

Verify with plot_landmarks() before generating data.
"""
from collections import defaultdict
import numpy as np

BODY_DENSITY_KG_PER_M3 = 985.0

# search windows, as fractions of stature
CHEST_WINDOW = (0.08, 0.01)    # below axilla_y: [axilla - 0.08, axilla - 0.01]
HIPS_WINDOW = (0.03, 0.09)     # above crotch_y
WAIST_MIN_ABOVE_CROTCH = 0.10  # waist search starts here python -c "import pandas as pd; m=pd.read_csv('data/bodym/train/measurements.csv'); h=pd.read_csv('data/bodym/train/hwg_metadata.csv'); d=m.merge(h,on='subject_id'); print(d.groupby('gender')[['height','weight_kg','chest','waist','hip','thigh']].agg(['std','min','max']).round(1).T.to_string())"...
WAIST_BELOW_CHEST = 0.08       # ... and ends this far below the chest level
THIGH_BELOW_CROTCH = 0.06
SCAN_STEP = 0.0025             # level spacing (fraction of stature)
SHOULDER_RADIUS = 0.07         # metres around the shoulder joint searched for the shoulder cap
L_SHOULDER_JOINT, R_SHOULDER_JOINT = 16, 17


# ----------------------------------------------------------------------
# plane / mesh intersection
# ----------------------------------------------------------------------
def _slice_loops(v: np.ndarray, f: np.ndarray, y: float):
    """Closed loops of the mesh cut by the plane Y=y.
    Returns a list of (edges (K,2) int, t (K,) float), points = v[a] + t*(v[b]-v[a])."""
    above = v[:, 1] > y
    s = above[f]                                   # (F,3)
    mixed = s.any(1) & ~s.all(1)
    if not mixed.any():
        return []
    fm, sm = f[mixed], s[mixed]
    e0 = np.stack([fm[:, 0], fm[:, 1], fm[:, 2]], 1)   # edge i = (e0[i], e1[i])
    e1 = np.stack([fm[:, 1], fm[:, 2], fm[:, 0]], 1)
    cross = sm != np.stack([sm[:, 1], sm[:, 2], sm[:, 0]], 1)   # (n,3)
    rows, cols = np.nonzero(cross)
    a = e0[rows, cols].reshape(-1, 2)
    b = e1[rows, cols].reshape(-1, 2)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keys = [[(int(lo[i, k]), int(hi[i, k])) for k in range(2)] for i in range(len(lo))]

    adj = defaultdict(list)
    for i, (k1, k2) in enumerate(keys):
        adj[k1].append(i)
        adj[k2].append(i)

    used = [False] * len(keys)
    loops = []
    for i in range(len(keys)):
        if used[i]:
            continue
        start = keys[i][0]
        key, cur, loop = start, i, []
        closed = False
        while True:
            used[cur] = True
            k1, k2 = keys[cur]
            loop.append(key)
            key = k2 if k1 == key else k1
            if key == start:
                closed = True
                break
            nxt = [j for j in adj[key] if not used[j]]
            if not nxt:
                break
            cur = nxt[0]
        if closed and len(loop) >= 3:
            ed = np.array(loop, dtype=np.int64)
            ya, yb = v[ed[:, 0], 1], v[ed[:, 1], 1]
            t = (y - ya) / (yb - ya)
            loops.append((ed, t))
    return loops


def _loop_points(v, ed, t):
    return v[ed[:, 0]] + t[:, None] * (v[ed[:, 1]] - v[ed[:, 0]])


def _loop_length(v, ed, t):
    p = _loop_points(v, ed, t)
    return float(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1).sum())


def _stats(v, loop):
    p = _loop_points(v, *loop)
    return p[:, 0].min(), p[:, 0].max(), _loop_length(v, *loop)


def volume(vertices, faces):
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


# ----------------------------------------------------------------------
# landmark set
# ----------------------------------------------------------------------
class MeasurementSet:
    def __init__(self, loops, shoulder_vtx, levels, floor_y, stature):
        self.loops = loops                # name -> (edges, t)
        self.shoulder_vtx = shoulder_vtx  # (left_idx, right_idx)
        self.levels = levels              # name -> y on the template (for debugging)
        self.template_floor = floor_y
        self.template_stature = stature

    @classmethod
    def from_template(cls, tv: np.ndarray, faces: np.ndarray, joints: np.ndarray = None):
        tv = np.asarray(tv, dtype=np.float64)
        floor, top = tv[:, 1].min(), tv[:, 1].max()
        H = top - floor
        ys = floor + H * np.arange(0.02, 0.92, SCAN_STEP)

        central, nloops = {}, {}
        for y in ys:
            loops = _slice_loops(tv, faces, y)
            nloops[y] = len(loops)
            best = None
            for lp in loops:
                xmin, xmax, per = _stats(tv, lp)
                if xmin < 0 < xmax and (xmax - xmin) > 0.10 * H:
                    if best is None or per > best[0]:
                        best = (per, lp)
            if best is not None:
                central[y] = best

        if not central:
            raise RuntimeError("No torso loop found; is the template upright in Y?")
        crotch_y = min(central)
        # highest level below the neck where arms are separate loops
        cand = [y for y in ys if y > crotch_y and y < floor + 0.85 * H and nloops[y] >= 3]
        if not cand:
            raise RuntimeError("Could not find the armpit level; are the arms separated from the torso?")
        axilla_y = max(cand)

        def pick(lo, hi, fn):
            sel = [y for y in central if lo <= y <= hi]
            if not sel:
                raise RuntimeError(f"Empty search window [{lo:.3f},{hi:.3f}]")
            per = {y: central[y][0] for y in sel}
            return fn(per, key=per.get)

        chest_y = pick(axilla_y - CHEST_WINDOW[0] * H, axilla_y - CHEST_WINDOW[1] * H, max)
        hips_y = pick(crotch_y + HIPS_WINDOW[0] * H, crotch_y + HIPS_WINDOW[1] * H, max)
        waist_y = pick(crotch_y + WAIST_MIN_ABOVE_CROTCH * H, chest_y - WAIST_BELOW_CHEST * H, min)

        loops = {
            "chest": central[chest_y][1],
            "hips": central[hips_y][1],
            "waist": central[waist_y][1],
        }

        thigh_y = crotch_y - THIGH_BELOW_CROTCH * H
        cands = []
        for lp in _slice_loops(tv, faces, thigh_y):
            p = _loop_points(tv, *lp)
            if p[:, 0].mean() > 0.02:                      # left leg (+x)
                cands.append((_loop_length(tv, *lp), lp))
        if not cands:
            raise RuntimeError("Thigh loop not found")
        loops["thigh"] = max(cands, key=lambda c: c[0])[1]

        if joints is None:
            raise ValueError("joints are required to place the shoulder points")
        picks = []
        for jidx, sign in ((L_SHOULDER_JOINT, 1.0), (R_SHOULDER_JOINT, -1.0)):
            j = np.asarray(joints[jidx], dtype=np.float64)
            d = np.linalg.norm(tv - j, axis=1)
            sel = np.where((d < SHOULDER_RADIUS) & (sign * (tv[:, 0] - j[0]) > -0.02))[0]
            if len(sel) == 0:
                raise RuntimeError("No vertices near the shoulder joint")
            picks.append(int(sel[np.argmax(tv[sel, 1])]))
        shoulder = tuple(picks)

        levels = dict(crotch=crotch_y, axilla=axilla_y, chest=chest_y,
                      hips=hips_y, waist=waist_y, thigh=thigh_y)
        return cls(loops, shoulder, levels, floor, H)

    def measure(self, vertices: np.ndarray, faces: np.ndarray) -> dict:
        v = np.asarray(vertices, dtype=np.float64)
        out = {k: _loop_length(v, *lp) * 100.0 for k, lp in self.loops.items()}
        out["height"] = (v[:, 1].max() - v[:, 1].min()) * 100.0
        out["weight"] = volume(v, faces) * BODY_DENSITY_KG_PER_M3
        l, r = self.shoulder_vtx
        out["biacromial"] = float(np.linalg.norm(v[l] - v[r])) * 100.0
        return out


def plot_landmarks(ms: MeasurementSet, vertices: np.ndarray, save_path: str):
    """Front and side 2D views of the mesh with every measurement loop overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(chest="red", waist="blue", hips="green", thigh="orange")
    fig, axes = plt.subplots(1, 2, figsize=(9, 7))
    for ax, hx, title in zip(axes, (0, 2), ("front (x-y)", "side (z-y)")):
        ax.scatter(vertices[:, hx], vertices[:, 1], s=0.5, color="lightgray")
        for name, lp in ms.loops.items():
            p = _loop_points(vertices, *lp)
            ax.scatter(p[:, hx], p[:, 1], s=6, color=colors[name], label=name)
        if hx == 0:
            l, r = ms.shoulder_vtx
            ax.scatter(vertices[[l, r], 0], vertices[[l, r], 1], s=40, color="purple",
                       marker="x", label="biacromial pts")
        ax.set_aspect("equal")
        ax.set_title(title)
    axes[0].legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
