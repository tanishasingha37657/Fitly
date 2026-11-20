"""
Orthographic silhouette renderer + IoU against BodyM masks (Stage 4).

Conventions (SMPL, A-posed mesh, metres): y up, subject faces +z, subject's left is +x.
  front view : image x = +x   (viewer in front sees the subject's left on the image right)
  side view  : image x = +z   (BodyM side masks: subject faces RIGHT)
If the overlay check shows the side silhouette mirrored, set SIDE_FACES_RIGHT = False.

Scaling: each mask is scaled separately. Pixels-per-metre = (mask bbox height in px) / (height_cm / 100).
Alignment: feet (lowest vertex) on the bottom row of the mask bbox; horizontal shift so that the
centroid column of the predicted silhouette equals the mask centroid column.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

SIDE_FACES_RIGHT = True

# SMPL joint indices (pelvis = 0; body_pose covers joints 1..23)
L_HIP, R_HIP = 1, 2
L_COLLAR, R_COLLAR = 13, 14
L_SHOULDER, R_SHOULDER = 16, 17


def make_render_pose(arm_deg, leg_deg, collar_deg=0.0):
    """(69,) axis-angle pose used ONLY for drawing silhouettes.

    arm_deg    : angle of each arm away from the torso (T-pose = 90)
    leg_deg    : extra hip abduction per leg (negative = legs closer together)
    collar_deg : collar rotation, positive = shoulders lowered (same sign convention as the arms)
    SMPL betas do not depend on pose, so the render pose can differ from the pose used to
    generate the synthetic training data without affecting the beta predictions.
    """
    pose = np.zeros((23, 3), dtype=np.float32)
    drop = np.deg2rad(90.0 - arm_deg)
    pose[L_SHOULDER - 1, 2] = -drop
    pose[R_SHOULDER - 1, 2] = +drop
    c = np.deg2rad(collar_deg)
    pose[L_COLLAR - 1, 2] = -c
    pose[R_COLLAR - 1, 2] = +c
    s = np.deg2rad(leg_deg)
    pose[L_HIP - 1, 2] = +s
    pose[R_HIP - 1, 2] = -s
    return pose.reshape(-1)


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def mask_geometry(mask: np.ndarray):
    """(top_row, bottom_row, centroid_col) of a non-empty bool mask."""
    rows = np.flatnonzero(mask.any(1))
    cols = np.nonzero(mask)[1]
    return int(rows[0]), int(rows[-1]), float(cols.mean())


def _shift_x(img: np.ndarray, dx: int) -> np.ndarray:
    if dx == 0:
        return img
    out = np.zeros_like(img)
    if dx > 0:
        out[:, dx:] = img[:, :-dx]
    else:
        out[:, :dx] = img[:, -dx:]
    return out


def render_silhouette(verts: np.ndarray, faces: np.ndarray, view: str,
                      mask: np.ndarray, height_cm: float) -> np.ndarray:
    """Predicted-body silhouette (bool, same shape as mask), scaled and aligned to `mask`."""
    H, W = mask.shape
    top, bottom, cx = mask_geometry(mask)
    scale = (bottom - top + 1) / (height_cm / 100.0)          # px per metre

    if view == "front":
        u = verts[:, 0]
    elif view == "side":
        u = verts[:, 2] if SIDE_FACES_RIGHT else -verts[:, 2]
    else:
        raise ValueError(view)
    y = verts[:, 1]
    px = (u - u.mean()) * scale + W / 2.0
    py = (bottom + 1) - (y - y.min()) * scale
    pts = np.stack([px, py], 1)

    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for tri in pts[faces]:
        d.polygon(tri.reshape(-1).tolist(), fill=1)
    sil = np.array(img).astype(bool)
    if not sil.any():
        return sil
    dx = int(round(cx - np.nonzero(sil)[1].mean()))
    return _shift_x(sil, dx)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def overlay_image(mask: np.ndarray, sil: np.ndarray) -> Image.Image:
    """Mask only = red, prediction only = blue, both = white."""
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    rgb[mask & ~sil] = (230, 60, 60)
    rgb[sil & ~mask] = (60, 90, 230)
    rgb[mask & sil] = (255, 255, 255)
    return Image.fromarray(rgb)
