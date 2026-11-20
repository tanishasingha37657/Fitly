"""
Extract anthropometric measurements from a SMPL mesh (vertices array).

Approach: rather than hard-coding a handful of single vertex indices (which
vary in reliability across sources and are easy to get subtly wrong), this
uses horizontal cross-sections of the mesh at anatomically-approximate
height fractions of total stature, then measures either the perimeter of
that cross-section (for circumferences) or its width (for breadths).

This is a heuristic, standard technique in anthropometric mesh measurement,
but the height fractions below are approximate population averages, not
measured from your specific SMPL topology. Use `debug_plot_landmarks()` to
visually sanity-check the slices against a rendered mesh before trusting
numbers in your report, and adjust HEIGHT_FRACTIONS if a slice clearly
lands on the wrong body part.
"""
import numpy as np
from scipy.spatial import ConvexHull

# Approximate height fraction (from foot to point, as fraction of total
# stature) for each anatomical landmark used below. Rough values based on
# standard anthropometric proportion tables (e.g. bust/waist/hip location as
# fraction of height); calibrate against your own mesh if precision matters.
HEIGHT_FRACTIONS = {
    "chest": 0.72,     # bust/chest line
    "waist": 0.62,     # natural waist
    "hips": 0.53,      # widest point of pelvis
    "thigh": 0.45,     # upper thigh, just below hip crease
    "shoulder": 0.82,  # acromion / shoulder line
    "iliac_crest": 0.53,  # same band as hips, measured as width not perimeter
}

SLICE_THICKNESS_FRAC = 0.01  # thickness of the horizontal band to sample, as fraction of stature


def _get_stature_and_floor(vertices: np.ndarray):
    """SMPL's up-axis is Y. Returns (height, floor_y)."""
    floor_y = vertices[:, 1].min()
    top_y = vertices[:, 1].max()
    return (top_y - floor_y), floor_y


def _slice_at_height(vertices: np.ndarray, target_y: float, thickness: float) -> np.ndarray:
    """Return the subset of vertices within a thin horizontal band around target_y."""
    mask = np.abs(vertices[:, 1] - target_y) <= thickness / 2
    band = vertices[mask]
    if band.shape[0] < 3:
        # widen the band if too few vertices were captured
        mask = np.abs(vertices[:, 1] - target_y) <= thickness
        band = vertices[mask]
    return band


def _perimeter_at_height(vertices: np.ndarray, height_frac: float, stature: float, floor_y: float) -> float:
    """Circumference estimate: convex hull perimeter of the X-Z cross-section."""
    target_y = floor_y + height_frac * stature
    band = _slice_at_height(vertices, target_y, SLICE_THICKNESS_FRAC * stature)
    points_2d = band[:, [0, 2]]  # X-Z plane (horizontal cross-section)
    if points_2d.shape[0] < 3:
        raise ValueError(f"Not enough vertices to compute cross-section at height_frac={height_frac}")
    hull = ConvexHull(points_2d)
    hull_pts = points_2d[hull.vertices]
    perim = np.sum(np.linalg.norm(np.diff(np.vstack([hull_pts, hull_pts[0]]), axis=0), axis=1))
    return perim


def _width_at_height(vertices: np.ndarray, height_frac: float, stature: float, floor_y: float,
                      axis: int = 0) -> float:
    """Breadth estimate: max extent along `axis` (0=X, left-right) within the slice."""
    target_y = floor_y + height_frac * stature
    band = _slice_at_height(vertices, target_y, SLICE_THICKNESS_FRAC * stature)
    if band.shape[0] < 2:
        raise ValueError(f"Not enough vertices to compute width at height_frac={height_frac}")
    return band[:, axis].max() - band[:, axis].min()


def _estimate_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed-volume-of-mesh method (divergence theorem), assumes a closed watertight mesh."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1, v2)
    signed_vol = np.sum(np.einsum("ij,ij->i", v0, cross)) / 6.0
    return abs(signed_vol)


BODY_DENSITY_KG_PER_M3 = 985.0  # average human body density approximation


def extract_measurements(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """
    vertices: (6890, 3) numpy array of SMPL mesh vertices, in meters.
    faces: (F, 3) numpy array of SMPL face indices (needed only for weight/volume).

    Returns a dict with the 8 measurements. height/chest/waist/hips/thigh in cm,
    weight in kg.
    """
    stature, floor_y = _get_stature_and_floor(vertices)

    chest_m = _perimeter_at_height(vertices, HEIGHT_FRACTIONS["chest"], stature, floor_y)
    waist_m = _perimeter_at_height(vertices, HEIGHT_FRACTIONS["waist"], stature, floor_y)
    hips_m = _perimeter_at_height(vertices, HEIGHT_FRACTIONS["hips"], stature, floor_y)
    thigh_m = _perimeter_at_height(vertices, HEIGHT_FRACTIONS["thigh"], stature, floor_y)
    biacromial_m = _width_at_height(vertices, HEIGHT_FRACTIONS["shoulder"], stature, floor_y, axis=0)
    bicristal_m = _width_at_height(vertices, HEIGHT_FRACTIONS["iliac_crest"], stature, floor_y, axis=0)

    volume_m3 = _estimate_volume(vertices, faces)
    weight_kg = volume_m3 * BODY_DENSITY_KG_PER_M3

    return {
        "chest": chest_m * 100.0,
        "waist": waist_m * 100.0,
        "hips": hips_m * 100.0,
        "height": stature * 100.0,
        "weight": weight_kg,
        "biacromial": biacromial_m * 100.0,
        "bicristal": bicristal_m * 100.0,
        "thigh": thigh_m * 100.0,
    }


def debug_plot_landmarks(vertices: np.ndarray, save_path: str = None):
    """Quick 3D scatter plot marking the horizontal slices used, for visual sanity-checking."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    stature, floor_y = _get_stature_and_floor(vertices)
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(vertices[:, 0], vertices[:, 2], vertices[:, 1], s=1, alpha=0.15, color="gray")

    colors = {"chest": "red", "waist": "blue", "hips": "green", "thigh": "orange",
              "shoulder": "purple", "iliac_crest": "brown"}
    for name, frac in HEIGHT_FRACTIONS.items():
        target_y = floor_y + frac * stature
        band = _slice_at_height(vertices, target_y, SLICE_THICKNESS_FRAC * stature)
        ax.scatter(band[:, 0], band[:, 2], band[:, 1], s=8, color=colors.get(name, "black"), label=name)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y (up)")
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
