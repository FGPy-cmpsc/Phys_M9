from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from optics import trace_through_system, Screen


@dataclass
class ImageResult:
    image: np.ndarray
    extent: tuple
    object_image: np.ndarray
    object_extent: tuple
    paths_xz: list
    n_rays_total: int
    n_rays_passed: int


def make_arrow_object(size: float, resolution: int):
    arrow = np.zeros((resolution, resolution))

    cx = resolution // 2
    shaft_w = max(1, resolution // 16)
    head_h = max(2, resolution // 4)
    shaft_top = head_h + 1
    shaft_bot = max(shaft_top + 2, 4 * resolution // 5)
    arrow[shaft_top:shaft_bot, cx-shaft_w:cx+shaft_w+1] = 1.0

    for i in range(head_h):
        w = i + 1
        row = shaft_top - head_h + i
        if 0 <= row < resolution:
            lo = max(0, cx - w)
            hi = min(resolution, cx + w + 1)
            arrow[row, lo:hi] = 1.0

    ys, xs = np.where(arrow > 0)
    x_coords = (xs - resolution / 2 + 0.5) / resolution * size
    y_coords = -(ys - resolution / 2 + 0.5) / resolution * size
    bright_pts = np.column_stack([x_coords, y_coords])
    return arrow, bright_pts


def sample_cone_directions(n_rays: int, max_angle: float, rng):
    cos_min = np.cos(max_angle)
    u = rng.uniform(cos_min, 1.0, n_rays)
    phi = rng.uniform(0, 2 * np.pi, n_rays)
    sin_t = np.sqrt(1 - u**2)
    dx = sin_t * np.cos(phi)
    dy = sin_t * np.sin(phi)
    dz = u
    return np.stack([dx, dy, dz], axis=1)


def form_image(object_points, elements, screen_z: float,
               rays_per_point: int, aperture_radius: float,
               first_element_z: float, image_res: int,
               image_extent: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    image = np.zeros((image_res, image_res))

    screen = Screen(z=screen_z)
    full_elements = list(elements) + [screen]

    max_angle = np.arctan(aperture_radius / max(first_element_z - 0.0, 1e-9))

    paths_xz_collected = []

    for pt_idx, (x_obj, y_obj) in enumerate(object_points):
        origins = np.tile([x_obj, y_obj, 0.0], (rays_per_point, 1))

        directions = sample_cone_directions(rays_per_point, max_angle, rng)

        first_axis_dir = np.array([0.0 - x_obj, 0.0 - y_obj, first_element_z])
        first_axis_dir /= np.linalg.norm(first_axis_dir)

        target_zaxis = first_axis_dir
        up = np.array([0.0, 0.0, 1.0])
        cosang = up @ target_zaxis
        if cosang > 1 - 1e-9:
            R = np.eye(3)
        elif cosang < -1 + 1e-9:
            R = np.diag([1.0, -1.0, -1.0])
        else:
            ax = np.cross(up, target_zaxis)
            ax /= np.linalg.norm(ax)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            s = np.sqrt(1 - cosang**2)
            R = np.eye(3) + s * K + (1 - cosang) * K @ K
        directions = directions @ R.T

        paths, final_dirs, valid = trace_through_system(origins, directions, full_elements)

        n_rays_valid = int(valid.sum())
        if n_rays_valid == 0:
            continue

        final_hits = paths[-1][valid]
        xi = final_hits[:, 0]
        yi = final_hits[:, 1]

        ix = ((xi + image_extent / 2) / image_extent * image_res).astype(int)
        iy = ((-yi + image_extent / 2) / image_extent * image_res).astype(int)
        mask = (ix >= 0) & (ix < image_res) & (iy >= 0) & (iy < image_res)
        np.add.at(image, (iy[mask], ix[mask]), 1.0)

        if pt_idx < 60:
            valid_idx = np.flatnonzero(valid)[:8]
            for k in valid_idx:
                xs_path = [p[k, 0] for p in paths]
                zs_path = [p[k, 2] for p in paths]
                if all(np.isfinite(xs_path)) and all(np.isfinite(zs_path)):
                    paths_xz_collected.append((zs_path, xs_path))

    return image, paths_xz_collected
