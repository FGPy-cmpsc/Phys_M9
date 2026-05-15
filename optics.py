from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ThinLens:
    z: float
    f: float
    aperture: float

    def trace(self, origins, directions):
        dz = self.z - origins[:, 2]
        t = dz / directions[:, 2]
        hit = origins + t[:, None] * directions
        x, y = hit[:, 0], hit[:, 1]
        r2 = x**2 + y**2
        valid = r2 <= self.aperture**2

        dx = directions[:, 0] - x * directions[:, 2] / self.f
        dy = directions[:, 1] - y * directions[:, 2] / self.f
        dz_new = directions[:, 2]
        new_dirs = np.stack([dx, dy, dz_new], axis=1)
        norm = np.linalg.norm(new_dirs, axis=1, keepdims=True)
        new_dirs = new_dirs / norm

        return hit, new_dirs, valid


@dataclass
class SphericalSurface:
    z_vertex: float
    radius: float
    aperture: float
    n_before: float
    n_after: float

    def _center(self):
        return np.array([0.0, 0.0, self.z_vertex + self.radius])

    def trace(self, origins, directions):
        C = self._center()
        oc = origins - C
        b = np.einsum('ij,ij->i', oc, directions)
        c = np.einsum('ij,ij->i', oc, oc) - self.radius**2
        disc = b**2 - c
        valid = disc > 0

        sq = np.sqrt(np.maximum(disc, 0.0))
        t1 = -b - sq
        t2 = -b + sq

        eps_t = 1e-12
        t = np.where(t1 > eps_t, t1, t2)
        valid &= (t > eps_t)

        hit = origins + t[:, None] * directions

        valid &= (hit[:, 0]**2 + hit[:, 1]**2) <= self.aperture**2

        normal = (hit - C) / np.abs(self.radius)
        cos_i = -np.einsum('ij,ij->i', directions, normal)
        flip = cos_i < 0
        normal = np.where(flip[:, None], -normal, normal)
        cos_i = np.where(flip, -cos_i, cos_i)

        n1, n2 = self.n_before, self.n_after
        eta = n1 / n2
        sin2_t = eta**2 * (1 - cos_i**2)
        tir = sin2_t > 1.0
        valid &= ~tir

        cos_t = np.sqrt(np.maximum(1 - sin2_t, 0.0))
        new_dirs = eta * directions + (eta * cos_i - cos_t)[:, None] * normal
        norm = np.linalg.norm(new_dirs, axis=1, keepdims=True)
        new_dirs = new_dirs / np.maximum(norm, 1e-30)

        return hit, new_dirs, valid


@dataclass
class ThickLens:
    z: float
    R1: float
    R2: float
    thickness: float
    n: float
    aperture: float

    def surfaces(self):
        s1 = SphericalSurface(
            z_vertex=self.z,
            radius=self.R1,
            aperture=self.aperture,
            n_before=1.0,
            n_after=self.n,
        )
        s2 = SphericalSurface(
            z_vertex=self.z + self.thickness,
            radius=self.R2,
            aperture=self.aperture,
            n_before=self.n,
            n_after=1.0,
        )
        return [s1, s2]

    def trace(self, origins, directions):
        s1, s2 = self.surfaces()
        pos1, dir1, v1 = s1.trace(origins, directions)
        pos2, dir2, v2 = s2.trace(pos1, dir1)
        return pos2, dir2, (v1 & v2)


@dataclass
class Screen:
    z: float

    def trace(self, origins, directions):
        dz = self.z - origins[:, 2]
        t = dz / directions[:, 2]
        hit = origins + t[:, None] * directions
        valid = t > 0
        return hit, directions, valid


def trace_through_system(origins, directions, elements):
    paths = [origins.copy()]
    valid = np.ones(len(origins), dtype=bool)
    cur_pos = origins
    cur_dir = directions
    for el in elements:
        new_pos, new_dir, new_valid = el.trace(cur_pos, cur_dir)
        valid &= new_valid
        cur_pos = new_pos
        cur_dir = new_dir
        paths.append(cur_pos.copy())
    return paths, cur_dir, valid


def thin_lens_image_distance(f: float, d_obj: float) -> float:
    if d_obj <= f:
        return float('inf')
    return 1.0 / (1.0 / f - 1.0 / d_obj)


def thin_lens_magnification(f: float, d_obj: float) -> float:
    d_img = thin_lens_image_distance(f, d_obj)
    if not np.isfinite(d_img):
        return float('inf')
    return -d_img / d_obj
