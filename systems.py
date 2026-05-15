from __future__ import annotations
import numpy as np

from config import SimulationConfig
from optics import ThinLens, ThickLens, thin_lens_image_distance


def build_system(cfg: SimulationConfig):
    if cfg.system == 'thin':
        return _build_thin(cfg)
    if cfg.system == 'microscope':
        return _build_microscope(cfg)
    if cfg.system == 'thick':
        return _build_thick(cfg)
    raise ValueError(f'Unknown system: {cfg.system}')


def _build_thin(cfg: SimulationConfig):
    z_lens = cfg.object_distance
    lens = ThinLens(z=z_lens, f=cfg.lens_f, aperture=cfg.lens_aperture)

    d_img = thin_lens_image_distance(cfg.lens_f, cfg.object_distance)
    if cfg.screen_distance > 0:
        z_screen = z_lens + cfg.screen_distance
    else:
        z_screen = z_lens + d_img

    mag = -d_img / cfg.object_distance

    info = {
        'd_obj': cfg.object_distance,
        'd_img_theory': d_img,
        'magnification_theory': mag,
        'first_element_z': z_lens,
        'first_aperture': cfg.lens_aperture,
        'screen_z': z_screen,
    }
    return [lens], info


def _build_microscope(cfg: SimulationConfig):
    z_obj_lens = cfg.object_distance
    z_oc_lens = z_obj_lens + cfg.obj_to_oc

    d_img1 = thin_lens_image_distance(cfg.f_obj, cfg.object_distance)
    z_intermediate = z_obj_lens + d_img1
    mag1 = -d_img1 / cfg.object_distance

    d_obj2 = z_oc_lens - z_intermediate
    d_img2 = thin_lens_image_distance(cfg.f_oc, d_obj2)
    mag2 = -d_img2 / d_obj2 if np.isfinite(d_img2) else float('inf')

    if cfg.screen_distance > 0:
        z_screen = z_oc_lens + cfg.screen_distance
    elif np.isfinite(d_img2) and d_img2 > 0:
        z_screen = z_oc_lens + d_img2
    else:
        z_screen = z_oc_lens + 0.30

    mag_total = mag1 * mag2 if np.isfinite(mag2) else float('inf')

    obj_lens = ThinLens(z=z_obj_lens, f=cfg.f_obj, aperture=cfg.obj_aperture)
    oc_lens = ThinLens(z=z_oc_lens, f=cfg.f_oc, aperture=cfg.oc_aperture)

    info = {
        'd_obj': cfg.object_distance,
        'd_img_obj_theory': d_img1,
        'mag_obj_theory': mag1,
        'z_intermediate': z_intermediate,
        'd_obj_oc': d_obj2,
        'd_img_oc_theory': d_img2,
        'mag_oc_theory': mag2,
        'magnification_theory': mag_total,
        'first_element_z': z_obj_lens,
        'first_aperture': cfg.obj_aperture,
        'screen_z': z_screen,
    }
    return [obj_lens, oc_lens], info


def _build_thick(cfg: SimulationConfig):
    R1, R2, d, n = cfg.lens_R1, cfg.lens_R2, cfg.lens_thickness, cfg.lens_n
    inv_f = (n - 1) * (1 / R1 - 1 / R2 + (n - 1) * d / (n * R1 * R2))
    f_eff = 1.0 / inv_f if inv_f != 0 else float('inf')

    z_lens = cfg.object_distance

    lens = ThickLens(z=z_lens, R1=R1, R2=R2, thickness=d, n=n,
                     aperture=cfg.lens_thick_aperture)

    d_img = thin_lens_image_distance(f_eff, cfg.object_distance)
    if cfg.screen_distance > 0:
        z_screen = z_lens + cfg.screen_distance
    else:
        z_screen = z_lens + d + d_img

    mag = -d_img / cfg.object_distance

    info = {
        'd_obj': cfg.object_distance,
        'f_eff_theory': f_eff,
        'd_img_theory': d_img,
        'magnification_theory': mag,
        'first_element_z': z_lens,
        'first_aperture': cfg.lens_thick_aperture,
        'screen_z': z_screen,
    }
    return [lens], info
