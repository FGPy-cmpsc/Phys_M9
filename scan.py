from __future__ import annotations
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import (SimulationConfig, validate_output_prefix,
                    THICKNESS_MIN, THICKNESS_MAX, APERTURE_MIN, APERTURE_MAX)
from systems import build_system
from tracer import form_image, make_arrow_object

SCAN_STEPS_MIN, SCAN_STEPS_MAX = 3, 50


def parse_args():
    p = argparse.ArgumentParser(description='Развёртка по толщине и апертуре толстой линзы')
    p.add_argument('--object-distance', type=float, default=0.30)
    p.add_argument('--object-size', type=float, default=0.005)
    p.add_argument('--lens-R1', type=float, default=0.10)
    p.add_argument('--lens-R2', type=float, default=-0.10)
    p.add_argument('--lens-n', type=float, default=1.52)

    p.add_argument('--thickness-min', type=float, default=1e-3)
    p.add_argument('--thickness-max', type=float, default=4e-2)
    p.add_argument('--thickness-steps', type=int, default=8)

    p.add_argument('--aperture-min', type=float, default=5e-3)
    p.add_argument('--aperture-max', type=float, default=4e-2)
    p.add_argument('--aperture-steps', type=int, default=8)

    p.add_argument('--object-resolution', type=int, default=20)
    p.add_argument('--rays-per-point', type=int, default=200)
    p.add_argument('--output', type=str, default='scan')
    return p.parse_args()


def validate_args(args):
    errors = []

    for name, val in [('object-distance', args.object_distance),
                      ('object-size', args.object_size),
                      ('lens-R1', args.lens_R1), ('lens-R2', args.lens_R2),
                      ('lens-n', args.lens_n),
                      ('thickness-min', args.thickness_min),
                      ('thickness-max', args.thickness_max),
                      ('aperture-min', args.aperture_min),
                      ('aperture-max', args.aperture_max)]:
        if not np.isfinite(val):
            errors.append(f'{name}: должно быть конечным числом')

    if errors:
        raise ValueError('Ошибки в параметрах:\n  ' + '\n  '.join(errors))

    if args.thickness_min <= 0 or args.thickness_max <= 0:
        errors.append('thickness-min и thickness-max должны быть > 0')
    if args.thickness_min >= args.thickness_max:
        errors.append('thickness-min должно быть < thickness-max')
    if args.thickness_min < THICKNESS_MIN or args.thickness_max > THICKNESS_MAX:
        errors.append(f'диапазон толщины должен быть в [{THICKNESS_MIN}, {THICKNESS_MAX}] м')

    if args.aperture_min <= 0 or args.aperture_max <= 0:
        errors.append('aperture-min и aperture-max должны быть > 0')
    if args.aperture_min >= args.aperture_max:
        errors.append('aperture-min должно быть < aperture-max')
    if args.aperture_min < APERTURE_MIN or args.aperture_max > APERTURE_MAX:
        errors.append(f'диапазон апертуры должен быть в [{APERTURE_MIN}, {APERTURE_MAX}] м')

    for name, val in [('thickness-steps', args.thickness_steps),
                      ('aperture-steps', args.aperture_steps)]:
        if not isinstance(val, int) or val < SCAN_STEPS_MIN or val > SCAN_STEPS_MAX:
            errors.append(f'{name}: должно быть целым в [{SCAN_STEPS_MIN}, {SCAN_STEPS_MAX}]')

    total = args.thickness_steps * args.aperture_steps
    if total > 200:
        errors.append(f'thickness-steps * aperture-steps = {total} слишком велико (макс 200)')

    if args.object_resolution < 5 or args.object_resolution > 50:
        errors.append('object-resolution должно быть в [5, 50]')
    if args.rays_per_point < 10 or args.rays_per_point > 5000:
        errors.append('rays-per-point должно быть в [10, 5000]')

    errors.extend(validate_output_prefix(args.output))

    if errors:
        raise ValueError('Ошибки в параметрах:\n  ' + '\n  '.join(errors))


def compute_blur_metric(image):
    if image.sum() == 0:
        return float('nan')
    ys, xs = np.indices(image.shape)
    w = image
    cy = (ys * w).sum() / w.sum()
    cx = (xs * w).sum() / w.sum()
    r2 = ((ys - cy)**2 + (xs - cx)**2) * w
    return float(np.sqrt(r2.sum() / w.sum()))


def main():
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as e:
        print(e)
        return

    thicknesses = np.linspace(args.thickness_min, args.thickness_max, args.thickness_steps)
    apertures = np.linspace(args.aperture_min, args.aperture_max, args.aperture_steps)

    blur_map = np.full((len(apertures), len(thicknesses)), np.nan)
    rays_map = np.zeros((len(apertures), len(thicknesses)), dtype=int)

    total = len(thicknesses) * len(apertures)
    print(f'Расчёт: {len(thicknesses)} x {len(apertures)} = {total} точек')
    print(f'Толщина: {args.thickness_min*1e3:.2f} - {args.thickness_max*1e3:.2f} мм')
    print(f'Апертура: {args.aperture_min*1e3:.2f} - {args.aperture_max*1e3:.2f} мм')

    done = 0
    for i, ap in enumerate(apertures):
        for j, d in enumerate(thicknesses):
            cfg = SimulationConfig(
                system='thick',
                object_distance=args.object_distance, object_size=args.object_size,
                lens_R1=args.lens_R1, lens_R2=args.lens_R2,
                lens_thickness=d, lens_n=args.lens_n, lens_thick_aperture=ap,
                object_resolution=args.object_resolution,
                rays_per_point=args.rays_per_point,
            )
            try:
                cfg.validate()
                elements, info = build_system(cfg)
                arrow, pts = make_arrow_object(cfg.object_size, cfg.object_resolution)
                mag_t = info.get('magnification_theory', 1.0)
                if not np.isfinite(mag_t):
                    mag_t = 1.0
                image_extent = abs(mag_t) * cfg.object_size * 3.0
                image, _ = form_image(
                    pts, elements, info['screen_z'],
                    cfg.rays_per_point, info['first_aperture'],
                    info['first_element_z'], 100, image_extent,
                )
                blur_map[i, j] = compute_blur_metric(image) * image_extent / 100 * 1e3
                rays_map[i, j] = int(image.sum())
            except Exception:
                blur_map[i, j] = np.nan
                rays_map[i, j] = 0

            done += 1
            if done % max(1, total // 10) == 0:
                print(f'  {done}/{total}')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Толстая линза: влияние толщины и апертуры (R1={args.lens_R1*1e3:.0f} мм, R2={args.lens_R2*1e3:.0f} мм, n={args.lens_n})',
                 fontsize=13)

    T_mesh, A_mesh = np.meshgrid(thicknesses, apertures)

    ax1 = axes[0]
    pcm1 = ax1.pcolormesh(T_mesh * 1e3, A_mesh * 1e3, blur_map,
                           shading='auto', cmap='inferno')
    fig.colorbar(pcm1, ax=ax1, label='размытие пятна, мм (RMS)')
    ax1.set_xlabel('толщина линзы, мм')
    ax1.set_ylabel('апертура, мм')
    ax1.set_title('Среднеквадратичное размытие изображения')

    ax2 = axes[1]
    pcm2 = ax2.pcolormesh(T_mesh * 1e3, A_mesh * 1e3, rays_map,
                           shading='auto', cmap='viridis')
    fig.colorbar(pcm2, ax=ax2, label='число лучей на экране')
    ax2.set_xlabel('толщина линзы, мм')
    ax2.set_ylabel('апертура, мм')
    ax2.set_title('Сбор света (виньетирование)')

    fig.tight_layout()
    out_png = f'{args.output}.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Сохранено: {out_png}')
    print('\nГотово.')


if __name__ == '__main__':
    main()
