from __future__ import annotations
import argparse
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import SimulationConfig, validate_output_prefix, validate_config_path
from systems import build_system
from tracer import form_image, make_arrow_object


def parse_args():
    p = argparse.ArgumentParser(description='Трассировка лучей')
    p.add_argument('--system', type=str, default=None,
                   choices=['thin', 'microscope', 'thick'])
    p.add_argument('--object-distance', type=float, default=None)
    p.add_argument('--object-size', type=float, default=None)
    p.add_argument('--lens-f', type=float, default=None)
    p.add_argument('--lens-aperture', type=float, default=None)
    p.add_argument('--f-obj', type=float, default=None)
    p.add_argument('--f-oc', type=float, default=None)
    p.add_argument('--obj-to-oc', type=float, default=None)
    p.add_argument('--obj-aperture', type=float, default=None)
    p.add_argument('--oc-aperture', type=float, default=None)
    p.add_argument('--lens-R1', type=float, default=None)
    p.add_argument('--lens-R2', type=float, default=None)
    p.add_argument('--lens-thickness', type=float, default=None)
    p.add_argument('--lens-n', type=float, default=None)
    p.add_argument('--lens-thick-aperture', type=float, default=None)
    p.add_argument('--screen-distance', type=float, default=None)
    p.add_argument('--object-resolution', type=int, default=None)
    p.add_argument('--rays-per-point', type=int, default=None)
    p.add_argument('--output', type=str, default='trace')
    p.add_argument('--config', type=str, default=None)
    return p.parse_args()


def build_config(args) -> SimulationConfig:
    if args.config:
        cfg = SimulationConfig.load(args.config)
    else:
        cfg = SimulationConfig()

    overrides = {
        'system': args.system,
        'object_distance': args.object_distance,
        'object_size': args.object_size,
        'lens_f': args.lens_f,
        'lens_aperture': args.lens_aperture,
        'f_obj': args.f_obj,
        'f_oc': args.f_oc,
        'obj_to_oc': args.obj_to_oc,
        'obj_aperture': args.obj_aperture,
        'oc_aperture': args.oc_aperture,
        'lens_R1': args.lens_R1,
        'lens_R2': args.lens_R2,
        'lens_thickness': args.lens_thickness,
        'lens_n': args.lens_n,
        'lens_thick_aperture': args.lens_thick_aperture,
        'screen_distance': args.screen_distance,
        'object_resolution': args.object_resolution,
        'rays_per_point': args.rays_per_point,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def plot_layout(ax, cfg, elements, info, paths_xz):
    z_max = info['screen_z'] + 0.02

    for path in paths_xz[:200]:
        z, x = path
        ax.plot(z, x, 'y-', linewidth=0.3, alpha=0.4)

    ax.plot([0, 0], [-cfg.object_size/2, cfg.object_size/2],
            'b-', linewidth=2.5, label='предмет')
    ax.plot([info['screen_z'], info['screen_z']],
            [-cfg.object_size, cfg.object_size],
            'r-', linewidth=2.5, label='экран')

    for el in elements:
        if hasattr(el, 'f'):
            ax.plot([el.z, el.z], [-el.aperture, el.aperture], 'g-', linewidth=2,
                    label=f'тонкая линза f={el.f*1e3:.0f} мм' if el is elements[0] else None)
        else:
            ax.plot([el.z, el.z + el.thickness],
                    [-el.aperture, -el.aperture], 'g-', linewidth=1.5)
            ax.plot([el.z, el.z + el.thickness],
                    [el.aperture, el.aperture], 'g-', linewidth=1.5)
            ax.plot([el.z, el.z], [-el.aperture, el.aperture], 'g-', linewidth=0.5)
            ax.plot([el.z + el.thickness, el.z + el.thickness],
                    [-el.aperture, el.aperture], 'g-', linewidth=0.5,
                    label=f'толстая линза d={el.thickness*1e3:.1f} мм')

    ax.set_xlim(-0.02, z_max)
    ax.set_xlabel('z, м')
    ax.set_ylabel('x, м')
    ax.set_title('Ход лучей (сечение xz)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)


def plot_object_image(ax_obj, ax_img, arrow, image, cfg, info, image_extent):
    obj_extent = cfg.object_size
    ax_obj.imshow(arrow, cmap='gray', origin='upper',
                  extent=[-obj_extent/2*1e3, obj_extent/2*1e3,
                          -obj_extent/2*1e3, obj_extent/2*1e3])
    ax_obj.set_xlabel('x, мм')
    ax_obj.set_ylabel('y, мм')
    ax_obj.set_title(f'Предмет (размер {obj_extent*1e3:.1f} мм)')

    img_max = image.max() if image.max() > 0 else 1
    img_norm = image / img_max
    ax_img.imshow(img_norm, cmap='gray', origin='upper',
                  extent=[-image_extent/2*1e3, image_extent/2*1e3,
                          -image_extent/2*1e3, image_extent/2*1e3])
    mag_t = info.get('magnification_theory', None)
    if mag_t is None or not np.isfinite(mag_t):
        mag_str = 'inf'
    else:
        mag_str = f'{mag_t:.2f}'
    ax_img.set_xlabel('x, мм')
    ax_img.set_ylabel('y, мм')
    ax_img.set_title(f'Изображение (теор. увеличение {mag_str}x)')


def measure_numerical_magnification(image, image_extent, arrow, object_size):
    if image.max() == 0:
        return None
    threshold = 0.1 * image.max()
    mask = image > threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    nrows = image.shape[0]
    xs_m = (xs - nrows/2 + 0.5) / nrows * image_extent
    ys_m = -(ys - nrows/2 + 0.5) / nrows * image_extent
    extent_x_img = float(xs_m.max() - xs_m.min())
    extent_y_img = float(ys_m.max() - ys_m.min())

    obj_rows, obj_cols = np.where(arrow > 0)
    if len(obj_rows) == 0:
        return None
    obj_size_pixels = arrow.shape[0]
    arrow_height = (obj_rows.max() - obj_rows.min() + 1) / obj_size_pixels * object_size
    arrow_width = (obj_cols.max() - obj_cols.min() + 1) / obj_size_pixels * object_size

    M_height = extent_y_img / arrow_height if arrow_height > 0 else 0
    M_width = extent_x_img / arrow_width if arrow_width > 0 else 0
    M_measured = (M_height + M_width) / 2

    return M_measured, extent_x_img, extent_y_img


def plot_info_panel(ax, cfg, info, image, image_extent, arrow):
    ax.axis('off')

    lines = [f'Система: {cfg.system}', '']
    lines.append(f'Расстояние до предмета: {cfg.object_distance*1e3:.1f} мм')
    lines.append(f'Размер предмета: {cfg.object_size*1e3:.2f} мм')
    lines.append('')

    if cfg.system == 'thin':
        lines.append(f'Тонкая линза:  f = {cfg.lens_f*1e3:.1f} мм,  апертура = {cfg.lens_aperture*1e3:.1f} мм')
        lines.append(f'Расстояние до изображения (теория): d_i = {info["d_img_theory"]*1e3:.2f} мм')
        lines.append(f'Увеличение (теория): M = {info["magnification_theory"]:+.3f}')
    elif cfg.system == 'microscope':
        lines.append(f'Объектив:  f = {cfg.f_obj*1e3:.1f} мм')
        lines.append(f'Окуляр:    f = {cfg.f_oc*1e3:.1f} мм')
        lines.append(f'Расстояние между линзами: {cfg.obj_to_oc*1e3:.1f} мм')
        lines.append('')
        lines.append(f'Объектив:    d_i = {info["d_img_obj_theory"]*1e3:.2f} мм,  M = {info["mag_obj_theory"]:+.3f}')
        lines.append(f'Окуляр:      d_i = {info["d_img_oc_theory"]*1e3:.2f} мм,  M = {info["mag_oc_theory"]:+.3f}')
        lines.append(f'Полное увеличение (теория): M = {info["magnification_theory"]:+.3f}')
    elif cfg.system == 'thick':
        lines.append(f'Толстая линза: R1 = {cfg.lens_R1*1e3:.1f} мм, R2 = {cfg.lens_R2*1e3:.1f} мм')
        lines.append(f'Толщина: d = {cfg.lens_thickness*1e3:.2f} мм, n = {cfg.lens_n:.3f}')
        lines.append(f'Эффективное f (формула линзовара): {info["f_eff_theory"]*1e3:.2f} мм')
        lines.append(f'd_i (теория параксиальная): {info["d_img_theory"]*1e3:.2f} мм')
        lines.append(f'Увеличение (теория): M = {info["magnification_theory"]:+.3f}')

    lines.append('')
    measured = measure_numerical_magnification(image, image_extent, arrow, cfg.object_size)
    if measured is not None:
        m_total, ex, ey = measured
        lines.append(f'Численно измеренное увеличение: M ~ {m_total:.3f}')
        lines.append(f'Размер изображения: {ex*1e3:.2f} x {ey*1e3:.2f} мм')

    lines.append('')
    lines.append(f'Лучей на точку: {cfg.rays_per_point}')
    lines.append(f'Разрешение предмета: {cfg.object_resolution}x{cfg.object_resolution}')

    ax.text(0.0, 1.0, '\n'.join(lines),
            transform=ax.transAxes,
            fontsize=9, family='monospace',
            verticalalignment='top')


def main():
    args = parse_args()

    pre_errors = []
    pre_errors.extend(validate_output_prefix(args.output))
    if args.config is not None:
        pre_errors.extend(validate_config_path(args.config))
    if pre_errors:
        print('Ошибки в параметрах:\n  ' + '\n  '.join(pre_errors))
        return

    try:
        cfg = build_config(args)
        cfg.validate()
    except ValueError as e:
        print(e)
        return

    print('=== Трассировка лучей ===')
    print(f'Система: {cfg.system}')
    print(f'Предмет: {cfg.object_size*1e3:.2f} мм на расстоянии {cfg.object_distance*1e3:.1f} мм')
    print(f'Разрешение: {cfg.object_resolution}x{cfg.object_resolution}, лучей на точку: {cfg.rays_per_point}')
    print()

    elements, info = build_system(cfg)

    arrow, pts = make_arrow_object(cfg.object_size, cfg.object_resolution)
    print(f'Точек предмета: {len(pts)}')

    mag_theory = abs(info.get('magnification_theory', 1.0))
    if not np.isfinite(mag_theory):
        mag_theory = 5.0
    image_extent = mag_theory * cfg.object_size * 2.5

    print('Трассировка...')
    image, paths_xz = form_image(
        pts, elements, info['screen_z'],
        cfg.rays_per_point, info['first_aperture'],
        info['first_element_z'], 200, image_extent,
    )
    print(f'Лучей в изображении: {int(image.sum())}')

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Трассировка лучей: {cfg.system}', fontsize=14)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    ax_layout = fig.add_subplot(gs[0, :])
    plot_layout(ax_layout, cfg, elements, info, paths_xz)

    ax_obj = fig.add_subplot(gs[1, 0])
    ax_img = fig.add_subplot(gs[1, 1])
    plot_object_image(ax_obj, ax_img, arrow, image, cfg, info, image_extent)

    ax_info = fig.add_subplot(gs[1, 2])
    plot_info_panel(ax_info, cfg, info, image, image_extent, arrow)

    out_png = f'{args.output}.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Сохранено: {out_png}')

    summary = {
        'system': cfg.system,
        'object_distance': cfg.object_distance,
        'object_size': cfg.object_size,
        'magnification_theory': float(info.get('magnification_theory', 0.0))
            if np.isfinite(info.get('magnification_theory', 0.0)) else None,
        'd_img_theory': float(info.get('d_img_theory', 0.0))
            if np.isfinite(info.get('d_img_theory', 0.0)) else None,
        'rays_in_image': int(image.sum()),
    }
    measured = measure_numerical_magnification(image, image_extent, arrow, cfg.object_size)
    if measured is not None:
        summary['magnification_measured'] = measured[0]
    with open(f'{args.output}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Сохранено: {args.output}.json')

    print('\nГотово.')


if __name__ == '__main__':
    main()
