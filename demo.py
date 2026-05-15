from __future__ import annotations
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import SimulationConfig
from systems import build_system
from tracer import form_image, make_arrow_object

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'demo_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


SCENARIOS = [
    {
        'name': 'Тонкая линза',
        'cfg': SimulationConfig(
            system='thin',
            object_distance=0.30, object_size=0.005,
            lens_f=0.10, lens_aperture=0.025,
            object_resolution=30, rays_per_point=400,
        ),
    },
    {
        'name': 'Микроскоп (объектив + окуляр)',
        'cfg': SimulationConfig(
            system='microscope',
            object_distance=0.0229, object_size=0.0008,
            f_obj=0.020, f_oc=0.025, obj_to_oc=0.190,
            obj_aperture=0.015, oc_aperture=0.025,
            object_resolution=25, rays_per_point=300,
        ),
    },
    {
        'name': 'Толстая сферическая линза',
        'cfg': SimulationConfig(
            system='thick',
            object_distance=0.30, object_size=0.005,
            lens_R1=0.10, lens_R2=-0.10,
            lens_thickness=0.020, lens_n=1.52, lens_thick_aperture=0.015,
            object_resolution=30, rays_per_point=400,
        ),
    },
]


def magnification_from_image(image, image_extent, arrow, object_size):
    if image.max() == 0:
        return None
    threshold = 0.2 * image.max()
    mask = image > threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    nrows = image.shape[0]
    ys_m = -(ys - nrows/2 + 0.5) / nrows * image_extent
    extent_y_img = float(ys_m.max() - ys_m.min())
    obj_rows, _ = np.where(arrow > 0)
    if len(obj_rows) == 0:
        return None
    n_obj = arrow.shape[0]
    arrow_height = (obj_rows.max() - obj_rows.min() + 1) / n_obj * object_size
    if arrow_height <= 0:
        return None
    return extent_y_img / arrow_height


def main():
    print('=== Демонстрация: трассировка лучей ===\n')

    results = []
    for sc in SCENARIOS:
        sc['cfg'].validate()
        print(f'{sc["name"]}:')
        elements, info = build_system(sc['cfg'])
        arrow, pts = make_arrow_object(sc['cfg'].object_size, sc['cfg'].object_resolution)

        mag_t = info.get('magnification_theory', 1.0)
        if not np.isfinite(mag_t):
            mag_t = 5.0
        image_extent = abs(mag_t) * sc['cfg'].object_size * 2.5

        image, paths = form_image(
            pts, elements, info['screen_z'],
            sc['cfg'].rays_per_point, info['first_aperture'],
            info['first_element_z'], 200, image_extent,
        )
        m_meas = magnification_from_image(image, image_extent, arrow, sc['cfg'].object_size)
        print(f'  Лучей в изображении: {int(image.sum())}')
        print(f'  Увеличение теор: {mag_t:+.3f}, числ: {m_meas}')
        results.append({
            'name': sc['name'],
            'cfg': sc['cfg'],
            'elements': elements,
            'info': info,
            'arrow': arrow,
            'image': image,
            'paths': paths,
            'image_extent': image_extent,
            'mag_measured': m_meas,
        })
        print()

    n = len(results)
    fig = plt.figure(figsize=(7 * n, 13))
    fig.suptitle('Трассировка лучей: три оптические системы', fontsize=14)
    gs = GridSpec(3, n, figure=fig, hspace=0.4, wspace=0.3,
                  height_ratios=[2, 1, 1])

    for col, r in enumerate(results):
        cfg = r['cfg']
        elements = r['elements']
        info = r['info']

        ax_layout = fig.add_subplot(gs[0, col])
        for path in r['paths'][:150]:
            z, x = path
            ax_layout.plot(z, x, 'y-', linewidth=0.3, alpha=0.4)
        ax_layout.plot([0, 0], [-cfg.object_size/2, cfg.object_size/2],
                       'b-', linewidth=2.5, label='предмет')
        ax_layout.plot([info['screen_z'], info['screen_z']],
                       [-cfg.object_size, cfg.object_size],
                       'r-', linewidth=2.5, label='экран')
        for el in elements:
            if hasattr(el, 'f'):
                ax_layout.plot([el.z, el.z],
                               [-el.aperture, el.aperture],
                               'g-', linewidth=2)
            else:
                ax_layout.plot([el.z, el.z + el.thickness],
                               [-el.aperture, -el.aperture], 'g-', linewidth=1.5)
                ax_layout.plot([el.z, el.z + el.thickness],
                               [el.aperture, el.aperture], 'g-', linewidth=1.5)
                ax_layout.plot([el.z, el.z],
                               [-el.aperture, el.aperture], 'g-', linewidth=0.5)
                ax_layout.plot([el.z + el.thickness, el.z + el.thickness],
                               [-el.aperture, el.aperture], 'g-', linewidth=0.5)
        ax_layout.set_xlim(-0.02, info['screen_z'] + 0.02)
        ax_layout.set_xlabel('z, м')
        ax_layout.set_ylabel('x, м')
        ax_layout.set_title(r['name'])
        ax_layout.grid(alpha=0.3)
        ax_layout.axhline(0, color='gray', linestyle=':', linewidth=0.5)

        ax_obj = fig.add_subplot(gs[1, col])
        obj_extent = cfg.object_size
        ax_obj.imshow(r['arrow'], cmap='gray', origin='upper',
                      extent=[-obj_extent/2*1e3, obj_extent/2*1e3,
                              -obj_extent/2*1e3, obj_extent/2*1e3])
        ax_obj.set_xlabel('x, мм')
        ax_obj.set_ylabel('y, мм')
        ax_obj.set_title(f'Предмет ({obj_extent*1e3:.2f} мм)')

        ax_img = fig.add_subplot(gs[2, col])
        img_max = r['image'].max() if r['image'].max() > 0 else 1
        ax_img.imshow(r['image'] / img_max, cmap='gray', origin='upper',
                      extent=[-r['image_extent']/2*1e3,
                              r['image_extent']/2*1e3,
                              -r['image_extent']/2*1e3,
                              r['image_extent']/2*1e3])
        mag_t = info.get('magnification_theory', 0)
        mag_str = f'{mag_t:+.2f}' if np.isfinite(mag_t) else 'inf'
        meas = r['mag_measured']
        meas_str = f' (числ. {meas:.2f})' if meas is not None else ''
        ax_img.set_xlabel('x, мм')
        ax_img.set_ylabel('y, мм')
        ax_img.set_title(f'Изображение, M_теор = {mag_str}x{meas_str}')

    path = os.path.join(OUTPUT_DIR, 'three_systems.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Сохранено: {path}')
    print('\nДемонстрация завершена.')


if __name__ == '__main__':
    main()
