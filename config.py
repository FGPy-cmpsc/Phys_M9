from __future__ import annotations
import json
import os
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List


SYSTEM_TYPES = ('thin', 'microscope', 'thick')


SIZE_MIN, SIZE_MAX = 1e-6, 1.0
F_MIN, F_MAX = 1e-3, 10.0
DIST_MIN, DIST_MAX = 1e-3, 100.0
APERTURE_MIN, APERTURE_MAX = 1e-4, 0.5
N_INDEX_MIN, N_INDEX_MAX = 1.0, 3.0
THICKNESS_MIN, THICKNESS_MAX = 1e-4, 0.5
RES_MIN, RES_MAX = 5, 200
RAYS_MIN, RAYS_MAX = 10, 100_000
TOTAL_RAYS_MAX = 5_000_000


def validate_output_prefix(prefix: str) -> list:
    errors = []
    if not prefix:
        errors.append('output: пустая строка')
        return errors
    if len(prefix) > 100:
        errors.append('output: длина больше 100 символов')
    if os.path.isabs(prefix):
        errors.append('output: абсолютные пути запрещены')
    if '..' in prefix.split(os.sep):
        errors.append('output: запрещена последовательность ".."')
    if not re.match(r'^[A-Za-z0-9_\-./]+$', prefix):
        errors.append('output: допустимы только буквы, цифры, _ - . /')
    return errors


def validate_config_path(path: str) -> list:
    errors = []
    if not path:
        errors.append('config: пустой путь')
        return errors
    if not os.path.isfile(path):
        errors.append(f'config: файл не существует: {path}')
        return errors
    try:
        with open(path, 'r', encoding='utf-8') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f'config: файл не является корректным JSON: {e}')
    except OSError as e:
        errors.append(f'config: не удалось прочитать файл: {e}')
    return errors


@dataclass
class SimulationConfig:
    system: str = 'thin'

    object_distance: float = 0.30
    object_size: float = 0.02

    lens_f: float = 0.10
    lens_aperture: float = 0.03

    f_obj: float = 0.020
    f_oc: float = 0.025
    obj_to_oc: float = 0.190
    obj_aperture: float = 0.015
    oc_aperture: float = 0.025

    lens_R1: float = 0.10
    lens_R2: float = -0.10
    lens_thickness: float = 0.020
    lens_n: float = 1.52
    lens_thick_aperture: float = 0.03

    screen_distance: float = 0.0
    object_resolution: int = 40
    rays_per_point: int = 800

    def validate(self) -> None:
        errors = []

        if self.system not in SYSTEM_TYPES:
            errors.append(f'system: должно быть из {SYSTEM_TYPES}, получено "{self.system}"')

        for name in ['object_distance', 'object_size', 'lens_f', 'lens_aperture',
                     'f_obj', 'f_oc', 'obj_to_oc', 'obj_aperture', 'oc_aperture',
                     'lens_R1', 'lens_R2', 'lens_thickness', 'lens_n',
                     'lens_thick_aperture', 'screen_distance']:
            val = getattr(self, name)
            if not isinstance(val, (int, float)):
                errors.append(f'{name}: должно быть числом')
                continue
            if not np.isfinite(val):
                errors.append(f'{name}: должно быть конечным числом')

        if not isinstance(self.object_resolution, int):
            errors.append('object_resolution: должно быть целым')
        if not isinstance(self.rays_per_point, int):
            errors.append('rays_per_point: должно быть целым')

        if errors:
            raise ValueError('Ошибки в параметрах:\n  ' + '\n  '.join(errors))

        if not (DIST_MIN <= self.object_distance <= DIST_MAX):
            errors.append(f'object_distance: в [{DIST_MIN}, {DIST_MAX}] м')
        if not (SIZE_MIN <= self.object_size <= SIZE_MAX):
            errors.append(f'object_size: в [{SIZE_MIN}, {SIZE_MAX}] м')
        if not (RES_MIN <= self.object_resolution <= RES_MAX):
            errors.append(f'object_resolution: в [{RES_MIN}, {RES_MAX}]')
        if not (RAYS_MIN <= self.rays_per_point <= RAYS_MAX):
            errors.append(f'rays_per_point: в [{RAYS_MIN}, {RAYS_MAX}]')

        approx_points = max(1, int(self.object_resolution * self.object_resolution * 0.25))
        total_rays_est = approx_points * self.rays_per_point
        if total_rays_est > TOTAL_RAYS_MAX:
            errors.append(
                f'object_resolution^2 * rays_per_point дает оценку ~{total_rays_est:.0e} лучей, '
                f'максимум {TOTAL_RAYS_MAX:.0e}. Уменьшите rays_per_point или object_resolution')

        if self.system == 'thin':
            if not (F_MIN <= self.lens_f <= F_MAX):
                errors.append(f'lens_f: в [{F_MIN}, {F_MAX}] м')
            if not (APERTURE_MIN <= self.lens_aperture <= APERTURE_MAX):
                errors.append(f'lens_aperture: в [{APERTURE_MIN}, {APERTURE_MAX}] м')
            if self.object_distance <= self.lens_f:
                errors.append(
                    f'object_distance ({self.object_distance}) должно быть больше lens_f ({self.lens_f}), '
                    f'иначе изображение мнимое и не строится на экране')
        elif self.system == 'microscope':
            if not (F_MIN <= self.f_obj <= F_MAX):
                errors.append(f'f_obj: в [{F_MIN}, {F_MAX}] м')
            if not (F_MIN <= self.f_oc <= F_MAX):
                errors.append(f'f_oc: в [{F_MIN}, {F_MAX}] м')
            if not (DIST_MIN <= self.obj_to_oc <= DIST_MAX):
                errors.append(f'obj_to_oc: в [{DIST_MIN}, {DIST_MAX}] м')
            if not (APERTURE_MIN <= self.obj_aperture <= APERTURE_MAX):
                errors.append(f'obj_aperture: в [{APERTURE_MIN}, {APERTURE_MAX}] м')
            if not (APERTURE_MIN <= self.oc_aperture <= APERTURE_MAX):
                errors.append(f'oc_aperture: в [{APERTURE_MIN}, {APERTURE_MAX}] м')
            if self.object_distance <= self.f_obj:
                errors.append(
                    f'object_distance ({self.object_distance}) должно быть больше f_obj ({self.f_obj})')
            if self.obj_to_oc <= self.f_obj + self.f_oc:
                errors.append(
                    f'obj_to_oc должно быть больше f_obj + f_oc = {self.f_obj + self.f_oc:.3f}, '
                    f'иначе промежуточное изображение не попадёт между линзами')
        elif self.system == 'thick':
            if not (N_INDEX_MIN <= self.lens_n <= N_INDEX_MAX):
                errors.append(f'lens_n: в [{N_INDEX_MIN}, {N_INDEX_MAX}]')
            if not (THICKNESS_MIN <= self.lens_thickness <= THICKNESS_MAX):
                errors.append(f'lens_thickness: в [{THICKNESS_MIN}, {THICKNESS_MAX}] м')
            if not (APERTURE_MIN <= self.lens_thick_aperture <= APERTURE_MAX):
                errors.append(f'lens_thick_aperture: в [{APERTURE_MIN}, {APERTURE_MAX}] м')
            if abs(self.lens_R1) < self.lens_thick_aperture:
                errors.append(
                    f'|lens_R1| ({abs(self.lens_R1)}) должно быть >= lens_thick_aperture')
            if abs(self.lens_R2) < self.lens_thick_aperture:
                errors.append(
                    f'|lens_R2| ({abs(self.lens_R2)}) должно быть >= lens_thick_aperture')
            if self.lens_thickness >= min(abs(self.lens_R1), abs(self.lens_R2)):
                errors.append(
                    f'lens_thickness должно быть меньше min(|R1|, |R2|)')

        if self.screen_distance < 0:
            errors.append('screen_distance: не может быть отрицательным')
        if self.screen_distance > 0 and self.screen_distance > DIST_MAX:
            errors.append(f'screen_distance: <= {DIST_MAX} м')

        if errors:
            raise ValueError('Ошибки в параметрах:\n  ' + '\n  '.join(errors))

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d: dict) -> SimulationConfig:
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> SimulationConfig:
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
