from __future__ import annotations
import unittest
import numpy as np

from config import SimulationConfig
from optics import (ThinLens, SphericalSurface, ThickLens, Screen,
                    trace_through_system,
                    thin_lens_image_distance, thin_lens_magnification)
from systems import build_system
from tracer import make_arrow_object, form_image


class TestThinLensFormula(unittest.TestCase):
    def test_image_distance_formula(self):
        f = 0.1
        d_obj = 0.3
        d_img = thin_lens_image_distance(f, d_obj)
        self.assertAlmostEqual(1/f, 1/d_obj + 1/d_img, places=10)

    def test_magnification_formula(self):
        f = 0.1
        d_obj = 0.3
        m = thin_lens_magnification(f, d_obj)
        self.assertAlmostEqual(m, -0.5, places=10)

    def test_image_at_2f(self):
        f = 0.05
        d_obj = 2 * f
        d_img = thin_lens_image_distance(f, d_obj)
        m = thin_lens_magnification(f, d_obj)
        self.assertAlmostEqual(d_img, 2 * f, places=10)
        self.assertAlmostEqual(m, -1.0, places=10)

    def test_at_focal_plane(self):
        f = 0.1
        d_img = thin_lens_image_distance(f, f)
        self.assertTrue(np.isinf(d_img))


class TestThinLensRayTrace(unittest.TestCase):
    def test_central_ray_through_center(self):
        lens = ThinLens(z=0.1, f=0.05, aperture=0.02)
        origins = np.array([[0.005, 0.0, 0.0]])
        directions = np.array([[-0.005/0.1, 0.0, 1.0]])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        screen = Screen(z=0.1 + 0.05 * 2)
        paths, _, valid = trace_through_system(origins, directions, [lens, screen])
        self.assertTrue(valid[0])
        x_img = paths[-1][0][0]
        x_obj = origins[0][0]
        M = x_img / x_obj
        self.assertAlmostEqual(M, -1.0, places=2)

    def test_object_at_2f_produces_inverted_unity(self):
        f = 0.05
        d_obj = 2 * f
        lens = ThinLens(z=d_obj, f=f, aperture=0.02)
        d_img = thin_lens_image_distance(f, d_obj)
        screen = Screen(z=d_obj + d_img)

        x_obj = 0.003
        origins = np.tile([x_obj, 0.0, 0.0], (5, 1))
        directions = np.array([
            [0.0, 0.0, 1.0],
            [0.05, 0.0, 1.0],
            [-0.05, 0.0, 1.0],
            [0.0, 0.05, 1.0],
            [0.0, -0.05, 1.0],
        ])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        paths, _, valid = trace_through_system(origins, directions, [lens, screen])
        xs_img = paths[-1][valid, 0]
        self.assertGreater(len(xs_img), 0)
        x_mean = np.mean(xs_img)
        M = x_mean / x_obj
        self.assertAlmostEqual(M, -1.0, places=2)


class TestSnellLaw(unittest.TestCase):
    def test_no_refraction_at_normal_incidence(self):
        surf = SphericalSurface(z_vertex=0.0, radius=0.1, aperture=0.05,
                                n_before=1.0, n_after=1.5)
        origins = np.array([[0.0, 0.0, -0.01]])
        directions = np.array([[0.0, 0.0, 1.0]])
        _, new_dirs, valid = surf.trace(origins, directions)
        self.assertTrue(valid[0])
        np.testing.assert_allclose(new_dirs[0], [0.0, 0.0, 1.0], atol=1e-9)

    def test_refraction_obeys_snell(self):
        surf = SphericalSurface(z_vertex=0.0, radius=1.0, aperture=0.5,
                                n_before=1.0, n_after=1.5)
        theta_i = np.radians(30)
        origins = np.array([[0.0, 0.0, -0.1]])
        directions = np.array([[np.sin(theta_i), 0.0, np.cos(theta_i)]])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        hit, new_dirs, valid = surf.trace(origins, directions)
        self.assertTrue(valid[0])
        center = np.array([0.0, 0.0, 1.0])
        normal = (hit[0] - center) / 1.0
        cos_i = -np.dot(directions[0], normal)
        cos_t = np.dot(new_dirs[0], normal)
        sin_i = np.sqrt(1 - cos_i**2)
        sin_t = np.sqrt(1 - cos_t**2)
        self.assertAlmostEqual(1.0 * sin_i, 1.5 * sin_t, places=5)


class TestSystems(unittest.TestCase):
    def test_thin_system_inverts_and_magnifies(self):
        cfg = SimulationConfig(system='thin',
                                object_distance=0.20, object_size=0.005,
                                lens_f=0.05, lens_aperture=0.02,
                                object_resolution=20, rays_per_point=200)
        cfg.validate()
        elements, info = build_system(cfg)
        arrow, pts = make_arrow_object(cfg.object_size, cfg.object_resolution)
        ext = abs(info['magnification_theory']) * cfg.object_size * 2.5
        image, _ = form_image(pts, elements, info['screen_z'],
                              cfg.rays_per_point, info['first_aperture'],
                              info['first_element_z'], 200, ext)
        self.assertGreater(image.sum(), 0)
        ys_obj, _ = np.where(arrow > 0)
        n_obj = arrow.shape[0]
        arrow_h = (ys_obj.max() - ys_obj.min() + 1) / n_obj * cfg.object_size
        ys, _ = np.where(image > 0.2 * image.max())
        n_img = image.shape[0]
        img_h = (ys.max() - ys.min() + 1) / n_img * ext
        M_meas = img_h / arrow_h
        M_th = abs(info['magnification_theory'])
        self.assertLess(abs(M_meas - M_th) / M_th, 0.15)

    def test_microscope_total_magnification(self):
        cfg = SimulationConfig(
            system='microscope',
            object_distance=0.0229, object_size=0.0008,
            f_obj=0.020, f_oc=0.025, obj_to_oc=0.190,
            obj_aperture=0.015, oc_aperture=0.025,
            object_resolution=20, rays_per_point=200)
        cfg.validate()
        elements, info = build_system(cfg)
        self.assertGreater(abs(info['magnification_theory']), 5)


class TestValidation(unittest.TestCase):
    def test_valid_thin(self):
        SimulationConfig(system='thin').validate()

    def test_valid_microscope(self):
        SimulationConfig(
            system='microscope',
            object_distance=0.0229, object_size=0.0008,
            f_obj=0.020, f_oc=0.025, obj_to_oc=0.190,
            obj_aperture=0.015, oc_aperture=0.025,
        ).validate()

    def test_valid_thick(self):
        SimulationConfig(system='thick').validate()

    def test_negative_object_distance(self):
        with self.assertRaises(ValueError):
            SimulationConfig(system='thin', object_distance=-1).validate()

    def test_object_inside_focal_plane(self):
        with self.assertRaises(ValueError):
            SimulationConfig(system='thin', object_distance=0.05, lens_f=0.10).validate()

    def test_unknown_system(self):
        with self.assertRaises(ValueError):
            SimulationConfig(system='foo').validate()

    def test_nan_value(self):
        with self.assertRaises(ValueError):
            SimulationConfig(system='thin', lens_f=float('nan')).validate()

    def test_inf_value(self):
        with self.assertRaises(ValueError):
            SimulationConfig(system='thin', object_distance=float('inf')).validate()

    def test_thick_aperture_too_big(self):
        with self.assertRaises(ValueError):
            SimulationConfig(system='thick', lens_R1=0.05,
                             lens_thick_aperture=0.06).validate()

    def test_thickness_too_big(self):
        with self.assertRaises(ValueError):
            SimulationConfig(system='thick', lens_R1=0.05, lens_R2=-0.05,
                             lens_thickness=0.10).validate()

    def test_microscope_lenses_too_close(self):
        with self.assertRaises(ValueError):
            SimulationConfig(
                system='microscope',
                object_distance=0.025, object_size=0.0008,
                f_obj=0.020, f_oc=0.025, obj_to_oc=0.040,
                obj_aperture=0.015, oc_aperture=0.025,
            ).validate()

    def test_total_rays_cap(self):
        with self.assertRaises(ValueError):
            SimulationConfig(system='thin',
                             object_resolution=200,
                             rays_per_point=100_000).validate()

    def test_resolution_out_of_range(self):
        with self.assertRaises(ValueError):
            SimulationConfig(system='thin', object_resolution=2).validate()

    def test_rays_out_of_range(self):
        with self.assertRaises(ValueError):
            SimulationConfig(system='thin', rays_per_point=5).validate()

    def test_negative_screen_distance(self):
        with self.assertRaises(ValueError):
            SimulationConfig(system='thin', screen_distance=-0.1).validate()

    def test_invalid_n_index(self):
        with self.assertRaises(ValueError):
            SimulationConfig(system='thick', lens_n=0.5).validate()

    def test_save_and_load(self):
        import os
        import tempfile
        cfg = SimulationConfig(system='thin', object_distance=0.25)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tmpname = f.name
        try:
            cfg.save(tmpname)
            cfg2 = SimulationConfig.load(tmpname)
            self.assertEqual(cfg.system, cfg2.system)
            self.assertEqual(cfg.object_distance, cfg2.object_distance)
        finally:
            os.unlink(tmpname)


class TestOutputValidation(unittest.TestCase):
    def test_valid_relative_prefix(self):
        from config import validate_output_prefix
        self.assertEqual(validate_output_prefix('output'), [])
        self.assertEqual(validate_output_prefix('sub/dir/file'), [])
        self.assertEqual(validate_output_prefix('name-1.test_'), [])

    def test_empty_prefix(self):
        from config import validate_output_prefix
        self.assertTrue(len(validate_output_prefix('')) > 0)

    def test_absolute_path_rejected(self):
        from config import validate_output_prefix
        self.assertTrue(len(validate_output_prefix('/tmp/out')) > 0)

    def test_parent_dir_rejected(self):
        from config import validate_output_prefix
        self.assertTrue(len(validate_output_prefix('../out')) > 0)

    def test_invalid_chars_rejected(self):
        from config import validate_output_prefix
        self.assertTrue(len(validate_output_prefix('out;rm')) > 0)
        self.assertTrue(len(validate_output_prefix('out file')) > 0)

    def test_too_long_prefix(self):
        from config import validate_output_prefix
        self.assertTrue(len(validate_output_prefix('a' * 101)) > 0)


class TestConfigPathValidation(unittest.TestCase):
    def test_nonexistent_file(self):
        from config import validate_config_path
        self.assertTrue(len(validate_config_path('/nonexistent/path/x.json')) > 0)

    def test_empty_path(self):
        from config import validate_config_path
        self.assertTrue(len(validate_config_path('')) > 0)

    def test_invalid_json(self):
        import os
        import tempfile
        from config import validate_config_path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{not valid json')
            tmpname = f.name
        try:
            self.assertTrue(len(validate_config_path(tmpname)) > 0)
        finally:
            os.unlink(tmpname)

    def test_valid_json(self):
        import os
        import tempfile
        from config import validate_config_path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"system": "thin"}')
            tmpname = f.name
        try:
            self.assertEqual(validate_config_path(tmpname), [])
        finally:
            os.unlink(tmpname)


class TestArrowObject(unittest.TestCase):
    def test_arrow_has_nonzero_pixels(self):
        arrow, pts = make_arrow_object(0.01, 40)
        self.assertGreater((arrow > 0).sum(), 0)
        self.assertGreater(len(pts), 0)

    def test_arrow_within_bounds(self):
        size = 0.005
        _, pts = make_arrow_object(size, 30)
        self.assertTrue(np.all(np.abs(pts[:, 0]) <= size / 2 + 1e-9))
        self.assertTrue(np.all(np.abs(pts[:, 1]) <= size / 2 + 1e-9))


class TestThickLensFormula(unittest.TestCase):
    def test_lensmaker_thin_limit(self):
        cfg = SimulationConfig(system='thick',
                                lens_R1=0.10, lens_R2=-0.10,
                                lens_thickness=1e-3, lens_n=1.5,
                                lens_thick_aperture=0.02)
        cfg.validate()
        _, info = build_system(cfg)
        f_thin_approx = 1.0 / ((1.5 - 1) * (1/0.10 - 1/(-0.10)))
        self.assertAlmostEqual(info['f_eff_theory'], f_thin_approx, places=3)


if __name__ == '__main__':
    unittest.main()
