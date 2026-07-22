import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "gaussian-splatting"))

from colmap_io import Camera
from distort_back_renders import build_distort_back_maps, copy_pinhole_scene
from prepare_round2_kaggle import ALL_SCENES, GROUP_SCENES
from render_test_poses import render_output_path, short_split_name
from undistort_simple_radial_dataset import (
    choose_undistorted_matrix,
    rewrite_test_poses,
)

camera_spec = importlib.util.spec_from_file_location(
    "camera_module", ROOT / "gaussian-splatting" / "scene" / "cameras.py"
)
camera_module = importlib.util.module_from_spec(camera_spec)
camera_spec.loader.exec_module(camera_module)
cache_image_tensor = camera_module.cache_image_tensor
image_tensor_to_float = camera_module.image_tensor_to_float


class CameraPipelineTests(unittest.TestCase):
    def test_auto_policy_preserves_full_fov_for_strong_negative_k(self):
        camera = Camera(
            id=1,
            model="SIMPLE_RADIAL",
            width=1320,
            height=989,
            params=np.array([925.477, 660.0, 494.5, -0.114794]),
        )
        original_k, _, undistorted_k, policy = choose_undistorted_matrix(camera, "auto", -0.05)
        self.assertEqual(policy, "full_fov")
        self.assertLess(undistorted_k[0, 0], original_k[0, 0])
        self.assertLess(undistorted_k[1, 1], original_k[1, 1])

    def test_auto_policy_keeps_intrinsics_for_small_positive_k(self):
        camera = Camera(
            id=1,
            model="SIMPLE_RADIAL",
            width=1320,
            height=989,
            params=np.array([928.197, 660.0, 494.5, 0.00891]),
        )
        original_k, _, undistorted_k, policy = choose_undistorted_matrix(camera, "auto", -0.05)
        self.assertEqual(policy, "keep_intrinsics")
        np.testing.assert_allclose(undistorted_k, original_k)

    def test_distort_map_is_identity_without_distortion(self):
        map_x, map_y = build_distort_back_maps(
            width=4,
            height=3,
            original_fx=10.0,
            original_fy=10.0,
            original_cx=2.0,
            original_cy=1.5,
            k=0.0,
        )
        expected_x, expected_y = np.meshgrid(np.arange(4), np.arange(3))
        np.testing.assert_allclose(map_x, expected_x, atol=1e-5)
        np.testing.assert_allclose(map_y, expected_y, atol=1e-5)

    def test_test_pose_intrinsics_are_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "input.csv"
            output_path = directory / "output.csv"
            fieldnames = [
                "image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
                "fx", "fy", "cx", "cy", "width", "height",
            ]
            with open(input_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({name: "0" for name in fieldnames} | {"image_name": "frame.JPG"})
            rewrite_test_poses(
                input_path,
                output_path,
                {"fx": 818.0, "fy": 819.0, "cx": 659.5, "cy": 494.0},
            )
            with open(output_path, encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(float(row["fx"]), 818.0)
            self.assertEqual(float(row["fy"]), 819.0)
            self.assertEqual(float(row["cx"]), 659.5)
            self.assertEqual(float(row["cy"]), 494.0)

    def test_render_intermediate_defaults_to_png_name(self):
        output = render_output_path(Path("renders"), "target.JPG", "png")
        self.assertEqual(output, Path("renders/target.png"))

    def test_private_split_model_prefix_is_stable(self):
        self.assertEqual(short_split_name("private_set1"), "private")
        self.assertEqual(short_split_name("public_set"), "public")

    def test_cpu_camera_cache_uses_compact_uint8_storage(self):
        source = torch.tensor([0.0, 1.0 / 255.0, 0.5, 1.0], dtype=torch.float32)
        cached = cache_image_tensor(source, torch.device("cpu"))
        self.assertEqual(cached.dtype, torch.uint8)
        self.assertEqual(cached.element_size(), 1)
        restored = image_tensor_to_float(cached, device="cpu")
        expected = (source * 255.0).round() / 255.0
        torch.testing.assert_close(restored, expected)

    def test_round2_groups_cover_all_added_scenes_without_overlap(self):
        self.assertFalse(GROUP_SCENES["C"] & GROUP_SCENES["D"])
        self.assertEqual(GROUP_SCENES["C"] | GROUP_SCENES["D"], ALL_SCENES)
        self.assertEqual(len(ALL_SCENES), 7)

    def test_pinhole_scene_is_copied_with_original_submission_name(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            scene_dir = directory / "chair"
            input_dir = directory / "renders"
            output_dir = directory / "submission" / "chair"
            (scene_dir / "test").mkdir(parents=True)
            input_dir.mkdir()
            with open(scene_dir / "test" / "test_poses.csv", "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["image_name"])
                writer.writeheader()
                writer.writerow({"image_name": "frame_001.jpg"})
            source = np.full((8, 12, 3), 127, dtype=np.uint8)
            cv2.imwrite(str(input_dir / "frame_001.png"), source)
            args = SimpleNamespace(keep_original_extension=True, jpeg_quality=99)
            count, result_dir = copy_pinhole_scene(args, "private_set1", scene_dir, input_dir, output_dir)
            self.assertEqual(count, 1)
            self.assertEqual(result_dir, output_dir)
            output_path = output_dir / "frame_001.jpg"
            self.assertTrue(output_path.exists())
            rendered = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            self.assertEqual(rendered.shape, source.shape)


if __name__ == "__main__":
    unittest.main()
