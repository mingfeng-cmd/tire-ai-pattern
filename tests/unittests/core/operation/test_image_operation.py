"""
核心图像操作算法单元测试
"""

import unittest
import numpy as np
import cv2
from tire_ai_pattern.core.operation.image_operation import (
    apply_single_rib_operation,
    apply_rib_operations_sequence,
    repeat_vertically,
    apply_opacity,
    horizontal_concatenate,
    overlay_decoration
)
from tire_ai_pattern.models.enums import RibOperation


class TestImageOperation(unittest.TestCase):
    """核心图像操作算法测试"""

    def setUp(self):
        """测试前准备"""
        self.image_h = 100
        self.image_w = 100
        self.image_c = 3
        self.test_image = np.ones((self.image_h, self.image_w, self.image_c), dtype=np.uint8) * 128
        self.gray_image = np.ones((self.image_h, self.image_w), dtype=np.uint8) * 128

    def test_apply_single_rib_operation_none(self):
        """测试NONE操作"""
        result = apply_single_rib_operation(self.test_image, RibOperation.NONE)
        expected = self.test_image
        np.testing.assert_array_equal(result, expected)

    def test_apply_single_rib_operation_flip_lr(self):
        """测试FLIP_LR操作"""
        result = apply_single_rib_operation(self.test_image, RibOperation.FLIP_LR)
        expected = cv2.flip(self.test_image, 1)
        np.testing.assert_array_equal(result, expected)

    def test_apply_single_rib_operation_flip(self):
        """测试FLIP操作"""
        result = apply_single_rib_operation(self.test_image, RibOperation.FLIP)
        expected = cv2.rotate(self.test_image, cv2.ROTATE_180)
        np.testing.assert_array_equal(result, expected)

    def test_apply_single_rib_operation_offset_vertical_half(self):
        """测试纵向半高错位操作。"""
        image = np.arange(6, dtype=np.uint8).reshape(6, 1, 1)

        result = apply_single_rib_operation(image, RibOperation.OFFSET_VERTICAL_HALF)
        expected = np.roll(image, shift=3, axis=0)

        np.testing.assert_array_equal(result, expected)

    def test_apply_single_rib_operation_left_flip_lr_offset_right_half(self):
        """测试左半镜像到右侧后仅右侧纵向错位。"""
        left_half = np.array(
            [
                [[1], [2]],
                [[3], [4]],
                [[5], [6]],
                [[7], [8]],
            ],
            dtype=np.uint8,
        )
        right_half = np.zeros((4, 2, 1), dtype=np.uint8)
        image = np.concatenate([left_half, right_half], axis=1)

        result = apply_single_rib_operation(image, RibOperation.LEFT_FLIP_LR_OFFSET_RIGHT_HALF)
        expected_right = np.roll(cv2.flip(left_half, 1)[:, :, np.newaxis], shift=2, axis=0)

        np.testing.assert_array_equal(result[:, :2], left_half)
        np.testing.assert_array_equal(result[:, 2:], expected_right)

    def test_apply_single_rib_operation_resize_horizontal_2x(self):
        """测试RESIZE_HORIZONTAL_2X操作"""
        result = apply_single_rib_operation(self.test_image, RibOperation.RESIZE_HORIZONTAL_2X)
        expected_height = self.image_h
        expected_width = self.image_w * 2
        self.assertEqual(result.shape[0], expected_height)
        self.assertEqual(result.shape[1], expected_width)

    def test_apply_single_rib_operation_left_right(self):
        """测试LEFT和RIGHT操作"""
        expected_width = self.image_w // 2

        result_left = apply_single_rib_operation(self.test_image, RibOperation.LEFT)
        self.assertEqual(result_left.shape[1], expected_width)

        result_right = apply_single_rib_operation(self.test_image, RibOperation.RIGHT)
        self.assertEqual(result_right.shape[1], expected_width)

    def test_apply_single_rib_operation_fractions(self):
        """测试分数操作"""
        # LEFT_1_3
        result = apply_single_rib_operation(self.test_image, RibOperation.LEFT_1_3)
        expected_width = int(self.image_w * 1 / 3)
        self.assertEqual(result.shape[1], expected_width)

        # RIGHT_1_3
        result = apply_single_rib_operation(self.test_image, RibOperation.RIGHT_1_3)
        expected_width = self.image_w - int(self.image_w * 2 / 3)
        self.assertEqual(result.shape[1], expected_width)

    def test_apply_rib_operations_sequence(self):
        """测试操作序列 — resize_2x + left → 恢复原尺寸"""
        operations = (RibOperation.RESIZE_HORIZONTAL_2X, RibOperation.LEFT)
        result = apply_rib_operations_sequence(self.test_image, operations)
        expected_height = self.image_h
        expected_width = self.image_w
        self.assertEqual(result.shape[0], expected_height)
        self.assertEqual(result.shape[1], expected_width)

    def test_repeat_vertically(self):
        """测试纵向重复"""
        num_repeats = 3
        result = repeat_vertically(self.test_image, num_repeats)
        expected_height = self.image_h * num_repeats
        expected_width = self.image_w
        self.assertEqual(result.shape[0], expected_height)
        self.assertEqual(result.shape[1], expected_width)

    def test_apply_opacity(self):
        """测试透明度应用"""
        opacity = 128
        result = apply_opacity(self.test_image, opacity)
        expected_channels = 4  # BGRA
        self.assertEqual(result.shape[2], expected_channels)
        expected_alpha = np.full((self.image_h, self.image_w), opacity, dtype=np.uint8)
        np.testing.assert_array_equal(result[:, :, 3], expected_alpha)

    def test_horizontal_concatenate(self):
        """测试横向拼接"""
        images = [self.test_image, self.test_image]
        result = horizontal_concatenate(images)
        expected_height = self.image_h
        expected_width = self.image_w * 2
        self.assertEqual(result.shape[0], expected_height)
        self.assertEqual(result.shape[1], expected_width)

    def test_overlay_decoration(self):
        """测试装饰覆盖 — 不改变图像分辨率，在底图左右边缘原地覆盖装饰图"""
        left_dec = np.ones((self.image_h, 50, 3), dtype=np.uint8) * 200
        right_dec = np.ones((self.image_h, 50, 3), dtype=np.uint8) * 50
        result = overlay_decoration(self.test_image, left_dec, right_dec)
        expected_height = self.image_h
        expected_width = self.image_w
        self.assertEqual(result.shape[0], expected_height)
        self.assertEqual(result.shape[1], expected_width)

    def test_error_cases(self):
        """测试错误情况"""
        # 测试空图像
        with self.assertRaises(ValueError):
            apply_single_rib_operation(None, RibOperation.NONE)

        with self.assertRaises(ValueError):
            apply_single_rib_operation(np.array([]), RibOperation.NONE)

        # 测试无效操作
        with self.assertRaises(RuntimeError):
            apply_single_rib_operation(self.test_image, "invalid_operation")

        # 测试无效重复次数
        with self.assertRaises(ValueError):
            repeat_vertically(self.test_image, 0)

        with self.assertRaises(ValueError):
            repeat_vertically(self.test_image, -1)

        # 测试无效透明度
        with self.assertRaises(ValueError):
            apply_opacity(self.test_image, -1)

        with self.assertRaises(ValueError):
            apply_opacity(self.test_image, 256)


if __name__ == '__main__':
    unittest.main()
