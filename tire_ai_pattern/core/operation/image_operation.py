"""
核心图像操作算法模块

实现纯粹的图像处理算法，输入输出均为基本类型（np.ndarray, base64字符串等）
不依赖任何业务数据类，与业务逻辑完全解耦。
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from tire_ai_pattern.models.enums import RibOperation


def apply_single_rib_operation(image: np.ndarray, operation: RibOperation) -> np.ndarray:
    """
    执行单个RIB原子操作

    Args:
        image: 输入图像 (np.ndarray, BGR格式)
        operation: RIB原子操作枚举值

    Returns:
        np.ndarray: 处理后的图像

    Raises:
        ValueError: 当操作不支持时
        RuntimeError: 当图像处理失败时
    """
    if image is None or image.size == 0:
        raise ValueError("输入图像不能为空")

    h, w = image.shape[:2]

    try:
        if operation == RibOperation.NONE:
            return image.copy()

        elif operation == RibOperation.FLIP_LR:
            # 左右对称
            return cv2.flip(image, 1)

        elif operation == RibOperation.FLIP:
            # 旋转180度
            return cv2.rotate(image, cv2.ROTATE_180)

        elif operation == RibOperation.OFFSET_VERTICAL_HALF:
            # 纵向错位半高度
            if h < 2:
                return image.copy()
            return np.roll(image, shift=h // 2, axis=0)

        elif operation == RibOperation.LEFT_FLIP_LR:
            # 左半左右对称覆盖右侧
            if w < 2:
                return image.copy()
            left_half = image[:, :w//2]
            flipped_left = cv2.flip(left_half, 1)
            result = image.copy()
            # 处理奇数宽度：右半部分可能比左半部分少1像素
            right_width = w - w//2
            if flipped_left.shape[1] >= right_width:
                result[:, w//2:] = flipped_left[:, :right_width]
            else:
                # 如果翻转后的左半部分不够宽，用最后一列填充
                result[:, w//2:] = np.tile(flipped_left[:, -1:], (1, right_width))
            return result

        elif operation == RibOperation.LEFT_FLIP:
            # 左半旋转180覆盖右侧
            if w < 2:
                return image.copy()
            left_half = image[:, :w//2]
            rotated_left = cv2.rotate(left_half, cv2.ROTATE_180)
            result = image.copy()
            right_width = w - w//2
            if rotated_left.shape[1] >= right_width:
                result[:, w//2:] = rotated_left[:, :right_width]
            else:
                result[:, w//2:] = np.tile(rotated_left[:, -1:], (1, right_width))
            return result

        elif operation == RibOperation.RESIZE_HORIZONTAL_2X:
            # 横向拉伸2倍
            return cv2.resize(image, (w * 2, h))

        elif operation == RibOperation.LEFT:
            # 截取左边 (等同于 LEFT_1_1)
            if w < 2:
                return image.copy()
            return image[:, :w//2]

        elif operation == RibOperation.RIGHT:
            # 截取右边 (等同于 RIGHT_1_1)
            if w < 2:
                return image.copy()
            return image[:, w//2:]

        elif operation == RibOperation.RESIZE_HORIZONTAL_1_5X:
            # 横向拉伸1.5倍
            new_width = int(w * 1.5)
            if new_width <= 0:
                new_width = 1
            return cv2.resize(image, (new_width, h))

        elif operation == RibOperation.RESIZE_HORIZONTAL_3X:
            # 横向拉伸3倍
            return cv2.resize(image, (w * 3, h))

        elif operation == RibOperation.LEFT_2_3:
            # 截取左2/3
            target_width = int(w * 2/3)
            if target_width <= 0:
                target_width = 1
            return image[:, :target_width]

        elif operation == RibOperation.RIGHT_2_3:
            # 截取右2/3
            start_col = int(w * 1/3)
            if start_col >= w:
                start_col = w - 1
            return image[:, start_col:]

        elif operation == RibOperation.LEFT_1_3:
            # 截取左1/3
            target_width = int(w * 1/3)
            if target_width <= 0:
                target_width = 1
            return image[:, :target_width]

        elif operation == RibOperation.RIGHT_1_3:
            # 截取右1/3
            start_col = int(w * 2/3)
            if start_col >= w:
                start_col = w - 1
            return image[:, start_col:]

        elif operation == RibOperation._RESIZE_AS_FIRST_RIB:
            # 图片大小向第一张图对齐 - 此操作需要额外参数，在拼接后处理中实现
            raise NotImplementedError("_RESIZE_AS_FIRST_RIB 需要在拼接上下文中处理")

        else:
            raise ValueError(f"Unsupported operation: {operation}")

    except Exception as e:
        raise RuntimeError(f"执行RIB操作 {operation} 时发生错误: {str(e)}")


def apply_rib_operations_sequence(
    image: np.ndarray,
    operations: Tuple[RibOperation, ...]
) -> np.ndarray:
    """
    按顺序执行RIB操作序列

    示例:
    - ("resize_horizontal_2x", "left") → 先横向拉伸2倍，再截取左边
    - ("",) 或 () → 无操作

    Args:
        image: 输入图像 (np.ndarray)
        operations: 操作序列元组

    Returns:
        np.ndarray: 处理后的图像
    """
    if not operations:
        return image.copy()

    current_image = image.copy()
    for operation in operations:
        if operation != RibOperation.NONE:  # 跳过空操作
            current_image = apply_single_rib_operation(current_image, operation)

    return current_image


def repeat_vertically(image: np.ndarray, num_times: int) -> np.ndarray:
    """
    纵向重复图像指定次数

    Args:
        image: 输入图像 (np.ndarray)
        num_times: 重复次数，必须为正整数

    Returns:
        np.ndarray: 纵向重复后的图像
    """
    if num_times <= 0:
        raise ValueError("重复次数必须为正整数")
    if image is None or image.size == 0:
        raise ValueError("输入图像不能为空")

    if num_times == 1:
        return image.copy()

    return np.tile(image, (num_times, 1, 1))


def apply_opacity(image: np.ndarray, opacity: int) -> np.ndarray:
    """
    应用透明度 (0-255)

    Args:
        image: 输入图像 (np.ndarray, BGR格式)
        opacity: 透明度值，0-255 (0=完全透明, 255=完全不透明)

    Returns:
        np.ndarray: 应用透明度后的图像 (BGRA格式)
    """
    if opacity < 0 or opacity > 255:
        raise ValueError("透明度值必须在0-255范围内")
    if image is None or image.size == 0:
        raise ValueError("输入图像不能为空")

    # 转换为BGRA格式
    if len(image.shape) == 2:
        # 灰度图
        bgra = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    else:
        # BGR图
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # 设置alpha通道
    bgra[:, :, 3] = opacity

    return bgra


def horizontal_concatenate(images: List[np.ndarray]) -> np.ndarray:
    """
    横向拼接多个图像

    Args:
        images: 图像列表，所有图像必须具有相同的高度

    Returns:
        np.ndarray: 横向拼接后的图像

    Raises:
        ValueError: 当图像高度不一致或列表为空时
    """
    if not images:
        raise ValueError("图像列表不能为空")

    if len(images) == 1:
        return images[0].copy()

    # 验证所有图像高度一致
    heights = [img.shape[0] for img in images]
    if len(set(heights)) > 1:
        raise ValueError(f"所有图像高度必须一致，当前高度: {heights}")

    # 拼接图像
    return np.concatenate(images, axis=1)


def overlay_decoration(
    base_image: np.ndarray,
    left_decoration: np.ndarray,
    right_decoration: np.ndarray
) -> np.ndarray:
    """
    在基础图像左右边缘应用半透明装饰覆盖（不改变图像分辨率）

    Args:
        base_image: 基础图像 (np.ndarray, BGR格式)
        left_decoration: 左侧装饰图像 (np.ndarray, 可以是BGR或BGRA格式)
        right_decoration: 右侧装饰图像 (np.ndarray, 可以是BGR或BGRA格式)

    Returns:
        np.ndarray: 应用半透明装饰后的图像（BGR格式，尺寸与base_image相同）
    """
    if base_image is None or left_decoration is None or right_decoration is None:
        raise ValueError("输入图像不能为空")

    base_h, base_w = base_image.shape[:2]

    # 创建结果图像
    result = base_image.copy().astype(np.float32)

    # 处理左侧装饰
    if left_decoration.shape[2] == 4:  # BGRA格式，有alpha通道
        left_h, left_w = left_decoration.shape[:2]
        if left_h != base_h:
            raise ValueError("左侧装饰图像高度必须与基础图像高度一致")
        if left_w > base_w:
            raise ValueError("左侧装饰图像宽度不能超过基础图像宽度")

        # 提取RGB和Alpha通道
        left_rgb = left_decoration[:, :, :3].astype(np.float32)
        left_alpha = left_decoration[:, :, 3] / 255.0

        # 应用左侧半透明覆盖
        for c in range(3):  # B, G, R 通道
            result[:, :left_w, c] = (
                left_alpha * left_rgb[:, :, c] +
                (1 - left_alpha) * result[:, :left_w, c]
            )
    else:  # BGR格式，无透明度，直接覆盖
        left_h, left_w = left_decoration.shape[:2]
        if left_h != base_h:
            raise ValueError("左侧装饰图像高度必须与基础图像高度一致")
        if left_w > base_w:
            raise ValueError("左侧装饰图像宽度不能超过基础图像宽度")

        result[:, :left_w] = left_decoration.astype(np.float32)

    # 处理右侧装饰
    if right_decoration.shape[2] == 4:  # BGRA格式，有alpha通道
        right_h, right_w = right_decoration.shape[:2]
        if right_h != base_h:
            raise ValueError("右侧装饰图像高度必须与基础图像高度一致")
        if right_w > base_w:
            raise ValueError("右侧装饰图像宽度不能超过基础图像宽度")

        # 提取RGB和Alpha通道
        right_rgb = right_decoration[:, :, :3].astype(np.float32)
        right_alpha = right_decoration[:, :, 3] / 255.0

        # 应用右侧半透明覆盖
        for c in range(3):  # B, G, R 通道
            result[:, base_w - right_w:, c] = (
                right_alpha * right_rgb[:, :, c] +
                (1 - right_alpha) * result[:, base_w - right_w:, c]
            )
    else:  # BGR格式，无透明度，直接覆盖
        right_h, right_w = right_decoration.shape[:2]
        if right_h != base_h:
            raise ValueError("右侧装饰图像高度必须与基础图像高度一致")
        if right_w > base_w:
            raise ValueError("右侧装饰图像宽度不能超过基础图像宽度")

        result[:, base_w - right_w:] = right_decoration.astype(np.float32)

    # 转换回uint8格式
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result
