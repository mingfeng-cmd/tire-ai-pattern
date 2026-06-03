from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tire_ai_pattern.common.exceptions import InputDataError, InputTypeError
from tire_ai_pattern.models.enums import (
    ImageFormatEnum,
    ImageModeEnum,
    LevelEnum,
    RegionEnum,
    SourceTypeEnum,
)
from tire_ai_pattern.models.image_models import BigImage, ImageBiz, ImageMeta, SmallImage
from tire_ai_pattern.models.rule_models import Rule6Feature, Rule9Config, Rule9Feature, Rule9Score
from tire_ai_pattern.rules.executors.rule9 import Rule9Executor
from tire_ai_pattern.utils.image_utils import base64_to_ndarray, load_image_to_base64


IMAGE_SIZE = 128
DATASET_ROOT = Path("tests/datasets/task_rule9_vis")
BASELINE_PATH = DATASET_ROOT / "baseline.json"


def load_rule9_baseline_cases() -> list[dict]:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["cases"]


def get_rule9_dataset_path(relative_path: str) -> Path:
    path = Path(relative_path)
    rst = {
        "is_relative": not path.is_absolute(),
        "stays_in_rule9_dataset": ".." not in path.parts,
    }
    expect_rst = {
        "is_relative": True,
        "stays_in_rule9_dataset": True,
    }
    assert rst == expect_rst
    return DATASET_ROOT / path


def make_rule9_config(
    *,
    max_score: int = 4,
    transverse_sipe_width: float = 0.6,
    min_sipe_count_rib1_5: int = 0,
    max_sipe_count_rib1_5: int = 2,
    min_sipe_count_rib2_4: int = 0,
    max_sipe_count_rib2_4: int = 3,
) -> Rule9Config:
    return Rule9Config(
        max_score=max_score,
        transverse_sipe_width=transverse_sipe_width,
        min_sipe_count_rib1_5=min_sipe_count_rib1_5,
        max_sipe_count_rib1_5=max_sipe_count_rib1_5,
        min_sipe_count_rib2_4=min_sipe_count_rib2_4,
        max_sipe_count_rib2_4=max_sipe_count_rib2_4,
    )


def make_meta(width: int = IMAGE_SIZE, height: int = IMAGE_SIZE, size: int = 1) -> ImageMeta:
    return ImageMeta(
        width=width,
        height=height,
        channels=3,
        mode=ImageModeEnum.RGB,
        format=ImageFormatEnum.PNG,
        size=size,
    )


def make_small_image(
    region: RegionEnum | None = RegionEnum.CENTER,
    image_base64: str = "data:image/png;base64,small",
    meta: ImageMeta | None = None,
) -> SmallImage:
    source_type = SourceTypeEnum.ORIGINAL if region is not None else SourceTypeEnum.CONCAT
    return SmallImage(
        image_base64=image_base64,
        meta=meta or make_meta(),
        biz=ImageBiz(level=LevelEnum.SMALL, region=region, source_type=source_type),
    )


def make_big_image() -> BigImage:
    return BigImage(
        image_base64="data:image/png;base64,big",
        meta=make_meta(),
        biz=ImageBiz(level=LevelEnum.BIG, source_type=SourceTypeEnum.CONCAT),
    )


def make_baseline_small_image(baseline_case: dict) -> SmallImage:
    image_path = get_rule9_dataset_path(baseline_case["image_path"])
    return make_small_image(
        region=RegionEnum(baseline_case["region"]),
        image_base64=load_image_to_base64(image_path),
        meta=make_meta(size=image_path.stat().st_size),
    )


def test_baseline_includes_all_rule9_input_images():
    """Every Rule9 dataset input image should be covered by baseline comparisons."""
    baseline_image_paths = {case["image_path"] for case in load_rule9_baseline_cases()}
    dataset_image_paths = {
        image_path.relative_to(DATASET_ROOT).as_posix()
        for image_dir in (DATASET_ROOT / "center_inf", DATASET_ROOT / "side_inf")
        for image_path in image_dir.glob("*.png")
    }

    rst = {
        "missing_from_baseline": sorted(dataset_image_paths - baseline_image_paths),
        "extra_in_baseline": sorted(baseline_image_paths - dataset_image_paths),
    }
    expect_rst = {
        "missing_from_baseline": [],
        "extra_in_baseline": [],
    }
    assert rst == expect_rst


def test_baseline_includes_all_rule9_debug_golden_images():
    """Every Rule9 debug golden image should be referenced by a baseline case."""
    baseline_debug_paths = {case["debug_image_path"] for case in load_rule9_baseline_cases()}
    dataset_debug_paths = {
        image_path.relative_to(DATASET_ROOT).as_posix()
        for image_path in (DATASET_ROOT / "debug_baseline").rglob("*.png")
    }

    rst = {
        "missing_from_baseline": sorted(dataset_debug_paths - baseline_debug_paths),
        "extra_in_baseline": sorted(baseline_debug_paths - dataset_debug_paths),
    }
    expect_rst = {
        "missing_from_baseline": [],
        "extra_in_baseline": [],
    }
    assert rst == expect_rst


def test_exec_feature_converts_center_detector_result_to_rib2_4_feature(monkeypatch):
    """Rule9 center 小图应把横向钢片数量映射到 RIB2/3/4 字段。"""
    decoded_image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    calls = {"base64": [], "detector": []}

    def fake_base64_to_ndarray(image_base64: str) -> np.ndarray:
        calls["base64"].append(image_base64)
        return decoded_image

    def fake_detect_transverse_sipes(image_array: np.ndarray, **kwargs):
        calls["detector"].append({"received_decoded_image": image_array is decoded_image, **kwargs})
        return 2, [32.0, 96.0], [1.0, 1.0], None, None

    monkeypatch.setattr("tire_ai_pattern.rules.executors.rule9.base64_to_ndarray", fake_base64_to_ndarray)
    monkeypatch.setattr("tire_ai_pattern.rules.executors.rule9.detect_transverse_sipes", fake_detect_transverse_sipes)

    feature = Rule9Executor().exec_feature(make_small_image(RegionEnum.CENTER), make_rule9_config())

    rst = {
        "feature": feature,
        "calls": calls,
    }
    expect_rst = {
        "feature": Rule9Feature(
            num_transverse_sipes_rib1_5=0,
            num_transverse_sipes_rib2_4=2,
            is_count_valid=True,
            region=RegionEnum.CENTER,
        ),
        "calls": {
            "base64": ["data:image/png;base64,small"],
            "detector": [
                {
                    "received_decoded_image": True,
                    "nominal_width_px": 0.6,
                    "is_debug": False,
                }
            ],
        },
    }
    assert rst == expect_rst


def test_exec_feature_converts_side_detector_result_to_rib1_5_feature(monkeypatch):
    """Rule9 side 小图应把横向钢片数量映射到 RIB1/5 字段。"""
    decoded_image = np.full((40, 80, 3), 255, dtype=np.uint8)
    calls = {"base64": [], "detector": []}

    def fake_base64_to_ndarray(image_base64: str) -> np.ndarray:
        calls["base64"].append(image_base64)
        return decoded_image

    def fake_detect_transverse_sipes(image_array: np.ndarray, **kwargs):
        calls["detector"].append({"shape": image_array.shape, **kwargs})
        return 1, [20.0], [1.0], None, None

    monkeypatch.setattr("tire_ai_pattern.rules.executors.rule9.base64_to_ndarray", fake_base64_to_ndarray)
    monkeypatch.setattr("tire_ai_pattern.rules.executors.rule9.detect_transverse_sipes", fake_detect_transverse_sipes)

    feature = Rule9Executor().exec_feature(make_small_image(RegionEnum.SIDE), make_rule9_config(transverse_sipe_width=1.2))

    rst = {
        "feature": feature,
        "calls": calls,
    }
    expect_rst = {
        "feature": Rule9Feature(
            num_transverse_sipes_rib1_5=1,
            num_transverse_sipes_rib2_4=0,
            is_count_valid=True,
            region=RegionEnum.SIDE,
        ),
        "calls": {
            "base64": ["data:image/png;base64,small"],
            "detector": [
                {
                    "shape": (40, 80, 3),
                    "nominal_width_px": 1.2,
                    "is_debug": False,
                }
            ],
        },
    }
    assert rst == expect_rst


def test_exec_feature_passes_debug_and_returns_visualization(monkeypatch):
    """Rule9 应透传 is_debug，并只在 debug 模式下填充可视化结果。"""
    decoded_image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    debug_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    calls = {"detector": []}

    def fake_base64_to_ndarray(_image_base64: str) -> np.ndarray:
        return decoded_image

    def fake_detect_transverse_sipes(image_array: np.ndarray, **kwargs):
        calls["detector"].append({"received_decoded_image": image_array is decoded_image, **kwargs})
        return 1, [64.0], [1.0], None, debug_image

    monkeypatch.setattr("tire_ai_pattern.rules.executors.rule9.base64_to_ndarray", fake_base64_to_ndarray)
    monkeypatch.setattr("tire_ai_pattern.rules.executors.rule9.detect_transverse_sipes", fake_detect_transverse_sipes)

    feature = Rule9Executor().exec_feature(make_small_image(RegionEnum.CENTER), make_rule9_config(), is_debug=True)

    rst = {
        "feature_fields": {
            "num_transverse_sipes_rib2_4": feature.num_transverse_sipes_rib2_4,
            "vis_names": feature.vis_names,
            "vis_image_prefix": feature.vis_images[0].split(",", 1)[0] if feature.vis_images else None,
        },
        "calls": calls,
    }
    expect_rst = {
        "feature_fields": {
            "num_transverse_sipes_rib2_4": 1,
            "vis_names": ["rule9_transverse_sipes.png"],
            "vis_image_prefix": "data:image/png;base64",
        },
        "calls": {
            "detector": [
                {
                    "received_decoded_image": True,
                    "nominal_width_px": 0.6,
                    "is_debug": True,
                }
            ],
        },
    }
    assert rst == expect_rst


@pytest.mark.parametrize("baseline_case", load_rule9_baseline_cases(), ids=lambda case: case["image_path"])
def test_exec_feature_and_score_match_real_image_baseline(baseline_case: dict):
    """真实图片应输出已固化的 Rule9 特征和评分 baseline。"""
    executor = Rule9Executor()
    config = make_rule9_config()
    small_image = make_baseline_small_image(baseline_case)

    feature = executor.exec_feature(small_image, config)
    score = executor.exec_score(config, feature)

    rst = {
        "feature": feature.model_dump(mode="json"),
        "score": score.model_dump(mode="json"),
    }
    expect_rst = {
        "feature": baseline_case["feature"],
        "score": baseline_case["score"],
    }
    assert rst == expect_rst


@pytest.mark.parametrize("baseline_case", load_rule9_baseline_cases(), ids=lambda case: case["image_path"])
def test_exec_feature_debug_vis_matches_rule9_golden(baseline_case: dict):
    """Rule9 debug output should match its own image golden baseline."""
    executor = Rule9Executor()
    config = make_rule9_config()
    small_image = make_baseline_small_image(baseline_case)

    feature = executor.exec_feature(small_image, config, is_debug=True)

    rst = {
        "vis_names": feature.vis_names,
        "vis_images_count": len(feature.vis_images) if feature.vis_images is not None else 0,
    }
    expect_rst = {
        "vis_names": ["rule9_transverse_sipes.png"],
        "vis_images_count": 1,
    }
    assert rst == expect_rst

    debug_array = base64_to_ndarray(feature.vis_images[0])
    golden_path = get_rule9_dataset_path(baseline_case["debug_image_path"])
    golden = cv2.imread(str(golden_path), cv2.IMREAD_COLOR)
    if golden is None:
        pytest.fail(f"failed to read Rule9 debug golden: {golden_path}")

    rst = {
        "debug_shape": debug_array.shape,
        "golden_shape": golden.shape,
        "matches_golden": np.array_equal(debug_array, golden),
    }
    expect_rst = {
        "debug_shape": golden.shape,
        "golden_shape": golden.shape,
        "matches_golden": True,
    }
    assert rst == expect_rst


@pytest.mark.parametrize(
    ("region", "rib1_5_count", "rib2_4_count", "expected_score"),
    [
        (RegionEnum.SIDE, 0, 0, 4),
        (RegionEnum.SIDE, 2, 0, 4),
        (RegionEnum.SIDE, 3, 0, 0),
        (RegionEnum.CENTER, 0, 0, 4),
        (RegionEnum.CENTER, 0, 3, 4),
        (RegionEnum.CENTER, 0, 4, 0),
    ],
)
def test_exec_score_uses_region_specific_count_limit(
    region: RegionEnum,
    rib1_5_count: int,
    rib2_4_count: int,
    expected_score: int,
):
    """Rule9 评分应按 side/center 选择不同数量上限。"""
    score = Rule9Executor().exec_score(
        make_rule9_config(),
        Rule9Feature(
            num_transverse_sipes_rib1_5=rib1_5_count,
            num_transverse_sipes_rib2_4=rib2_4_count,
            is_count_valid=expected_score > 0,
            region=region,
        ),
    )

    rst = score
    expect_rst = Rule9Score(score=expected_score)
    assert rst == expect_rst


def test_exec_feature_rejects_non_small_image():
    """Rule9 是小图规则，不能直接接收 BigImage。"""
    with pytest.raises(InputTypeError, match="SmallImage"):
        Rule9Executor().exec_feature(make_big_image(), make_rule9_config())


def test_exec_feature_rejects_missing_region():
    """Rule9 选择 RIB 组需要 center/side 区域信息。"""
    with pytest.raises(InputDataError, match="image.biz.region"):
        Rule9Executor().exec_feature(make_small_image(None), make_rule9_config())


def test_exec_feature_rejects_invalid_debug_flag():
    """Rule9 debug 开关必须是 bool。"""
    with pytest.raises(InputTypeError, match="is_debug"):
        Rule9Executor().exec_feature(make_small_image(), make_rule9_config(), is_debug=1)  # type: ignore[arg-type]


def test_exec_score_rejects_wrong_feature_type():
    """Rule9 打分只接受 Rule9Feature。"""
    with pytest.raises(InputTypeError, match="Rule9Feature"):
        Rule9Executor().exec_score(make_rule9_config(), Rule6Feature(is_continuous=True))


def test_exec_score_rejects_invalid_feature_region():
    """绕过模型校验构造的非法 region 应在评分入口被拒绝。"""
    feature = Rule9Feature.model_construct(
        num_transverse_sipes_rib1_5=1,
        num_transverse_sipes_rib2_4=0,
        is_count_valid=True,
        region=None,
    )

    with pytest.raises(InputDataError, match="feature.region"):
        Rule9Executor().exec_score(make_rule9_config(), feature)


def test_exec_score_rejects_negative_feature_count():
    """绕过模型校验构造的非法数量应在评分入口被拒绝。"""
    feature = Rule9Feature.model_construct(
        num_transverse_sipes_rib1_5=-1,
        num_transverse_sipes_rib2_4=0,
        is_count_valid=False,
        region=RegionEnum.SIDE,
    )

    with pytest.raises(InputDataError, match="num_transverse_sipes_rib1_5"):
        Rule9Executor().exec_score(make_rule9_config(), feature)


def test_exec_score_rejects_invalid_config_limit():
    """Rule9 配置中的数量下限不能大于上限。"""
    config = make_rule9_config(min_sipe_count_rib1_5=3, max_sipe_count_rib1_5=2)
    feature = Rule9Feature(
        num_transverse_sipes_rib1_5=2,
        num_transverse_sipes_rib2_4=0,
        is_count_valid=True,
        region=RegionEnum.SIDE,
    )

    with pytest.raises(InputDataError, match="min_sipe_count_rib1_5"):
        Rule9Executor().exec_score(config, feature)
