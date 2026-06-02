from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tire_ai_pattern.models.enums import ContinuityModeName
from tire_ai_pattern.models.enums import ImageFormatEnum, ImageModeEnum, LevelEnum, SourceTypeEnum
from tire_ai_pattern.models.image_models import BigImage, ImageBiz, ImageMeta
from tire_ai_pattern.models.rule_models import (
    DecorationItem,
    GrooveSizeItem,
    RibSizeItem,
    Rule1Config,
    Rule2Config,
    Rule3Config,
    Rule6Config,
    Rule8Config,
    Rule11Config,
    Rule16Config,
    Rule100Config,
    Rule101Config,
    Rule102Config,
)
from tire_ai_pattern.models.tire_struct import TireStruct
from tire_ai_pattern.piplines.pipline1 import run_pipeline1
from tire_ai_pattern.piplines.pipline4 import run_pipeline4
from tire_ai_pattern.rules.executors import load_all_executors
from tire_ai_pattern.utils.image_utils import base64_to_ndarray, load_image_to_base64, save_base64_to_image


INPUT_IMAGE = Path("tests/datasets/tire_design_images/images/testcase_001.png")
DEFAULT_OUTPUT_DIR = Path(".results/pipeline4_to_pipeline1/testcase_001")


def run_pipeline4_to_pipeline1_case(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    image_path: Path = INPUT_IMAGE,
    top_n: int = 20,
) -> dict[str, Any]:
    """Run testcase_001 through Pipeline-4 and Pipeline-1, saving visible artifacts."""

    load_all_executors()
    output_dir = output_dir.resolve()
    small_dir = output_dir / "pipeline4_small_images"
    ranked_big_dir = output_dir / "pipeline1_ranked_big_images"
    small_dir.mkdir(parents=True, exist_ok=True)
    ranked_big_dir.mkdir(parents=True, exist_ok=True)
    _clear_pngs(small_dir)
    _clear_pngs(ranked_big_dir)

    tire_struct = _tire_struct_from_big_image(image_path)
    run_pipeline4(tire_struct)
    pipeline4_flag = tire_struct.flag
    pipeline4_err_msg = tire_struct.err_msg
    if tire_struct.flag is not True:
        return _write_manifest(output_dir, tire_struct, pipeline4_flag, pipeline4_err_msg, None, None, [])

    small_image_paths = []
    for index, small_image in enumerate(tire_struct.small_images, 1):
        region = small_image.biz.region.value if small_image.biz.region is not None else "unknown"
        path = small_dir / f"{index:02d}_{region}.png"
        save_base64_to_image(small_image.image_base64, path)
        small_image_paths.append(str(path))

    ranked_results = []
    original_big_image = tire_struct.big_image.model_copy(deep=True)
    split_small_images = [image.model_copy(deep=True) for image in tire_struct.small_images]
    geometry_config = _derive_geometry_config(base64_to_ndarray(original_big_image.image_base64))
    rules_config = _pipeline1_rules_config(geometry_config)

    for rank in range(1, top_n + 1):
        ranked_struct = TireStruct(
            big_image=original_big_image.model_copy(deep=True),
            small_images=[image.model_copy(deep=True) for image in split_small_images],
            rules_config=[config.model_copy(deep=True) for config in rules_config],
            scheme_rank=rank,
            is_debug=False,
        )
        run_pipeline1(ranked_struct)
        big_image_path = ranked_big_dir / f"rank_{rank:02d}_big_image.png"
        if ranked_struct.flag is True and ranked_struct.big_image is not None:
            save_base64_to_image(ranked_struct.big_image.image_base64, big_image_path)
        ranked_results.append(
            {
                "rank": rank,
                "flag": ranked_struct.flag,
                "err_msg": ranked_struct.err_msg,
                "current_score": (
                    ranked_struct.big_image.evaluation.current_score
                    if ranked_struct.big_image is not None and ranked_struct.big_image.evaluation is not None
                    else None
                ),
                 "compliance": (
                    _extract_compliance_score(ranked_struct.big_image)
                    if ranked_struct.big_image is not None
                    else None
                ),
                "big_image_path": str(big_image_path),
            }
        )

    return _write_manifest(
        output_dir,
        tire_struct,
        pipeline4_flag,
        pipeline4_err_msg,
        geometry_config["summary"],
        ranked_results,
        small_image_paths,
    )


def _tire_struct_from_big_image(image_path: Path) -> TireStruct:
    image_base64 = load_image_to_base64(image_path, with_prefix=True)
    image = base64_to_ndarray(image_base64)
    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]

    return TireStruct(
        big_image=BigImage(
            image_base64=image_base64,
            meta=ImageMeta(
                width=width,
                height=height,
                channels=channels,
                mode=ImageModeEnum.GRAY if channels == 1 else ImageModeEnum.RGB,
                format=ImageFormatEnum.PNG,
                size=len(image_base64.split(",", 1)[1]),
            ),
            biz=ImageBiz(level=LevelEnum.BIG, source_type=SourceTypeEnum.CONCAT),
        ),
        rules_config=[],
        scheme_rank=1,
        is_debug=False,
    )


def _pipeline1_rules_config(geometry_config: dict[str, Any]) -> list[Any]:
    summary = geometry_config["summary"]
    height = summary["target_height"]
    return [
        Rule1Config(),
        Rule2Config(),
        Rule3Config(),
        Rule6Config(max_score=10),
        Rule8Config(max_score=4, groove_width_center=25.0, groove_width_side=13.0),
        Rule11Config(
            groove_width=1,
            min_width_offset_px=1,
            edge_margin_ratio=0.1,
            min_segment_length_ratio=0.5,
            max_angle_from_vertical=10,
            max_count_center=3,
            max_count_side=2,
        ),
        Rule16Config(
            continuity_mode_list=[
                ContinuityModeName.CONTINUITY_0,
                ContinuityModeName.CONTINUITY_1,
                ContinuityModeName.CONTINUITY_2,
            ],
        ),
        Rule100Config(
            rib_number=5,
            rib_sizes=[
                RibSizeItem(
                    rib_name=f"rib{index + 1}",
                    num_pitchs=5 if index == 0 else 6,
                    rib_width=width,
                    rib_height=height,
                )
                for index, width in enumerate(summary["rib_widths"])
            ],
        ),
        Rule101Config(
            groove_sizes=[
                GrooveSizeItem(groove_width=width, groove_height=height)
                for width in summary["groove_widths"]
            ],
        ),
        Rule102Config(
            decorations=[
                DecorationItem(
                    position="left",
                    decoration_width=summary["decoration_widths"][0],
                    decoration_height=height,
                    decoration_opacity=128,
                ),
                DecorationItem(
                    position="right",
                    decoration_width=summary["decoration_widths"][1],
                    decoration_height=height,
                    decoration_opacity=128,
                ),
            ],
        ),
    ]


def _write_manifest(
    output_dir: Path,
    tire_struct: TireStruct,
    pipeline4_flag: bool | None,
    pipeline4_err_msg: str | None,
    config_summary: dict[str, Any] | None,
    ranked_results: list[dict[str, Any]] | None,
    small_image_paths: list[str],
) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    ranked_results = ranked_results or []
    ranked_paths = [
        result["big_image_path"]
        for result in ranked_results
        if result.get("flag") is True
    ]
    manifest = {
        "input_image": str(INPUT_IMAGE.resolve()),
        "manifest_path": str(manifest_path),
        "pipeline4": {
            "flag": pipeline4_flag,
            "err_msg": pipeline4_err_msg,
            "small_image_count": len(tire_struct.small_images),
            "regions": [
                image.biz.region.value if image.biz.region is not None else "unknown"
                for image in tire_struct.small_images
            ],
            "small_image_paths": small_image_paths,
        },
        "pipeline1": {
            "flag": len(ranked_paths) == len(ranked_results) and len(ranked_results) > 0,
            "config_summary": config_summary,
            "rank_count": len(ranked_results),
            "ranked_results": ranked_results,
            "ranked_big_image_paths": ranked_paths,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _clear_pngs(directory: Path) -> None:
    for path in directory.glob("*.png"):
        path.unlink()


def _derive_geometry_config(image: np.ndarray) -> dict[str, Any]:
    height, width = image.shape[:2]
    black_segments = _detect_black_column_segments(image)
    main_grooves = sorted(
        sorted(black_segments, key=lambda segment: segment["width"], reverse=True)[:4],
        key=lambda segment: segment["start"],
    )
    keep_segments = _remaining_segments(width, main_grooves)
    decoration_widths = _derive_decoration_widths(black_segments, main_grooves, width)
    summary = {
        "target_width": width,
        "target_height": height,
        "rib_widths": [segment["width"] for segment in keep_segments],
        "groove_widths": [segment["width"] for segment in main_grooves],
        "decoration_widths": decoration_widths,
        "symmetry_rules": ["rule1", "rule2", "rule3"],
        "continuity_modes": [
            ContinuityModeName.CONTINUITY_0.value,
            ContinuityModeName.CONTINUITY_1.value,
            ContinuityModeName.CONTINUITY_2.value,
        ],
    }
    return {"summary": summary}


def _detect_black_column_segments(image: np.ndarray) -> list[dict[str, int]]:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb_image.shape[:2]
    dark_pixels = np.all(rgb_image < 10, axis=2)
    black_mask = (dark_pixels.sum(axis=0) / height) > 0.95

    segments = []
    start = None
    for index in range(width):
        if black_mask[index] and start is None:
            start = index
        elif not black_mask[index] and start is not None:
            if index - start >= 5:
                segments.append({"start": start, "end": index - 1, "width": index - start})
            start = None
    if start is not None and width - start >= 5:
        segments.append({"start": start, "end": width - 1, "width": width - start})
    return segments


def _remaining_segments(width: int, removed_segments: list[dict[str, int]]) -> list[dict[str, int]]:
    keep_mask = np.ones(width, dtype=bool)
    for segment in removed_segments:
        keep_mask[segment["start"]: segment["end"] + 1] = False

    segments = []
    start = None
    for index in range(width):
        if keep_mask[index] and start is None:
            start = index
        elif not keep_mask[index] and start is not None:
            segments.append({"start": start, "end": index - 1, "width": index - start})
            start = None
    if start is not None:
        segments.append({"start": start, "end": width - 1, "width": width - start})
    return segments


def _derive_decoration_widths(
    black_segments: list[dict[str, int]],
    main_grooves: list[dict[str, int]],
    width: int,
) -> list[int]:
    main_groove_ids = {(segment["start"], segment["end"]) for segment in main_grooves}
    boundary_segments = [
        segment
        for segment in black_segments
        if (segment["start"], segment["end"]) not in main_groove_ids
    ]
    if len(boundary_segments) >= 2:
        left = boundary_segments[0]["end"] + 1
        right = width - boundary_segments[-1]["start"]
        return [left, right]
    return [0, 0]

def _extract_compliance_score(big_image: BigImage) -> int | None:
    """从 BigImage 中提取 compliance 分数"""
    if big_image.scores is None:
        return None
    for score in big_image.scores:
        if hasattr(score, 'compliance'):
            return score.compliance
    return None

if __name__ == "__main__":
    result = run_pipeline4_to_pipeline1_case()
    print(json.dumps(result, ensure_ascii=False, indent=2))
