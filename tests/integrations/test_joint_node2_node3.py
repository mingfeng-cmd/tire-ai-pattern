"""
Node2 + Node3 联调集成测试

端到端验证 generate_stitch_scheme → stitch_big_image 的完整管线。

结果输出到 .results/test_joint_node2_node3/ :
  - {case}_input.json     — 测试输入元数据
  - {case}_node2.json     — Node2 产出的 lineage 结构
  - {case}_node3.json     — Node3 产出摘要
  - {case}_stitched.png   — 拼接大图
  - report.md             — 可读测试报告
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tire_ai_pattern.models.enums import (
    ContinuityModeName,
    ImageFormatEnum,
    ImageModeEnum,
    LevelEnum,
    RegionEnum,
    RibOperation,
    SourceTypeEnum,
    StitchingSchemeName,
)
from tire_ai_pattern.models.image_models import (
    BigImage,
    ImageBiz,
    ImageEvaluation,
    ImageLineage,
    ImageMeta,
    RuleEvaluation,
    SmallImage,
)
from tire_ai_pattern.models.rule_models import (
    DecorationItem,
    GrooveSizeItem,
    RibSizeItem,
    Rule1Config,
    Rule1Score,
    Rule2Config,
    Rule100Config,
    Rule101Config,
    Rule102Config,
    Rule12Config,
    Rule16Config,
    Rule17Config,
    Rule4Config,
)
from tire_ai_pattern.nodes.big_image_stitcher import stitch_big_image
from tire_ai_pattern.nodes.stitch_scheme_generator import generate_stitch_scheme
from tire_ai_pattern.utils.image_utils import base64_to_ndarray, load_image_to_base64, ndarray_to_base64
from tire_ai_pattern.utils.logger import get_logger


logger = get_logger("joint_test")

DATASET_DIR = Path("tests/datasets/stitching")
RESULTS_DIR = Path(".results/test_joint_node2_node3")


# ---------- 规则配置 ----------

RULES_CONFIG_V1 = [
    Rule1Config(description="rib无对称", max_score=10),
    Rule2Config(description="rib中心对称", max_score=10),
    Rule100Config(
        description="RIB 节距与尺寸配置",
        rib_number=5,
        rib_sizes=[
            RibSizeItem(rib_name="rib1", num_pitchs=5, rib_width=400, rib_height=640),
            RibSizeItem(rib_name="rib2", num_pitchs=6, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib3", num_pitchs=6, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib4", num_pitchs=6, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib5", num_pitchs=6, rib_width=200, rib_height=640),
        ],
    ),
    Rule101Config(
        description="主沟尺寸配置",
        groove_sizes=[
            GrooveSizeItem(groove_width=10, groove_height=640),
            GrooveSizeItem(groove_width=10, groove_height=640),
            GrooveSizeItem(groove_width=10, groove_height=640),
            GrooveSizeItem(groove_width=10, groove_height=640),
        ],
    ),
    Rule102Config(
        description="装饰边框尺寸与透明度配置",
        decorations=[
            DecorationItem(
                position="left",
                decoration_width=300,
                decoration_height=640,
                decoration_opacity=128,
            )
        ],
    ),
]

RULES_CONFIG_CASE1 = [
    Rule1Config(description="rib无对称", max_score=10),
    Rule100Config(
        description="RIB 节距与尺寸配置",
        rib_number=5,
        rib_sizes=[
            RibSizeItem(rib_name="rib1", num_pitchs=5, rib_width=400, rib_height=640),
            RibSizeItem(rib_name="rib2", num_pitchs=5, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib3", num_pitchs=5, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib4", num_pitchs=5, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib5", num_pitchs=5, rib_width=400, rib_height=640),
        ],
    ),
    Rule101Config(
        description="主沟尺寸配置",
        groove_sizes=[
            GrooveSizeItem(groove_width=20, groove_height=640),
            GrooveSizeItem(groove_width=20, groove_height=640),
            GrooveSizeItem(groove_width=20, groove_height=640),
            GrooveSizeItem(groove_width=20, groove_height=640),
        ],
    ),
    Rule102Config(
        description="装饰边框尺寸与透明度配置",
        decorations=[
            DecorationItem(
                position="left",
                decoration_width=300,
                decoration_height=640,
                decoration_opacity=128,
            )
        ],
    ),
]

RULES_CONFIG_CASE2 = [
    Rule1Config(description="rib无对称", max_score=10),
    Rule100Config(
        description="RIB 节距与尺寸配置",
        rib_number=5,
        rib_sizes=[
            RibSizeItem(rib_name="rib1", num_pitchs=5, rib_width=400, rib_height=640),
            RibSizeItem(rib_name="rib2", num_pitchs=5, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib3", num_pitchs=5, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib4", num_pitchs=5, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib5", num_pitchs=5, rib_width=400, rib_height=640),
        ],
    ),
    Rule101Config(
        description="主沟尺寸配置",
        groove_sizes=[
            GrooveSizeItem(groove_width=20, groove_height=640),
            GrooveSizeItem(groove_width=20, groove_height=640),
            GrooveSizeItem(groove_width=20, groove_height=640),
            GrooveSizeItem(groove_width=20, groove_height=640),
        ],
    ),
    Rule102Config(
        description="装饰边框尺寸与透明度配置",
        decorations=[
            DecorationItem(
                position="left",
                decoration_width=300,
                decoration_height=640,
                decoration_opacity=128,
            )
        ],
    ),
    Rule12Config(
        description="两个RIB间横向钢片及横沟连续性占比",
        max_score=6,
        continuity_ratio_upper=0.7,
        continuity_ratio_lower=0.6,
        continuity_mode_list=[ContinuityModeName.CONTINUITY_0, ContinuityModeName.CONTINUITY_1],
    ),
    Rule16Config(
        description="中心RIB上的横沟或横向钢片可任意组合连续性",
        max_score=4,
        continuity_mode_list=[ContinuityModeName.CONTINUITY_0, ContinuityModeName.CONTINUITY_1],
    ),
    Rule17Config(
        description="边缘RIB上的横沟或横向钢片可任意组合连续性",
        max_score=6,
        continuity_mode_list=[ContinuityModeName.CONTINUITY_0],
    ),
]

RULES_CONFIG_RULE4_5RIB = [
    Rule4Config(description="中心线镜像对称可错位", max_score=10),
    Rule100Config(
        description="RIB 节距与尺寸配置",
        rib_number=5,
        rib_sizes=[
            RibSizeItem(rib_name="rib1", num_pitchs=5, rib_width=400, rib_height=640),
            RibSizeItem(rib_name="rib2", num_pitchs=5, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib3", num_pitchs=5, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib4", num_pitchs=5, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib5", num_pitchs=5, rib_width=400, rib_height=640),
        ],
    ),
    Rule101Config(
        description="主沟尺寸配置",
        groove_sizes=[
            GrooveSizeItem(groove_width=20, groove_height=640),
            GrooveSizeItem(groove_width=20, groove_height=640),
            GrooveSizeItem(groove_width=20, groove_height=640),
            GrooveSizeItem(groove_width=20, groove_height=640),
        ],
    ),
    Rule102Config(
        description="透明装饰占位配置",
        decorations=[
            DecorationItem(
                position="left",
                decoration_width=1,
                decoration_height=640,
                decoration_opacity=0,
            )
        ],
    ),
]

RULES_CONFIG_RULE4_4RIB = [
    Rule4Config(description="中心线镜像对称可错位", max_score=10),
    Rule100Config(
        description="RIB 节距与尺寸配置",
        rib_number=4,
        rib_sizes=[
            RibSizeItem(rib_name="rib1", num_pitchs=5, rib_width=400, rib_height=640),
            RibSizeItem(rib_name="rib2", num_pitchs=5, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib3", num_pitchs=5, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib4", num_pitchs=5, rib_width=400, rib_height=640),
        ],
    ),
    Rule101Config(
        description="主沟尺寸配置",
        groove_sizes=[
            GrooveSizeItem(groove_width=20, groove_height=640),
            GrooveSizeItem(groove_width=20, groove_height=640),
            GrooveSizeItem(groove_width=20, groove_height=640),
        ],
    ),
    Rule12Config(
        description="4RIB 连续性占位配置",
        max_score=6,
        continuity_ratio_upper=1.0,
        continuity_ratio_lower=0.0,
        continuity_mode_list=[ContinuityModeName.CONTINUITY_4],
    ),
    Rule102Config(
        description="透明装饰占位配置",
        decorations=[
            DecorationItem(
                position="left",
                decoration_width=1,
                decoration_height=640,
                decoration_opacity=0,
            )
        ],
    ),
]


# ---------- 工具函数 ----------

def _short_hash(value: str, n: int = 8) -> str:
    """截取 value 的前 n 位作为短标识。"""
    return value[:n] if value else "None"


def _read_image_dims(filepath: Path) -> tuple[int, int]:
    """读取图片文件的 (height, width)。"""
    img = cv2.imread(str(filepath))
    if img is None:
        raise FileNotFoundError(f"无法加载图片: {filepath}")
    return img.shape[0], img.shape[1]


def make_real_small_image(
    region: RegionEnum,
    image_filename: str,
    score: int,
) -> SmallImage:
    """用真实图片文件构建带评分的 SmallImage。"""
    filepath = DATASET_DIR / image_filename
    img = cv2.imread(str(filepath))
    if img is None:
        raise FileNotFoundError(f"无法加载图片: {filepath}")
    base64_str = ndarray_to_base64(img)
    return SmallImage(
        image_base64=base64_str,
        meta=ImageMeta(
            width=img.shape[1],
            height=img.shape[0],
            channels=img.shape[2],
            mode=ImageModeEnum.RGB,
            format=ImageFormatEnum.PNG,
            size=len(base64_str),
        ),
        biz=ImageBiz(
            level=LevelEnum.SMALL,
            region=region,
            source_type=SourceTypeEnum.ORIGINAL,
        ),
        evaluation=ImageEvaluation(
            rules=[
                RuleEvaluation(
                    name="rule1",
                    config=Rule1Config(description="rib无对称", max_score=10),
                    score=Rule1Score(score=score),
                )
            ],
            current_score=score,
        ),
    )


def make_big_image_placeholder() -> BigImage:
    """构建占位 BigImage。"""
    return BigImage(
        image_base64="data:image/png;base64,placeholder",
        meta=ImageMeta(
            width=1, height=1, channels=3,
            mode=ImageModeEnum.RGB, format=ImageFormatEnum.PNG, size=0,
        ),
        biz=ImageBiz(
            level=LevelEnum.BIG,
            region=RegionEnum.CENTER,
            source_type=SourceTypeEnum.ORIGINAL,
        ),
    )


def run_joint_pipeline(
    small_images: list[SmallImage],
    rules_config: list,
    scheme_rank: int = 1,
) -> BigImage:
    """执行 Node2 → Node3 完整管线。"""
    logger.info(">>> Step 1: generate_stitch_scheme (Node2)")
    big_image = generate_stitch_scheme(
        big_image=make_big_image_placeholder(),
        small_images=small_images,
        rules_config=rules_config,
        scheme_rank=scheme_rank,
    )

    assert big_image.lineage is not None, "Node2 必须生成 lineage"

    logger.info(">>> Step 2: stitch_big_image (Node3)")
    result = stitch_big_image(big_image)

    logger.info(">>> 联调完成")
    return result


# ---------- 结果导出 ----------

def _save_json(filepath: Path, data: dict | list):
    """保存 JSON 到指定路径。"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_input_summary(
    small_images: list[SmallImage],
    rules_config: list,
    scheme_rank: int,
) -> dict:
    """构建输入摘要。"""
    images_data = []
    for i, img in enumerate(small_images):
        images_data.append({
            "index": i,
            "region": img.biz.region.value,
            "score": img.evaluation.current_score,
            "image_base64_hash": _short_hash(img.image_base64, 16),
            "meta": {
                "width": img.meta.width,
                "height": img.meta.height,
                "channels": img.meta.channels,
            },
        })

    rib_sizes = []
    groove_sizes = []
    decorations = []
    active_rules = []

    for conf in rules_config:
        name = conf.__class__.__name__.replace("Config", "")
        active_rules.append({"class": name, "description": conf.description})

        if isinstance(conf, Rule100Config):
            rib_sizes = [
                {"name": r.rib_name, "num_pitchs": r.num_pitchs, "width": r.rib_width, "height": r.rib_height}
                for r in conf.rib_sizes
            ]
        elif isinstance(conf, Rule101Config):
            groove_sizes = [
                {"width": g.groove_width, "height": g.groove_height}
                for g in conf.groove_sizes
            ]
        elif isinstance(conf, Rule102Config):
            decorations = [
                {"position": d.position, "width": d.decoration_width, "height": d.decoration_height, "opacity": d.decoration_opacity}
                for d in conf.decorations
            ]

    return {
        "scheme_rank": scheme_rank,
        "small_images": images_data,
        "active_rules": active_rules,
        "rib_config": rib_sizes,
        "groove_config": groove_sizes,
        "decoration_config": decorations,
    }


def _build_node2_summary(lineage: ImageLineage) -> dict:
    """构建 Node2 lineage 摘要。"""
    scheme_name = lineage.stitching_scheme.stitching_scheme_abstract.name.value
    rib_number = lineage.stitching_scheme.stitching_scheme_abstract.rib_number

    ribs_data = []
    for rib in lineage.stitching_scheme.ribs_scheme_implementation:
        ribs_data.append({
            "rib_name": rib.rib_name,
            "rib_source": rib.rib_source,
            "same_as": rib.rib_same_as,
            "operation": [op.value if hasattr(op, "value") else str(op) for op in rib.rib_operation],
            "rib_width": rib.rib_width,
            "rib_height": rib.rib_height,
            "num_pitchs": rib.num_pitchs,
            "before_image_hash": _short_hash(rib.before_image or "", 16),
            "has_before_image": rib.before_image is not None and rib.before_image.startswith("data:image/"),
        })

    grooves_data = []
    for i, g in enumerate(lineage.main_groove_scheme.main_groove_implementation):
        grooves_data.append({
            "index": i,
            "groove_width": g.groove_width,
            "groove_height": g.groove_height,
            "before_image_hash": _short_hash(g.before_image or "", 16),
            "has_before_image": g.before_image is not None and g.before_image.startswith("data:image/"),
        })

    decs_data = []
    for i, d in enumerate(lineage.decoration_scheme.decoration_implementation):
        decs_data.append({
            "index": i,
            "decoration_width": d.decoration_width,
            "decoration_height": d.decoration_height,
            "decoration_opacity": d.decoration_opacity,
            "before_image_hash": _short_hash(d.before_image or "", 16),
            "has_before_image": d.before_image is not None and d.before_image.startswith("data:image/"),
        })

    return {
        "scheme_name": scheme_name,
        "rib_number": rib_number,
        "rib_count": len(ribs_data),
        "groove_count": len(grooves_data),
        "decoration_count": len(decs_data),
        "ribs": ribs_data,
        "grooves": grooves_data,
        "decorations": decs_data,
    }


def _build_node3_summary(big_image: BigImage) -> dict:
    """构建 Node3 输出摘要。"""
    lineage = big_image.lineage

    rst = base64_to_ndarray(big_image.image_base64)
    output_shape = {"height": rst.shape[0], "width": rst.shape[1]}
    if len(rst.shape) >= 3:
        output_shape["channels"] = rst.shape[2]

    ribs_after = []
    for rib in lineage.stitching_scheme.ribs_scheme_implementation:
        dims = None
        if rib.after_image:
            try:
                arr = base64_to_ndarray(rib.after_image)
                dims = {"height": arr.shape[0], "width": arr.shape[1]}
                if len(arr.shape) >= 3:
                    dims["channels"] = arr.shape[2]
            except Exception:
                pass
        ribs_after.append({
            "rib_name": rib.rib_name,
            "has_after_image": rib.after_image is not None and rib.after_image.startswith("data:image/"),
            "after_image_hash": _short_hash(rib.after_image or "", 16),
            "after_dimensions": dims,
        })

    grooves_after = []
    for i, g in enumerate(lineage.main_groove_scheme.main_groove_implementation):
        dims = None
        if g.after_image:
            try:
                arr = base64_to_ndarray(g.after_image)
                dims = {"height": arr.shape[0], "width": arr.shape[1]}
            except Exception:
                pass
        grooves_after.append({
            "index": i,
            "has_after_image": g.after_image is not None and g.after_image.startswith("data:image/"),
            "after_dimensions": dims,
        })

    decs_after = []
    for i, d in enumerate(lineage.decoration_scheme.decoration_implementation):
        dims = None
        if d.after_image:
            try:
                arr = base64_to_ndarray(d.after_image)
                dims = {"height": arr.shape[0], "width": arr.shape[1]}
            except Exception:
                pass
        decs_after.append({
            "index": i,
            "has_after_image": d.after_image is not None and d.after_image.startswith("data:image/"),
            "after_dimensions": dims,
        })

    return {
        "output_shape": output_shape,
        "image_base64_prefix": big_image.image_base64[:15] + "...",
        "ribs_after": ribs_after,
        "grooves_after": grooves_after,
        "decorations_after": decs_after,
    }


def _save_stitched_png(filepath: Path, image_base64: str):
    """保存拼接大图为 PNG 文件。"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    rst = base64_to_ndarray(image_base64)
    cv2.imwrite(str(filepath), rst)


def _export_case_results(
    case_name: str,
    small_images: list[SmallImage],
    rules_config: list,
    scheme_rank: int,
    big_image: BigImage,
    expected_width: int,
    expected_height: int,
):
    """导出单个测试用例的所有结果文件。"""
    # 3.1 输入 JSON
    input_summary = _build_input_summary(small_images, rules_config, scheme_rank)
    _save_json(RESULTS_DIR / f"{case_name}_input.json", input_summary)

    # 3.2 Node2 结果 JSON
    node2_summary = _build_node2_summary(big_image.lineage)
    _save_json(RESULTS_DIR / f"{case_name}_node2.json", node2_summary)

    # 3.3 Node3 结果 JSON
    node3_summary = _build_node3_summary(big_image)
    node3_summary["expected_shape"] = {"height": expected_height, "width": expected_width, "channels": 3}
    _save_json(RESULTS_DIR / f"{case_name}_node3.json", node3_summary)

    # 3.4 拼接大图 PNG
    _save_stitched_png(RESULTS_DIR / f"{case_name}_stitched.png", big_image.image_base64)


def _generate_markdown_report(cases_data: list[dict]):
    """生成可读的 Markdown 测试报告。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Node2 + Node3 联调集成测试报告",
        "",
        f"## 测试数据集",
        f"",
        f"数据集路径: `tests/datasets/stitching/`",
        f"",
        "| 图片 | 原始尺寸 | 用途 |",
        "|------|----------|------|",
    ]
    for fname in ["rib1.png", "rib2.png", "rib3.png", "rib4.png", "rib5.png"]:
        h, w = _read_image_dims(DATASET_DIR / fname)
        lines.append(f"| `{fname}` | {w}×{h} | SIDE/CENTER 候选 |")

    lines.extend(["", "---", ""])

    for cd in cases_data:
        case_name = cd["case_name"]
        expected_width = cd["expected_width"]
        expected_height = cd["expected_height"]
        small_images = cd["small_images"]
        rules_config = cd["rules_config"]
        big_image = cd["big_image"]

        lineage = big_image.lineage
        scheme_name = lineage.stitching_scheme.stitching_scheme_abstract.name.value

        lines.extend([
            f"## 用例: {case_name}",
            "",
            f"### 输入",
            "",
            "#### 小图清单",
            "",
            "| # | 区域 | 评分 | 文件 | 尺寸 |",
            "|---|------|------|------|------|",
        ])
        for i, img in enumerate(small_images):
            lines.append(
                f"| {i} | {img.biz.region.value} | {img.evaluation.current_score} "
                f"| — | {img.meta.width}×{img.meta.height} |"
            )

        lines.extend([
            "",
            "#### 规则配置",
            "",
            "| 规则 | 说明 |",
            "|------|------|",
        ])
        for conf in rules_config:
            name = conf.__class__.__name__
            lines.append(f"| {name} | {conf.description} |")

        # RIB 尺寸
        for conf in rules_config:
            if isinstance(conf, Rule100Config):
                lines.extend(["", "**RIB 尺寸**:", ""])
                for r in conf.rib_sizes:
                    lines.append(f"- {r.rib_name}: {r.rib_width}×{r.rib_height}, pitchs={r.num_pitchs}")
            elif isinstance(conf, Rule101Config):
                lines.extend(["", f"**主沟**: {len(conf.groove_sizes)} 个, 各 {conf.groove_sizes[0].groove_width}×{conf.groove_sizes[0].groove_height}"])
            elif isinstance(conf, Rule102Config):
                for d in conf.decorations:
                    lines.extend(["", f"**装饰**: {d.position}, {d.decoration_width}×{d.decoration_height}, opacity={d.decoration_opacity}"])

        lines.extend([
            "",
            f"### Node2 输出",
            "",
            f"- **方案**: {scheme_name}",
            "",
            "| RIB | 来源 | 继承自 | 操作 | 尺寸 | before_image |",
            "|-----|------|--------|------|------|-------------|",
        ])
        for rib in lineage.stitching_scheme.ribs_scheme_implementation:
            ops = ", ".join(op.value if hasattr(op, "value") else str(op) for op in rib.rib_operation)
            lines.append(
                f"| {rib.rib_name} | {rib.rib_source} | {rib.rib_same_as or '—'} "
                f"| {ops} | {rib.rib_width}×{rib.rib_height} "
                f"| `{_short_hash(rib.before_image or '', 8)}` |"
            )

        lines.extend([
            "",
            f"### Node3 输出",
            "",
        ])

        # 输出尺寸
        rst = base64_to_ndarray(big_image.image_base64)
        actual_w = rst.shape[1]
        actual_h = rst.shape[0]
        actual_c = rst.shape[2] if len(rst.shape) >= 3 else 1
        dim_ok = actual_w == expected_width and actual_h == expected_height

        lines.extend([
            f"- **预期尺寸**: {expected_width}×{expected_height}×3",
            f"- **实际尺寸**: {actual_w}×{actual_h}×{actual_c}",
            f"- **尺寸校验**: {'✅ 通过' if dim_ok else '❌ 不符'}",
            "",
            "#### after_image 填充状态",
            "",
            "| 组件 | 已填充 | 尺寸 |",
            "|------|--------|------|",
        ])

        for rib in lineage.stitching_scheme.ribs_scheme_implementation:
            filled = "✅" if rib.after_image and rib.after_image.startswith("data:image/") else "❌"
            dims_str = "—"
            if rib.after_image:
                try:
                    arr = base64_to_ndarray(rib.after_image)
                    dims_str = f"{arr.shape[1]}×{arr.shape[0]}"
                except Exception:
                    dims_str = "解码失败"
            lines.append(f"| rib/{rib.rib_name} | {filled} | {dims_str} |")

        for i, g in enumerate(lineage.main_groove_scheme.main_groove_implementation):
            filled = "✅" if g.after_image and g.after_image.startswith("data:image/") else "❌"
            dims_str = "—"
            if g.after_image:
                try:
                    arr = base64_to_ndarray(g.after_image)
                    dims_str = f"{arr.shape[1]}×{arr.shape[0]}"
                except Exception:
                    dims_str = "解码失败"
            lines.append(f"| groove/{i} | {filled} | {dims_str} |")

        for i, d in enumerate(lineage.decoration_scheme.decoration_implementation):
            filled = "✅" if d.after_image and d.after_image.startswith("data:image/") else "❌"
            dims_str = "—"
            if d.after_image:
                try:
                    arr = base64_to_ndarray(d.after_image)
                    dims_str = f"{arr.shape[1]}×{arr.shape[0]}"
                except Exception:
                    dims_str = "解码失败"
            lines.append(f"| decoration/{i} | {filled} | {dims_str} |")

        lines.extend([
            "",
            f"### 输出文件",
            "",
            f"- 拼接大图: `{case_name}_stitched.png`",
            f"- 输入摘要: `{case_name}_input.json`",
            f"- Node2 血缘: `{case_name}_node2.json`",
            f"- Node3 结果: `{case_name}_node3.json`",
            "",
            "---",
            "",
        ])

    # Write report
    report_path = RESULTS_DIR / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("测试报告已保存到 %s", report_path)


# ---------- 测试用例 ----------

class TestJointNode2Node3:

    @pytest.fixture(autouse=True)
    def setup_results_dir(self):
        """确保结果目录存在。"""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _collected_cases: list[dict] = []

    def test_joint_node2_produces_expected_lineage(self):
        """用例 1：反推 test_large_image_stitching.py，验证 Node2 产出 Node3 期望格式。

        从 _build_lineage_with_black_decoration 反推 Node2 输入：
        - 仅 Rule1（Symmetry0），无 Rule2
        - rib5 width=400，groove_width=20
        - 恰好 2 SIDE + 3 CENTER（Symmetry0 所需最少原始图片数）
        - 所有 score=5，num_pitchs=5
        """
        expected_prefix = "data:image/"

        small_images = [
            make_real_small_image(RegionEnum.SIDE, "rib1.png", 5),
            make_real_small_image(RegionEnum.CENTER, "rib2.png", 5),
            make_real_small_image(RegionEnum.CENTER, "rib3.png", 5),
            make_real_small_image(RegionEnum.CENTER, "rib4.png", 5),
            make_real_small_image(RegionEnum.SIDE, "rib5.png", 5),
        ]

        result = run_joint_pipeline(small_images, RULES_CONFIG_CASE1, scheme_rank=11)

        # 1. lineage 结构校验
        lineage: ImageLineage = result.lineage
        assert lineage.stitching_scheme.stitching_scheme_abstract.name == \
            StitchingSchemeName.SYMMETRY_0

        # 2. RIB/主沟/装饰数量
        ribs = lineage.stitching_scheme.ribs_scheme_implementation
        assert len(ribs) == 5
        grooves = lineage.main_groove_scheme.main_groove_implementation
        assert len(grooves) == 4
        decs = lineage.decoration_scheme.decoration_implementation
        assert len(decs) == 1

        # 3. 装饰配置对齐
        assert decs[0].decoration_width == 300
        assert decs[0].decoration_height == 640
        assert decs[0].decoration_opacity == 128

        # 4. before_image 和 after_image
        self._assert_before_images_filled(lineage)
        self._assert_after_images_filled(lineage, expected_prefix)

        # 5. 输出尺寸（rib5=400，groove=20）
        expected_width = 400 + 200 + 200 + 200 + 400 + 20 * 4
        expected_height = 640
        expected_channels = 3
        expected_shape = (expected_height, expected_width, expected_channels)
        rst = base64_to_ndarray(result.image_base64)
        assert rst.shape == expected_shape

        # 6. 逐像素对比预期图片（来自 _build_lineage_with_black_decoration 的已知正确输出）
        expected_image_path = DATASET_DIR / "correct_black_decoration.png"
        expected_image = cv2.imread(str(expected_image_path))
        assert expected_image is not None, f"预期图片不存在: {expected_image_path}"
        np.testing.assert_array_equal(rst, expected_image)

        # 7. 导出结果
        _export_case_results(
            "case1_lineage", small_images, RULES_CONFIG_CASE1, 1,
            result, expected_width, expected_height,
        )
        TestJointNode2Node3._collected_cases.append({
            "case_name": "case1_lineage (格式兼容 - 仅 Rule1 Symmetry0, score=5, pitchs=5)",
            "expected_width": expected_width,
            "expected_height": expected_height,
            "small_images": small_images,
            "rules_config": RULES_CONFIG_CASE1,
            "big_image": result,
        })

    def test_joint_continuity1_operations(self):
        """用例 2：Symmetry0 + Continuity1 操作管线，验证 RESIZE_H_2X 和 same_as 继承。

        配置同用例 1（RULES_CONFIG_CASE2，相同 small_images），scheme_rank=14 选中 Continuity1。
        Continuity1 = RIB2-RIB3 连续：rib2 和 rib3 共享同一张 CENTER 图片，
        rib2 执行 RESIZE_H_2X + LEFT，rib3(2) 执行 RESIZE_H_2X + RIGHT。
        """
        expected_prefix = "data:image/"

        small_images = [
            make_real_small_image(RegionEnum.SIDE, "rib1.png", 5),
            make_real_small_image(RegionEnum.CENTER, "rib2.png", 5),
            make_real_small_image(RegionEnum.CENTER, "rib3.png", 5),
            make_real_small_image(RegionEnum.CENTER, "rib4.png", 5),
            make_real_small_image(RegionEnum.SIDE, "rib5.png", 5),
        ]

        result = run_joint_pipeline(small_images, RULES_CONFIG_CASE2, scheme_rank=14)

        lineage: ImageLineage = result.lineage
        assert lineage.stitching_scheme.stitching_scheme_abstract.name == \
            StitchingSchemeName.SYMMETRY_0

        ribs = lineage.stitching_scheme.ribs_scheme_implementation
        assert len(ribs) == 5

        # Continuity1 特有：rib2 和 rib3 共享图片 (same_as)，且操作包含 RESIZE_H_2X
        rib2 = ribs[1]
        rib3 = ribs[2]
        assert "resize_horizontal_2x" in [op.value for op in rib2.rib_operation], \
            "rib2 应包含 RESIZE_H_2X (Continuity1)"
        assert rib3.rib_same_as == rib2.rib_name, \
            "rib3 应继承自 rib2 (Continuity1)"
        assert "resize_horizontal_2x" in [op.value for op in rib3.rib_operation], \
            "rib3 应包含 RESIZE_H_2X (Continuity1)"
        assert rib3.before_image == rib2.before_image, \
            "rib3 的 before_image 应与 rib2 相同 (共享图片)"

        # 其他 rib 无继承
        assert ribs[0].rib_same_as is None  # rib1
        assert ribs[3].rib_same_as is None  # rib4
        assert ribs[4].rib_same_as is None  # rib5

        # before_image 和 after_image
        self._assert_before_images_filled(lineage)
        self._assert_after_images_filled(lineage, expected_prefix)

        # 输出尺寸
        expected_width = 400 + 200 + 200 + 200 + 400 + 20 * 4
        expected_height = 640
        expected_channels = 3
        expected_shape = (expected_height, expected_width, expected_channels)
        rst = base64_to_ndarray(result.image_base64)
        assert rst.shape == expected_shape

        # 逐像素对比预期图片
        expected_image_path = DATASET_DIR / "correct_continuity1.png"
        expected_image = cv2.imread(str(expected_image_path))
        assert expected_image is not None, f"预期图片不存在: {expected_image_path}"
        np.testing.assert_array_equal(rst, expected_image)

        # 导出结果
        _export_case_results(
            "case2_continuity1", small_images, RULES_CONFIG_CASE2, 1,
            result, expected_width, expected_height,
        )
        TestJointNode2Node3._collected_cases.append({
            "case_name": "case2_continuity1 (Symmetry0 + Continuity1, rank=1)",
            "expected_width": expected_width,
            "expected_height": expected_height,
            "small_images": small_images,
            "rules_config": RULES_CONFIG_CASE2,
            "big_image": result,
        })

    def test_joint_smoke_node2_to_node3(self):
        """用例 3：冒烟测试——Node2 正常输出 → Node3 不出错。

        沿用 test_stitch_scheme_generator.py 的输入构建模式：
        3 SIDE + 4 CENTER，Rule1+Rule2 双模板竞争。
        """
        expected_prefix = "data:image/"

        small_images = [
            make_real_small_image(RegionEnum.SIDE, "rib1.png", 5),
            make_real_small_image(RegionEnum.SIDE, "rib5.png", 8),
            make_real_small_image(RegionEnum.SIDE, "rib1.png", 3),
            make_real_small_image(RegionEnum.CENTER, "rib2.png", 3),
            make_real_small_image(RegionEnum.CENTER, "rib3.png", 10),
            make_real_small_image(RegionEnum.CENTER, "rib4.png", 10),
            make_real_small_image(RegionEnum.CENTER, "rib2.png", 2),
        ]

        result = run_joint_pipeline(small_images, RULES_CONFIG_V1)

        # 1. 返回值
        assert result is not None

        # 2. image_base64 已更新
        assert result.image_base64[:len(expected_prefix)] == expected_prefix
        placeholder = "data:image/png;base64,placeholder"
        assert result.image_base64 != placeholder

        # 3. 最终大图尺寸
        expected_width = 400 + 200 + 200 + 200 + 200 + 10 * 4
        expected_height = 640
        expected_channels = 3
        expected_shape = (expected_height, expected_width, expected_channels)
        rst = base64_to_ndarray(result.image_base64)
        assert rst.shape == expected_shape

        # 4. before_image 和 after_image
        self._assert_before_images_filled(result.lineage)
        self._assert_after_images_filled(result.lineage, expected_prefix)

        # 5. 导出结果
        _export_case_results(
            "case3_smoke", small_images, RULES_CONFIG_V1, 1,
            result, expected_width, expected_height,
        )
        TestJointNode2Node3._collected_cases.append({
            "case_name": "case3_smoke (冒烟测试 - Rule1+Rule2)",
            "expected_width": expected_width,
            "expected_height": expected_height,
            "small_images": small_images,
            "rules_config": RULES_CONFIG_V1,
            "big_image": result,
        })

    def test_joint_rule4_symmetry3_real_images(self):
        """用真实 RIB 图片验证 Node2 + Node3 可生成 5RIB Rule4 错位镜像模板。"""
        expected_prefix = "data:image/"
        small_images = [
            make_real_small_image(RegionEnum.SIDE, "rib1.png", 5),
            make_real_small_image(RegionEnum.CENTER, "rib2.png", 5),
            make_real_small_image(RegionEnum.CENTER, "rib3.png", 5),
        ]

        result = run_joint_pipeline(small_images, RULES_CONFIG_RULE4_5RIB, scheme_rank=1)
        lineage: ImageLineage = result.lineage

        assert lineage.stitching_scheme.stitching_scheme_abstract.name == StitchingSchemeName.SYMMETRY_3
        ribs = lineage.stitching_scheme.ribs_scheme_implementation
        assert [rib.rib_name for rib in ribs] == ["rib1", "rib2", "rib3", "rib4", "rib5"]
        assert ribs[2].rib_operation == (RibOperation.LEFT_FLIP_LR_OFFSET_RIGHT_HALF,)
        assert ribs[3].rib_same_as == "rib2"
        assert ribs[3].rib_operation == (RibOperation.FLIP_LR, RibOperation.OFFSET_VERTICAL_HALF)
        assert ribs[4].rib_same_as == "rib1"
        assert ribs[4].rib_operation == (RibOperation.FLIP_LR, RibOperation.OFFSET_VERTICAL_HALF)

        self._assert_before_images_filled(lineage)
        self._assert_after_images_filled(lineage, expected_prefix)

        expected_width = 400 + 200 + 200 + 200 + 400 + 20 * 4
        expected_height = 640
        rst = base64_to_ndarray(result.image_base64)
        assert rst.shape == (expected_height, expected_width, 3)

        _export_case_results(
            "case_rule4_symmetry3_5rib", small_images, RULES_CONFIG_RULE4_5RIB, 1,
            result, expected_width, expected_height,
        )

    def test_joint_rule4_symmetry7_real_images(self):
        """用真实 RIB 图片验证 Node2 + Node3 可生成 4RIB Rule4 错位镜像模板。"""
        expected_prefix = "data:image/"
        small_images = [
            make_real_small_image(RegionEnum.SIDE, "rib1.png", 5),
            make_real_small_image(RegionEnum.CENTER, "rib2.png", 5),
        ]

        result = run_joint_pipeline(small_images, RULES_CONFIG_RULE4_4RIB, scheme_rank=1)
        lineage: ImageLineage = result.lineage

        assert lineage.stitching_scheme.stitching_scheme_abstract.name == StitchingSchemeName.SYMMETRY_7
        ribs = lineage.stitching_scheme.ribs_scheme_implementation
        assert [rib.rib_name for rib in ribs] == ["rib1", "rib2", "rib3", "rib4"]
        assert ribs[2].rib_same_as == "rib2"
        assert ribs[2].rib_operation == (RibOperation.FLIP_LR, RibOperation.OFFSET_VERTICAL_HALF)
        assert ribs[3].rib_same_as == "rib1"
        assert ribs[3].rib_operation == (RibOperation.FLIP_LR, RibOperation.OFFSET_VERTICAL_HALF)

        self._assert_before_images_filled(lineage)
        self._assert_after_images_filled(lineage, expected_prefix)

        expected_width = 400 + 200 + 200 + 400 + 20 * 3
        expected_height = 640
        rst = base64_to_ndarray(result.image_base64)
        assert rst.shape == (expected_height, expected_width, 3)

        _export_case_results(
            "case_rule4_symmetry7_4rib", small_images, RULES_CONFIG_RULE4_4RIB, 1,
            result, expected_width, expected_height,
        )

    # ---------- 共享断言辅助 ----------

    def _assert_before_images_filled(self, lineage: ImageLineage):
        """验证 lineage 中所有 before_image 已填充、格式合法。"""
        expected_prefix = "data:image/"

        ribs = lineage.stitching_scheme.ribs_scheme_implementation
        for i, rib in enumerate(ribs):
            assert rib.before_image is not None, f"rib[{i}] before_image 为空"
            assert rib.before_image[:len(expected_prefix)] == expected_prefix, \
                f"rib[{i}] before_image 格式不正确"

        grooves = lineage.main_groove_scheme.main_groove_implementation
        for i, groove in enumerate(grooves):
            assert groove.before_image is not None, f"groove[{i}] before_image 为空"
            assert groove.before_image[:len(expected_prefix)] == expected_prefix, \
                f"groove[{i}] before_image 格式不正确"

        decs = lineage.decoration_scheme.decoration_implementation
        for i, dec in enumerate(decs):
            assert dec.before_image is not None, f"decoration[{i}] before_image 为空"
            assert dec.before_image[:len(expected_prefix)] == expected_prefix, \
                f"decoration[{i}] before_image 格式不正确"

    def _assert_after_images_filled(self, lineage: ImageLineage, expected_prefix: str):
        """验证 lineage 中所有 after_image 已填充、格式合法。"""
        ribs = lineage.stitching_scheme.ribs_scheme_implementation
        for i, rib in enumerate(ribs):
            assert rib.after_image is not None, f"rib[{i}] after_image 为空"
            assert rib.after_image[:len(expected_prefix)] == expected_prefix, \
                f"rib[{i}] after_image 格式不正确"

        grooves = lineage.main_groove_scheme.main_groove_implementation
        for i, groove in enumerate(grooves):
            assert groove.after_image is not None, f"groove[{i}] after_image 为空"
            assert groove.after_image[:len(expected_prefix)] == expected_prefix, \
                f"groove[{i}] after_image 格式不正确"

        decs = lineage.decoration_scheme.decoration_implementation
        for i, dec in enumerate(decs):
            assert dec.after_image is not None, f"decoration[{i}] after_image 为空"
            assert dec.after_image[:len(expected_prefix)] == expected_prefix, \
                f"decoration[{i}] after_image 格式不正确"


# ---------- 报告生成（在所有测试完成后） ----------

@pytest.fixture(scope="class", autouse=True)
def _generate_final_report(request):
    """测试类执行完毕后生成报告。"""
    yield
    cases = TestJointNode2Node3._collected_cases
    if cases:
        _generate_markdown_report(cases)
