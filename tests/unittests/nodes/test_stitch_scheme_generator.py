from __future__ import annotations

import pytest

from tire_ai_pattern.common.exceptions import InputDataError
from tire_ai_pattern.models.enums import (
    ContinuityModeName,
    ImageFormatEnum,
    ImageModeEnum,
    LevelEnum,
    RegionEnum,
    SourceTypeEnum,
    StitchingSchemeName,
    RibOperation,
)
from tire_ai_pattern.models.image_models import (
    BigImage,
    ImageLineage,
    ImageBiz,
    ImageEvaluation,
    ImageMeta,
    RuleEvaluation,
    SmallImage,
)
from tire_ai_pattern.models.rule_models import (
    BaseRuleConfig,
    DecorationItem,
    GrooveSizeItem,
    RibSizeItem,
    Rule1Config,
    Rule12Config,
    Rule1Score,
    Rule16Config,
    Rule2Config,
    Rule3Config,
    Rule4Config,
    Rule100Config,
    Rule101Config,
    Rule102Config,
)
from tire_ai_pattern.models.scheme_models import (
    Continuity0,
    Continuity1,
    Continuity2,
    RibTemplate,
    StitchingTemplate,
    Symmetry0,
    Symmetry1,
    Symmetry2,
    Symmetry3,
    Symmetry6,
    Symmetry7,
)
from tire_ai_pattern.models.template_registry import get_stitching_templates
from tire_ai_pattern.nodes.base import STITCH_SCHEME_GENERATOR_CONFIGS, select_node_configs
from tire_ai_pattern.nodes.stitch_scheme_generator import (
    _TemplateCombination,
    _filter_templates,
    _filter_symmetry_templates_by_image_count,
    _filter_continuity_templates_by_image_count,
    _merge_template_combination_ribs,
    _build_candidates,
    _instantiate_stitching_scheme,
    _candidate_total_score,
    _candidate_count_formula_summary,
    _small_image_content_hash,
    _CandidateScheme,
    generate_stitch_scheme,
)
from tire_ai_pattern.utils.image_utils import base64_to_ndarray


def make_small_image(region: RegionEnum, payload: str, score: int) -> SmallImage:
    return SmallImage(
        image_base64=f"data:image/png;base64,{payload}",
        meta=ImageMeta(
            width=10,
            height=10,
            channels=3,
            mode=ImageModeEnum.RGB,
            format=ImageFormatEnum.PNG,
            size=10,
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
                    config=Rule1Config(),
                    score=Rule1Score(score=score),
                )
            ],
            current_score=score,
        ),
    )


def make_big_image() -> BigImage:
    return BigImage(
        image_base64="data:image/png;base64,big",
        meta=ImageMeta(
            width=10,
            height=10,
            channels=3,
            mode=ImageModeEnum.RGB,
            format=ImageFormatEnum.PNG,
            size=10,
        ),
        biz=ImageBiz(
            level=LevelEnum.BIG,
            region=RegionEnum.CENTER,
            source_type=SourceTypeEnum.ORIGINAL,
        ),
    )


def generate_lineage_for_test(
    small_images: list[SmallImage],
    rules_config: list[BaseRuleConfig],
    scheme_rank: int,
) -> ImageLineage:
    result = generate_stitch_scheme(make_big_image(), small_images, rules_config, scheme_rank)

    assert isinstance(result, BigImage)
    assert isinstance(result.lineage, ImageLineage)
    return result.lineage


class Rule3Template(StitchingTemplate):
    name: ContinuityModeName = ContinuityModeName.CONTINUITY_1
    description: str = "rule3 test template"
    rib_number: int = 5
    mode: str = "test"
    matching_rule_names: tuple[str, ...] = (Rule3Config().name,)
    rib_template_list: list[RibTemplate] = Symmetry0().rib_template_list


def test_matching_rule_names_are_derived_from_rule_config_types():
    """模板匹配规则名应由 RuleConfig 类型推导，避免直接写规则名字符串。"""

    expected_symmetry0_rule_names = (Rule1Config().name,)
    expected_symmetry1_rule_names = (Rule2Config().name,)
    expected_rule3_template_names = (Rule3Config().name,)
    expected_symmetry3_rule_names = (Rule4Config().name,)

    assert Symmetry0().matching_rule_names == expected_symmetry0_rule_names
    assert Symmetry1().matching_rule_names == expected_symmetry1_rule_names
    assert Rule3Template().matching_rule_names == expected_rule3_template_names
    assert Symmetry3().matching_rule_names == expected_symmetry3_rule_names


def test_filter_templates_filters_symmetry_by_enabled_rules():
    """规则配置齐全时保留全部对称性模板，缺少配置时过滤对应模板。"""

    all_symmetry_templates, continuity_templates = _filter_templates(
        [Symmetry0(), Symmetry1(), Symmetry3(), Continuity0(), Continuity1(), Continuity2(), Rule3Template()],
        target_rib_number=5,
        configs=[Rule1Config(), Rule2Config(), Rule3Config(), Rule4Config()],
    )
    missing_rule2_symmetry_templates, _ = _filter_templates(
        [Symmetry0(), Symmetry1(), Symmetry3(), Continuity0(), Continuity1(), Continuity2(), Rule3Template()],
        target_rib_number=5,
        configs=[Rule1Config(), Rule3Config(), Rule4Config()],
    )

    expected_all_symmetry_template_names = [
        StitchingSchemeName.SYMMETRY_0,
        StitchingSchemeName.SYMMETRY_1,
        StitchingSchemeName.SYMMETRY_3,
    ]
    expected_missing_rule2_symmetry_template_names = [
        StitchingSchemeName.SYMMETRY_0,
        StitchingSchemeName.SYMMETRY_3,
    ]
    expected_continuity_template_names = [
        ContinuityModeName.CONTINUITY_0,
    ]

    assert [template.name for template in all_symmetry_templates] == expected_all_symmetry_template_names
    assert [template.name for template in missing_rule2_symmetry_templates] == expected_missing_rule2_symmetry_template_names
    assert [template.name for template in continuity_templates] == expected_continuity_template_names


def test_rule4_template_adds_vertical_offset_to_distinguish_from_rule3():
    rule3_5rib_operations = [rib.operation_template for rib in Symmetry2().rib_template_list]
    rule4_5rib_operations = [rib.operation_template for rib in Symmetry3().rib_template_list]
    rule3_4rib_operations = [rib.operation_template for rib in Symmetry6().rib_template_list]
    rule4_4rib_operations = [rib.operation_template for rib in Symmetry7().rib_template_list]

    assert rule4_5rib_operations != rule3_5rib_operations
    assert rule4_5rib_operations[2] == (RibOperation.OFFSET_VERTICAL_HALF,)
    assert RibOperation.OFFSET_VERTICAL_HALF in rule4_5rib_operations[3]
    assert RibOperation.OFFSET_VERTICAL_HALF in rule4_5rib_operations[4]
    assert rule4_4rib_operations != rule3_4rib_operations
    assert RibOperation.OFFSET_VERTICAL_HALF in rule4_4rib_operations[2]
    assert RibOperation.OFFSET_VERTICAL_HALF in rule4_4rib_operations[3]


def test_filter_templates_uses_user_continuity_mode_names():
    _, continuity_templates = _filter_templates(
        [Symmetry0(), Continuity0(), Continuity1(), Continuity2()],
        target_rib_number=5,
        configs=[
            Rule1Config(),
            Rule12Config(
                continuity_ratio_lower=0.0,
                continuity_ratio_upper=1.0,
                continuity_mode_list=[ContinuityModeName.CONTINUITY_1],
            ),
            Rule16Config(continuity_mode_list=[ContinuityModeName.CONTINUITY_2]),
        ],
    )

    assert [template.name for template in continuity_templates] == [
        ContinuityModeName.CONTINUITY_1,
        ContinuityModeName.CONTINUITY_2,
    ]


def test_small_image_content_hash_ignores_data_url_prefix():
    """哈希只依赖真实图片内容，不应被 data-url 前缀差异影响。"""

    first = _small_image_content_hash("data:image/png;base64,abc")
    second = _small_image_content_hash("data:image/jpeg;base64,abc")

    assert first == second


def test_filter_continuity_templates_by_image_count_keeps_templates_with_feasible_combination():
    """选图按对称性入口判断；对称性缺图时连续性模板也不可行。"""

    small_images = [
        make_small_image(RegionEnum.SIDE, "side-a", 5),
        make_small_image(RegionEnum.SIDE, "side-b", 4),
        make_small_image(RegionEnum.CENTER, "center-a", 3),
        make_small_image(RegionEnum.CENTER, "center-b", 2),
    ]

    result = _filter_continuity_templates_by_image_count(
        [Symmetry0()],
        [Continuity0(), Continuity1()],
        small_images,
    )

    expected_template_names = []

    assert [template.name for template in result] == expected_template_names


def test_filter_symmetry_templates_by_image_count_matches_center_image_requirements():
    """中心图数量按对称性入口判断，中心图 2 张时仅 Symmetry1 可保留。"""

    images_with_three_centers = [
        make_small_image(RegionEnum.SIDE, "side-a", 5),
        make_small_image(RegionEnum.SIDE, "side-b", 4),
        make_small_image(RegionEnum.SIDE, "side-c", 3),
        make_small_image(RegionEnum.CENTER, "center-a", 3),
        make_small_image(RegionEnum.CENTER, "center-b", 2),
        make_small_image(RegionEnum.CENTER, "center-c", 1),
    ]
    images_with_two_centers = [
        make_small_image(RegionEnum.SIDE, "side-a", 5),
        make_small_image(RegionEnum.SIDE, "side-b", 4),
        make_small_image(RegionEnum.SIDE, "side-c", 3),
        make_small_image(RegionEnum.CENTER, "center-a", 2),
        make_small_image(RegionEnum.CENTER, "center-b", 1),
    ]

    result_with_three_centers = _filter_symmetry_templates_by_image_count(
        [Symmetry0(), Symmetry1()],
        images_with_three_centers,
    )
    result_with_two_centers = _filter_symmetry_templates_by_image_count(
        [Symmetry0(), Symmetry1()],
        images_with_two_centers,
    )

    expected_template_names_with_three_centers = [
        StitchingSchemeName.SYMMETRY_0,
        StitchingSchemeName.SYMMETRY_1,
    ]
    expected_template_names_with_two_centers = [
        StitchingSchemeName.SYMMETRY_1,
    ]

    assert [template.name for template in result_with_three_centers] == expected_template_names_with_three_centers
    assert [template.name for template in result_with_two_centers] == expected_template_names_with_two_centers


def test_rank_candidates_prefers_score_then_stable_hash():
    """候选方案先按分数排序，同分时再使用稳定哈希保证结果可重复。"""

    image_a = make_small_image(RegionEnum.SIDE, "aaa", 5)
    image_b = make_small_image(RegionEnum.CENTER, "bbb", 4)
    image_c = make_small_image(RegionEnum.CENTER, "ccc", 4)
    higher = _CandidateScheme(_TemplateCombination(Symmetry0(), Continuity0()), (image_a, image_b), total_score=9)
    tied_a = _CandidateScheme(_TemplateCombination(Symmetry0(), Continuity0()), (image_a, image_c), total_score=8)
    tied_b = _CandidateScheme(_TemplateCombination(Symmetry1(), Continuity0()), (image_a, image_c), total_score=8)

    first = _CandidateScheme.rank([tied_b, higher, tied_a])
    second = _CandidateScheme.rank([tied_a, tied_b, higher])

    assert first[0] == higher
    assert first == second


def test_rank_candidates_changes_tie_break_order_when_image_content_changes():
    """同分方案中，图片内容变化会改变稳定哈希，从而影响排序结果。"""

    fixed_candidate = _CandidateScheme(
        _TemplateCombination(Symmetry0(), Continuity0()),
        (
            make_small_image(RegionEnum.SIDE, "fixed-side", 1),
            make_small_image(RegionEnum.CENTER, "fixed-center", 1),
        ),
        total_score=2,
    )
    before_change_candidate = _CandidateScheme(
        _TemplateCombination(Symmetry0(), Continuity0()),
        (
            make_small_image(RegionEnum.SIDE, "var-a", 1),
            make_small_image(RegionEnum.CENTER, "same-center", 1),
        ),
        total_score=2,
    )
    after_change_candidate = _CandidateScheme(
        _TemplateCombination(Symmetry0(), Continuity0()),
        (
            make_small_image(RegionEnum.SIDE, "var-0", 1),
            make_small_image(RegionEnum.CENTER, "same-center", 1),
        ),
        total_score=2,
    )

    before_change_ranked = _CandidateScheme.rank([fixed_candidate, before_change_candidate])
    after_change_ranked = _CandidateScheme.rank([fixed_candidate, after_change_candidate])

    expected_before_change_ranked = [before_change_candidate, fixed_candidate]
    expected_after_change_ranked = [fixed_candidate, after_change_candidate]

    assert before_change_ranked == expected_before_change_ranked
    assert after_change_ranked == expected_after_change_ranked


def test_candidate_total_score_counts_reused_images_for_each_output_rib():
    """重复复用同一张图时，分数要按最终输出 rib 次数重复累计。"""

    small_images = (
        make_small_image(RegionEnum.SIDE, "side-a", 8),
        make_small_image(RegionEnum.CENTER, "center-a", 3),
        make_small_image(RegionEnum.CENTER, "center-b", 2),
    )

    result = _candidate_total_score(
        _TemplateCombination(Symmetry1(), Continuity0()),
        small_images,
    )

    expected_total_score = 24

    assert result == expected_total_score


def test_build_candidates_fills_final_rib_positions_with_reused_images():
    """候选方案应按对称性入口选图，再填满 5 个 rib 位置。"""

    side_image = make_small_image(RegionEnum.SIDE, "side-a", 8)
    center_image = make_small_image(RegionEnum.CENTER, "center-a", 10)
    center_fill_image = make_small_image(RegionEnum.CENTER, "center-b", 1)
    candidate_template = _TemplateCombination(Symmetry1(), Continuity1())

    result = _build_candidates(
        [candidate_template],
        [side_image, center_image, center_fill_image],
    )

    first_candidate = result[0]
    expected_selected_payloads = [
        "data:image/png;base64,side-a",
        "data:image/png;base64,center-a",
        "data:image/png;base64,center-b",
        "data:image/png;base64,center-a",
        "data:image/png;base64,side-a",
    ]

    assert [image.image_base64 for image in first_candidate.selected_images] == expected_selected_payloads


def test_registered_templates_include_builtin_templates():
    """内置模板应在注册中心中可被自动发现。"""

    template_names = {template.name for template in get_stitching_templates()}

    assert {
        StitchingSchemeName.SYMMETRY_0,
        StitchingSchemeName.SYMMETRY_1,
        ContinuityModeName.CONTINUITY_0,
        ContinuityModeName.CONTINUITY_2,
    } <= template_names


def test_filter_templates_uses_configs_selected_by_stitch_scheme_generator_registry():
    """模板过滤应使用 Node2 注册清单筛选后的配置，而不是自行重新筛选。"""

    rules_config = [
        Rule1Config(),
        Rule3Config(),
        Rule100Config(
            rib_number=1,
            rib_sizes=[RibSizeItem(rib_name="rib1", num_pitchs=1, rib_width=1, rib_height=1)],
        ),
        Rule101Config(groove_sizes=[GrooveSizeItem(groove_width=1, groove_height=1)]),
        Rule102Config(
            decorations=[
                DecorationItem(
                    position="left",
                    decoration_width=1,
                    decoration_height=1,
                    decoration_opacity=1,
                )
            ]
        ),
    ]

    configs = select_node_configs(rules_config, STITCH_SCHEME_GENERATOR_CONFIGS)
    symmetry_templates, continuity_templates = _filter_templates(
        [Symmetry0(), Symmetry1(), Continuity0(), Continuity1(), Continuity2(), Rule3Template()],
        target_rib_number=5,
        configs=configs,
    )
    expected_config_names = [
        config.name
        for config in rules_config
        if type(config) in STITCH_SCHEME_GENERATOR_CONFIGS
    ]
    expected_symmetry_template_names = [
        StitchingSchemeName.SYMMETRY_0,
    ]
    expected_continuity_template_names = [
        ContinuityModeName.CONTINUITY_0,
    ]

    assert [config.name for config in configs] == expected_config_names
    assert [template.name for template in symmetry_templates] == expected_symmetry_template_names
    assert [template.name for template in continuity_templates] == expected_continuity_template_names


def test_candidate_count_formula_handles_templates_with_different_image_requirements():
    """排列公式应分别计算不同对称性模板的入口选图数量，再汇总。"""

    image_counts = {
        "side": 3,
        "center": 3,
    }
    template_combinations = [
        _TemplateCombination(Symmetry0(), Continuity0()),
        _TemplateCombination(Symmetry1(), Continuity0()),
    ]

    result = _candidate_count_formula_summary(template_combinations, image_counts)

    expected_formula_summary = "A[2,3]*A[3,3] + A[1,3]*A[2,3] = 36 + 18 = 54"

    assert result == expected_formula_summary


def test_generate_stitch_scheme_returns_requested_ranked_scheme():
    """完整流程应返回指定排名的方案，并保留 rib、主沟、装饰的实现数据。"""

    small_images = [
        make_small_image(RegionEnum.SIDE, "side-a", 5),
        make_small_image(RegionEnum.SIDE, "side-b", 3),
        make_small_image(RegionEnum.CENTER, "center-a", 4),
        make_small_image(RegionEnum.CENTER, "center-b", 2),
        make_small_image(RegionEnum.CENTER, "center-c", 1),
    ]
    rules_config = [
        Rule1Config(),
        Rule100Config(
                rib_number=5,
                rib_sizes=[
                    RibSizeItem(rib_name="rib1", num_pitchs=5, rib_width=400, rib_height=640),
                    RibSizeItem(rib_name="rib2", num_pitchs=6, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib3", num_pitchs=7, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib4", num_pitchs=6, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib5", num_pitchs=5, rib_width=400, rib_height=640),
                ],
        ),
        Rule101Config(
                groove_sizes=[
                    GrooveSizeItem(groove_width=10, groove_height=640),
                    GrooveSizeItem(groove_width=20, groove_height=640),
                ],
        ),
        Rule102Config(
                decorations=[
                    DecorationItem(
                        position="left",
                        decoration_width=300,
                        decoration_height=640,
                        decoration_opacity=128,
                    ),
                    DecorationItem(
                        position="right",
                        decoration_width=200,
                        decoration_height=640,
                        decoration_opacity=128,
                    ),
                ],
        ),
    ]

    result_big_image = generate_stitch_scheme(make_big_image(), small_images, rules_config, 1)

    assert isinstance(result_big_image, BigImage)
    assert isinstance(result_big_image.lineage, ImageLineage)
    result = result_big_image.lineage
    expected_scheme_name = StitchingSchemeName.SYMMETRY_0
    expected_rib_count = 5
    expected_rib_sizes = [
        ("rib1", 5, 400, 640),
        ("rib2", 6, 200, 640),
        ("rib3", 7, 200, 640),
        ("rib4", 6, 200, 640),
        ("rib5", 5, 400, 640),
    ]
    expected_groove_sizes = [(10, 640), (20, 640)]
    expected_decoration_sizes = [(300, 640, 128), (200, 640, 128)]

    assert result.stitching_scheme.stitching_scheme_abstract.name == expected_scheme_name
    assert len(result.stitching_scheme.ribs_scheme_implementation) == expected_rib_count
    assert {rib.before_image for rib in result.stitching_scheme.ribs_scheme_implementation} <= {
        "data:image/png;base64,side-a",
        "data:image/png;base64,side-b",
        "data:image/png;base64,center-a",
        "data:image/png;base64,center-b",
        "data:image/png;base64,center-c",
    }
    assert [
        (rib.rib_name, rib.num_pitchs, rib.rib_width, rib.rib_height)
        for rib in result.stitching_scheme.ribs_scheme_implementation
    ] == expected_rib_sizes
    assert [
        (groove.groove_width, groove.groove_height)
        for groove in result.main_groove_scheme.main_groove_implementation
    ] == expected_groove_sizes
    assert [
        (decoration.decoration_width, decoration.decoration_height, decoration.decoration_opacity)
        for decoration in result.decoration_scheme.decoration_implementation
    ] == expected_decoration_sizes
    assert all(
        not base64_to_ndarray(groove.before_image).any()
        for groove in result.main_groove_scheme.main_groove_implementation
    )
    assert all(
        not base64_to_ndarray(decoration.before_image).any()
        for decoration in result.decoration_scheme.decoration_implementation
    )


def test_generate_stitch_scheme_uses_best_subset_when_input_images_are_more_than_needed():
    """输入图片多于模板需求时，应优先选取得分更高的子集生成方案。"""

    small_images = [
        make_small_image(RegionEnum.SIDE, "side-a", 5),
        make_small_image(RegionEnum.SIDE, "side-b", 4),
        make_small_image(RegionEnum.SIDE, "side-c", 1),
        make_small_image(RegionEnum.CENTER, "center-a", 6),
        make_small_image(RegionEnum.CENTER, "center-b", 5),
        make_small_image(RegionEnum.CENTER, "center-c", 4),
        make_small_image(RegionEnum.CENTER, "center-d", 1),
        make_small_image(RegionEnum.CENTER, "center-e", 0),
    ]
    rules_config = [
        Rule1Config(),
        Rule100Config(
                rib_number=5,
                rib_sizes=[
                    RibSizeItem(rib_name="rib1", num_pitchs=5, rib_width=400, rib_height=640),
                    RibSizeItem(rib_name="rib2", num_pitchs=6, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib3", num_pitchs=7, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib4", num_pitchs=6, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib5", num_pitchs=5, rib_width=400, rib_height=640),
                ],
        ),
        Rule101Config(groove_sizes=[GrooveSizeItem(groove_width=10, groove_height=640)]),
        Rule102Config(
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

    result = generate_lineage_for_test(small_images, rules_config, 1)

    assert {rib.before_image for rib in result.stitching_scheme.ribs_scheme_implementation} <= {
        "data:image/png;base64,side-a",
        "data:image/png;base64,side-b",
        "data:image/png;base64,center-a",
        "data:image/png;base64,center-b",
        "data:image/png;base64,center-c",
    }


def test_generate_stitch_scheme_rejects_rank_out_of_range():
    """请求的方案排名超过候选总数时，应拒绝并提示 scheme_rank。"""

    small_images = [
        make_small_image(RegionEnum.SIDE, "side-a", 5),
        make_small_image(RegionEnum.SIDE, "side-b", 4),
        make_small_image(RegionEnum.CENTER, "center-a", 3),
        make_small_image(RegionEnum.CENTER, "center-b", 2),
        make_small_image(RegionEnum.CENTER, "center-c", 1),
    ]
    rules_config = [
        Rule1Config(),
        Rule100Config(
                rib_number=5,
                rib_sizes=[
                    RibSizeItem(rib_name="rib1", num_pitchs=5, rib_width=400, rib_height=640),
                    RibSizeItem(rib_name="rib2", num_pitchs=6, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib3", num_pitchs=7, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib4", num_pitchs=6, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib5", num_pitchs=5, rib_width=400, rib_height=640),
                ],
            ),
        Rule101Config(groove_sizes=[GrooveSizeItem(groove_width=10, groove_height=640)]),
        Rule102Config(
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

    out_of_range_scheme_rank = 37

    with pytest.raises(InputDataError, match="scheme_rank"):
        generate_stitch_scheme(make_big_image(), small_images, rules_config, out_of_range_scheme_rank)


def test_generate_stitch_scheme_supports_inherited_ribs():
    """对称性继承 rib 时，最终方案仍应补齐全部 rib 并标记来源。"""

    small_images = [
        make_small_image(RegionEnum.SIDE, "side-a", 5),
        make_small_image(RegionEnum.CENTER, "center-a", 4),
        make_small_image(RegionEnum.CENTER, "center-b", 3),
    ]
    rules_config = [
        Rule2Config(),
        Rule100Config(
                rib_number=5,
                rib_sizes=[
                    RibSizeItem(rib_name="rib1", num_pitchs=5, rib_width=400, rib_height=640),
                    RibSizeItem(rib_name="rib2", num_pitchs=6, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib3", num_pitchs=7, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib4", num_pitchs=6, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib5", num_pitchs=5, rib_width=400, rib_height=640),
                ],
        ),
        Rule101Config(groove_sizes=[GrooveSizeItem(groove_width=10, groove_height=640)]),
        Rule102Config(
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

    result = generate_lineage_for_test(small_images, rules_config, 1)

    expected_rib_names = [
        "rib1",
        "rib2",
        "rib3",
        "rib4",
        "rib5",
    ]
    expected_rib4_source = "rib2"
    expected_rib5_source = "rib1"

    assert [rib.rib_name for rib in result.stitching_scheme.ribs_scheme_implementation] == expected_rib_names
    assert result.stitching_scheme.ribs_scheme_implementation[3].rib_same_as == expected_rib4_source
    assert result.stitching_scheme.ribs_scheme_implementation[4].rib_same_as == expected_rib5_source


def test_instantiate_stitching_scheme_combines_symmetry_and_continuity_operations():
    """实例化时应把对称性操作和连续性操作按 rib 正确叠加。"""

    small_images = [
        make_small_image(RegionEnum.SIDE, "side-a", 5),
        make_small_image(RegionEnum.CENTER, "center-a", 4),
        make_small_image(RegionEnum.CENTER, "center-b", 3),
    ]
    rule100_config = Rule100Config(
        rib_number=5,
        rib_sizes=[
            RibSizeItem(rib_name="rib1", num_pitchs=5, rib_width=400, rib_height=640),
            RibSizeItem(rib_name="rib2", num_pitchs=6, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib3", num_pitchs=7, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib4", num_pitchs=6, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib5", num_pitchs=5, rib_width=400, rib_height=640),
        ],
    )
    candidate = _CandidateScheme(
        _TemplateCombination(Symmetry1(), Continuity1()),
        (
            small_images[0],
            small_images[1],
            small_images[2],
            small_images[1],
            small_images[0],
        ),
        total_score=12,
    )

    result = _instantiate_stitching_scheme(candidate, rule100_config)

    expected_rib_operations = [
        ("rib1", None, (RibOperation.NONE,)),
        ("rib2", None, (RibOperation.NONE, RibOperation.RESIZE_HORIZONTAL_2X, RibOperation.LEFT)),
        ("rib3", "rib2", (RibOperation.NONE, RibOperation.RESIZE_HORIZONTAL_2X, RibOperation.RIGHT)),
        ("rib4", "rib2", (RibOperation.FLIP,)),
        ("rib5", "rib1", (RibOperation.FLIP,)),
    ]

    assert [
        (rib.rib_name, rib.rib_same_as, rib.rib_operation)
        for rib in result.ribs_scheme_implementation
    ] == expected_rib_operations


def test_instantiate_stitching_scheme_merges_symmetry_inherit_chain_before_continuity():
    """Symmetry1 与 Continuity2 组合时，应先解析对称性继承链，再叠加连续性操作。"""

    small_images = [
        make_small_image(RegionEnum.SIDE, "side-a", 5),
        make_small_image(RegionEnum.CENTER, "center-a", 4),
        make_small_image(RegionEnum.CENTER, "center-b", 3),
    ]
    rule100_config = Rule100Config(
        rib_number=5,
        rib_sizes=[
            RibSizeItem(rib_name="rib1", num_pitchs=5, rib_width=400, rib_height=640),
            RibSizeItem(rib_name="rib2", num_pitchs=6, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib3", num_pitchs=7, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib4", num_pitchs=6, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib5", num_pitchs=5, rib_width=400, rib_height=640),
        ],
    )
    candidate = _CandidateScheme(
        _TemplateCombination(Symmetry1(), Continuity2()),
        (
            small_images[0],
            small_images[1],
            small_images[2],
            small_images[1],
            small_images[0],
        ),
        total_score=12,
    )

    result = _instantiate_stitching_scheme(candidate, rule100_config)

    expected_rib_operations = [
        ("rib1", None, (RibOperation.NONE,)),
        ("rib2", None, (RibOperation.NONE,)),
        ("rib3", "rib2", (RibOperation.FLIP, RibOperation.RESIZE_HORIZONTAL_2X, RibOperation.LEFT)),
        ("rib4", "rib2", (RibOperation.FLIP, RibOperation.RESIZE_HORIZONTAL_2X, RibOperation.RIGHT)),
        ("rib5", "rib1", (RibOperation.FLIP,)),
    ]
    expected_before_images = [
        "data:image/png;base64,side-a",
        "data:image/png;base64,center-a",
        "data:image/png;base64,center-a",
        "data:image/png;base64,center-a",
        "data:image/png;base64,side-a",
    ]

    assert [
        (rib.rib_name, rib.rib_same_as, rib.rib_operation)
        for rib in result.ribs_scheme_implementation
    ] == expected_rib_operations
    assert [
        rib.before_image
        for rib in result.ribs_scheme_implementation
    ] == expected_before_images


def test_merge_template_combination_ribs_flattens_symmetry_and_continuity_templates():
    """模板组合应先合并成最终 5 个 RibTemplate，再交给运行时方案实例化。"""

    result = _merge_template_combination_ribs(
        _TemplateCombination(Symmetry1(), Continuity2()),
    )

    expected_final_ribs = [
        ("rib1", SourceTypeEnum.ORIGINAL, None, (RibOperation.NONE,)),
        ("rib2", SourceTypeEnum.ORIGINAL, None, (RibOperation.NONE,)),
        ("rib3", SourceTypeEnum.INHERIT, "rib2", (RibOperation.FLIP, RibOperation.RESIZE_HORIZONTAL_2X, RibOperation.LEFT)),
        ("rib4", SourceTypeEnum.INHERIT, "rib2", (RibOperation.FLIP, RibOperation.RESIZE_HORIZONTAL_2X, RibOperation.RIGHT)),
        ("rib5", SourceTypeEnum.INHERIT, "rib1", (RibOperation.FLIP,)),
    ]

    assert [
        (
            rib.rib_name,
            rib.source_type,
            rib.inherit_from,
            rib.operation_template,
        )
        for rib in result
    ] == expected_final_ribs


def test_instantiate_stitching_scheme_keeps_symmetry_image_count_when_continuity_merges_sources():
    """连续性合并来源后，选图仍沿用对称性模板的原始图片数量。"""

    small_images = [
        make_small_image(RegionEnum.SIDE, "side-a", 5),
        make_small_image(RegionEnum.CENTER, "center-a", 4),
        make_small_image(RegionEnum.CENTER, "center-b", 3),
        make_small_image(RegionEnum.CENTER, "center-c", 2),
        make_small_image(RegionEnum.SIDE, "side-b", 1),
    ]
    rule100_config = Rule100Config(
        rib_number=5,
        rib_sizes=[
            RibSizeItem(rib_name="rib1", num_pitchs=5, rib_width=400, rib_height=640),
            RibSizeItem(rib_name="rib2", num_pitchs=6, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib3", num_pitchs=7, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib4", num_pitchs=6, rib_width=200, rib_height=640),
            RibSizeItem(rib_name="rib5", num_pitchs=5, rib_width=400, rib_height=640),
        ],
    )
    candidate = _CandidateScheme(
        _TemplateCombination(Symmetry0(), Continuity1()),
        (
            small_images[0],
            small_images[1],
            small_images[1],
            small_images[3],
            small_images[4],
        ),
        total_score=14,
    )

    result = _instantiate_stitching_scheme(candidate, rule100_config)

    expected_before_images = [
        "data:image/png;base64,side-a",
        "data:image/png;base64,center-a",
        "data:image/png;base64,center-a",
        "data:image/png;base64,center-c",
        "data:image/png;base64,side-b",
    ]

    assert [rib.before_image for rib in result.ribs_scheme_implementation] == expected_before_images


def test_generate_stitch_scheme_rejects_missing_rib_size_config():
    """Rule100 未覆盖全部输出 rib 时，应拒绝生成不完整方案。"""

    small_images = [
        make_small_image(RegionEnum.SIDE, "side-a", 5),
        make_small_image(RegionEnum.SIDE, "side-b", 4),
        make_small_image(RegionEnum.CENTER, "center-a", 3),
        make_small_image(RegionEnum.CENTER, "center-b", 2),
        make_small_image(RegionEnum.CENTER, "center-c", 1),
    ]
    rules_config = [
        Rule1Config(),
        Rule100Config(
                rib_number=5,
                rib_sizes=[
                    RibSizeItem(rib_name="rib1", num_pitchs=5, rib_width=400, rib_height=640),
                    RibSizeItem(rib_name="rib2", num_pitchs=6, rib_width=200, rib_height=640),
                ],
        ),
        Rule101Config(groove_sizes=[GrooveSizeItem(groove_width=10, groove_height=640)]),
        Rule102Config(
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

    with pytest.raises(InputDataError, match="rib_sizes"):
        generate_stitch_scheme(make_big_image(), small_images, rules_config, 1)


def test_generate_stitch_scheme_logs_key_execution_steps(caplog):
    """主流程日志应保留关键步骤，便于逐段检查模板、排列与得分。"""

    small_images = [
        make_small_image(RegionEnum.SIDE, "side-a", 5),
        make_small_image(RegionEnum.SIDE, "side-b", 4),
        make_small_image(RegionEnum.CENTER, "center-a", 3),
        make_small_image(RegionEnum.CENTER, "center-b", 2),
        make_small_image(RegionEnum.CENTER, "center-c", 1),
    ]
    rules_config = [
        Rule1Config(),
        Rule100Config(
                rib_number=5,
                rib_sizes=[
                    RibSizeItem(rib_name="rib1", num_pitchs=5, rib_width=400, rib_height=640),
                    RibSizeItem(rib_name="rib2", num_pitchs=6, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib3", num_pitchs=7, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib4", num_pitchs=6, rib_width=200, rib_height=640),
                    RibSizeItem(rib_name="rib5", num_pitchs=5, rib_width=400, rib_height=640),
                ],
        ),
        Rule101Config(groove_sizes=[GrooveSizeItem(groove_width=10, groove_height=640)]),
        Rule102Config(
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

    with caplog.at_level("DEBUG", logger="拼接方案"):
        generate_stitch_scheme(make_big_image(), small_images, rules_config, 1)

    messages = [record.message for record in caplog.records]
    expected_continuity_config_log = "连续性配置: continuity_mode_list=[continuity_0]"
    expected_formula_log = "公式=A[2,2]*A[3,3] = 12 = 12"
    expected_candidate_count_log = "生成方案数量: 12"
    expected_top_score_templates_log = "最高分模板组合:"
    expected_rib_lineage_log = "lineage.rib[1]:"
    expected_groove_lineage_log = "lineage.groove[1]:"
    expected_decoration_lineage_log = "lineage.decoration[1]:"

    assert any("小图数量:" in message for message in messages)
    assert any(expected_continuity_config_log in message for message in messages)
    assert any("准备工作:" in message for message in messages)
    assert any("排列思路:" in message for message in messages)
    assert any(expected_formula_log in message for message in messages)
    assert any(expected_candidate_count_log in message for message in messages)
    assert any("得分统计:" in message for message in messages)
    assert any(expected_top_score_templates_log in message for message in messages)
    assert any(expected_rib_lineage_log in message for message in messages)
    assert any(expected_groove_lineage_log in message for message in messages)
    assert any(expected_decoration_lineage_log in message for message in messages)
