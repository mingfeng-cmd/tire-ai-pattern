"""节点层通用配置和评估工具。

本模块为具体 node 提供通用的规则配置筛选、重复校验、评分汇总和
图片评估能力。具体 node 只需要声明自己支持的规则顺序，并调用这些
helper 完成规则执行。
"""

from __future__ import annotations

from tire_ai_pattern.common.exceptions import InputDataError
from tire_ai_pattern.models.enums import RuleTypeEnum
from tire_ai_pattern.models.image_models import BaseImage, ImageEvaluation, RuleEvaluation
from tire_ai_pattern.models.rule_models import (
    BaseRuleConfig,
    Rule1Config,
    Rule2Config,
    Rule3Config,
    Rule4Config,
    # Rule5Config,  # 已注释
    Rule6AConfig,
    Rule6Config,
    Rule7Config,
    Rule8Config,
    Rule9Config,
    Rule10Config,
    Rule11Config,
    Rule12Config,
    Rule13Config,
    Rule14Config,
    Rule15Config,
    Rule16Config,
    Rule17Config,
    Rule18Config,
    Rule19Config,
    Rule20Config,
    Rule21Config,
    Rule22Config,
    Rule100Config,
    Rule101Config,
    Rule102Config,
)
from tire_ai_pattern.rules.runner import RuleRunner


ALL_RULE_CONFIG_TYPES: list[type[BaseRuleConfig]] = [
    Rule1Config,
    Rule2Config,
    Rule3Config,
    Rule4Config,
    # Rule5Config,  # 已注释
    Rule6AConfig,
    Rule6Config,
    Rule7Config,
    Rule8Config,
    Rule9Config,
    Rule10Config,
    Rule11Config,
    Rule12Config,
    Rule13Config,
    Rule14Config,
    Rule15Config,
    Rule16Config,
    Rule17Config,
    Rule18Config,
    Rule19Config,
    Rule20Config,
    Rule21Config,
    Rule22Config,
    Rule100Config,
    Rule101Config,
    Rule102Config,
]


def get_rule_config_types_by_rule_type(
    rule_type: RuleTypeEnum,
) -> list[type[BaseRuleConfig]]:
    """Return rule config classes whose declared ``rule_type`` matches."""

    return [
        config_type
        for config_type in ALL_RULE_CONFIG_TYPES
        if config_type.model_fields["rule_type"].default == rule_type
    ]


SMALL_IMAGE_EVALUATOR_CONFIGS: list[type[BaseRuleConfig]] = (
    get_rule_config_types_by_rule_type(RuleTypeEnum.SMALL_IMAGE)
)

BIG_IMAGE_EVALUATOR_CONFIGS: list[type[BaseRuleConfig]] = (
    get_rule_config_types_by_rule_type(RuleTypeEnum.BIG_IMAGE)
)

DEFAULT_RULE_CONFIGS: list[type[BaseRuleConfig]] = (
    get_rule_config_types_by_rule_type(RuleTypeEnum.DEFAULT)
)

STITCH_SCHEME_GENERATOR_CONFIGS: list[type[BaseRuleConfig]] = [
    Rule1Config,
    Rule2Config,
    Rule3Config,
    Rule4Config,
    # Rule5Config,  # 已注释
    Rule6AConfig,
    Rule7Config,
    Rule12Config,
    Rule16Config,
    Rule17Config,
    Rule19Config,
    Rule100Config,
    Rule101Config,
    Rule102Config,
]


def validate_no_duplicate_config_types(configs: list[BaseRuleConfig]) -> None:
    """校验规则配置列表中不存在重复的规则配置类型。

    Args:
        configs: 待校验的规则配置列表。列表中的每一项必须是
            ``BaseRuleConfig`` 的具体子类实例。

    Raises:
        InputDataError: 当同一种规则配置类型在列表中重复出现时抛出。
            异常字段为重复规则的 ``config.name``。
    """

    seen: set[type[BaseRuleConfig]] = set()
    for config in configs:
        config_type = type(config)
        if config_type in seen:
            raise InputDataError(
                "rules_config",
                config.name,
                "duplicate rule config",
            )
        seen.add(config_type)


def select_node_configs(
    rules_config: list[BaseRuleConfig],
    ordered_config_types: list[type[BaseRuleConfig]],
) -> list[BaseRuleConfig]:
    """按节点声明的规则顺序筛选本节点需要执行的规则配置。

    Args:
        rules_config: 用户传入的完整规则配置列表。
        ordered_config_types: 当前节点支持的规则配置类型列表。列表顺序
            即节点执行规则的固定顺序。

    Returns:
        按 ``ordered_config_types`` 排序后的规则配置列表。用户未传入的
        规则配置会被忽略，不会生成默认配置。

    """
    config_by_type = {type(config): config for config in rules_config}

    return [
        config_by_type[config_type]
        for config_type in ordered_config_types
        if config_type in config_by_type
    ]


def recalculate_current_score(evaluation: ImageEvaluation) -> None:
    """根据每条规则已有的 score 重新计算图片当前总分。

    Args:
        evaluation: 待更新的图片评估结果对象。函数会原地更新
            ``evaluation.current_score``。

    Notes:
        缺少 ``score`` 的规则会被跳过，不参与总分计算。
    """

    evaluation.current_score = sum(
        rule.score.score for rule in evaluation.rules
        if rule.score is not None and rule.score.score is not None
    )


def evaluate_image_with_configs(
    image: BaseImage,
    configs: list[BaseRuleConfig],
    is_debug: bool = False,
) -> ImageEvaluation:
    """使用指定规则配置完整评估一张图片。

    函数会按 ``configs`` 顺序依次调用 ``RuleRunner.exec_feature`` 和
    ``RuleRunner.exec_score``，并将每条规则的 config、feature、score
    组装成 ``RuleEvaluation``，最后汇总得到 ``ImageEvaluation``。

    Args:
        image: 待评估的图片对象，可以是小图或大图。
        configs: 当前节点筛选后的规则配置列表，执行顺序与列表顺序一致。
        is_debug: 是否在规则特征中附带 debug 可视化结果。

    Returns:
        新生成的图片评估结果对象。调用方负责将其写回到图片的
        ``image.evaluation`` 字段。

    Raises:
        Exception: 本函数不捕获 ``RuleRunner`` 或具体规则执行器抛出的
            异常，异常会原样向上透传。
    """

    rule_evaluations: list[RuleEvaluation] = []

    for config in configs:
        feature = RuleRunner.exec_feature(image, config, is_debug=is_debug)
        score = RuleRunner.exec_score(config, feature)
        rule_evaluations.append(
            RuleEvaluation(
                name=config.name,
                config=config,
                feature=feature,
                score=score,
            )
        )

    evaluation = ImageEvaluation(rules=rule_evaluations)
    recalculate_current_score(evaluation)
    return evaluation
