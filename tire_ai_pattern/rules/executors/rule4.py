from __future__ import annotations

from tire_ai_pattern.models.image_models import BaseImage, BigImage
from tire_ai_pattern.models.rule_models import Rule4Config, Rule4Feature, Rule4Score
from tire_ai_pattern.models.enums import StitchingSchemeName
from tire_ai_pattern.rules.base import RuleExecutor
from tire_ai_pattern.rules.registry import register_rule_executor


@register_rule_executor
class Rule4Executor(RuleExecutor):
    rule_cls = Rule4Config

    def exec_feature(self, image: BaseImage, config: Rule4Config, is_debug: bool = False) -> Rule4Feature:
        """根据血缘中的 StitchingSchemeName 判断是否匹配中心线镜像对称可错位方案。"""
        if not isinstance(image, BigImage) or image.lineage is None:
            return Rule4Feature(is_active=False)

        scheme_name = image.lineage.stitching_scheme.stitching_scheme_abstract.name
        is_active = scheme_name in (
            StitchingSchemeName.SYMMETRY_3,
            StitchingSchemeName.SYMMETRY_7,
        )
        return Rule4Feature(is_active=is_active)

    def exec_score(self, config: Rule4Config, feature: Rule4Feature) -> Rule4Score:
        score = config.max_score if feature.is_active else 0
        return Rule4Score(score=score)
