from __future__ import annotations

from tire_ai_pattern.common.exceptions import InputDataError, InputTypeError
from tire_ai_pattern.core.detection.transverse_sipe import detect_transverse_sipes
from tire_ai_pattern.models.enums import RegionEnum
from tire_ai_pattern.models.image_models import BaseImage, SmallImage
from tire_ai_pattern.models.rule_models import (
    BaseRuleFeature,
    BaseRuleScore,
    Rule9Config,
    Rule9Feature,
    Rule9Score,
)
from tire_ai_pattern.rules.base import RuleExecutor
from tire_ai_pattern.rules.registry import register_rule_executor
from tire_ai_pattern.utils.image_utils import base64_to_ndarray, ndarray_to_base64


FEATURE_FUNCTION = "Rule9Executor.exec_feature"
SCORE_FUNCTION = "Rule9Executor.exec_score"
DEBUG_IMAGE_NAME = "rule9_transverse_sipes.png"


@register_rule_executor
class Rule9Executor(RuleExecutor):
    rule_cls = Rule9Config

    def exec_feature(
        self,
        image: BaseImage,
        config: Rule9Config,
        is_debug: bool = False,
    ) -> BaseRuleFeature:
        if not isinstance(image, SmallImage):
            raise InputTypeError(FEATURE_FUNCTION, "image", "SmallImage", type(image).__name__)
        if not isinstance(config, Rule9Config):
            raise InputTypeError(FEATURE_FUNCTION, "config", "Rule9Config", type(config).__name__)
        if not isinstance(is_debug, bool):
            raise InputTypeError(FEATURE_FUNCTION, "is_debug", "bool", type(is_debug).__name__)

        region = image.biz.region
        if region not in (RegionEnum.CENTER, RegionEnum.SIDE):
            raise InputDataError(FEATURE_FUNCTION, "image.biz.region", "must be center or side", region)

        image_array = base64_to_ndarray(image.image_base64)
        sipe_count, _positions, _widths, _line_mask, debug_image = detect_transverse_sipes(
            image_array,
            nominal_width_px=config.transverse_sipe_width,
            is_debug=is_debug,
        )

        num_transverse_sipes_rib1_5 = sipe_count if region == RegionEnum.SIDE else 0
        num_transverse_sipes_rib2_4 = sipe_count if region == RegionEnum.CENTER else 0
        is_count_valid = self._is_count_valid(
            config=config,
            region=region,
            num_transverse_sipes_rib1_5=num_transverse_sipes_rib1_5,
            num_transverse_sipes_rib2_4=num_transverse_sipes_rib2_4,
            function_name=FEATURE_FUNCTION,
        )

        feature_data = {
            "num_transverse_sipes_rib1_5": num_transverse_sipes_rib1_5,
            "num_transverse_sipes_rib2_4": num_transverse_sipes_rib2_4,
            "is_count_valid": is_count_valid,
            "region": region,
        }
        if is_debug and debug_image is not None:
            feature_data["vis_names"] = [DEBUG_IMAGE_NAME]
            feature_data["vis_images"] = [ndarray_to_base64(debug_image)]

        return Rule9Feature(**feature_data)

    def exec_score(
        self,
        config: Rule9Config,
        feature: Rule9Feature,
    ) -> BaseRuleScore:
        if not isinstance(config, Rule9Config):
            raise InputTypeError(SCORE_FUNCTION, "config", "Rule9Config", type(config).__name__)
        if not isinstance(feature, Rule9Feature):
            raise InputTypeError(SCORE_FUNCTION, "feature", "Rule9Feature", type(feature).__name__)

        self._validate_count("feature.num_transverse_sipes_rib1_5", feature.num_transverse_sipes_rib1_5)
        self._validate_count("feature.num_transverse_sipes_rib2_4", feature.num_transverse_sipes_rib2_4)
        is_count_valid = self._is_count_valid(
            config=config,
            region=feature.region,
            num_transverse_sipes_rib1_5=feature.num_transverse_sipes_rib1_5,
            num_transverse_sipes_rib2_4=feature.num_transverse_sipes_rib2_4,
            function_name=SCORE_FUNCTION,
        )

        score = config.max_score if is_count_valid else 0
        return Rule9Score(score=score)

    @staticmethod
    def _is_count_valid(
        config: Rule9Config,
        region: RegionEnum,
        num_transverse_sipes_rib1_5: int,
        num_transverse_sipes_rib2_4: int,
        function_name: str,
    ) -> bool:
        if region == RegionEnum.SIDE:
            min_count = config.min_sipe_count_rib1_5
            max_count = config.max_sipe_count_rib1_5
            count = num_transverse_sipes_rib1_5
            min_field_name = "config.min_sipe_count_rib1_5"
        elif region == RegionEnum.CENTER:
            min_count = config.min_sipe_count_rib2_4
            max_count = config.max_sipe_count_rib2_4
            count = num_transverse_sipes_rib2_4
            min_field_name = "config.min_sipe_count_rib2_4"
        else:
            raise InputDataError(function_name, "feature.region", "must be center or side", region)

        Rule9Executor._validate_count(min_field_name, min_count, function_name=function_name)
        Rule9Executor._validate_count(min_field_name.replace("min", "max"), max_count, function_name=function_name)
        if min_count > max_count:
            raise InputDataError(function_name, min_field_name, "must be less than or equal to max count", min_count)

        return min_count <= count <= max_count

    @staticmethod
    def _validate_count(field_name: str, value: int, function_name: str = SCORE_FUNCTION) -> None:
        if value < 0:
            raise InputDataError(function_name, field_name, "must be >= 0", value)
