import pytest
import tire_ai_pattern.rules.executors as executors

from tire_ai_pattern.models.enums import ImageFormatEnum, ImageModeEnum, LevelEnum, RegionEnum
from tire_ai_pattern.models.image_models import BigImage, ImageBiz, ImageMeta
from tire_ai_pattern.models.rule_models import (
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
)
from tire_ai_pattern.rules.base import RuleExecutor
from tire_ai_pattern.rules.registry import get_rule_executor
from tire_ai_pattern.rules.executors.rule19 import Rule19Executor


executors.load_all_executors()


ALL_RULE_CONFIGS = [
    Rule1Config,
    Rule2Config,
    Rule3Config,
    Rule4Config,
    # Rule5Config,  # 已注释
    Rule6Config,
    Rule6AConfig,
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
]


def make_big_image() -> BigImage:
    return BigImage(
        image_base64="data:image/png;base64,original",
        meta=ImageMeta(
            width=100,
            height=40,
            channels=3,
            mode=ImageModeEnum.RGB,
            format=ImageFormatEnum.PNG,
            size=20,
        ),
        biz=ImageBiz(level=LevelEnum.BIG, region=RegionEnum.CENTER),
    )


def make_config() -> Rule19Config:
    return Rule19Config(
        tire_design_width=80,
        decoration_border_alpha=0.5,
        decoration_gray_color=128,
    )


def test_rule19_executor_has_no_rule_name_attribute():
    """验证 Rule19Executor 不再声明单独的 rule_name 属性。"""
    assert not hasattr(Rule19Executor(), "rule_name")


def test_rule19_executor_defines_rule_cls():
    """验证 Rule19Executor 通过 rule_cls 绑定 Rule19Config。"""
    assert Rule19Executor.rule_cls is Rule19Config


def test_rule19_executor_inherits_rule_executor():
    """验证 Rule19Executor 是 RuleExecutor 的具体实现类。"""
    assert isinstance(Rule19Executor(), RuleExecutor)


def test_all_rule_executors_are_registered():
    """验证 Rule1 到 Rule22 的 executor 都已按 config 名称注册。"""
    for config_cls in ALL_RULE_CONFIGS:
        rule_name = config_cls.__name__.lower().replace("config", "")

        executor = get_rule_executor(rule_name)

        assert isinstance(executor, RuleExecutor)
        assert executor.rule_cls is config_cls



def test_unimplemented_rule_uses_base_not_implemented_methods():
    """验证未落地规则使用 RuleExecutor 默认未实现方法。"""
    # Rule20 是一个未实现功能的规则，使用基类的默认实现
    config = Rule20Config(
        prompt="test",
        num_images=1,
        output_width=512,
        output_height=512,
    )
    executor = get_rule_executor(config.name)
    image = make_big_image()

    assert type(executor).exec_feature is RuleExecutor.exec_feature
    assert type(executor).exec_score is RuleExecutor.exec_score
    with pytest.raises(NotImplementedError, match="rule20.exec_feature is not implemented"):
        executor.exec_feature(image, config)



def test_rule19_executor_is_registered_by_config_name():
    """验证可以通过 Rule19Config.name 从全局注册表取回 Rule19Executor。"""
    assert isinstance(get_rule_executor(make_config().name), Rule19Executor)
