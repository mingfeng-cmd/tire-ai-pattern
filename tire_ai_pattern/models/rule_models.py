# ============================================================
# 规则模型 - 规则层
#
# 包含：
#   - 注册机制（_FEATURE_REGISTRY / _SCORE_REGISTRY / 装饰器）
#   - BaseRuleConfig / BaseRuleFeature / BaseRuleScore（基类）
#   - Rule1-22Config / Rule1-22Feature / Rule1-22Score（22个规则）
#   - RibSizeItem / GrooveSizeItem / DecorationItem（Rule100-102 辅助模型）
#   - Rule100-102Config / Rule100-102Feature / Rule100-102Score（3个纯配置型规则）
#
# 注意：
#   - Feature 类必须使用 @register_rule_feature 装饰器
#   - Score 类必须使用 @register_rule_score 装饰器
#   - name 属性从类名自动提取，不要手动定义
# ============================================================

from typing import Optional, List, Dict, Type
from pydantic import BaseModel, Field
from .enums import ContinuityModeName, RuleTypeEnum, DecorationPositionEnum

from tire_ai_pattern.models.enums import RegionEnum


# ============================================================
# 第一部分：注册机制
# ============================================================

_FEATURE_REGISTRY: Dict[str, Type["BaseRuleFeature"]] = {}
_SCORE_REGISTRY: Dict[str, Type["BaseRuleScore"]] = {}


def register_rule_feature(cls: Type["BaseRuleFeature"]) -> Type["BaseRuleFeature"]:
    """注册规则特征类到全局注册表"""
    _FEATURE_REGISTRY[cls.__name__] = cls
    return cls


def register_rule_score(cls: Type["BaseRuleScore"]) -> Type["BaseRuleScore"]:
    """注册规则评分类到全局注册表"""
    _SCORE_REGISTRY[cls.__name__] = cls
    return cls


def get_feature_class(name: str) -> Type["BaseRuleFeature"]:
    """
    根据规则名获取特征类

    Args:
        name: 规则名，支持以下格式：
            - 全小写：rule8, rule6a
            - 首字母大写：Rule8, Rule6A

    Returns:
        特征类或 None
    """
    # 尝试直接匹配类名
    class_name = f"{name}Feature"
    if class_name in _FEATURE_REGISTRY:
        return _FEATURE_REGISTRY[class_name]

    # 尝试首字母大写后匹配
    class_prefix = name[0].upper() + name[1:] if name else name
    class_name = f"{class_prefix}Feature"
    if class_name in _FEATURE_REGISTRY:
        return _FEATURE_REGISTRY[class_name]

    # 遍历注册表，通过类名提取的 name 匹配（避免实例化）
    name_lower = name.lower()
    for cls_name, cls in _FEATURE_REGISTRY.items():
        # 从类名提取 name: Rule6AFeature -> rule6a
        extracted_name = cls_name.lower().replace("feature", "")
        if extracted_name == name_lower:
            return cls

    return None


def get_score_class(name: str) -> Type["BaseRuleScore"]:
    """
    根据规则名获取评分类

    Args:
        name: 规则名，支持以下格式：
            - 全小写：rule8, rule6a
            - 首字母大写：Rule8, Rule6A

    Returns:
        评分类或 None
    """
    # 尝试直接匹配类名
    class_name = f"{name}Score"
    if class_name in _SCORE_REGISTRY:
        return _SCORE_REGISTRY[class_name]

    # 尝试首字母大写后匹配
    class_prefix = name[0].upper() + name[1:] if name else name
    class_name = f"{class_prefix}Score"
    if class_name in _SCORE_REGISTRY:
        return _SCORE_REGISTRY[class_name]

    # 遍历注册表，通过类名提取的 name 匹配（避免实例化）
    name_lower = name.lower()
    for cls_name, cls in _SCORE_REGISTRY.items():
        # 从类名提取 name: Rule6AScore -> rule6a
        extracted_name = cls_name.lower().replace("score", "")
        if extracted_name == name_lower:
            return cls

    return None


# ============================================================
# 第二部分：基类定义
# ============================================================

class BaseRuleConfig(BaseModel):
    """
    规则配置基类

    所有规则配置的公共基类，定义通用字段。
    子类继承并扩展具体阈值字段。
    name属性自动从类名提取（如Rule8Config → rule8）。
    """

    description: str = Field(description="规则描述")
    max_score: Optional[int] = Field(default=None, ge=0, description="最大可得分，None表示非打分规则")
    rule_type: Optional[RuleTypeEnum] = Field(default=None, description="规则类型枚举")
    @property
    def name(self) -> str:
        """规则名称，从类名自动提取（如Rule8Config → rule8）"""
        return self.__class__.__name__.lower().replace("config", "")

    @property
    def log_display(self) -> str:
        return f'{self.__class__.__name__}[{self.description}]'


class BaseRuleFeature(BaseModel):
    """
    规则特征基类

    所有规则特征的公共基类。
    name属性自动从类名提取（如Rule8Feature → rule8）。

    注意：
    - Feature只包含用于打分的特征数据，不能访问图片
    - 支持debug模式，输出可视化图片（vis_names/vis_images）
    """

    vis_names: Optional[List[str]] = Field(default=None, description="可视化名称列表，debug模式使用")
    vis_images: Optional[List[str]] = Field(default=None, description="可视化图片base64列表，debug模式使用")

    @property
    def name(self) -> str:
        """规则名称，从类名自动提取（如Rule8Feature → rule8）"""
        return self.__class__.__name__.lower().replace("feature", "")


class BaseRuleScore(BaseModel):
    """
    规则评分基类

    所有规则评分的公共基类。
    name属性自动从类名提取（如Rule8Score → rule8）。
    """

    score: Optional[int] = Field(default=None, description="得分，None表示不参与评分")

    @property
    def name(self) -> str:
        """规则名称，从类名自动提取（如Rule8Score → rule8）"""
        return self.__class__.__name__.lower().replace("score", "")


# ============================================================
# 第三部分：Rule1-5 横图拼接子规则
# ============================================================

class Rule1Config(BaseRuleConfig):
    """Rule1：rib无对称"""
    description: str = "rib无对称"
    max_score: int = 10
    rule_type: RuleTypeEnum = RuleTypeEnum.BIG_IMAGE


class Rule2Config(BaseRuleConfig):
    """Rule2：rib中心对称"""
    description: str = "rib中心对称"
    max_score: int = 10
    rule_type: RuleTypeEnum = RuleTypeEnum.BIG_IMAGE


class Rule3Config(BaseRuleConfig):
    """Rule3：rib左右对称"""
    description: str = "rib左右对称"
    max_score: int = 10
    rule_type: RuleTypeEnum = RuleTypeEnum.BIG_IMAGE


# class Rule4Config(BaseRuleConfig):
#     """Rule4：未实现"""
#     description: str = "未实现"
#     # max_score 继承 BaseRuleConfig，默认 None


# class Rule5Config(BaseRuleConfig):
#     """Rule5：分数已合并入rule1～4"""
#     description: str = "分数已合并入rule1～4"
#     # max_score 继承 BaseRuleConfig，默认 None
@register_rule_feature
class Rule1Feature(BaseRuleFeature):
    """Rule1特征：rib无对称"""
    is_active: bool = Field(description="是否生效")


@register_rule_feature
class Rule2Feature(BaseRuleFeature):
    """Rule2特征：rib中心对称"""
    is_active: bool = Field(description="是否生效")


@register_rule_feature
class Rule3Feature(BaseRuleFeature):
    """Rule3特征：rib左右对称"""
    is_active: bool = Field(description="是否生效")


# @register_rule_feature
# class Rule4Feature(BaseRuleFeature):
#     """Rule4特征：未实现"""
#     pass


# @register_rule_feature
# class Rule5Feature(BaseRuleFeature):
#     """Rule5特征：分数已合并入rule1～4"""
#     pass


@register_rule_score
class Rule1Score(BaseRuleScore):
    """Rule1评分"""
    pass


@register_rule_score
class Rule2Score(BaseRuleScore):
    """Rule2评分"""
    pass


@register_rule_score
class Rule3Score(BaseRuleScore):
    """Rule3评分"""
    pass


# @register_rule_score
# class Rule4Score(BaseRuleScore):
#     """Rule4评分：未实现"""
#     pass


# @register_rule_score
# class Rule5Score(BaseRuleScore):
#     """Rule5评分：分数已合并入rule1～4"""
#     pass


# ============================================================
# 第四部分：Rule6-6A 连续性检测与拼接节距
# ============================================================

class Rule6Config(BaseRuleConfig):
    """Rule6：节距纵向关系无缝拼接/图案连续性检测"""
    description: str = "节距纵向关系无缝拼接 / 图案连续性检测"
    max_score: int = 10
    rule_type: RuleTypeEnum = RuleTypeEnum.SMALL_IMAGE


class Rule6AConfig(BaseRuleConfig):
    """Rule6A：拼接节距 - 非打分规则，后续可能摘出为新类"""
    description: str = "拼接节距"
    max_score: int = 0
    rule_type: RuleTypeEnum = RuleTypeEnum.SMALL_IMAGE
    default_stitch_count: int = Field(description="拼接次数")
    target_resolution_width: int = Field(description="目标分辨率宽度(像素)")


@register_rule_feature
class Rule6Feature(BaseRuleFeature):
    """Rule6特征：连续性检测"""
    is_continuous: bool = Field(description="是否连续")


@register_rule_feature
class Rule6AFeature(BaseRuleFeature):
    """Rule6A特征：截距数量"""
    num_pitchs: int = Field(description="节距数量")


@register_rule_score
class Rule6Score(BaseRuleScore):
    """Rule6（连续性检测）评分"""
    pass


@register_rule_score
class Rule6AScore(BaseRuleScore):
    """Rule6A（拼接节距）评分"""
    pass


# ============================================================
# 第五部分：Rule7 主沟宽度/数量（已被融入）
# ============================================================

class Rule7Config(BaseRuleConfig):
    """Rule7：主沟宽度/数量 - 现被rule1to5横图拼接、rule12/16/17连续性拼接吸收"""
    description: str = "主沟宽度/数量约束"
    max_score: int = 0
    rule_type: RuleTypeEnum = RuleTypeEnum.BIG_IMAGE
    main_groove_width: float = Field(description="主沟宽度(像素)")
    main_groove_width_min: float = Field(description="主沟宽度下限(像素)")
    main_groove_width_max: float = Field(description="主沟宽度上限(像素)")
    main_groove_count_min: int = Field(description="主沟数量下限")
    main_groove_count_max: int = Field(description="主沟数量上限")
    preferred_groove_count: int = Field(description="优先目标数量")
    preferred_groove_ratio: float = Field(description="优先目标占比")


@register_rule_feature
class Rule7Feature(BaseRuleFeature):
    """Rule7特征：主沟宽度/数量，已被融入"""
    num_main_grooves: int = Field(description="主沟数量")
    main_groove_widths: List[float] = Field(description="主沟宽度列表")
    main_groove_positions: List[int] = Field(description="主沟位置列表")
    all_widths_in_range: bool = Field(description="所有宽度是否在范围内")
    preferred_count_hit: bool = Field(description="优先数量是否命中")


@register_rule_score
class Rule7Score(BaseRuleScore):
    """Rule7（主沟宽度/数量）评分，已被融入"""
    pass


# ============================================================
# 第六部分：Rule8 横沟数量
# ============================================================

class Rule8Config(BaseRuleConfig):
    """Rule8：横沟数量约束"""
    description: str = "横沟数量约束"
    max_score: int = 4
    rule_type: RuleTypeEnum = RuleTypeEnum.SMALL_IMAGE
    groove_width_center: float = Field(gt=0, description="center横沟宽度(像素)")
    groove_width_side: float = Field(gt=0, description="side横沟宽度(像素)")


@register_rule_feature
class Rule8Feature(BaseRuleFeature):
    """Rule8特征：横沟数量"""
    num_transverse_grooves: int = Field(ge=0, description="横沟数量")


@register_rule_score
class Rule8Score(BaseRuleScore):
    """Rule8（横沟数量）评分"""
    pass


# ============================================================
# 第七部分：Rule9-10 横向钢片
# ============================================================

class Rule9Config(BaseRuleConfig):
    """Rule9：横向钢片数量约束"""
    description: str = "横向钢片数量约束"
    max_score: int = 4
    rule_type: RuleTypeEnum = RuleTypeEnum.SMALL_IMAGE
    transverse_sipe_width: float = Field(default=0.6, gt=0, description="横向钢片宽度(像素)")
    min_sipe_count_side: int = Field(default=0, ge=0, description="side区域钢片数量下限")
    max_sipe_count_side: int = Field(default=2, ge=0, description="side区域钢片数量上限")
    min_sipe_count_center: int = Field(default=0, ge=0, description="center区域钢片数量下限")
    max_sipe_count_center: int = Field(default=3, ge=0, description="center区域钢片数量上限")


class Rule10Config(BaseRuleConfig):
    """Rule10：横向钢片位置需均分两横沟之间花纹块（未实现）"""
    description: str = "横向钢片位置需均分两横沟之间花纹块"
    max_score: int = 0
    rule_type: RuleTypeEnum = RuleTypeEnum.SMALL_IMAGE
    transverse_sipe_width: float = Field(description="横向钢片宽度(像素)")
    position_tolerance_ratio: float = Field(description="允许偏离均分位置的比例阈值")
    min_adjacent_groove_count: int = Field(description="至少需要两条横沟才能判断均分")


@register_rule_feature
class Rule9Feature(BaseRuleFeature):
    """Rule9特征：横向钢片数量"""
    num_transverse_sipes_side: int = Field(ge=0, description="side区域横向钢片数量")
    num_transverse_sipes_center: int = Field(ge=0, description="center区域横向钢片数量")
    is_count_valid: bool = Field(description="数量是否合规")
    region: RegionEnum = Field(description="小图区域：center/side")


@register_rule_feature
class Rule10Feature(BaseRuleFeature):
    """Rule10特征：横向钢片位置，未实现"""
    sipe_positions: List[float] = Field(description="钢片位置列表")
    groove_positions: List[float] = Field(description="横沟位置列表")
    position_deviation_ratio: float = Field(description="位置偏差比例")
    is_evenly_distributed: bool = Field(description="是否均匀分布")


@register_rule_score
class Rule9Score(BaseRuleScore):
    """Rule9（横向钢片数量）评分"""
    pass


@register_rule_score
class Rule10Score(BaseRuleScore):
    """Rule10（横向钢片位置）评分，未实现"""
    pass


# ============================================================
# 第八部分：Rule11 纵向细沟&纵向钢片
# ============================================================

class Rule11Config(BaseRuleConfig):
    """Rule11：纵向钢片与纵向细沟数量约束"""
    description: str = "纵向钢片与纵向细沟数量约束"
    max_score: int = 4
    rule_type: RuleTypeEnum = RuleTypeEnum.SMALL_IMAGE
    groove_width: float = Field(description="纵向线条名义宽度(像素)")
    min_width_offset_px: int = Field(description="宽度下限偏移(像素)")
    edge_margin_ratio: float = Field(description="左右边缘排除比例")
    min_segment_length_ratio: float = Field(description="连续线段最小长度比例")
    max_angle_from_vertical: float = Field(description="主轴允许偏离竖直方向的最大角度(度)")
    max_count_center: int = Field(description="center允许的最大纵向线条数")
    max_count_side: int = Field(description="side允许的最大纵向线条数")


@register_rule_feature
class Rule11Feature(BaseRuleFeature):
    """Rule11特征：纵向细沟&纵向钢片数量"""
    num_longitudinal_grooves: int = Field(description="纵向线条数量")
    region: RegionEnum = Field(description="小图区域：center/side")


@register_rule_score
class Rule11Score(BaseRuleScore):
    """Rule11（纵向细沟&纵向钢片）评分"""
    pass


# ============================================================
# 第九部分：Rule12 RIB横向连续性
# ============================================================

class Rule12Config(BaseRuleConfig):
    """Rule12：两个RIB间横向钢片及横沟连续性占比是否满足要求"""
    description: str = "两个RIB间横向钢片及横沟连续性占比是否满足要求"
    max_score: int = 6
    rule_type: RuleTypeEnum = RuleTypeEnum.BIG_IMAGE
    continuity_ratio_upper: float = Field(description="连续性占比上界")
    continuity_ratio_lower: float = Field(description="连续性占比下界")
    continuity_mode_list: List[ContinuityModeName] = Field(description="连续性模式列表")


@register_rule_feature
class Rule12Feature(BaseRuleFeature):
    """Rule12特征：RIB横向连续性占比"""
    continuity_ratio: float = Field(description="连续性占比")


@register_rule_score
class Rule12Score(BaseRuleScore):
    """Rule12（RIB横向连续性）评分"""
    pass


# ============================================================
# 第十部分：Rule13 海陆比/陆地占比
# ============================================================

class Rule13Config(BaseRuleConfig):
    """Rule13：1个节距TDW范围内海陆比在28%-35%"""
    description: str = "1个节距TDW范围内海陆比在28%-35%"
    max_score: int = 2
    rule_type: RuleTypeEnum = RuleTypeEnum.BIG_IMAGE
    land_ratio_min: float = Field(description="合格陆地占比下限")
    land_ratio_max: float = Field(description="合格陆地占比上限")


@register_rule_feature
class Rule13Feature(BaseRuleFeature):
    """Rule13特征：海陆比（其实是陆地占比）"""
    land_ratio: float = Field(description="陆地占比")


@register_rule_score
class Rule13Score(BaseRuleScore):
    """Rule13（海陆比/陆地占比）评分"""
    pass


# ============================================================
# 第十一部分：Rule14 交点数量
# ============================================================

class Rule14Config(BaseRuleConfig):
    """Rule14：钢片&横沟与其他线条交点数量<=2"""
    description: str = "钢片&横沟与其他线条交点数量≤2"
    max_score: int = 2
    rule_type: RuleTypeEnum = RuleTypeEnum.SMALL_IMAGE
    max_intersections: int = Field(description="允许的最大交叉点数量")


@register_rule_feature
class Rule14Feature(BaseRuleFeature):
    """Rule14特征：交点数量"""
    num_intersections: int = Field(description="交叉点数量")


@register_rule_score
class Rule14Score(BaseRuleScore):
    """Rule14（交点数量）评分"""
    pass


# ============================================================
# 第十二部分：Rule15 花纹块面积比例（未实现）
# ============================================================

class Rule15Config(BaseRuleConfig):
    """Rule15：各节距中细沟&横沟分割出的花纹块面积比例（未实现）"""
    description: str = "各节距中细沟&横沟分割出的花纹块面积比例≤1:1.2"
    max_score: int = 0
    rule_type: RuleTypeEnum = RuleTypeEnum.SMALL_IMAGE
    max_block_area_ratio: float = Field(description="各花纹块面积比例上限：1.2")
    min_block_area_px: int = Field(description="过滤噪声用最小花纹块面积(像素)")
    rib_index_start: int = Field(description="RIB起始索引，默认RIB1")
    rib_index_end: int = Field(description="RIB结束索引，默认RIB5")


@register_rule_feature
class Rule15Feature(BaseRuleFeature):
    """Rule15特征：花纹块面积比例，未实现"""
    block_area_ratios: List[float] = Field(description="各花纹块面积比例列表")
    max_block_area_ratio: float = Field(description="最大面积比例")
    is_area_ratio_valid: bool = Field(description="面积比例是否合规")


@register_rule_score
class Rule15Score(BaseRuleScore):
    """Rule15（花纹块面积比例）评分，未实现"""
    pass


# ============================================================
# 第十三部分：Rule16-17 连续性拼接
# ============================================================

class Rule16Config(BaseRuleConfig):
    """Rule16：中心RIB上的横沟或横向钢片可任意组合连续性"""
    description: str = "中心RIB上的横沟或横向钢片可任意组合连续性"
    max_score: int = 4
    rule_type: RuleTypeEnum = RuleTypeEnum.BIG_IMAGE
    continuity_mode_list: List[ContinuityModeName] = Field(description="连续性模式列表")


class Rule17Config(BaseRuleConfig):
    """Rule17：边缘RIB上的横沟或横向钢片可任意组合连续性"""
    description: str = "边缘RIB上的横沟或横向钢片可任意组合连续性"
    max_score: int = 6
    rule_type: RuleTypeEnum = RuleTypeEnum.BIG_IMAGE
    continuity_mode_list: List[ContinuityModeName] = Field(description="连续性模式列表")


@register_rule_feature
class Rule16Feature(BaseRuleFeature):
    """Rule16特征：RIB2/3/4任意组合连续性"""
    is_continuous: bool = Field(description="是否连续")


@register_rule_feature
class Rule17Feature(BaseRuleFeature):
    """Rule17特征：边缘RIB任意组合连续性"""
    is_continuous: bool = Field(description="是否连续")


@register_rule_score
class Rule16Score(BaseRuleScore):
    """Rule16（RIB2/3/4任意组合连续性）评分"""
    pass


@register_rule_score
class Rule17Score(BaseRuleScore):
    """Rule17（RIB1/2与RIB4/5概率连续）评分"""
    pass

# ============================================================
# 第十四部分：Rule18 颜色灰度变化（未实现）
# ============================================================

class Rule18Config(BaseRuleConfig):
    """Rule18：颜色灰度变化用于表征沟的深浅（未实现）"""
    description: str = "颜色灰度变化用于表征沟的深浅"
    max_score: int = 0
    rule_type: RuleTypeEnum = RuleTypeEnum.BIG_IMAGE
    enable_gray_depth: bool = Field(description="是否启用灰度深度")
    min_gray_value: int = Field(description="最小灰度值")
    max_gray_value: int = Field(description="最大灰度值")
    depth_levels: int = Field(description="灰度深度级别数")


@register_rule_feature
class Rule18Feature(BaseRuleFeature):
    """Rule18特征：颜色灰度变化，未实现"""
    gray_value_min: int = Field(description="最小灰度值")
    gray_value_max: int = Field(description="最大灰度值")
    gray_depth_levels: int = Field(description="灰度深度级别数")
    has_gray_depth_variation: bool = Field(description="是否有灰度深度变化")


@register_rule_score
class Rule18Score(BaseRuleScore):
    """Rule18（颜色灰度变化）评分，未实现"""
    pass


# ============================================================
# 第十五部分：Rule19 装饰边框
# ============================================================

class Rule19Config(BaseRuleConfig):
    """Rule19：PDW与TDW之间边缘灰色区域可结合横沟或钢片做装饰性造型 - 纯流程类规则，无评分"""
    description: str = "PDW与TDW之间边缘灰色区域可结合横沟或钢片做装饰性造型"
    max_score: int = 0
    rule_type: RuleTypeEnum = RuleTypeEnum.BIG_IMAGE
    tire_design_width: int = Field(description="花纹有效宽度(像素)")
    decoration_border_alpha: float = Field(ge=0, le=1, description="透明度")
    decoration_gray_color: int = Field(description="灰色RGB值")


@register_rule_feature
class Rule19Feature(BaseRuleFeature):
    """Rule19特征：装饰边框，流程类，无可评分特征"""
    pass


@register_rule_score
class Rule19Score(BaseRuleScore):
    """Rule19（装饰边框）评分，纯流程类无评分"""
    pass


# ============================================================
# 第十六部分：Rule20 文生图（未实现）
# ============================================================

class Rule20Config(BaseRuleConfig):
    """Rule20：用户输入文字生成合理的花纹概念图片（未实现）"""
    description: str = "用户输入文字生成合理的花纹概念图片"
    max_score: int = 0
    rule_type: RuleTypeEnum = RuleTypeEnum.DEFAULT
    prompt: str = Field(description="提示词")
    negative_prompt: Optional[str] = Field(default=None, description="负面提示词")
    num_images: int = Field(description="生成图片数量")
    seed: Optional[int] = Field(default=None, description="随机种子")
    output_width: int = Field(description="输出宽度(像素)")
    output_height: int = Field(description="输出高度(像素)")


@register_rule_feature
class Rule20Feature(BaseRuleFeature):
    """Rule20特征：文生图，未实现"""
    prompt_used: str = Field(description="使用的提示词")
    generated_image_count: int = Field(description="生成图片数量")
    generation_success: bool = Field(description="生成是否成功")


@register_rule_score
class Rule20Score(BaseRuleScore):
    """Rule20（文生图）评分，未实现"""
    pass


# ============================================================
# 第十七部分：Rule21-22 已被融入
# ============================================================

class Rule21Config(BaseRuleConfig):
    """Rule21：针对用户提出的业务目标量化要求支持配置 - 现被configs与各RuleConfig吸收"""
    description: str = "针对用户提出的业务目标量化要求支持配置"
    max_score: int = 0
    configurable_rule_names: List[str] = Field(description="可配置规则名称列表")
    strict_validation: bool = Field(description="是否严格校验")
    allow_partial_override: bool = Field(description="是否允许部分覆盖")


class Rule22Config(BaseRuleConfig):
    """Rule22：能够根据需要输出指定清晰度的图片 - 现被rule6a纵图拼接resolution配置吸收"""
    description: str = "能够根据需要输出指定清晰度的图片"
    max_score: int = 0
    rule_type: RuleTypeEnum = RuleTypeEnum.DEFAULT
    target_width: int = Field(description="目标宽度(像素)")
    target_height: int = Field(description="目标高度(像素)")
    keep_aspect_ratio: bool = Field(description="是否保持宽高比")
    output_format: str = Field(description="输出格式")


@register_rule_feature
class Rule21Feature(BaseRuleFeature):
    """Rule21特征：参数可配置，已被融入"""
    configurable_rule_names: List[str] = Field(description="可配置规则名称列表")
    validated_config_count: int = Field(description="已校验配置数量")
    invalid_config_messages: List[str] = Field(description="无效配置消息列表")
    is_config_valid: bool = Field(description="配置是否有效")


@register_rule_feature
class Rule22Feature(BaseRuleFeature):
    """Rule22特征：图片分辨率，已被融入"""
    output_width: int = Field(description="输出宽度(像素)")
    output_height: int = Field(description="输出高度(像素)")
    resolution_matched: bool = Field(description="分辨率是否匹配")


@register_rule_score
class Rule21Score(BaseRuleScore):
    """Rule21（参数可配置）评分，已被融入"""
    pass


@register_rule_score
class Rule22Score(BaseRuleScore):
    """Rule22（图片分辨率）评分，已被融入"""
    pass


# ============================================================
# 第十八部分：Rule100-102 辅助 Item 模型
# ============================================================

class RibSizeItem(BaseModel):
    """单个 RIB 的尺寸配置"""
    rib_name: str = Field(description="RIB名称，如 rib1、rib2")
    num_pitchs: int = Field(ge=1, description="节距数量")
    rib_width: int = Field(ge=1, description="纵向图宽度(像素)")
    rib_height: int = Field(ge=1, description="纵向图高度(像素)")


class GrooveSizeItem(BaseModel):
    """单个主沟的尺寸配置"""
    groove_width: int = Field(ge=1, description="主沟宽度(像素)")
    groove_height: int = Field(ge=1, description="主沟高度(像素)")


class DecorationItem(BaseModel):
    """单个装饰的尺寸与透明度配置"""
    position: DecorationPositionEnum = Field(description="装饰位置：left / right")
    decoration_width: int = Field(ge=1, description="装饰宽度(像素)")
    decoration_height: int = Field(ge=1, description="装饰高度(像素)")
    decoration_opacity: int = Field(ge=0, le=255, description="装饰透明度(0-255)")


# ============================================================
# 第十九部分：Rule100-102 纯配置型规则
# ============================================================

class Rule100Config(BaseRuleConfig):
    """Rule100：RIB 节距/尺寸配置"""
    description: str = "RIB 节距与尺寸配置"
    rib_number: int = Field(ge=1, description="RIB 数量")
    rib_sizes: List[RibSizeItem] = Field(min_length=1, description="每个RIB的尺寸配置列表")
    rule_type: RuleTypeEnum = RuleTypeEnum.PROCESSING


class Rule101Config(BaseRuleConfig):
    """Rule101：主沟尺寸配置"""
    description: str = "主沟尺寸配置"
    groove_sizes: List[GrooveSizeItem] = Field(min_length=1, description="每个主沟的尺寸配置列表")
    rule_type: RuleTypeEnum = RuleTypeEnum.PROCESSING


class Rule102Config(BaseRuleConfig):
    """Rule102：装饰边框配置"""
    description: str = "装饰边框尺寸与透明度配置"
    decorations: List[DecorationItem] = Field(min_length=1, description="左右装饰配置列表")
    rule_type: RuleTypeEnum = RuleTypeEnum.PROCESSING


@register_rule_feature
class Rule100Feature(BaseRuleFeature):
    """Rule100 特征（纯配置型，无特征提取）"""
    pass


@register_rule_feature
class Rule101Feature(BaseRuleFeature):
    """Rule101 特征（纯配置型，无特征提取）"""
    pass


@register_rule_feature
class Rule102Feature(BaseRuleFeature):
    """Rule102 特征（纯配置型，无特征提取）"""
    pass


@register_rule_score
class Rule100Score(BaseRuleScore):
    """Rule100 评分（纯配置型，无评分）"""
    pass


@register_rule_score
class Rule101Score(BaseRuleScore):
    """Rule101 评分（纯配置型，无评分）"""
    pass


@register_rule_score
class Rule102Score(BaseRuleScore):
    """Rule102 评分（纯配置型，无评分）"""
    pass
