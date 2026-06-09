# ============================================================
# 枚举定义
#
# 所有枚举继承 str, Enum
# 每个枚举值必须有注释
# 枚举命名以 Enum 结尾
# ============================================================

from enum import Enum


class LevelEnum(str, Enum):
    """图像层级枚举"""
    SMALL = "small"  # 小图
    BIG = "big"      # 大图


class RegionEnum(str, Enum):
    """实际区域类型枚举"""
    SIDE = "side"      # 侧边区域
    CENTER = "center"  # 中心区域

class RuleTypeEnum(str, Enum):
    """规则类型枚举"""
    BIG_IMAGE = "big_image"      # 大图
    SMALL_IMAGE = "small_image"  # 小图
    DEFAULT = "default"  # 特例（留给rule20、22）
    PROCESSING = "processing"  # 图像过程

class SourceTypeEnum(str, Enum):
    """数据来源类型枚举"""
    ORIGINAL = "original"  # 原始输入
    INHERIT = "inherit"    # 继承自其他RIB
    CONCAT = "concat"      # 拼接生成


class ImageModeEnum(str, Enum):
    """图像颜色模式枚举"""
    GRAY = "GRAY"    # 灰度图
    RGB = "RGB"      # RGB三通道
    RGBA = "RGBA"    # RGBA四通道


class ImageFormatEnum(str, Enum):
    """图像格式枚举"""
    JPG = "jpg"  # JPEG格式
    PNG = "png"  # PNG格式
    BMP = "bmp"  # BMP格式
    RAW = "raw"  # 原始数据


class DecorationPositionEnum(str, Enum):
    """装饰位置枚举"""
    LEFT = "left"    # 左侧装饰
    RIGHT = "right"  # 右侧装饰


class StitchingSchemeName(str, Enum):
    """拼接方案名称枚举"""
    SYMMETRY_0 = "symmetry_0"          # 无对称
    SYMMETRY_1 = "symmetry_1"          # 中心旋转180°对称
    SYMMETRY_2 = "symmetry_2"          # 左右镜像对称
    SYMMETRY_3 = "symmetry_3"          # 中心线镜像对称可错位
    
    SYMMETRY_4 = "symmetry_4"          # 无对称
    SYMMETRY_5 = "symmetry_5"          # 中心旋转180°对称
    SYMMETRY_6 = "symmetry_6"          # 左右镜像对称
    SYMMETRY_7 = "symmetry_7"          # 中心线镜像对称可错位


    _CONCATENATE_0 = "_concatenate_0"  # 内部：两张图拼接（单下划线避免名称修饰）



class ContinuityModeName(str, Enum):
    """连续性模式名称枚举"""
    CONTINUITY_0 = "continuity_0"      # 无连续性
    CONTINUITY_1 = "continuity_1"      # RIB2-3中间全连续
    CONTINUITY_2 = "continuity_2"      # RIB3-4中间全连续
    CONTINUITY_3 = "continuity_3"      # 4rib，RIB2-3中间全连续

class RibOperation(str, Enum):
    """RIB原子操作枚举"""
    NONE = ""                               # 无操作
    FLIP_LR = "fliplr"                      # 左右对称
    FLIP = "flip"                           # 旋转180度
    LEFT_FLIP_LR = "left_fliplr"            # 左半左右对称覆盖右侧
    LEFT_FLIP = "left_flip"                 # 左半旋转180覆盖右侧
    OFFSET_VERTICAL_HALF = "offset_vertical_half"  # 纵向错位半高度
    RESIZE_HORIZONTAL_2X = "resize_horizontal_2x"      # 横向拉伸2倍
    LEFT = "left"                           # 截取左边
    RIGHT = "right"                         # 截取右边
    RESIZE_HORIZONTAL_1_5X = "resize_horizontal_1.5x"  # 横向拉伸1.5倍
    RESIZE_HORIZONTAL_3X = "resize_horizontal_3x"      # 横向拉伸3倍
    LEFT_2_3 = "left_2/3"                   # 截取左2/3
    RIGHT_2_3 = "right_2/3"                 # 截取右2/3
    LEFT_1_3 = "left_1/3"                   # 截取左1/3
    RIGHT_1_3 = "right_1/3"                 # 截取右1/3
    _RESIZE_AS_FIRST_RIB = "_resize_as_first_rib"  # 内部：图片大小向第一张图对齐（单下划线避免名称修饰）
