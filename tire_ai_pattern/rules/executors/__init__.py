from __future__ import annotations

from importlib import import_module


_EXECUTOR_MODULES = {
    "Rule1Executor": "tire_ai_pattern.rules.executors.rule1",
    "Rule2Executor": "tire_ai_pattern.rules.executors.rule2",
    "Rule3Executor": "tire_ai_pattern.rules.executors.rule3",
    "Rule4Executor": "tire_ai_pattern.rules.executors.rule4",
    # "Rule5Executor": "tire_ai_pattern.rules.executors.rule5",  # 已注释
    "Rule6Executor": "tire_ai_pattern.rules.executors.rule6",
    "Rule6AExecutor": "tire_ai_pattern.rules.executors.rule6a",
    "Rule7Executor": "tire_ai_pattern.rules.executors.rule7",
    "Rule8Executor": "tire_ai_pattern.rules.executors.rule8",
    "Rule9Executor": "tire_ai_pattern.rules.executors.rule9",
    "Rule10Executor": "tire_ai_pattern.rules.executors.rule10",
    "Rule11Executor": "tire_ai_pattern.rules.executors.rule11",
    "Rule12Executor": "tire_ai_pattern.rules.executors.rule12",
    "Rule13Executor": "tire_ai_pattern.rules.executors.rule13",
    "Rule14Executor": "tire_ai_pattern.rules.executors.rule14",
    "Rule15Executor": "tire_ai_pattern.rules.executors.rule15",
    "Rule16Executor": "tire_ai_pattern.rules.executors.rule16",
    "Rule17Executor": "tire_ai_pattern.rules.executors.rule17",
    "Rule18Executor": "tire_ai_pattern.rules.executors.rule18",
    "Rule19Executor": "tire_ai_pattern.rules.executors.rule19",
    "Rule20Executor": "tire_ai_pattern.rules.executors.rule20",
    "Rule21Executor": "tire_ai_pattern.rules.executors.rule21",
    "Rule22Executor": "tire_ai_pattern.rules.executors.rule22",
    "Rule100Executor": "tire_ai_pattern.rules.executors.rule100",
    "Rule101Executor": "tire_ai_pattern.rules.executors.rule101",
    "Rule102Executor": "tire_ai_pattern.rules.executors.rule102",
}

__all__ = list(_EXECUTOR_MODULES)


def __getattr__(name: str):
    try:
        module_name = _EXECUTOR_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    return getattr(module, name)


def load_all_executors() -> None:
    for executor_name in __all__:
        getattr(import_module(_EXECUTOR_MODULES[executor_name]), executor_name)
