"""Identify tasks supported by the tabletop Omni model before applying actions."""
from __future__ import annotations

import re


MODEL_ID = "so100_smolvla_omni"


class UnsupportedInstruction(ValueError):
    pass


_CUP = re.compile(r"\b(?:cups?|mugs?|tumblers?)\b|杯|随行杯", re.I)
_BLOCK = re.compile(r"\b(?:cubes?|blocks?)\b|方块|积木|木块", re.I)
_COLOR = re.compile(
    r"\b(?:blue|orange|yellow|green|red|purple|pink|black|white|brown|gray|grey|silver|gold)\b"
    r"|蓝|藍|橙|橘|黄|黃|绿|綠|红|紅|紫|粉|黑|白|棕|灰|银|銀|金", re.I,
)
_BLUE = {"blue", "蓝", "藍"}
_ORANGE = {"orange", "橙", "橘"}
_CUP_VARIANTS = {
    "standard": re.compile(r"\bstandard\b|标准", re.I),
    "tall": re.compile(r"\btall\b|高杯|高款", re.I),
    "wide": re.compile(r"\bwide\b|宽杯|宽口", re.I),
    "imported/cup-web": re.compile(r"\b(?:ceramic|tumbler)\b|陶瓷|随行杯", re.I),
    "imported/lux3d-blue-live-5887561": re.compile(r"\bblue\b|蓝|藍", re.I),
}
_INTO_TRAY = re.compile(
    r"\b(?:in|into|inside|to|on|onto)\s+(?:(?:the|a|its)\s+)?tray\b"
    r"|(?:入|进|進|到|在|回)\s*(?:(?:这|那)个)?托盘", re.I,
)
_MOVE = re.compile(r"\b(?:put|place|move|carry|transfer|bring|set|pick)\b|放|搬|移|夹|夾|抓|拿", re.I)
_UNSUPPORTED = re.compile(
    r"\b(?:not|never|without|except|instead|remove|flip|throw|stack|turn over|out of|from\s+(?:the\s+)?tray)\b"
    r"|\b(?:don['’]t|do\s+not)\b|不要|别|別|搬出|移出|夹出|夾出|拿出|取出|倒出|翻转|翻轉|扔|堆叠", re.I,
)


def validate_target(instruction: str | None, *, cup_asset: str, dynamic_ids: set[str]) -> str:
    """Return the single object the request can be evaluated against."""
    text = instruction or ""
    if not text.strip():
        raise UnsupportedInstruction("请输入搬运指令。")
    cup, block = _CUP.search(text), _BLOCK.search(text)
    colors = {match.group().lower() for match in _COLOR.finditer(text)}
    allowed_colors = _ORANGE if block else _BLUE if cup_asset == "imported/lux3d-blue-live-5887561" else set()
    if colors - allowed_colors:
        raise UnsupportedInstruction("场景里没有该颜色的目标物体。")
    if bool(cup) == bool(block):
        raise UnsupportedInstruction("请指定一个目标物体，每次可搬运一个。")
    if block:
        if "companion_cube" not in dynamic_ids:
            raise UnsupportedInstruction("当前场景没有可搬运的橙色方块。")
        target = "companion_cube"
    else:
        if cup_asset not in _CUP_VARIANTS or any(
            asset != cup_asset and pattern.search(text) for asset, pattern in _CUP_VARIANTS.items()
        ):
            raise UnsupportedInstruction("当前场景或模型不支持指令里的杯型。")
        target = "cup"
    if _UNSUPPORTED.search(text) or not (_MOVE.search(text) and _INTO_TRAY.search(text)):
        raise UnsupportedInstruction("当前模型支持把目标放入托盘。请给出一个入托盘的指令。")
    return target
