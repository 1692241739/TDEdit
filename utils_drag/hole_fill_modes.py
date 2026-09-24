from typing import Mapping


HOLE_FILL_MODE_CHOICES = (
    "sgf",
    "lama",
    "bnni_all",
)

DEFAULT_HOLE_FILL_MODE = "sgf"

DEFAULT_HOLE_FILL_MODE_BY_MODE: Mapping[str, str] = {
    "text": "sgf",
    "drag": "sgf",
    "joint": "sgf",
}


def default_hole_fill_mode_for_mode(mode: str) -> str:
    key = str(mode or "").strip().lower()
    return str(DEFAULT_HOLE_FILL_MODE_BY_MODE.get(key, DEFAULT_HOLE_FILL_MODE))


def normalize_hole_fill_mode(mode: str, *, default: str = DEFAULT_HOLE_FILL_MODE) -> str:
    value = str(mode or default).strip().lower()
    canonical_default = default_hole_fill_mode_for_mode(default)
    # 历史兼容：已移除模式统一回落到 sgf。
    if value in {"sgf+bnni", "sgf+context_bnni", "fastdrag_residual_bnni"}:
        return "sgf"
    if value in {"lama_sgf", "sgf_lama", "lama-fill", "lama_fill"}:
        return "lama"
    if value not in HOLE_FILL_MODE_CHOICES:
        return canonical_default if canonical_default in HOLE_FILL_MODE_CHOICES else DEFAULT_HOLE_FILL_MODE
    return value
