import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from tdedit_paths import DRAGBENCH_ROOT, PIEBENCH_ROOT, OUTPUT_ROOT, EVAL_SD_PATH, EVAL_SCRIPT_DIR

from utils_drag.hole_fill_modes import (
    HOLE_FILL_MODE_CHOICES,
    default_hole_fill_mode_for_mode,
    normalize_hole_fill_mode,
)


DEFAULT_DATA_ROOT_DRAGBENCH = DRAGBENCH_ROOT
DEFAULT_DATA_ROOT_PIEBENCH = PIEBENCH_ROOT
DEFAULT_EXP_ROOT = OUTPUT_ROOT
DEFAULT_DRAG_SD_PATH = EVAL_SD_PATH

# 模式默认参数（text 对齐 InfEdit 评估脚本 run_pie_bench.py）
DEFAULT_STRENGTH_BY_MODE = {
    "text": 1.0,
    "drag": 0.75,
    "joint": 0.7,
}
DEFAULT_GUIDANCE_T_BY_MODE = {
    "text": 2.3,
    "drag": 1.0,
    "joint": 2.0,
}
DEFAULT_CROSS_REPLACE_BY_MODE = {
    "text": 0.7,
    "drag": 0.7,
    "joint": 0.7,
}
DEFAULT_EXPANDED_SUBJECT_FILL_PX_BY_MODE = {
    "text": 6,
    "drag": 4,
    "joint": 6,
}
DEFAULT_ANNOTATION_VIEW_BY_MODE = {
    "text": "source",
    "drag": "modified",
    "joint": "modified",
}


@dataclass(frozen=True)
class Task:
    # 任务维度：数据集 + 编辑模式
    dataset: str  # piebench / dragbench
    mode: str  # text / drag / joint


@dataclass(frozen=True)
class DatasetConfig:
    # 每个数据集对应的路径配置
    dataset_bucket: str
    dataset_name: str
    data_root: Path
    mapping_file: Path


def mode_title(mode: str) -> str:
    return {"text": "Text", "drag": "Drag", "joint": "Joint"}[mode]


def normalize_dataset(raw: str) -> str:
    value = raw.strip().lower()
    if value in {"pie", "piebench", "pie-bench"}:
        return "piebench"
    if value in {"drag", "dragbench", "drag-bench"}:
        return "dragbench"
    raise ValueError(f"invalid dataset: {raw} (use pie|dragbench)")


def normalize_mode(raw: str) -> str:
    value = raw.strip().lower()
    if value in {"text", "drag", "joint"}:
        return value
    raise ValueError(f"invalid mode: {raw} (use text|drag|joint)")


def parse_gpu_list(raw: str) -> List[int]:
    tokens = raw.replace(",", " ").split()
    if not tokens:
        raise ValueError("GPU list cannot be empty")
    gpus: List[int] = []
    for tok in tokens:
        if not tok.isdigit():
            raise ValueError(f"invalid GPU id: {tok}")
        gpus.append(int(tok))
    return gpus


def parse_tasks(raw: str) -> List[Task]:
    """解析 tasks 字符串，例如: pie:joint,dragbench:drag"""
    tasks: List[Task] = []
    for chunk in raw.split(","):
        task = chunk.strip()
        if not task:
            continue
        if ":" not in task:
            raise ValueError(f"task must be dataset:mode, got: {task}")
        ds_raw, mode_raw = task.split(":", 1)
        tasks.append(Task(dataset=normalize_dataset(ds_raw), mode=normalize_mode(mode_raw)))
    if not tasks:
        raise ValueError("no tasks provided")
    return tasks


def resolve_skip_no_points(mode: str, option: str) -> bool:
    # text 模式天然无拖拽点，因此 auto 时不跳图；drag/joint 则跳过无点样本
    if option == "on":
        return True
    if option == "off":
        return False
    return mode != "text"


def resolve_mode_policy(mode: str, text_val: str, drag_val: str, joint_val: str) -> str:
    policy_map = {
        "text": text_val,
        "drag": drag_val,
        "joint": joint_val,
    }
    return policy_map[mode]


def resolve_mode_float_override(
    mode: str,
    text_val: float,
    drag_val: float,
    joint_val: float,
) -> float:
    value_map = {
        "text": text_val,
        "drag": drag_val,
        "joint": joint_val,
    }
    return value_map[mode]


def resolve_mode_int_override(
    mode: str,
    text_val: int,
    drag_val: int,
    joint_val: int,
) -> int:
    value_map = {
        "text": text_val,
        "drag": drag_val,
        "joint": joint_val,
    }
    return value_map[mode]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一评估入口：先生成，再评估。"
    )

    # 核心任务配置
    parser.add_argument("--tasks", default="pie:joint", help='Comma list, e.g. "pie:joint,dragbench:drag"')
    parser.add_argument("--gen-gpus", default="4 5 6", help='GPU list for generation, e.g. "4 5 6"')
    parser.add_argument("--eval-gpus", default=None, help='GPU list for evaluation, default same as --gen-gpus')
    parser.add_argument("--skip-generate", action="store_true", help="Skip generation stage")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation stage")
    parser.add_argument(
        "--existing-output-dir",
        default="",
        help="仅在 --skip-generate 时使用：指向已生成结果的实验目录（目录内应包含 results 子目录）",
    )
    parser.add_argument("--method", default="TDEdit")
    parser.add_argument("--exp-root", default=DEFAULT_EXP_ROOT)

    # 生成阶段参数（与 run_batch.py 对齐）
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--fixed-first-n", type=int, default=-1, help="固定取前 N 张，<=0 表示不启用")
    parser.add_argument("--random-sample-size", type=int, default=-1, help="随机抽样数量，<=0 表示不启用")
    parser.add_argument("--random-sample-seed", type=int, default=42, help="随机抽样种子")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-text", type=int, default=12, help="Text 模式 steps（InfEdit 评估默认 12）")
    parser.add_argument("--steps-drag", type=int, default=17, help="Drag 模式 steps")
    parser.add_argument("--steps-joint", type=int, default=None, help="Joint 模式 steps")
    parser.add_argument("--seed-text", type=int, default=42, help="Text 模式 seed")
    parser.add_argument("--seed-drag", type=int, default=42, help="Drag 模式 seed")
    parser.add_argument("--seed-joint", type=int, default=None, help="Joint 模式 seed")

    # 全局回退参数：当 mode 专属参数为空时才使用
    parser.add_argument("--strength", type=float, default=None, help="全局强度回退值")
    parser.add_argument("--guidance-s", type=float, default=1.0)
    parser.add_argument("--guidance-t", type=float, default=None, help="全局 guidance_t 回退值")
    parser.add_argument("--cross-replace", type=float, default=None, help="全局 cross_replace 回退值")

    # mode 专属参数（优先级高于全局）
    parser.add_argument("--strength-text", type=float, default=None, help="Text 模式 strength")
    parser.add_argument("--strength-drag", type=float, default=None, help="Drag 模式 strength")
    parser.add_argument("--strength-joint", type=float, default=None, help="Joint 模式 strength")

    parser.add_argument("--guidance-t-text", type=float, default=None, help="Text 模式 guidance_t")
    parser.add_argument("--guidance-t-drag", type=float, default=None, help="Drag 模式 guidance_t")
    parser.add_argument("--guidance-t-joint", type=float, default=None, help="Joint 模式 guidance_t")

    parser.add_argument("--cross-replace-text", type=float, default=None, help="Text 模式 cross_replace")
    parser.add_argument("--cross-replace-drag", type=float, default=None, help="Drag 模式 cross_replace")
    parser.add_argument("--cross-replace-joint", type=float, default=None, help="Joint 模式 cross_replace")

    parser.add_argument("--positive-prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--self-replace", type=float, default=0.7)
    parser.add_argument("--text-cross-replace-steps", type=float, default=-1.0, help=">=0 时覆盖 Text Branch 的 cross schedule")
    parser.add_argument("--text-self-replace-steps", type=float, default=-1.0, help=">=0 时覆盖 Text Branch 的 self schedule")
    parser.add_argument(
        "--drag-cross-replace-steps",
        type=float,
        default=0.0,
        help="Drag Branch 的 cross schedule；默认 0.0，即关闭 layout early mix",
    )
    parser.add_argument(
        "--drag-self-replace-steps",
        type=float,
        default=-1.0,
        help=">=0 时覆盖 Drag Branch 的 self schedule（即 GSAC 起点）",
    )
    parser.add_argument(
        "--ref-kv-injection",
        dest="ref_kv_injection",
        action="store_true",
        help="启用 target self-attn 上的 reference K/V 注入（默认开启）",
    )
    parser.add_argument(
        "--no-ref-kv-injection",
        dest="ref_kv_injection",
        action="store_false",
        help="关闭 target self-attn 上的 reference K/V 注入",
    )
    parser.add_argument(
        "--drag-target-q-layout-mix",
        dest="drag_target_q_layout_mix",
        action="store_true",
        help="启用 target Q 与 layout Q 的调度混合（LQM，默认开启）",
    )
    parser.add_argument(
        "--no-drag-target-q-layout-mix",
        dest="drag_target_q_layout_mix",
        action="store_false",
        help="关闭 target-layout Q 混合（LQM 消融）",
    )
    parser.add_argument(
        "--ref-target-denoise-mix",
        dest="ref_target_denoise_mix",
        action="store_true",
        help="启用后期 layout-side 与 reference 细节噪声混合，用于 layout->target DDCM（默认开启）",
    )
    parser.add_argument(
        "--no-ref-target-denoise-mix",
        dest="ref_target_denoise_mix",
        action="store_false",
        help="关闭后期 layout-side 与 reference 细节噪声混合",
    )
    parser.add_argument("--ref-target-denoise-mix-max", type=float, default=0.7)
    parser.add_argument("--ref-target-denoise-mix-start", type=float, default=0.9)
    parser.add_argument(
        "--joint-target-refine-mix",
        dest="joint_target_refine_mix",
        action="store_true",
        help="Joint 模式启用 drag-aware 的 reference->target 蒸馏（默认关闭）",
    )
    parser.add_argument(
        "--no-joint-target-refine-mix",
        dest="joint_target_refine_mix",
        action="store_false",
        help="Joint 模式关闭 drag-aware 的 reference->target 蒸馏",
    )
    parser.add_argument("--joint-target-refine-mix-start", type=float, default=0.30)
    parser.add_argument("--joint-target-refine-mix-max-out", type=float, default=0.45)
    parser.add_argument("--joint-target-refine-mix-max-in", type=float, default=0.10)
    parser.add_argument(
        "--drag-layout-latents",
        dest="drag_layout_latents",
        action="store_true",
        help="Drag/Joint 模式对 layout 分支的 noisy latents 应用拖拽（target 默认跟随 layout）",
    )
    parser.add_argument(
        "--no-drag-layout-latents",
        dest="drag_layout_latents",
        action="store_false",
        help="Drag/Joint 模式保持 layout 分支的 noisy latents 为未拖拽版本",
    )
    parser.add_argument(
        "--drag-target-latents",
        dest="drag_target_latents",
        action="store_true",
        help="Drag/Joint 模式对 target 分支的 noisy latents 应用拖拽（默认跟随 layout）",
    )
    parser.add_argument(
        "--no-drag-target-latents",
        dest="drag_target_latents",
        action="store_false",
        help="Drag/Joint 模式保持 target 分支的 noisy latents 为未拖拽版本",
    )
    parser.add_argument(
        "--drag-clean-latents",
        dest="drag_clean_latents",
        action="store_true",
        help="Drag/Joint 模式对 clean latents 应用拖拽，作为 DDCM 的 x0 锚点（默认开启）",
    )
    parser.add_argument(
        "--no-drag-clean-latents",
        dest="drag_clean_latents",
        action="store_false",
        help="Drag/Joint 模式保持 clean latents 为未拖拽版本",
    )
    parser.add_argument("--local-blend-thresh-e", type=float, default=0.3, help="非 Text 模式默认 local blend thresh_e")
    parser.add_argument("--local-blend-thresh-m", type=float, default=0.3, help="非 Text 模式默认 local blend thresh_m")
    parser.add_argument("--local-blend-thresh-e-text", type=float, default=0.55, help="Text 模式 local blend thresh_e")
    parser.add_argument("--local-blend-thresh-m-text", type=float, default=0.6, help="Text 模式 local blend thresh_m")
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--start-layer", type=int, default=10)
    parser.add_argument("--drag-type", default="2D-Non-Rigid")
    parser.add_argument("--influence-range", type=float, default=0.5)
    parser.add_argument(
        "--pointcloud-domain",
        choices=["auto", "latent", "image"],
        default="image",
        help="Point Cloud Domain（默认 image）",
    )
    parser.add_argument(
        "--attn-switch-mode",
        default="hard",
        help="注意力切换模式固定为 hard（保留参数仅为兼容旧命令）",
    )
    parser.add_argument(
        "--hole-fill-mode",
        type=str,
        choices=["auto", *HOLE_FILL_MODE_CHOICES],
        default="auto",
        help=(
            "补洞模式。canonical: "
            + "/".join(HOLE_FILL_MODE_CHOICES)
            + "；默认按 mode 自动选择：text=sgf, drag=sgf, joint=sgf"
        ),
    )
    parser.add_argument(
        "--use-expanded-subject-fill",
        dest="use_expanded_subject_fill",
        action="store_true",
        help="启用 mask 扩展补全（默认开启）",
    )
    parser.add_argument(
        "--no-use-expanded-subject-fill",
        dest="use_expanded_subject_fill",
        action="store_false",
        help="关闭 mask 扩展补全",
    )
    parser.add_argument(
        "--expanded-subject-fill-px",
        type=int,
        default=None,
        help="单次运行时 mask 扩展像素（默认按 mode 自动选择：text=6, drag=4）",
    )
    parser.set_defaults(use_expanded_subject_fill=True)
    parser.add_argument(
        "--use-drag-guided-prefill",
        dest="use_drag_guided_prefill",
        action="store_true",
        help="启用拖拽方向背景预填充（默认开启）",
    )
    parser.add_argument(
        "--no-use-drag-guided-prefill",
        dest="use_drag_guided_prefill",
        action="store_false",
        help="关闭拖拽方向背景预填充",
    )
    parser.set_defaults(use_drag_guided_prefill=True)
    parser.set_defaults(ref_kv_injection=True)
    parser.set_defaults(drag_target_q_layout_mix=True)
    parser.set_defaults(ref_target_denoise_mix=True)
    parser.set_defaults(joint_target_refine_mix=False)
    parser.set_defaults(drag_layout_latents=False)
    parser.set_defaults(drag_target_latents=None)
    parser.set_defaults(drag_clean_latents=True)
    parser.add_argument(
        "--skip-no-points",
        choices=["auto", "on", "off"],
        default="auto",
        help='auto: text=off, drag/joint=on',
    )
    parser.add_argument(
        "--save-reference-as-result",
        action="store_true",
        help="兼容旧参数：等价于 --result-branch reference",
    )
    parser.add_argument(
        "--result-branch",
        choices=["auto", "target", "reference", "layout", "mutual", "source"],
        default="auto",
        help="保存到 results 的最终分支；auto 会按 topology 自动选择",
    )
    parser.add_argument(
        "--annotation-view",
        choices=["auto", "effective", "source", "modified", "user_study"],
        default="auto",
        help="读取标注视图：默认按 mode 自动选择；text 固定 source，drag 默认 modified",
    )

    # JSON 拖拽参数读取开关：全局 + mode 专属
    parser.add_argument(
        "--use-saved-drag-params",
        action="store_true",
        help="全局开关：启用后，从 JSON 读取 drag_type/influence_range（仅 drag/joint 生效）",
    )
    parser.add_argument(
        "--use-saved-drag-params-drag",
        choices=["inherit", "on", "off"],
        default="on",
        help="Drag 模式是否读取 JSON 拖拽参数：inherit/on/off",
    )
    parser.add_argument(
        "--use-saved-drag-params-joint",
        choices=["inherit", "on", "off"],
        default="on",
        help="Joint 模式是否读取 JSON 拖拽参数：inherit/on/off",
    )
    parser.add_argument(
        "--saved-drag-type",
        choices=["inherit", "on", "off"],
        default="inherit",
        help="是否从 JSON 读取 drag_type；inherit 时沿用 mode 专属 saved-drag-params 策略。",
    )
    parser.add_argument(
        "--saved-influence-range",
        choices=["inherit", "on", "off"],
        default="inherit",
        help="是否从 JSON 读取 influence_range；inherit 时沿用 mode 专属 saved-drag-params 策略。",
    )

    parser.add_argument(
        "--denoise-text",
        choices=["inherit", "on", "off"],
        default="off",
        help="Text 模式 denoise 策略：inherit/on/off",
    )
    parser.add_argument(
        "--denoise-drag",
        choices=["inherit", "on", "off"],
        default="on",
        help="Drag 模式 denoise 策略：inherit/on/off",
    )
    parser.add_argument(
        "--denoise-joint",
        choices=["inherit", "on", "off"],
        default="inherit",
        help="Joint 模式 denoise 策略：inherit/on/off",
    )

    parser.set_defaults(denoise=True, low_randomness=False, visualize_process=False, visualize_drag=False)
    parser.add_argument("--denoise", dest="denoise", action="store_true")
    parser.add_argument("--no-denoise", dest="denoise", action="store_false")
    parser.add_argument("--low-randomness", dest="low_randomness", action="store_true")
    parser.add_argument("--visualize-process", dest="visualize_process", action="store_true")
    parser.add_argument("--visualize-drag", dest="visualize_drag", action="store_true")

    # 数据与评估路径
    parser.add_argument("--data-root-pie", default=DEFAULT_DATA_ROOT_PIEBENCH)
    parser.add_argument("--mapping-pie", default=None)
    parser.add_argument("--data-root-drag", default=DEFAULT_DATA_ROOT_DRAGBENCH)
    parser.add_argument("--mapping-drag", default=None)
    parser.add_argument("--pie-metrics-json", default=None)
    parser.add_argument("--drag-metrics-json", default=None)
    parser.add_argument("--drag-sd-path", default=DEFAULT_DRAG_SD_PATH)
    parser.add_argument("--eval-script-dir", default=EVAL_SCRIPT_DIR,
                        help="Directory containing optional eval_pie.py and eval_drag.py evaluators")
    parser.add_argument(
        "--metrics-as-percent",
        action="store_true",
        help="评估展示系数开关：Distance×1e3, LPIPS×1e3, MSE×1e4, SSIM×1e2",
    )

    # 调试工具
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    return parser


def command_to_text(cmd: List[str]) -> str:
    return shlex.join(cmd)


def run_command(cmd: List[str], dry_run: bool) -> None:
    # dry-run 仅打印不执行，方便先核对参数
    print(f"[run_eval] $ {command_to_text(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def resolve_dataset_config(task: Task, args: argparse.Namespace) -> DatasetConfig:
    # 根据任务中的 dataset 解析路径
    if task.dataset == "piebench":
        data_root = Path(args.data_root_pie)
        mapping_file = Path(args.mapping_pie) if args.mapping_pie else data_root / "pie_bench.json"
        return DatasetConfig(
            dataset_bucket="2_PIE-Bench",
            dataset_name="PIE-Bench",
            data_root=data_root,
            mapping_file=mapping_file,
        )
    data_root = Path(args.data_root_drag)
    mapping_file = Path(args.mapping_drag) if args.mapping_drag else data_root / "dragbench.json"
    return DatasetConfig(
        dataset_bucket="1_Drag-Bench",
        dataset_name="Drag-Bench",
        data_root=data_root,
        mapping_file=mapping_file,
    )


def resolve_mode_defaults(mode: str, args: argparse.Namespace) -> Tuple[float, float, float, bool, int, int, bool]:
    """
    参数优先级：
    1) mode 专属参数
    2) 全局参数
    3) 代码默认值
    """
    strength = resolve_mode_float_override(
        mode,
        args.strength_text,
        args.strength_drag,
        args.strength_joint,
    )
    if strength is None:
        strength = args.strength if args.strength is not None else DEFAULT_STRENGTH_BY_MODE[mode]

    guidance_t = resolve_mode_float_override(
        mode,
        args.guidance_t_text,
        args.guidance_t_drag,
        args.guidance_t_joint,
    )
    if guidance_t is None:
        guidance_t = args.guidance_t if args.guidance_t is not None else DEFAULT_GUIDANCE_T_BY_MODE[mode]

    cross_replace = resolve_mode_float_override(
        mode,
        args.cross_replace_text,
        args.cross_replace_drag,
        args.cross_replace_joint,
    )
    if cross_replace is None:
        cross_replace = args.cross_replace if args.cross_replace is not None else DEFAULT_CROSS_REPLACE_BY_MODE[mode]

    steps = resolve_mode_int_override(
        mode,
        args.steps_text,
        args.steps_drag,
        args.steps_joint,
    )
    if steps is None:
        steps = args.steps

    seed = resolve_mode_int_override(
        mode,
        args.seed_text,
        args.seed_drag,
        args.seed_joint,
    )
    if seed is None:
        seed = args.seed

    denoise_policy = resolve_mode_policy(
        mode,
        args.denoise_text,
        args.denoise_drag,
        args.denoise_joint,
    )
    if denoise_policy == "on":
        denoise = True
    elif denoise_policy == "off":
        denoise = False
    else:
        denoise = bool(args.denoise)

    skip_no_points = resolve_skip_no_points(mode, args.skip_no_points)
    return (
        float(strength),
        float(guidance_t),
        float(cross_replace),
        skip_no_points,
        int(steps),
        int(seed),
        bool(denoise),
    )


def resolve_mode_use_saved_drag_params(mode: str, args: argparse.Namespace) -> bool:
    """按 mode 决定是否读取 JSON 的 drag_type/influence_range。"""
    # text 模式与拖拽参数无关，始终关闭
    if mode == "text":
        return False

    policy = (
        args.use_saved_drag_params_drag
        if mode == "drag"
        else args.use_saved_drag_params_joint
    )
    if policy == "on":
        return True
    if policy == "off":
        return False
    return bool(args.use_saved_drag_params)


def resolve_saved_field_policy(policy: str, legacy_value: bool) -> bool:
    if policy == "on":
        return True
    if policy == "off":
        return False
    return bool(legacy_value)


def resolve_annotation_view_for_mode(mode: str, args: argparse.Namespace) -> str:
    # 与当前最优配置对齐：Text 固定使用 source；其余模式默认 modified，可显式覆盖。
    if mode == "text":
        return "source"
    if args.annotation_view in {"", "auto", None}:
        return DEFAULT_ANNOTATION_VIEW_BY_MODE[mode]
    return str(args.annotation_view)


def resolve_hole_fill_mode_for_mode(mode: str, args: argparse.Namespace) -> str:
    raw_value = args.hole_fill_mode
    if raw_value in {"", "auto", None}:
        raw_value = default_hole_fill_mode_for_mode(mode)
    return normalize_hole_fill_mode(raw_value)


def resolve_expanded_subject_fill_px_for_mode(mode: str, args: argparse.Namespace) -> int:
    px = args.expanded_subject_fill_px
    if px is None:
        px = DEFAULT_EXPANDED_SUBJECT_FILL_PX_BY_MODE[mode]
    return max(0, min(48, int(px)))


def resolve_result_branch_for_mode(mode: str, args: argparse.Namespace) -> str:
    raw = str(getattr(args, "result_branch", "auto") or "auto").strip().lower()
    if raw == "auto":
        if bool(getattr(args, "save_reference_as_result", False)):
            return "reference"
        if str(mode or "").strip().lower() == "text":
            return "reference"
        return "target"
    return raw


def build_generate_cmd(
    project_root: Path,
    dataset_cfg: DatasetConfig,
    mode: str,
    out_dir: Path,
    gen_gpus: List[int],
    strength: float,
    guidance_t: float,
    cross_replace: float,
    steps: int,
    seed: int,
    denoise: bool,
    skip_no_points: bool,
    use_saved_drag_params: bool,
    use_saved_drag_type: bool,
    use_saved_influence_range: bool,
    annotation_view: str,
    result_branch: str,
    save_reference_as_result: bool,
    hole_fill_mode: str,
    use_expanded_subject_fill: bool,
    expanded_subject_fill_px: int,
    use_drag_guided_prefill: bool,
    args: argparse.Namespace,
) -> List[str]:
    # 这里只负责组装 run_batch 命令，不做推理逻辑
    pointcloud_domain = str(args.pointcloud_domain)
    if pointcloud_domain == "auto":
        dataset_hint = f"{dataset_cfg.dataset_name}".lower()
        pointcloud_domain = "image" if "drag" in dataset_hint else "latent"
    local_blend_thresh_e = args.local_blend_thresh_e_text if mode == "text" else args.local_blend_thresh_e
    local_blend_thresh_m = args.local_blend_thresh_m_text if mode == "text" else args.local_blend_thresh_m
    force_text_only = (mode == "text")
    enable_ref_target_denoise_mix = bool(args.ref_target_denoise_mix) and (not force_text_only)
    enable_ref_kv_injection = bool(args.ref_kv_injection) and (not force_text_only)
    enable_drag_target_q_layout_mix = bool(args.drag_target_q_layout_mix) and (not force_text_only)
    enable_drag_layout_latents = bool(args.drag_layout_latents) and (not force_text_only)
    if force_text_only:
        enable_drag_target_latents = False
    else:
        enable_drag_target_latents = (
            enable_drag_layout_latents
            if args.drag_target_latents is None
            else bool(args.drag_target_latents)
        )
    enable_drag_clean_latents = bool(args.drag_clean_latents) and (not force_text_only)

    cmd = [
        sys.executable,
        str(project_root / "run_batch.py"),
        "--device",
        *[str(g) for g in gen_gpus],
        "--data_root",
        str(dataset_cfg.data_root),
        "--mapping_file",
        str(dataset_cfg.mapping_file),
        "--output_dir",
        str(out_dir),
        "--mode",
        mode,
        "--start_idx",
        str(args.start_idx),
        "--end_idx",
        str(args.end_idx),
        "--fixed_first_n",
        str(int(args.fixed_first_n)),
        "--random_sample_size",
        str(int(args.random_sample_size)),
        "--random_sample_seed",
        str(int(args.random_sample_seed)),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--strength",
        str(strength),
        "--guidance_s",
        str(args.guidance_s),
        "--guidance_t",
        str(guidance_t),
        "--positive_prompt",
        args.positive_prompt,
        "--negative_prompt",
        args.negative_prompt,
        "--start_step",
        str(args.start_step),
        "--start_layer",
        str(args.start_layer),
        "--cross_replace_steps",
        str(cross_replace),
        "--self_replace_steps",
        str(args.self_replace),
        "--ref_target_denoise_mix_max",
        str(float(args.ref_target_denoise_mix_max)),
        "--ref_target_denoise_mix_start",
        str(float(args.ref_target_denoise_mix_start)),
        "--local_blend_thresh_e",
        str(local_blend_thresh_e),
        "--local_blend_thresh_m",
        str(local_blend_thresh_m),
        "--drag_type",
        args.drag_type,
        "--influence_range",
        str(args.influence_range),
        "--pointcloud_domain",
        pointcloud_domain,
        "--attn_switch_mode",
        args.attn_switch_mode,
        "--hole_fill_mode",
        hole_fill_mode,
        "--expanded_subject_fill_px",
        str(int(expanded_subject_fill_px)),
        "--annotation_view",
        annotation_view,
        "--result_branch",
        str(result_branch),
        "--joint_target_refine_mix_start",
        str(float(args.joint_target_refine_mix_start)),
        "--joint_target_refine_mix_max_out",
        str(float(args.joint_target_refine_mix_max_out)),
        "--joint_target_refine_mix_max_in",
        str(float(args.joint_target_refine_mix_max_in)),
    ]
    if use_expanded_subject_fill:
        cmd.append("--use_expanded_subject_fill")
    else:
        cmd.append("--no-use_expanded_subject_fill")
    if use_drag_guided_prefill:
        cmd.append("--use_drag_guided_prefill")
    else:
        cmd.append("--no-use_drag_guided_prefill")
    if denoise:
        cmd.append("--denoise")
    else:
        cmd.append("--no-denoise")
    if enable_ref_target_denoise_mix:
        cmd.append("--ref_target_denoise_mix")
    else:
        cmd.append("--no-ref_target_denoise_mix")
    if enable_ref_kv_injection:
        cmd.append("--ref_kv_injection")
    else:
        cmd.append("--no-ref_kv_injection")
    if enable_drag_target_q_layout_mix:
        cmd.append("--drag_target_q_layout_mix")
    else:
        cmd.append("--no-drag_target_q_layout_mix")
    if enable_drag_layout_latents:
        cmd.append("--drag_layout_latents")
    else:
        cmd.append("--no-drag_layout_latents")
    if enable_drag_target_latents:
        cmd.append("--drag_target_latents")
    else:
        cmd.append("--no-drag_target_latents")
    if enable_drag_clean_latents:
        cmd.append("--drag_clean_latents")
    else:
        cmd.append("--no-drag_clean_latents")
    if float(args.drag_cross_replace_steps) >= 0.0:
        cmd.extend(["--drag_cross_replace_steps", str(float(args.drag_cross_replace_steps))])
    if float(args.drag_self_replace_steps) >= 0.0:
        cmd.extend(["--drag_self_replace_steps", str(float(args.drag_self_replace_steps))])
    if float(args.text_cross_replace_steps) >= 0.0:
        cmd.extend(["--text_cross_replace_steps", str(float(args.text_cross_replace_steps))])
    if float(args.text_self_replace_steps) >= 0.0:
        cmd.extend(["--text_self_replace_steps", str(float(args.text_self_replace_steps))])
    if args.low_randomness:
        cmd.append("--low_randomness")
    if args.visualize_process:
        cmd.append("--visualize_process")
    if args.visualize_drag:
        cmd.append("--visualize_drag")
    if skip_no_points:
        cmd.append("--skip_no_points")
    if use_saved_drag_params:
        cmd.append("--use_saved_drag_params")
    if use_saved_drag_type:
        cmd.append("--use_saved_drag_type")
    else:
        cmd.append("--no-use_saved_drag_type")
    if use_saved_influence_range:
        cmd.append("--use_saved_influence_range")
    else:
        cmd.append("--no-use_saved_influence_range")
    if save_reference_as_result:
        cmd.append("--save_reference_as_result")
    if args.joint_target_refine_mix:
        cmd.append("--joint_target_refine_mix")
    else:
        cmd.append("--no-joint_target_refine_mix")
    return cmd


def build_eval_cmds(
    project_root: Path,
    dataset_cfg: DatasetConfig,
    task: Task,
    out_dir: Path,
    eval_gpus: List[int],
    pie_metrics_json: Path,
    drag_metrics_json: Path,
    drag_sd_path: str,
    annotation_view: str,
    seed: int,
    metrics_as_percent: bool,
    eval_script_dir: Path = None,
) -> List[List[str]]:
    # 评估分发按编辑模式决定，而不是按数据集名决定：
    # text  -> PIE 指标
    # drag  -> Drag 指标
    # joint -> 两套都跑（PIE + Drag）
    mode_cap = mode_title(task.mode)
    eval_script_dir = Path(eval_script_dir or EVAL_SCRIPT_DIR)
    eval_cmds: List[List[str]] = []

    if task.mode in {"text", "joint"}:
        pie_cmd = [
            sys.executable,
            str(eval_script_dir / "eval_pie.py"),
            "--result_dirs",
            str(out_dir),
            "--data_root",
            str(dataset_cfg.data_root),
            "--mapping_file",
            str(dataset_cfg.mapping_file),
            "--mode",
            mode_cap,
            "--output_json",
            str(pie_metrics_json),
            "--annotation_view",
            annotation_view,
            "--seed",
            str(seed),
            "--device",
            *[str(g) for g in eval_gpus],
        ]
        if metrics_as_percent:
            pie_cmd.append("--metrics-as-percent")
        eval_cmds.append(pie_cmd)

    if task.mode in {"drag", "joint"}:
        drag_cmd = [
            sys.executable,
            str(eval_script_dir / "eval_drag.py"),
            "--result_dirs",
            str(out_dir),
            "--data_root",
            str(dataset_cfg.data_root),
            "--mapping_file",
            str(dataset_cfg.mapping_file),
            "--mode",
            mode_cap,
            "--output_json",
            str(drag_metrics_json),
            "--sd_path",
            drag_sd_path,
            "--annotation_view",
            annotation_view,
            "--seed",
            str(seed),
            "--device",
            *[str(g) for g in eval_gpus],
        ]
        if metrics_as_percent:
            drag_cmd.append("--metrics-as-percent")
        eval_cmds.append(drag_cmd)

    return eval_cmds


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # 软切换逻辑已移除，统一强制 hard。
    args.attn_switch_mode = "hard"
    project_root = Path(__file__).resolve().parent

    try:
        tasks = parse_tasks(args.tasks)
        gen_gpus = parse_gpu_list(args.gen_gpus)
        eval_gpus = parse_gpu_list(args.eval_gpus) if args.eval_gpus else gen_gpus
    except ValueError as exc:
        parser.error(str(exc))
        return

    if not args.skip_eval:
        required_evaluators = set()
        for task in tasks:
            if task.mode in {"text", "joint"}:
                required_evaluators.add("eval_pie.py")
            if task.mode in {"drag", "joint"}:
                required_evaluators.add("eval_drag.py")
        for name in sorted(required_evaluators):
            evaluator = Path(args.eval_script_dir).expanduser() / name
            if not evaluator.is_file():
                parser.error(f"Evaluator not found: {evaluator}. Set --eval-script-dir "
                             "or use --skip-eval for generation only.")

    exp_root = Path(args.exp_root)
    exp_root.mkdir(parents=True, exist_ok=True)
    default_pie_metrics_json = exp_root / "2_PIE-Bench" / args.method / "metrics_piebench.json"
    default_drag_metrics_json = exp_root / "1_Drag-Bench" / args.method / "metrics_dragbench.json"
    pie_metrics_json = Path(args.pie_metrics_json) if args.pie_metrics_json else default_pie_metrics_json
    drag_metrics_json = Path(args.drag_metrics_json) if args.drag_metrics_json else default_drag_metrics_json
    pie_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    drag_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    existing_output_dir = Path(args.existing_output_dir).expanduser().resolve() if args.existing_output_dir else None

    if args.skip_generate:
        if existing_output_dir is None:
            parser.error("--skip-generate requires --existing-output-dir")
        if len(tasks) != 1:
            parser.error("--skip-generate currently supports one task at a time with --existing-output-dir")
        if not existing_output_dir.is_dir():
            raise NotADirectoryError(f"existing output dir not found: {existing_output_dir}")

    print(f"[run_eval] Tasks: {args.tasks}")
    print(f"[run_eval] Gen GPUs: {gen_gpus}")
    print(f"[run_eval] Eval GPUs: {eval_gpus}")
    print(f"[run_eval] Skip Generate: {args.skip_generate} | Skip Eval: {args.skip_eval}")
    print(f"[run_eval] Dry Run: {args.dry_run}")
    print(f"[run_eval] Annotation View (global fallback): {args.annotation_view}")
    print(f"[run_eval] Use Saved Drag Params(Global): {args.use_saved_drag_params}")
    print(f"[run_eval] Metrics Display Scaling: {args.metrics_as_percent}")
    print(
        f"[run_eval] Expanded Subject Fill: enabled={bool(args.use_expanded_subject_fill)}, "
        f"px={'auto' if args.expanded_subject_fill_px is None else int(args.expanded_subject_fill_px)}"
    )
    print(f"[run_eval] Drag Guided Prefill: enabled={bool(args.use_drag_guided_prefill)}")

    for task in tasks:
        dataset_cfg = resolve_dataset_config(task, args)
        if not dataset_cfg.mapping_file.is_file():
            raise FileNotFoundError(f"mapping file not found: {dataset_cfg.mapping_file}")
        if not dataset_cfg.data_root.is_dir():
            raise NotADirectoryError(f"data root not found: {dataset_cfg.data_root}")

        (
            strength,
            guidance_t,
            cross_replace,
            skip_no_points,
            steps,
            seed,
            denoise,
        ) = resolve_mode_defaults(task.mode, args)
        task_annotation_view = resolve_annotation_view_for_mode(task.mode, args)
        use_saved_drag_params = resolve_mode_use_saved_drag_params(task.mode, args)
        use_saved_drag_type = resolve_saved_field_policy(args.saved_drag_type, use_saved_drag_params)
        use_saved_influence_range = resolve_saved_field_policy(args.saved_influence_range, use_saved_drag_params)
        hole_fill_mode = resolve_hole_fill_mode_for_mode(task.mode, args)
        result_branch = resolve_result_branch_for_mode(task.mode, args)
        expanded_px_values = [resolve_expanded_subject_fill_px_for_mode(task.mode, args)]

        mode_cap = mode_title(task.mode)
        task_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for expanded_px in expanded_px_values:
            if args.skip_generate:
                out_dir = existing_output_dir
            else:
                run_suffix = f"{dataset_cfg.dataset_name}-{task_ts}-maskpx{expanded_px:02d}"
                out_dir = exp_root / dataset_cfg.dataset_bucket / args.method / mode_cap / run_suffix
                out_dir.mkdir(parents=True, exist_ok=True)

            print("=" * 70)
            print(f"[run_eval] Task: {task.dataset}:{task.mode} | mask_expand_px={expanded_px}")
            print(f"[run_eval] Output: {out_dir}")
            print(f"[run_eval] data_root: {dataset_cfg.data_root}")
            print(f"[run_eval] mapping: {dataset_cfg.mapping_file}")
            print(
                f"[run_eval] mode config => strength={strength}, guidance_t={guidance_t}, "
                f"cross_replace={cross_replace}, steps={steps}, seed={seed}, denoise={denoise}, "
                f"skip_no_points={skip_no_points}, "
                f"annotation_view={task_annotation_view}, "
                f"result_branch={result_branch}, "
                f"save_reference_as_result={bool(args.save_reference_as_result)}, "
                f"use_saved_drag_params={use_saved_drag_params}, "
                f"use_saved_drag_type={use_saved_drag_type}, "
                f"use_saved_influence_range={use_saved_influence_range}, "
                f"attn_switch_mode={args.attn_switch_mode}, "
                f"ref_target_denoise_mix={bool(args.ref_target_denoise_mix)}, "
                f"ref_target_denoise_mix_max={float(args.ref_target_denoise_mix_max):.2f}, "
                f"ref_target_denoise_mix_start={float(args.ref_target_denoise_mix_start):.2f}, "
                f"ref_kv_injection={bool(args.ref_kv_injection)}, "
                f"drag_target_q_layout_mix={bool(args.drag_target_q_layout_mix)}, "
                f"text_cross_replace_steps={float(args.text_cross_replace_steps):.2f}, "
                f"text_self_replace_steps={float(args.text_self_replace_steps):.2f}, "
                f"drag_cross_replace_steps={float(args.drag_cross_replace_steps):.2f}, "
                f"drag_self_replace_steps={float(args.drag_self_replace_steps):.2f}, "
                f"joint_target_refine_mix={bool(args.joint_target_refine_mix)}, "
                f"joint_target_refine_mix_start={float(args.joint_target_refine_mix_start):.2f}, "
                f"joint_target_refine_mix_max_out={float(args.joint_target_refine_mix_max_out):.2f}, "
                f"joint_target_refine_mix_max_in={float(args.joint_target_refine_mix_max_in):.2f}, "
                f"drag_layout_latents={bool(args.drag_layout_latents)}, "
                f"drag_target_latents={'follow_layout' if args.drag_target_latents is None else bool(args.drag_target_latents)}, "
                f"drag_clean_latents={bool(args.drag_clean_latents)}, "
                f"hole_fill_mode={hole_fill_mode}, "
                f"use_expanded_subject_fill={bool(args.use_expanded_subject_fill)}, "
                f"use_drag_guided_prefill={bool(args.use_drag_guided_prefill)}"
            )
            print("=" * 70)

            if not args.skip_generate:
                gen_cmd = build_generate_cmd(
                    project_root=project_root,
                    dataset_cfg=dataset_cfg,
                    mode=task.mode,
                    out_dir=out_dir,
                    gen_gpus=gen_gpus,
                    strength=strength,
                    guidance_t=guidance_t,
                    cross_replace=cross_replace,
                    steps=steps,
                    seed=seed,
                    denoise=denoise,
                    skip_no_points=skip_no_points,
                    use_saved_drag_params=use_saved_drag_params,
                    use_saved_drag_type=use_saved_drag_type,
                    use_saved_influence_range=use_saved_influence_range,
                    annotation_view=task_annotation_view,
                    result_branch=result_branch,
                    save_reference_as_result=bool(args.save_reference_as_result),
                    hole_fill_mode=hole_fill_mode,
                    use_expanded_subject_fill=bool(args.use_expanded_subject_fill),
                    expanded_subject_fill_px=int(expanded_px),
                    use_drag_guided_prefill=bool(args.use_drag_guided_prefill),
                    args=args,
                )
                print("[run_eval] Generating images...")
                run_command(gen_cmd, args.dry_run)
            else:
                print("[run_eval] Skip generation.")

            if not args.skip_eval:
                results_dir = out_dir / "results"
                if not results_dir.is_dir():
                    print(f"[run_eval] results dir missing: {results_dir}")
                    print(f"[run_eval] evaluation skipped for task {task.dataset}:{task.mode} (px={expanded_px})")
                    continue
                eval_cmds = build_eval_cmds(
                    project_root=project_root,
                    dataset_cfg=dataset_cfg,
                    task=task,
                    out_dir=out_dir,
                    eval_gpus=eval_gpus,
                    pie_metrics_json=pie_metrics_json,
                    drag_metrics_json=drag_metrics_json,
                    drag_sd_path=args.drag_sd_path,
                    annotation_view=task_annotation_view,
                    seed=seed,
                    metrics_as_percent=args.metrics_as_percent,
                    eval_script_dir=Path(args.eval_script_dir).expanduser(),
                )
                if not eval_cmds:
                    print(f"[run_eval] No evaluator mapped for mode={task.mode}, skip.")
                for eval_cmd in eval_cmds:
                    eval_script = Path(eval_cmd[1]).name
                    print(f"[run_eval] Evaluating metrics with {eval_script}...")
                    run_command(eval_cmd, args.dry_run)
            else:
                print("[run_eval] Skip evaluation.")

    print("[run_eval] All tasks finished.")


if __name__ == "__main__":
    main()
