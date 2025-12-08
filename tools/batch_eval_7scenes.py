#!/usr/bin/env python3
"""
Batch evaluation script for 7-Scenes.

作用：
- 自动遍历 data_root/<dataset_dir> 下的所有 scene 和 seq-*
- 对每个 (scene, seq) 调一次 ros2 launch vslam_evals eval_7scenes_office.launch.py
- 实时监听 EvalNode 的输出，看到“DONE 标记”后，自动给 ros2 launch 发 Ctrl+C
- 方便你一键跑完所有 7-Scenes 的评估

使用前提：
- 你已经在终端中 source 了：
    source /opt/ros/humble/setup.bash
    source /home/jun/vslam/install/setup.bash
- EvalNode 在评估结束时会打印一行包含 EVAL_DONE_TOKEN 的日志
"""

import argparse
import subprocess
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional


# ----------------------------- 配置区 ---------------------------------

# EvalNode 打印的“评估完成”标记行中应该包含的字符串
# 建议在 EvalNode 里加一行：
#   RCLCPP_INFO(get_logger(), "[EvalNode] DONE scene=%s seq=%s ATE_RMSE=%.3f", ...);
# 然后这里用：
EVAL_DONE_TOKEN = "[EvalNode] DONE"

# 如果你暂时没改 EvalNode，可以先用这一行（匹配“Wrote results to ...”）：
# EVAL_DONE_TOKEN = "Wrote results to"

# （可选）EvalNode 写出的汇总 CSV（根据你自己的实现修改路径）
DEFAULT_EVAL_CSV = "src/vslam_evals/logs/evals_7scenes.csv"


# --------------------------- 参数解析 ---------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation for 7-Scenes scenes/seqs."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="根目录，例如 /home/jun/vslam/datasets（内部应有 7-scenes/office/...）",
    )
    parser.add_argument(
        "--launch_file",
        type=str,
        default="eval_7scenes_office.launch.py",
        help="vslam_evals 包中的 launch 文件名。",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="7-scenes",
        help="数据集所在的目录名（默认 7-scenes，可改为 7scenes 等）",
    )
    parser.add_argument(
        "--play_rate",
        type=float,
        default=0.3,
        help="传给 dataset player 的 play_rate。",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="*",
        default=None,
        help="只评估指定的 scene（如 office chess）。为空则自动遍历全部。",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印将要执行的命令，不真正运行。",
    )
    parser.add_argument(
        "--eval_csv",
        type=str,
        default=DEFAULT_EVAL_CSV,
        help="EvalNode 写出的评估结果 CSV 路径（可用于后续 collect）。",
    )
    return parser.parse_args()


# --------------------------- 场景 / 序列发现 ---------------------------


def discover_scenes_and_seqs(
    data_root: Path, dataset_dir: str, scenes_filter: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    在 data_root/<dataset_dir> 下发现所有 scene 和 seq-*。

    返回：
        { scene_name: [seq-01, seq-02, ...], ... }
    """
    scenes_dir = data_root / dataset_dir
    if not scenes_dir.is_dir():
        raise FileNotFoundError(f"{dataset_dir} directory not found: {scenes_dir}")

    scene_to_seqs: Dict[str, List[str]] = {}

    for scene_dir in sorted(scenes_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_name = scene_dir.name

        if scenes_filter is not None and scene_name not in scenes_filter:
            continue

        seq_names: List[str] = []
        for seq_dir in sorted(scene_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            if not seq_dir.name.startswith("seq-"):
                continue
            seq_names.append(seq_dir.name)

        if seq_names:
            scene_to_seqs[scene_name] = seq_names

    return scene_to_seqs


# --------------------------- 单次评估调用 ------------------------------


def run_eval_once_with_kill(
    scene: str,
    seq: str,
    data_root: Path,
    dataset_dir: str,
    launch_file: str,
    play_rate: float,
    dry_run: bool = False,
) -> int:
    """
    对单个 (scene, seq) 调一次 ros2 launch。

    行为：
    - 启动子进程运行：
        ros2 launch vslam_evals <launch_file> scene:=<scene> seq:=<seq> ...
    - 实时读 stdout/stderr
    - 一旦看到包含 EVAL_DONE_TOKEN 的行，就向子进程发送 SIGINT（等价于 Ctrl+C）
    - 等待进程退出，然后返回 returncode
    """
    cmd = [
        "ros2",
        "launch",
        "vslam_evals",
        launch_file,
        f"scene:={scene}",
        f"seq:={seq}",
        f"data_root:={str(data_root)}",
        f"play_rate:={play_rate}",
        f"dataset_dir:={dataset_dir}",
    ]

    print("=" * 80)
    print(f"[INFO] Evaluating scene={scene}, seq={seq}")
    print("[CMD]", " ".join(cmd))

    if dry_run:
        return 0

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,      # 逐行文本输出
        bufsize=1,      # 行缓冲
    )

    t0 = time.time()
    eval_done = False

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            # 原样打印 ros2 launch 的输出
            print(line, end="")

            # 监控 EvalNode 的“完成标记”
            if EVAL_DONE_TOKEN in line:
                eval_done = True
                print(
                    f"[INFO] Detected eval done token ({EVAL_DONE_TOKEN}) "
                    f"for {scene}/{seq}, sending SIGINT..."
                )
                # 给其他 node 一点收尾时间
                time.sleep(1.0)
                proc.send_signal(signal.SIGINT)

        # 输出读完，等待进程完全退出
        proc.wait()
    except KeyboardInterrupt:
        print("[WARN] Batch script interrupted, forwarding SIGINT to child...")
        proc.send_signal(signal.SIGINT)
        proc.wait()
    finally:
        dt = time.time() - t0
        print(
            f"[INFO] {scene}/{seq} finished in {dt:.1f}s, "
            f"returncode={proc.returncode}, eval_done={eval_done}"
        )

    return int(proc.returncode)


# ---------------------------- 结果收集（占位） --------------------------


def collect_eval_result(scene: str, seq: str, eval_csv: Path) -> None:
    """
    在这里解析 EvalNode 写出的 CSV，并做后续汇总。

    目前先留一个占位框架，方便你后续接着开发：
    - 你可以在这里读取 eval_csv（例如 logs/evals_7scenes.csv）
    - 找到 scene, seq 对应的那一行
    - 提取 ATE_RMSE 等字段
    - 写入一个 batch_results.csv

    这里先不实现细节，避免和你当前 eval_node 实现耦合太死。
    """
    # TODO(jun): 根据你自己的 CSV 格式实现解析和汇总
    # 比如：
    #   import csv
    #   with eval_csv.open() as f:
    #       reader = csv.DictReader(f)
    #       for row in reader:
    #           if row["scene"] == scene and row["seq"] == seq:
    #               print(f"[INFO] ATE_RMSE for {scene}/{seq} = {row['ATE_RMSE']}")
    #               break
    pass


# ------------------------------ 主逻辑 ---------------------------------


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    eval_csv = Path(args.eval_csv).expanduser().resolve()

    scene_to_seqs = discover_scenes_and_seqs(data_root, args.dataset_dir, args.scenes)
    if not scene_to_seqs:
        print(
            f"[ERROR] No scenes found under {data_root}/{args.dataset_dir} "
            f"(filter={args.scenes})"
        )
        return 1

    print("[INFO] Discovered scenes and sequences:")
    for scene, seqs in scene_to_seqs.items():
        print(f"  - {scene}: {', '.join(seqs)}")

    # 主循环：顺序跑每一个 (scene, seq)
    for scene, seqs in scene_to_seqs.items():
        for seq in seqs:
            ret = run_eval_once_with_kill(
                scene=scene,
                seq=seq,
                data_root=data_root,
                dataset_dir=args.dataset_dir,
                launch_file=args.launch_file,
                play_rate=args.play_rate,
                dry_run=args.dry_run,
            )
            if ret != 0:
                print(
                    f"[WARN] Eval failed for {scene}/{seq}, "
                    f"returncode={ret}"
                )
                # 这里选择继续跑后面的，避免全批次因为一条失败而中断
                continue

            # 可选：这里调用 collect_eval_result 做更细致的汇总
            # collect_eval_result(scene, seq, eval_csv)

    print("[INFO] Batch evaluation finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
