import os
from pathlib import Path
from typing import List, Dict

# 避免 matplotlib 写入不可写的全局缓存目录
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Bbox

# ------------- 数据：直接按示例图填充 ------------------------------------------

SCENES = ["chess", "fire", "heads", "office", "pumpkin", "kitchen", "stairs"]

# 每个方法一行，values 顺序与 SCENES 相同，末尾再跟 Avg
DATA: List[Dict] = [
    {
        "group": "Uncalib.",
        "method": "DROID-SLAM* [5]",
        "values": [0.047, 0.038, 0.034, 0.136, 0.166, 0.080, 0.044, 0.078],
    },
    {
        "group": "Uncalib.",
        "method": "MAS3R-SLAM* [4]",
        "values": [0.063, 0.046, 0.029, 0.103, 0.114, 0.074, 0.032, 0.066],
    },
    {
        "group": "Uncalib.",
        "method": "ORB-SLAM3* [1]",
        # 从 vslam/src/vslam_evals/logs/summary_ape_7scenes.csv 计算得到的场景均值 + 平均
        "values": [
            0.07653927514960998,
            0.05480031536362068,
            0.043645735049858934,
            0.1602452911899668,
            0.15407334234493694,
            0.13888709295891308,  # redkitchen 映射到 kitchen
            0.12017775218616497,
            0.10690982917758163,
        ],
    },
    {
        "group": "Uncalib.",
        "method": "MIT (SL(4), w = 8)* [3]",
        "values": [0.041, 0.060, 0.043, 0.106, 0.206, 0.054, 0.078, 0.084],
    },
    {
        "group": "Uncalib.",
        "method": "MIT (SL(4), w = 32)* [3]",
        "values": [0.036, 0.028, 0.018, 0.103, 0.133, 0.058, 0.093, 0.067],
    },
    {
        "group": "Uncalib.",
        "method": "Ours (1ms, w = 8)",
        # 从 vslam/src/vslam_evals/logs/evals_7scenes3.csv 聚合得到的场景均值 + 平均
        "values": [
            0.10170272379119581,
            0.08450312170561398,
            0.08964148218828422,
            0.20337057357845043,
            0.21812241093240312,
            0.1423118867122497,  # redkitchen 对应表中 kitchen
            0.11488683187706697,
            0.13636271868360916,
        ],
    },
    {
        "group": "Uncalib.",
        "method": "Ours (220ms, w = 8)",
        # 同上，为保持排序放在 MIT 之后
        "values": [
            0.06266936325816284,
            0.06631552757102717,
            0.057331024608948414,
            0.14478758837888364,
            0.1955085552862849,
            0.13324736601748086,  # redkitchen 对应表中 kitchen
            0.08253179464534068,
            0.10605588853801835,
        ],
    },
    {
        "group": "Uncalib.",
        "method": "Ours (1000ms, w = 8)",
        # 从 vslam/src/vslam_evals/logs/evals_7scenes4.csv 聚合得到的场景均值 + 平均
        "values": [
            0.055917108733395074,
            0.06494251166575242,
            0.051387743583596976,
            0.14032073830212866,
            0.14537757791915779,
            0.10026190686396258,  # redkitchen 对应表中 kitchen
            0.058467933926886456,
            0.08809650299926856,
        ],
    },
]

# ------------- 工具函数 -------------------------------------------------------

def compute_ranking() -> Dict[str, List[str]]:
    """计算每一列的最优与次优方法名，用于上色。"""
    ranking: Dict[str, List[str]] = {}
    cols = SCENES + ["Avg"]
    for i, col in enumerate(cols):
        sorted_methods = sorted(DATA, key=lambda d: d["values"][i])
        best = sorted_methods[0]["method"]
        second = sorted_methods[1]["method"] if len(sorted_methods) > 1 else None
        ranking[col] = [best, second]
    return ranking


def draw_table(outfile: str = "7scenes_ate.png") -> None:
    ranking = compute_ranking()

    # cellText 头两行为表头，后面为数据
    header_top = ["", "Method"] + [""] * len(SCENES) + ["Avg"]
    header_bottom = ["", ""] + SCENES + [""]
    cell_text: List[List[str]] = [header_top, header_bottom]
    row_methods: List[str] = [None, None]  # 与 cell_text 行对齐
    row_groups: List[str] = [None, None]  # 实际 group
    row_group_labels: List[str] = [None, None]  # 是否在这一行显示 group 文本

    spans_by_group: Dict[str, List[int]] = {}  # group -> [start_row_idx, end_row_idx]

    last_group = None
    current_row_idx = 2  # 数据行从索引 2 开始（前两行为表头）
    for row in DATA:
        group = row["group"]
        method = row["method"]
        vals = row["values"]
        group_label = group if group != last_group else ""
        last_group = group

        cell_text.append([group_label, method] + [f"{v:.3f}" for v in vals[:-1]] + [f"{vals[-1]:.3f}"])
        row_methods.append(method)
        row_groups.append(group)
        row_group_labels.append(group_label)
        if group not in spans_by_group:
            spans_by_group[group] = [current_row_idx, current_row_idx]
        else:
            spans_by_group[group][1] = current_row_idx
        current_row_idx += 1

    fig, ax = plt.subplots(figsize=(8.2, 1.7))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        loc="center",
        cellLoc="center",
        colWidths=[0.1, 0.2] + [0.09] * len(SCENES) + [0.09],  # 稍微加宽 Method 列
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    n_rows = len(cell_text)
    n_cols = len(cell_text[0])
    metric_cols = SCENES + ["Avg"]
    col_to_table_idx = {col: i + 2 for i, col in enumerate(metric_cols)}  # 场景列起始于 index 2

    # 样式：表头
    for r in range(2):
        for j in range(n_cols):
            c = table[r, j]
            c.set_facecolor("#f8f8f8")
            c.get_text().set_fontweight("bold")

    # 先触发布局计算，保证拿到准确的位置
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # 合并 Sequence 跨场景列
    span_start, span_end = 2, 2 + len(SCENES) - 1
    span_cells = [table[0, j] for j in range(span_start, span_end + 1)]

    # 清除这些单元格的边框和文字
    for cell in span_cells:
        cell.visible_edges = ""
        cell.get_text().set_text("")

    # 计算跨列区域的外接框（用像素坐标再转回 Axes 坐标，避免布局变动导致错位）
    bbox = Bbox.union([c.get_window_extent(renderer) for c in span_cells])
    x0, y0 = ax.transAxes.inverted().transform((bbox.x0, bbox.y0))
    x1, y1 = ax.transAxes.inverted().transform((bbox.x1, bbox.y1))
    width, height = x1 - x0, y1 - y0

    rect = patches.Rectangle(
        (x0, y0),
        width,
        height,
        transform=ax.transAxes,
        facecolor="#f8f8f8",
        edgecolor="black",
        linewidth=0.8,
        clip_on=False,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        x0 + width / 2,
        y0 + height / 2,
        "Sequence",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        transform=ax.transAxes,
        zorder=3,
    )

    best_color = "#fff3b0"
    second_color = "#c9e8d2"
    calib_text_color = "#8d8d8d"

    # 数据行样式、分组标记与高亮
    for r in range(2, n_rows):
        method = row_methods[r]
        group_actual = row_groups[r]
        group_label = row_group_labels[r]

        # 仅剩 Uncalib. 行
        text_color = "black"

        # 分组列先清空，稍后统一绘制跨行标签
        group_cell = table[r, 0]
        group_cell.get_text().set_text("")

        for j in range(1, n_cols):
            c = table[r, j]
            c.get_text().set_color(text_color)

        # 高亮 best / second
        for col_name, (best, second) in ranking.items():
            if method is None:
                continue
            j = col_to_table_idx[col_name]
            cell = table[r, j]
            if method == best:
                cell.set_facecolor(best_color)
            elif second and method == second:
                cell.set_facecolor(second_color)

    # 设置所有格子的边框线宽
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)

    # 绘制跨行标签（仅剩 Uncalib.）
    for label, (start_row, end_row) in spans_by_group.items():
        rows_span = list(range(start_row, end_row + 1))
        color = calib_text_color if label == "Calib." else "black"
        bboxes = [table[r, 0].get_window_extent(renderer) for r in rows_span]
        bbox = Bbox.union(bboxes)
        x0, y0 = ax.transAxes.inverted().transform((bbox.x0, bbox.y0))
        x1, y1 = ax.transAxes.inverted().transform((bbox.x1, bbox.y1))
        width, height = x1 - x0, y1 - y0
        rect = patches.Rectangle(
            (x0, y0),
            width,
            height,
            transform=ax.transAxes,
            facecolor="#f8f8f8",
            edgecolor="black",
            linewidth=0.8,
            clip_on=False,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + width / 2,
            y0 + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
            transform=ax.transAxes,
            zorder=3,
        )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.05)
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    out_path = Path(__file__).with_name("7scenes_ate.png")
    draw_table(str(out_path))
