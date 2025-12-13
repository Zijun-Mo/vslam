import csv
from collections import defaultdict
from pathlib import Path

def compute_scene_rmse_means(csv_path: str):
    totals = defaultdict(float)
    counts = defaultdict(int)

    with Path(csv_path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene = row["seq_name"].split("/")[0]  # e.g., "chess/seq-01" -> "chess"
            rmse = float(row["rmse"])
            totals[scene] += rmse
            counts[scene] += 1

    scene_means = {scene: totals[scene] / counts[scene] for scene in totals}
    overall_mean = sum(scene_means.values()) / len(scene_means) if scene_means else 0.0
    return scene_means, overall_mean

if __name__ == "__main__":
    csv_file = "/home/jun/vslam/src/vslam_evals/logs/evals_7scenes4.csv"  # adjust if needed
    scene_means, overall_mean = compute_scene_rmse_means(csv_file)

    for scene, mean_rmse in sorted(scene_means.items()):
        print(f"{scene}: {mean_rmse:.6f}")
    print(f"overall_mean: {overall_mean:.6f}")