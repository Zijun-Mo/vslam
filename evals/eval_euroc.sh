#!/bin/bash
# EuRoC evaluation wrapper for vslam, mirrored from VGGT-SLAM logic.
# Configure DATA_ROOT and RUN_CMD/RUN_ARGS as needed.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${DATA_ROOT:-$repo_root}"
dataset_path="${data_root%/}/datasets/euroc/"
gt_path="${data_root%/}/groundtruths/euroc/"

datasets=(
    MH_01_easy
    MH_02_easy
    MH_03_medium
    MH_04_difficult
    MH_05_difficult
    V1_01_easy
    V1_02_medium
    V1_03_difficult
    V2_01_easy
    V2_02_medium
    V2_03_difficult
)

run_cmd="${RUN_CMD:-}"
run_args_default="--max_loops 1 --conf_threshold 25 --min_disparity 50 --submap_size 16 --log_results"
run_args="${RUN_ARGS:-$run_args_default}"

for dataset in "${datasets[@]}"; do
    dataset_name="${dataset_path}${dataset}/mav0/cam0/data_rectified"
    est_file="${repo_root}/logs/${dataset}.txt"

    if [ -n "$run_cmd" ]; then
        echo "Running pipeline on $dataset"
        $run_cmd --image_folder "$dataset_name" $run_args --log_path "$est_file"
    else
        echo "RUN_CMD not set, skipping pipeline run for $dataset. Expecting existing log at $est_file"
    fi
done

total=0
count=0

for dataset in "${datasets[@]}"; do
    dataset_name="${dataset_path}${dataset}/"
    est_file="${repo_root}/logs/${dataset}.txt"
    echo "Processing ${dataset_name}"

    # Run evo_ape and extract RMSE translation error
    result=$(evo_ape tum "${gt_path}${dataset}.txt" "$est_file" -as)
    echo "$result"

    # Extract RMSE value (trans part) using grep/sed
    rmse=$(echo "$result" | grep "rmse" | head -1 | sed -E 's/.*rmse[^0-9]*([0-9.]+).*/\1/')

    if [[ ! -z "$rmse" ]]; then
        total=$(echo "$total + $rmse" | bc -l)
        count=$((count + 1))
    fi
done

if [[ $count -gt 0 ]]; then
    avg=$(echo "$total / $count" | bc -l)
    echo "Average RMSE translation APE over $count runs: $avg"
else
    echo "No valid results to average."
fi
