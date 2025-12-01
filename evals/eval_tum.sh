#!/bin/bash
# TUM RGB-D evaluation wrapper for vslam, mirrored from VGGT-SLAM logic.
# Set DATA_ROOT to dataset root and RUN_CMD/RUN_ARGS to your runner.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
submap_size=${1:-16}
data_root="${DATA_ROOT:-$repo_root}"
dataset_path="${data_root%/}/datasets/tum/"
gt_path="${data_root%/}/datasets/tum/"
log_dir="${repo_root}/logs"
log_path="${log_dir}/tum_results_w${submap_size}.txt"

mkdir -p "${log_dir}"

datasets=(
    rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_desk2
    rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_plant
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg1_rpy
    rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg1_xyz
)

# Number of full runs
n=${RUNS:-5}

run_cmd="${RUN_CMD:-}"
run_args_default="--max_loops 1 --min_disparity 50 --conf_threshold 25 --submap_size ${submap_size} --log_results --skip_dense_log"
run_args="${RUN_ARGS:-$run_args_default}"

# If file doesn't exist, write header
if [ ! -f "$log_path" ]; then
    echo "Run,Dataset,RMSE" > "$log_path"
fi

for run in $(seq 1 "$n"); do
    echo "==== Starting Run $run ===="

    total_rmse=0
    count=0

    for dataset in "${datasets[@]}"; do
        echo "Running pipeline on $dataset (Run $run)"
        dataset_name="${dataset_path}${dataset}/rgb"
        est_file="${log_dir}/${dataset}_run${run}_w${submap_size}.txt"

        if [ -n "$run_cmd" ]; then
            $run_cmd --image_folder "$dataset_name" $run_args --log_path "$est_file"
        else
            echo "RUN_CMD not set, skipping pipeline run for $dataset. Expecting existing log at $est_file"
        fi
    done

    for dataset in "${datasets[@]}"; do
        echo "Evaluating $dataset (Run $run)"
        est_path="${log_dir}/${dataset}_run${run}_w${submap_size}.txt"
        gt_file="${gt_path}${dataset}/groundtruth.txt"

        ape_result=$(evo_ape tum "$gt_file" "$est_path" -as)
        rmse=$(echo "$ape_result" | grep "rmse" | head -1 | sed -E 's/.*rmse[^0-9]*([0-9.]+).*/\1/')
        rmse=${rmse:-0}

        echo "$run,$dataset,$rmse" >> "$log_path"

        total_rmse=$(echo "$total_rmse + $rmse" | bc -l)
        count=$((count + 1))
    done

    avg_rmse=$(echo "$total_rmse / $count" | bc -l)
    echo "$run,Average,$avg_rmse" >> "$log_path"

    echo "==== Run $run complete ===="
    echo "Average RMSE for run $run: $avg_rmse"
done
