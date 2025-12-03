#!/bin/bash
# 7-Scenes evaluation wrapper for vslam, mirrored from VGGT-SLAM logic.
# Customize DATA_ROOT to point to datasets/groundtruths, and RUN_CMD/RUN_ARGS to run your pipeline.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
submap_size=${1:-16}

# DATA_ROOT 优先级：环境变量 DATA_ROOT -> ~/MASt3R-SLAM -> 仓库根
default_root="$HOME/MASt3R-SLAM"
if [ -n "${DATA_ROOT:-}" ]; then
    data_root="${DATA_ROOT}"
elif [ -d "$default_root" ]; then
    data_root="$default_root"
else
    data_root="$repo_root"
fi
dataset_path="${data_root%/}/datasets/7-scenes/"
gt_path="${data_root%/}/datasets/groundtruths/7-scenes/"
log_dir="${repo_root}/logs"
log_path="${log_dir}/7scenes_results_w${submap_size}.txt"

mkdir -p "${log_dir}"

datasets=(
    chess
    fire
    heads
    office
    pumpkin
    redkitchen
    stairs
)

# Number of full evaluation repetitions
n=${RUNS:-5}

# Pipeline command (set RUN_CMD to your runner, e.g. "python main.py" or a ros2 bag player wrapper)
run_cmd="${RUN_CMD:-}"
run_args_default="--max_loops 1 --min_disparity 50 --conf_threshold 25 --submap_size ${submap_size} --log_results"
run_args="${RUN_ARGS:-$run_args_default}"

# If file doesn't exist, write CSV header
if [ ! -f "$log_path" ]; then
    echo "Run,Dataset,RMSE,RMSE acc,RMSE comp,Chamfer" > "$log_path"
fi

for run in $(seq 1 "$n"); do
    echo "==== Starting Run $run ===="

    total_rmse=0
    total_rmse_acc=0
    total_rmse_comp=0
    total_chamfer=0
    count=0

    for dataset in "${datasets[@]}"; do
        dataset_name="${dataset_path}${dataset}/seq-01"
        est_file="${log_dir}/${dataset}_run${run}_w${submap_size}.txt"

        if [ -n "$run_cmd" ]; then
            echo "Running pipeline on $dataset (Run $run)"
            $run_cmd --image_folder "$dataset_name" $run_args --log_path "$est_file"
        else
            echo "RUN_CMD not set, skipping pipeline run for $dataset. Expecting existing log at $est_file"
        fi
    done

    for dataset in "${datasets[@]}"; do
        echo "Processing $dataset (Run $run)"

        est_file="${log_dir}/${dataset}_run${run}_w${submap_size}.txt"
        gt_file="${gt_path}${dataset}.txt"

        ape_result=$(evo_ape tum "$gt_file" "$est_file" -as)
        rmse=$(echo "$ape_result" | grep "rmse" | head -1 | sed -E 's/.*rmse[^0-9]*([0-9.]+).*/\1/')

        eval_output=$(python "$repo_root/evals/eval7_scenes_dense.py" --dataset "${dataset_path}${dataset}" --gt "$gt_file" --est "$est_file" --no-viz)

        rmse_acc=$(echo "$eval_output" | grep "RMSE acc" | awk '{print $3}')
        rmse_comp=$(echo "$eval_output" | grep "RMSE comp" | awk '{print $3}')
        chamfer=$(echo "$eval_output" | grep "Chamfer distance" | awk '{print $3}')

        rmse=${rmse:-0}
        rmse_acc=${rmse_acc:-0}
        rmse_comp=${rmse_comp:-0}
        chamfer=${chamfer:-0}

        echo "$run,$dataset,$rmse,$rmse_acc,$rmse_comp,$chamfer" >> "$log_path"

        total_rmse=$(echo "$total_rmse + $rmse" | bc -l)
        total_rmse_acc=$(echo "$total_rmse_acc + $rmse_acc" | bc -l)
        total_rmse_comp=$(echo "$total_rmse_comp + $rmse_comp" | bc -l)
        total_chamfer=$(echo "$total_chamfer + $chamfer" | bc -l)

        count=$((count + 1))
    done

    avg_rmse=$(echo "$total_rmse / $count" | bc -l)
    avg_rmse_acc=$(echo "$total_rmse_acc / $count" | bc -l)
    avg_rmse_comp=$(echo "$total_rmse_comp / $count" | bc -l)
    avg_chamfer=$(echo "$total_chamfer / $count" | bc -l)

    echo "$run,Average,$avg_rmse,$avg_rmse_acc,$avg_rmse_comp,$avg_chamfer" >> "$log_path"

    echo "==== Run $run complete ===="
    echo "Average RMSE: $avg_rmse"
    echo "Average RMSE acc: $avg_rmse_acc"
    echo "Average RMSE comp: $avg_rmse_comp"
    echo "Average Chamfer: $avg_chamfer"
done
