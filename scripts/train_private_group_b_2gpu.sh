#!/usr/bin/env bash

set -uo pipefail

mkdir -p logs

train_scene() {
    local gpu="$1"
    local port="$2"
    local scene="$3"

    echo "===== GPU ${gpu}: START ${scene}, PORT ${port} ====="

    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONUNBUFFERED=1 \
    python -u gaussian-splatting/train.py \
        -s "work_undistorted/phase1/private_set1/${scene}/train" \
        -m "outputs/prod_private_${scene}" \
        --iterations 30000 \
        -r 1 \
        --data_device cpu \
        --lambda_dssim 0.2 \
        --save_iterations 15000 30000 \
        --antialiasing \
        --port "${port}" \
        2>&1 | tee "logs/train_prod_private_${scene}.log"

    local status=${PIPESTATUS[0]}
    if [[ ${status} -eq 0 ]]; then
        echo "===== GPU ${gpu}: DONE ${scene} ====="
    else
        echo "===== GPU ${gpu}: FAILED ${scene}, STATUS ${status} ====="
    fi
    return "${status}"
}

run_gpu_queue() {
    local gpu="$1"
    local port="$2"
    shift 2
    local queue_status=0

    for scene in "$@"; do
        if ! train_scene "${gpu}" "${port}" "${scene}"; then
            queue_status=1
        fi
    done
    return "${queue_status}"
}

run_gpu_queue 0 6020 HNI0366 HCM1439 &
pid0=$!

run_gpu_queue 1 6021 HNI0437 HNI0265 &
pid1=$!

wait "${pid0}"
status0=$?

wait "${pid1}"
status1=$?

echo "GPU 0 queue status: ${status0}"
echo "GPU 1 queue status: ${status1}"

missing=0
for scene in HNI0366 HCM1439 HNI0437 HNI0265; do
    ply="outputs/prod_private_${scene}/point_cloud/iteration_30000/point_cloud.ply"
    if [[ -f "${ply}" ]]; then
        echo "MODEL OK: ${scene}"
    else
        echo "MODEL MISSING: ${scene}"
        missing=1
    fi
done

if [[ ${status0} -ne 0 || ${status1} -ne 0 || ${missing} -ne 0 ]]; then
    echo "PRIVATE GROUP B FAILED"
    exit 1
fi

echo "PRIVATE GROUP B COMPLETED"
