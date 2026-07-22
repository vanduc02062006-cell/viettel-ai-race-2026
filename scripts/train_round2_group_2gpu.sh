#!/usr/bin/env bash

set -uo pipefail

group="${ROUND2_GROUP:-C}"
case "${group}" in
    C)
        gpu0_scenes=(HCM0421 HCM0539)
        gpu1_scenes=(HCM0540 HCM0644)
        ;;
    D)
        gpu0_scenes=(HCM0674 chair)
        gpu1_scenes=(bonsai)
        ;;
    *)
        echo "ROUND2_GROUP must be C or D, got: ${group}"
        exit 2
        ;;
esac

required_scenes=("${gpu0_scenes[@]}" "${gpu1_scenes[@]}")
mkdir -p logs

train_scene() {
    local gpu="$1"
    local scene="$2"
    local scene_root="work_round2/phase1/private_set1/${scene}"
    local model_path="outputs/prod_private_${scene}"
    local final_ply="${model_path}/point_cloud/iteration_30000/point_cloud.ply"
    local log_path="logs/train_prod_private_${scene}.log"

    if [[ -f "${final_ply}" ]]; then
        echo "===== GPU ${gpu}: SKIP ${scene}, FINAL MODEL EXISTS ====="
        return 0
    fi
    if [[ ! -d "${scene_root}/train/images" ]]; then
        echo "===== GPU ${gpu}: FAILED ${scene}, TRAIN DATA MISSING ====="
        return 1
    fi

    echo "===== GPU ${gpu}: START ${scene} ====="
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONUNBUFFERED=1 \
    python -u gaussian-splatting/train.py \
        -s "${scene_root}/train" \
        -m "${model_path}" \
        --iterations 30000 \
        -r 1 \
        --data_device cpu \
        --lambda_dssim 0.2 \
        --save_iterations 30000 \
        --test_iterations 30001 \
        --antialiasing \
        --disable_viewer \
        > "${log_path}" 2>&1 &

    local train_pid=$!
    while kill -0 "${train_pid}" 2>/dev/null; do
        sleep 180
        if kill -0 "${train_pid}" 2>/dev/null; then
            local rss_kb
            local available_kb
            local disk_available_kb
            rss_kb=$(ps -o rss= -p "${train_pid}" | tr -d ' ')
            available_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
            disk_available_kb=$(df -Pk /kaggle/working | awk 'NR==2 {print $4}')
            echo "HEARTBEAT scene=${scene} gpu=${gpu} pid=${train_pid} rss_kb=${rss_kb:-unknown} mem_available_kb=${available_kb:-unknown} disk_available_kb=${disk_available_kb:-unknown}"
        fi
    done

    wait "${train_pid}"
    local status=$?
    if [[ ${status} -eq 0 && -f "${final_ply}" ]]; then
        local model_mib
        model_mib=$(du -m "${final_ply}" | awk '{print $1}')
        echo "===== GPU ${gpu}: DONE ${scene}, MODEL ${model_mib} MiB ====="
        rm -rf -- "${scene_root}/train/images"
        echo "Released training images for ${scene}"
        return 0
    fi

    echo "===== GPU ${gpu}: FAILED ${scene}, STATUS ${status} ====="
    if [[ ${status} -eq 137 ]]; then
        echo "DIAGNOSIS: ${scene} was killed by SIGKILL; check host RAM."
    fi
    if grep -q "No space left on device" "${log_path}"; then
        echo "DIAGNOSIS: ${scene} exhausted /kaggle/working disk space."
    fi
    if grep -q "CUDA out of memory" "${log_path}"; then
        echo "DIAGNOSIS: ${scene} exhausted GPU VRAM."
    fi
    echo "===== LAST 4000 LOG BYTES: ${scene} ====="
    tail -c 4000 "${log_path}" || true
    return 1
}

run_gpu_queue() {
    local gpu="$1"
    shift
    local queue_status=0
    for scene in "$@"; do
        if ! train_scene "${gpu}" "${scene}"; then
            queue_status=1
        fi
    done
    return "${queue_status}"
}

echo "ROUND 2 GROUP ${group}: GPU 0=${gpu0_scenes[*]} | GPU 1=${gpu1_scenes[*]}"
run_gpu_queue 0 "${gpu0_scenes[@]}" &
pid0=$!
run_gpu_queue 1 "${gpu1_scenes[@]}" &
pid1=$!

wait "${pid0}"
status0=$?
wait "${pid1}"
status1=$?

echo "GPU 0 queue status: ${status0}"
echo "GPU 1 queue status: ${status1}"

# Kaggle can occasionally kill one of two concurrent training processes because
# host RAM is shared between both GPUs. Once the parallel queues have drained,
# retry only missing scenes serially. Successful scenes are preserved and their
# deleted image folders also leave more RAM/disk headroom for the retry.
retry_count=0
for scene in "${required_scenes[@]}"; do
    final_ply="outputs/prod_private_${scene}/point_cloud/iteration_30000/point_cloud.ply"
    if [[ ! -f "${final_ply}" ]]; then
        retry_count=$((retry_count + 1))
        echo "===== SERIAL RETRY ${retry_count}: ${scene} ON GPU 0 ====="
        rm -rf -- "outputs/prod_private_${scene}"
        sync
        sleep 10
        if train_scene 0 "${scene}"; then
            echo "===== SERIAL RETRY RECOVERED: ${scene} ====="
        else
            echo "===== SERIAL RETRY FAILED: ${scene} ====="
        fi
    fi
done

missing=0
for scene in "${required_scenes[@]}"; do
    ply="outputs/prod_private_${scene}/point_cloud/iteration_30000/point_cloud.ply"
    if [[ -f "${ply}" ]]; then
        echo "MODEL OK: ${scene} ($(du -m "${ply}" | awk '{print $1}') MiB)"
    else
        echo "MODEL MISSING: ${scene}"
        missing=1
    fi
done

if [[ ${missing} -ne 0 ]]; then
    echo "ROUND 2 GROUP ${group} FAILED"
    exit 1
fi

if [[ ${status0} -ne 0 || ${status1} -ne 0 ]]; then
    echo "Parallel queue had a failure, but serial retry recovered every missing model."
fi
echo "ROUND 2 GROUP ${group} VERIFIED"
