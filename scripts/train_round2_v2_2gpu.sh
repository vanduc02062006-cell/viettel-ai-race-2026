#!/usr/bin/env bash

# Safe Round 2 V2 trainer for Kaggle T4 x2.
# - One active scene per GPU; scenes in each GPU queue run sequentially.
# - Disk-safe by default: no new optimizer checkpoints are written.
# - Existing checkpoints can still be resumed when attached as Kaggle inputs.
# - Skips completed scenes and resumes the newest checkpoint after interruption.
# - Retries missing scenes serially after both parallel queues have drained.

set -uo pipefail

group="${ROUND2_GROUP:-C}"
iterations="${V2_ITERATIONS:-40000}"
densify_until="${V2_DENSIFY_UNTIL:-20000}"
lambda_dssim="${V2_LAMBDA_DSSIM:-0.25}"
min_host_mem_gib="${V2_MIN_HOST_MEM_GIB:-8}"
min_gpu_mem_mib="${V2_MIN_GPU_MEM_MIB:-10000}"
min_disk_gib="${V2_MIN_DISK_GIB:-6}"
enable_checkpoints="${V2_ENABLE_CHECKPOINTS:-0}"

case "${group}" in
    C)
        gpu0_scenes=(HCM0421 HCM0539)
        gpu1_scenes=(HCM0540 HCM0644)
        ;;
    C1)
        gpu0_scenes=(HCM0421)
        gpu1_scenes=(HCM0540)
        ;;
    C2)
        gpu0_scenes=(HCM0539)
        gpu1_scenes=(HCM0644)
        ;;
    D)
        gpu0_scenes=(HCM0674 chair)
        gpu1_scenes=(bonsai)
        ;;
    *)
        echo "ROUND2_GROUP must be C, C1, C2, or D, got: ${group}"
        exit 2
        ;;
esac

required_scenes=("${gpu0_scenes[@]}" "${gpu1_scenes[@]}")
mkdir -p logs outputs

latest_checkpoint() {
    local model_path="$1"
    local best_iter=-1
    local best_path=""
    local checkpoint
    shopt -s nullglob
    for checkpoint in "${model_path}"/chkpnt*.pth; do
        local name="${checkpoint##*/}"
        local value="${name#chkpnt}"
        value="${value%.pth}"
        if [[ "${value}" =~ ^[0-9]+$ ]] && (( value < iterations )) && (( value > best_iter )); then
            best_iter="${value}"
            best_path="${checkpoint}"
        fi
    done
    shopt -u nullglob
    printf '%s' "${best_path}"
}

wait_for_resources() {
    local gpu="$1"
    local attempts=0
    while true; do
        local available_kb
        local available_gib
        local gpu_free_mib
        local disk_available_kb
        local disk_available_gib
        available_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
        available_gib=$(( ${available_kb:-0} / 1024 / 1024 ))
        gpu_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
        disk_available_kb=$(df -Pk /kaggle/working | awk 'NR==2 {print $4}')
        disk_available_gib=$(( ${disk_available_kb:-0} / 1024 / 1024 ))
        if (( available_gib >= min_host_mem_gib && ${gpu_free_mib:-0} >= min_gpu_mem_mib && disk_available_gib >= min_disk_gib )); then
            echo "RESOURCE GATE PASS gpu=${gpu} host_available_gib=${available_gib} gpu_free_mib=${gpu_free_mib} disk_available_gib=${disk_available_gib}"
            return 0
        fi
        attempts=$((attempts + 1))
        if (( attempts >= 10 )); then
            echo "RESOURCE GATE FAIL gpu=${gpu} host_available_gib=${available_gib} gpu_free_mib=${gpu_free_mib} disk_available_gib=${disk_available_gib}"
            return 1
        fi
        echo "RESOURCE WAIT gpu=${gpu} host_available_gib=${available_gib} gpu_free_mib=${gpu_free_mib} disk_available_gib=${disk_available_gib} attempt=${attempts}/10"
        sleep 30
    done
}

purge_checkpoints() {
    local model_path="$1"
    local checkpoint_files=()
    shopt -s nullglob
    checkpoint_files=("${model_path}"/chkpnt*.pth)
    shopt -u nullglob
    if (( ${#checkpoint_files[@]} > 0 )); then
        rm -f -- "${checkpoint_files[@]}"
        echo "Released ${#checkpoint_files[@]} optimizer checkpoint(s) from ${model_path}"
    fi
}

train_scene() {
    local gpu="$1"
    local scene="$2"
    local scene_root="work_round2/phase1/private_set1/${scene}"
    local model_path="outputs/prod_private_${scene}"
    local final_ply="${model_path}/point_cloud/iteration_${iterations}/point_cloud.ply"
    local log_path="logs/train_v2_private_${scene}.log"

    if [[ -f "${final_ply}" ]]; then
        purge_checkpoints "${model_path}"
        echo "===== GPU ${gpu}: SKIP ${scene}; ITERATION ${iterations} EXISTS ====="
        return 0
    fi
    if [[ ! -d "${scene_root}/train/images" ]]; then
        echo "===== GPU ${gpu}: FAILED ${scene}; TRAIN IMAGES MISSING ====="
        return 1
    fi
    wait_for_resources "${gpu}" || return 1

    local checkpoint
    checkpoint=$(latest_checkpoint "${model_path}")
    local checkpoint_args=()
    if [[ -n "${checkpoint}" ]]; then
        checkpoint_args=(--start_checkpoint "${checkpoint}")
        echo "===== GPU ${gpu}: RESUME ${scene} FROM ${checkpoint} ====="
    else
        echo "===== GPU ${gpu}: START ${scene} FROM SCRATCH ====="
    fi

    local checkpoint_iterations=()
    if [[ "${enable_checkpoints}" == "1" ]] && (( iterations > 30000 )); then
        checkpoint_iterations+=(30000)
    fi

    local checkpoint_cli=()
    if (( ${#checkpoint_iterations[@]} > 0 )); then
        checkpoint_cli=(--checkpoint_iterations "${checkpoint_iterations[@]}")
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MALLOC_ARENA_MAX=2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u gaussian-splatting/train.py \
        -s "${scene_root}/train" \
        -m "${model_path}" \
        --iterations "${iterations}" \
        --position_lr_max_steps "${iterations}" \
        --densify_until_iter "${densify_until}" \
        --densify_grad_threshold 0.0002 \
        -r 1 \
        --data_device cpu \
        --lambda_dssim "${lambda_dssim}" \
        --save_iterations "${iterations}" \
        --test_iterations "$((iterations + 1))" \
        "${checkpoint_cli[@]}" \
        "${checkpoint_args[@]}" \
        --antialiasing \
        --disable_viewer \
        > "${log_path}" 2>&1 &

    local train_pid=$!
    while kill -0 "${train_pid}" 2>/dev/null; do
        sleep 60
        if kill -0 "${train_pid}" 2>/dev/null; then
            local rss_kb
            local available_kb
            local disk_available_kb
            local gpu_used_mib
            rss_kb=$(ps -o rss= -p "${train_pid}" | tr -d ' ')
            available_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
            disk_available_kb=$(df -Pk /kaggle/working | awk 'NR==2 {print $4}')
            gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
            echo "HEARTBEAT scene=${scene} gpu=${gpu} pid=${train_pid} rss_kb=${rss_kb:-unknown} host_available_kb=${available_kb:-unknown} gpu_used_mib=${gpu_used_mib:-unknown} disk_available_kb=${disk_available_kb:-unknown}"
        fi
    done

    wait "${train_pid}"
    local status=$?
    if [[ ${status} -eq 0 && -f "${final_ply}" ]]; then
        local model_mib
        model_mib=$(du -m "${final_ply}" | awk '{print $1}')
        echo "===== GPU ${gpu}: DONE ${scene}; ITER=${iterations}; MODEL=${model_mib} MiB ====="
        purge_checkpoints "${model_path}"
        if [[ -d "${scene_root}/train/images" ]]; then
            rm -rf -- "${scene_root}/train/images"
            echo "Released prepared training images for ${scene}"
        fi
        return 0
    fi

    echo "===== GPU ${gpu}: FAILED ${scene}; STATUS=${status} ====="
    if [[ ${status} -eq 137 ]]; then
        echo "DIAGNOSIS: SIGKILL; likely shared host RAM or session limit. Serial retry will preserve checkpoints."
    fi
    if grep -q "CUDA out of memory" "${log_path}"; then
        echo "DIAGNOSIS: CUDA OOM on ${scene}."
    fi
    if grep -q "No space left on device" "${log_path}"; then
        echo "DIAGNOSIS: /kaggle/working ran out of disk."
    fi
    echo "===== LAST 6000 LOG BYTES: ${scene} ====="
    tail -c 6000 "${log_path}" || true
    return 1
}

run_gpu_queue() {
    local gpu="$1"
    shift
    local queue_status=0
    local scene
    for scene in "$@"; do
        if ! train_scene "${gpu}" "${scene}"; then
            queue_status=1
        fi
    done
    return "${queue_status}"
}

echo "ROUND 2 V2 GROUP ${group}: GPU0=${gpu0_scenes[*]} | GPU1=${gpu1_scenes[*]}"
echo "CONFIG iterations=${iterations} densify_until=${densify_until} lambda_dssim=${lambda_dssim} checkpoints=${enable_checkpoints} min_disk_gib=${min_disk_gib}"
run_gpu_queue 0 "${gpu0_scenes[@]}" &
pid0=$!
run_gpu_queue 1 "${gpu1_scenes[@]}" &
pid1=$!

wait "${pid0}"
status0=$?
wait "${pid1}"
status1=$?
echo "PARALLEL QUEUES FINISHED: gpu0_status=${status0} gpu1_status=${status1}"

# Both parallel jobs are finished here, so serial retries cannot contend for
# host RAM with another training process. Existing checkpoints are preserved.
for scene in "${required_scenes[@]}"; do
    final_ply="outputs/prod_private_${scene}/point_cloud/iteration_${iterations}/point_cloud.ply"
    if [[ ! -f "${final_ply}" ]]; then
        echo "===== SERIAL RETRY ${scene} ON GPU 0 ====="
        if ! train_scene 0 "${scene}"; then
            echo "===== SERIAL RETRY FAILED ${scene} ====="
        fi
    fi
done

missing=0
for scene in "${required_scenes[@]}"; do
    ply="outputs/prod_private_${scene}/point_cloud/iteration_${iterations}/point_cloud.ply"
    if [[ -f "${ply}" ]]; then
        echo "MODEL OK: ${scene} iteration=${iterations} size=$(du -m "${ply}" | awk '{print $1}') MiB"
    else
        echo "MODEL MISSING: ${scene} iteration=${iterations}"
        missing=1
    fi
done

if [[ ${missing} -ne 0 ]]; then
    echo "ROUND 2 V2 GROUP ${group} FAILED"
    exit 1
fi

echo "ROUND 2 V2 GROUP ${group} VERIFIED"
