# Viettel AI Race 2026 - Novel View Synthesis / 3D Gaussian Splatting

- Dataset path: `.\VAI_NVS_DATA`
- Nguyên tắc: Không sửa dataset gốc

## Pipeline tổng quát
inspect → COLMAP camera gate → setup 3DGS → train public debug → camera round-trip → render public → evaluate → private render → validate → zip

## Current blocker: local machine is CPU-only
- Kh�ng ch?y train tr�n CPU.
- Hu?ng ti?p theo l� ch?y tr�n GPU machine/cloud.
- C�c script c?n ch?y tr�n GPU d� du?c d?t trong scripts/ v� hu?ng d?n t?i docs/GPU_SETUP.md.
