Viettel AI Race 2026 - Private Group B

Files:
- viettel_private_group_b.ipynb: ready-to-import Kaggle notebook.
- scripts/train_private_group_b_2gpu.sh: two-GPU training script.

Kaggle steps:
1. Import viettel_private_group_b.ipynb as a new Kaggle notebook.
2. Create a small private Kaggle dataset from this Group B runner ZIP.
3. Add both inputs: the existing prod_v3 dataset and the Group B runner dataset.
4. Select GPU T4 x2 and enable Internet.
5. Save Version -> Save & Run All.
6. Do not launch Group A in this notebook.

GPU queues:
- GPU 0, port 6020: HNI0366 then HCM1439.
- GPU 1, port 6021: HNI0437 then HNI0265.

Success condition:
- The final log prints PRIVATE GROUP B COMPLETED.
- Four point_cloud.ply files exist under iteration_30000.
