VIETTEL AI RACE 2026 - TWO GPU TRAINING RUNNERS

This small runner package does not contain competition images. The existing
vai_nvs_kaggle_prod_v3.zip dataset is still required.

Run in two separate Kaggle notebook versions to avoid the 12-hour session limit:

1. Import viettel_private_group_a.ipynb.
2. Add both private Kaggle inputs: prod_v3 and this runner ZIP/dataset.
3. Select GPU T4 x2, enable Internet, then Save Version -> Save & Run All.
4. Wait for PRIVATE GROUP A COMPLETED.
5. Import viettel_private_group_b.ipynb and repeat the same setup.
6. Wait for PRIVATE GROUP B COMPLETED.

Group A:
- GPU 0 / port 6010: HCM0249, HCM0276
- GPU 1 / port 6011: HCM0254, HNI0131

Group B:
- GPU 0 / port 6020: HNI0366, HCM1439
- GPU 1 / port 6021: HNI0437, HNI0265

Keep the outputs of both notebook versions. They contain the eight production models.
