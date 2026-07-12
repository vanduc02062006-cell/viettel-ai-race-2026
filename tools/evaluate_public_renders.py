import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def psnr(pred, gt):
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def ssim(pred, gt):
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    scores = []
    for ch in range(3):
        x = pred[..., ch]
        y = gt[..., ch]
        mux = cv2.GaussianBlur(x, (11, 11), 1.5)
        muy = cv2.GaussianBlur(y, (11, 11), 1.5)
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy
        sigx2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mux2
        sigy2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - muy2
        sigxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - muxy
        score = ((2 * muxy + c1) * (2 * sigxy + c2)) / ((mux2 + muy2 + c1) * (sigx2 + sigy2 + c2))
        scores.append(float(score.mean()))
    return float(np.mean(scores))


def read_pose_names(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["image_name"] for row in reader]


def prediction_path(pred_root, layout, split_name, scene_name, image_name):
    if layout == "scene_only":
        return pred_root / scene_name / image_name
    if layout == "phase_split_scene":
        return pred_root / "phase1" / split_name / scene_name / image_name
    return pred_root / split_name / scene_name / image_name


def evaluate(args):
    data_root = Path(args.data_root)
    pred_root = Path(args.pred_root)
    rows = []
    split_name = "public_set"
    split_dir = data_root / "phase1" / split_name

    for scene_dir in sorted(split_dir.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name.startswith("._"):
            continue
        if args.scene and scene_dir.name != args.scene:
            continue

        gt_dir = scene_dir / "test" / "images_original_distorted"
        poses_path = scene_dir / "test" / "test_poses.csv"
        if not gt_dir.exists():
            print(f"Skipping {scene_dir.name}: missing {gt_dir}")
            continue

        for image_name in read_pose_names(poses_path):
            pred_path = prediction_path(pred_root, args.layout, split_name, scene_dir.name, image_name)
            gt_path = gt_dir / image_name
            if not pred_path.exists() or not gt_path.exists():
                rows.append({
                    "scene": scene_dir.name,
                    "image_name": image_name,
                    "psnr": "",
                    "ssim": "",
                    "status": "missing_prediction" if not pred_path.exists() else "missing_gt",
                })
                continue

            pred = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)
            gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
            if pred is None or gt is None:
                rows.append({
                    "scene": scene_dir.name,
                    "image_name": image_name,
                    "psnr": "",
                    "ssim": "",
                    "status": "read_error",
                })
                continue
            if pred.shape != gt.shape:
                if not args.resize_prediction:
                    rows.append({
                        "scene": scene_dir.name,
                        "image_name": image_name,
                        "psnr": "",
                        "ssim": "",
                        "status": f"shape_mismatch:{pred.shape}!={gt.shape}",
                    })
                    continue
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

            rows.append({
                "scene": scene_dir.name,
                "image_name": image_name,
                "psnr": f"{psnr(pred, gt):.6f}",
                "ssim": f"{ssim(pred, gt):.6f}",
                "status": "ok",
            })

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scene", "image_name", "psnr", "ssim", "status"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        psnrs = np.array([float(r["psnr"]) for r in ok_rows])
        ssims = np.array([float(r["ssim"]) for r in ok_rows])
        print(f"Images evaluated: {len(ok_rows)}")
        print(f"Mean PSNR: {psnrs.mean():.4f}")
        print(f"Mean SSIM: {ssims.mean():.4f}")
    print(f"Saved metrics to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate distorted public renders against public ground truth images.")
    parser.add_argument("--data_root", default="work_undistorted")
    parser.add_argument("--pred_root", default="outputs/test_pose_renders_distorted")
    parser.add_argument("--layout", default="split_scene", choices=["split_scene", "scene_only", "phase_split_scene"])
    parser.add_argument("--scene", default=None)
    parser.add_argument("--output_csv", default="outputs/evaluation/public_metrics.csv")
    parser.add_argument("--resize_prediction", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
