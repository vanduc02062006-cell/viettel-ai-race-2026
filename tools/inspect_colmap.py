import argparse
import sys
from pathlib import Path
from colmap_io import read_cameras_binary, read_images_binary

def inspect_colmap(data_root, split_mode, scene_opt):
    data_root = Path(data_root)
    log_file = Path('logs/colmap_inspect.txt')
    gate_file = Path('logs/camera_model_gate.txt')
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    splits = []
    if split_mode in ['public_set', 'all']:
        splits.append('public_set')
    if split_mode in ['private_set1', 'all']:
        splits.append('private_set1')

    supported_models = {'PINHOLE', 'SIMPLE_PINHOLE'}
    failed_cameras = []

    with open(log_file, 'w', encoding='utf-8') as lf:
        def lprint(msg):
            print(msg)
            lf.write(msg + '\n')

        for split in splits:
            split_dir = data_root / 'phase1' / split
            if not split_dir.exists():
                continue

            for scene_dir in split_dir.iterdir():
                if not scene_dir.is_dir() or scene_dir.name.startswith('._') or scene_dir.name in ['__MACOSX', '.DS_Store']:
                    continue
                if scene_opt and scene_dir.name != scene_opt:
                    continue

                cameras_bin = scene_dir / 'train' / 'sparse' / '0' / 'cameras.bin'
                images_bin = scene_dir / 'train' / 'sparse' / '0' / 'images.bin'

                if not cameras_bin.exists() or not images_bin.exists():
                    lprint(f"WARNING: Missing COLMAP bin files in {scene_dir}")
                    continue

                lprint(f"\n--- Scene: {scene_dir.name} ({split}) ---")
                
                try:
                    cameras = read_cameras_binary(cameras_bin)
                    lprint(f"Cameras count: {len(cameras)}")
                    for cam_id, cam in cameras.items():
                        lprint(f"  Camera {cam_id}: {cam.model}, {cam.width}x{cam.height}, params={cam.params}")
                        
                        if cam.model not in supported_models:
                            lprint("\nERROR: UNSUPPORTED_CAMERA_MODEL_FOR_GRAPHDECO_3DGS")
                            lprint(f"Scene: {scene_dir.name}, Camera ID: {cam_id}, Model: {cam.model}")
                            lprint("The original graphdeco-inria/gaussian-splatting repository will crash because it only supports undistorted PINHOLE/SIMPLE_PINHOLE models.")
                            lprint("Do NOT run training with the original repository until one of the following is done:")
                            lprint("  A. Undistort images and recreate the COLMAP model with PINHOLE/SIMPLE_PINHOLE.")
                            lprint("  B. Use a fork/renderer that supports distortion.")
                            lprint("")
                            failed_cameras.append((scene_dir.name, cam_id, cam.model))

                    images = read_images_binary(images_bin)
                    lprint(f"Registered images count: {len(images)}")
                    
                    img_list = list(images.values())
                    for img in img_list[:5]:
                        lprint(f"  Image {img.id}: {img.name}, cam_id={img.camera_id}, qvec={img.qvec}, tvec={img.tvec}")

                except Exception as e:
                    lprint(f"ERROR reading COLMAP models for {scene_dir.name}: {e}")

    with open(gate_file, 'w', encoding='utf-8') as gf:
        if not failed_cameras:
            gf.write("PASS\n")
        else:
            gf.write("FAIL\n")
            for scene, cam_id, model in failed_cameras:
                gf.write(f"{scene} / cam {cam_id} / {model}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='.\\VAI_NVS_DATA')
    parser.add_argument('--split', default='all', choices=['public_set', 'private_set1', 'all'])
    parser.add_argument('--scene', default=None)
    args = parser.parse_args()
    inspect_colmap(args.data_root, args.split, args.scene)
