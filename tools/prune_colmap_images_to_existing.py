import argparse
import shutil
from pathlib import Path

from colmap_io import read_images_binary, write_images_binary


def prune_scene(scene_train_path, backup):
    scene_train_path = Path(scene_train_path)
    images_dir = scene_train_path / "images"
    images_bin = scene_train_path / "sparse" / "0" / "images.bin"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not images_bin.exists():
        raise FileNotFoundError(f"Missing COLMAP images.bin: {images_bin}")

    existing = {
        path.name
        for path in images_dir.iterdir()
        if path.is_file() and not path.name.startswith("._")
    }
    images = read_images_binary(images_bin)
    kept = {image_id: image for image_id, image in images.items() if image.name in existing}
    missing = [image.name for image in images.values() if image.name not in existing]

    if backup:
        backup_path = images_bin.with_suffix(".bin.bak")
        if not backup_path.exists():
            shutil.copy2(images_bin, backup_path)

    write_images_binary(kept, images_bin)

    print(f"Scene train path: {scene_train_path}")
    print(f"Image files found: {len(existing)}")
    print(f"COLMAP images before: {len(images)}")
    print(f"COLMAP images after: {len(kept)}")
    print(f"Missing image entries removed: {len(missing)}")
    for name in missing[:20]:
        print(f"  removed: {name}")
    if len(missing) > 20:
        print(f"  ... {len(missing) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Prune COLMAP images.bin entries whose image files are missing from train/images."
    )
    parser.add_argument("--scene_train_path", required=True)
    parser.add_argument("--no_backup", action="store_true")
    args = parser.parse_args()
    prune_scene(args.scene_train_path, backup=not args.no_backup)


if __name__ == "__main__":
    main()
