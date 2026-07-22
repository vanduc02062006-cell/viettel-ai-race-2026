import argparse
import math
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np


def decode_image(data, name):
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {name}")
    return image


def main():
    parser = argparse.ArgumentParser(description="Recompress a JPEG-only submission ZIP to fit an upload limit.")
    parser.add_argument("input_zip")
    parser.add_argument("output_zip")
    parser.add_argument("--quality", type=int, default=99)
    parser.add_argument("--auto_quality", action="store_true")
    parser.add_argument("--max_bytes", type=int, default=350_000_000)
    args = parser.parse_args()

    input_path = Path(args.input_zip)
    output_path = Path(args.output_zip)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    total_squared_error = 0.0
    total_values = 0
    per_image_psnr = []
    expected_names = []

    chosen_quality = args.quality
    if args.auto_quality:
        with zipfile.ZipFile(input_path, "r") as source:
            all_infos = [info for info in source.infolist() if not info.is_dir()]
            stride = max(1, len(all_infos) // 80)
            sample_infos = all_infos[::stride]
            sample_images = [decode_image(source.read(info), info.filename) for info in sample_infos]
            sample_source_bytes = sum(info.file_size for info in sample_infos)
            total_source_bytes = sum(info.file_size for info in all_infos)
        estimates = {}
        for quality in range(99, 69, -1):
            params = [
                int(cv2.IMWRITE_JPEG_QUALITY),
                quality,
                int(cv2.IMWRITE_JPEG_PROGRESSIVE),
                1,
            ]
            sample_encoded_bytes = sum(
                len(cv2.imencode(".jpg", image, params)[1]) for image in sample_images
            )
            estimate = int(total_source_bytes * sample_encoded_bytes / sample_source_bytes * 1.06)
            estimates[quality] = estimate
            if estimate <= args.max_bytes:
                chosen_quality = quality
                break
        else:
            raise RuntimeError(f"Could not estimate a quality that fits {args.max_bytes} bytes: {estimates}")
        print(f"Auto quality estimates (bytes): {estimates}")
        print(f"Selected JPEG quality: {chosen_quality}")

    encode_params = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        chosen_quality,
        int(cv2.IMWRITE_JPEG_PROGRESSIVE),
        1,
    ]

    try:
        with zipfile.ZipFile(input_path, "r") as source, zipfile.ZipFile(
            temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as target:
            source_infos = [info for info in source.infolist() if not info.is_dir()]
            for index, info in enumerate(source_infos, start=1):
                archive_path = PurePosixPath(info.filename)
                if len(archive_path.parts) != 2 or archive_path.suffix.lower() not in {".jpg", ".jpeg"}:
                    raise ValueError(f"Unexpected submission path: {info.filename}")

                original = decode_image(source.read(info), info.filename)
                ok, encoded = cv2.imencode(".jpg", original, encode_params)
                if not ok:
                    raise ValueError(f"Could not encode {info.filename}")
                recompressed = decode_image(encoded.tobytes(), info.filename)
                if recompressed.shape != original.shape:
                    raise ValueError(f"Shape changed for {info.filename}")

                difference = original.astype(np.float32) - recompressed.astype(np.float32)
                squared_error = float(np.square(difference).sum())
                value_count = int(difference.size)
                total_squared_error += squared_error
                total_values += value_count
                mse = squared_error / value_count
                per_image_psnr.append(float("inf") if mse == 0 else 10.0 * math.log10((255.0**2) / mse))

                zip_info = zipfile.ZipInfo(info.filename, date_time=(2026, 7, 20, 0, 0, 0))
                zip_info.compress_type = zipfile.ZIP_DEFLATED
                zip_info.external_attr = 0o100644 << 16
                target.writestr(zip_info, encoded.tobytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
                expected_names.append(info.filename)
                if index % 50 == 0 or index == len(source_infos):
                    print(f"Encoded {index}/{len(source_infos)} images")

        size = temp_path.stat().st_size
        if size > args.max_bytes:
            raise RuntimeError(f"Output is {size} bytes, above limit {args.max_bytes} bytes")

        with zipfile.ZipFile(temp_path, "r") as result:
            result_infos = [info for info in result.infolist() if not info.is_dir()]
            result_names = [info.filename for info in result_infos]
            if result_names != expected_names:
                raise RuntimeError("Archive paths changed during recompression")
            bad_member = result.testzip()
            if bad_member is not None:
                raise RuntimeError(f"CRC failure: {bad_member}")
            for info in result_infos:
                decode_image(result.read(info), info.filename)

        if output_path.exists():
            output_path.unlink()
        temp_path.replace(output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    global_mse = total_squared_error / total_values
    global_psnr = float("inf") if global_mse == 0 else 10.0 * math.log10((255.0**2) / global_mse)
    print(f"Created: {output_path}")
    print(f"Images: {len(expected_names)}")
    print(f"Bytes: {output_path.stat().st_size}")
    print(f"JPEG quality: {chosen_quality}")
    print(f"Global PSNR versus quality-100 ZIP: {global_psnr:.4f} dB")
    print(f"Mean per-image PSNR versus quality-100 ZIP: {np.mean(per_image_psnr):.4f} dB")


if __name__ == "__main__":
    main()
