"""Download a small, resumable YOLO subset of RDD2022 from Hugging Face."""

from __future__ import annotations

import argparse
import random
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import HfApi
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPOSITORY = "dronefreak/RDD2022"
REVISION = "main"
CLASS_MAP = {0: 1, 1: 1, 2: 1, 3: 0}
CLASS_NAMES = {0: "pothole", 1: "crack"}
THREAD_LOCAL = threading.local()


def get_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        THREAD_LOCAL.session = session
    return session


def select_images(split: str, count: int, seed: int) -> list[str]:
    source_split = "valid" if split == "val" else "train"
    api = HfApi()
    image_files = [
        item.path
        for item in api.list_repo_tree(
            REPOSITORY,
            repo_type="dataset",
            revision=REVISION,
            path_in_repo=f"data/images/{source_split}",
            recursive=True,
        )
        if item.path.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        raise RuntimeError(f"No images found in the {source_split} split.")
    if len(image_files) <= count:
        return sorted(image_files)
    return sorted(random.Random(seed).sample(image_files, count))


def remap_label(contents: str, image_path: str) -> str:
    output = []
    for line_number, line in enumerate(contents.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label at {image_path}:{line_number}")
        try:
            source_class = int(parts[0])
            mapped_class = CLASS_MAP[source_class]
            coordinates = [float(value) for value in parts[1:]]
        except (ValueError, KeyError) as exc:
            raise ValueError(
                f"Unsupported label at {image_path}:{line_number}: {line!r}"
            ) from exc
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"Out-of-range box at {image_path}:{line_number}")
        output.append(f"{mapped_class} " + " ".join(f"{value:.6f}" for value in coordinates))
    return "\n".join(output) + ("\n" if output else "")


def download_pair(source_image: str, split: str, destination: Path) -> tuple[str, int]:
    filename = Path(source_image).name
    source_label = (
        source_image.replace("data/images/", "data/labels/", 1)
        .rsplit(".", 1)[0]
        + ".txt"
    )
    image_target = destination / "images" / split / filename
    label_target = destination / "labels" / split / f"{Path(filename).stem}.txt"
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)

    if image_target.is_file() and label_target.is_file():
        return "cached", label_target.stat().st_size

    base_url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/"
    session = get_session()
    temporary_image = image_target.with_suffix(image_target.suffix + ".part")
    with session.get(base_url + source_image, stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        with temporary_image.open("wb") as output:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if chunk:
                    output.write(chunk)
    temporary_image.replace(image_target)

    label_response = session.get(base_url + source_label, timeout=(20, 60))
    label_response.raise_for_status()
    mapped = remap_label(label_response.text, source_image)
    label_target.write_text(mapped, encoding="utf-8")
    return "downloaded", len(mapped)


def write_dataset_yaml(destination: Path) -> None:
    quoted_root = destination.resolve().as_posix().replace('"', '\\"')
    yaml = (
        f'path: "{quoted_root}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: pothole\n"
        "  1: crack\n"
    )
    (destination / "dataset.yaml").write_text(yaml, encoding="utf-8")


def write_attribution(destination: Path, train_count: int, val_count: int, seed: int) -> None:
    text = f"""RDD2022 pilot subset

Source: https://huggingface.co/datasets/{REPOSITORY}
Revision: {REVISION}
This is an unofficial YOLO-format redistribution, not the official RDD2022 release.
The source repository states the original imagery and annotations are CC BY-SA 4.0.
Review the source repository's current terms before redistributing this subset.

Original work: Deeksha Arya, Hiroya Maeda, Sanjay Kumar Ghosh, Durga Toshniwal,
and Yoshihide Sekimoto, "RDD2022: A multi-national image dataset for automatic
road damage detection," arXiv:2209.08538 (2022).
Original project: https://github.com/sekilab/RoadDamageDetector

Subset selection: random seed {seed}; {train_count} training images and
{val_count} validation images, selected from the source train and validation
splits. Original YOLO classes 0, 1, and 2 were merged as crack (class 1);
original class 3 was mapped to pothole (class 0). Other classes are not used.
"""
    (destination / "SOURCE.txt").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", type=int, default=2500)
    parser.add_argument("--val-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "rdd2022_yolo_subset",
    )
    args = parser.parse_args()
    if args.train_images < 1 or args.val_images < 1 or args.workers < 1:
        parser.error("Image counts and worker count must be positive.")

    train_files = select_images("train", args.train_images, args.seed)
    val_files = select_images("val", args.val_images, args.seed + 1)
    free_bytes = shutil.disk_usage(args.destination.parent).free
    minimum_free_bytes = 1_000_000_000
    if free_bytes < minimum_free_bytes:
        raise OSError(
            f"Only {free_bytes / 1e9:.1f} GB free; keep at least "
            f"{minimum_free_bytes / 1e9:.0f} GB available before downloading."
        )

    print(f"Source: https://huggingface.co/datasets/{REPOSITORY}")
    print(f"Selected {len(train_files)} train and {len(val_files)} validation images.")
    print(f"Destination: {args.destination.resolve()}")
    print(f"Available free space: {free_bytes / 1e9:.1f} GB")

    work = [(path, "train") for path in train_files]
    work.extend((path, "val") for path in val_files)
    downloaded = cached = label_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_pair, path, split, args.destination)
            for path, split in work
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            status, current_label_bytes = future.result()
            downloaded += status == "downloaded"
            cached += status == "cached"
            label_bytes += current_label_bytes
            if index % 100 == 0 or index == len(futures):
                print(
                    f"{index}/{len(futures)} filesets complete "
                    f"({downloaded} downloaded, {cached} already present)."
                )

    write_dataset_yaml(args.destination)
    write_attribution(args.destination, len(train_files), len(val_files), args.seed)
    print(f"YOLO dataset config: {args.destination / 'dataset.yaml'}")
    print(f"Mapped label text: {label_bytes / 1e6:.1f} MB")
    print("Class mapping: 3 RDD2022 crack classes -> crack; pothole -> pothole.")


if __name__ == "__main__":
    main()
