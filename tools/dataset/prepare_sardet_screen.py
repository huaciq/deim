"""Freeze SARDet-100K screening annotations without changing any boxes.

Reuse the existing 10% training image list, and sample only from the official
validation split. Images and all annotations belonging to them stay together.
Requires Python's standard library only.
"""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from statistics import median
import subprocess


def select_images(data, image_ids):
    return {
        **data,
        "images": [im for im in data["images"] if im["id"] in image_ids],
        "annotations": [a for a in data["annotations"] if a["image_id"] in image_ids],
    }


def size_bin(area):
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def density_bin(count):
    return "0" if count == 0 else "1" if count == 1 else "2-5" if count <= 5 else "6-15" if count <= 15 else "16+"


def image_groups(data, input_size):
    annotations = defaultdict(list)
    for annotation in data["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    groups = defaultdict(list)
    for image in data["images"]:
        objects = annotations[image["id"]]
        classes = tuple(sorted({a["category_id"] for a in objects}))
        areas = [a["bbox"][2] * a["bbox"][3] * input_size ** 2 /
                 (image["width"] * image["height"]) for a in objects]
        scale = size_bin(median(areas)) if areas else "empty"
        edge = max(image["width"], image["height"])
        resolution = "<=256" if edge <= 256 else "257-512" if edge <= 512 else "513-800" if edge <= 800 else ">800"
        groups[(classes, scale, density_bin(len(objects)), resolution)].append(image["id"])
    return groups


def stratified_ids(data, count, seed, input_size):
    """Proportional largest-remainder allocation, then seeded image sampling."""
    if not 0 < count <= len(data["images"]):
        raise ValueError("Validation image count must be positive and fit the official split")
    rng = random.Random(seed)
    groups = image_groups(data, input_size)
    keys = sorted(groups)
    exact = {key: count * len(groups[key]) / len(data["images"]) for key in keys}
    quota = {key: math.floor(exact[key]) for key in keys}
    # Seeded tie order prevents an arbitrary preference for low category IDs.
    tie_order = {key: rng.random() for key in keys}
    order = sorted(keys, key=lambda key: (-(exact[key] - quota[key]), tie_order[key]))
    for key in order[:count - sum(quota.values())]:
        quota[key] += 1
    selected = set()
    for key in keys:
        selected.update(rng.sample(sorted(groups[key]), quota[key]))
    return selected, len(groups)


def summarize(data, input_size):
    images = {im["id"]: im for im in data["images"]}
    per_image = Counter()
    class_images = defaultdict(set)
    class_objects = Counter()
    native_sizes = Counter()
    input_sizes = Counter()
    for annotation in data["annotations"]:
        image = images[annotation["image_id"]]
        category = annotation["category_id"]
        class_objects[category] += 1
        class_images[category].add(image["id"])
        per_image[image["id"]] += 1
        # bbox geometry is used for sampling; original area and bbox are retained.
        area = annotation["bbox"][2] * annotation["bbox"][3]
        native_sizes[size_bin(annotation.get("area", area))] += 1
        input_sizes[size_bin(area * input_size ** 2 / (image["width"] * image["height"]))] += 1
    return {
        "images": len(images),
        "annotations": len(data["annotations"]),
        "objects_per_class": {c["name"]: class_objects[c["id"]] for c in data["categories"]},
        "images_per_class": {c["name"]: len(class_images[c["id"]]) for c in data["categories"]},
        "object_sizes_native_coco": dict(native_sizes),
        "object_sizes_at_input": dict(input_sizes),
        "image_density": dict(Counter(density_bin(per_image[i]) for i in images)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/root/zxy/TCRA-SAR/datasets/SARDet_100K"))
    parser.add_argument("--train-subset", type=Path,
                        help="Existing 10%% training JSON; only its image IDs are reused")
    parser.add_argument("--val-images", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/root/zxy/experiments/data_subsets/sardet100k_screen_v1"))
    args = parser.parse_args()
    root = args.dataset_root
    annotations = root / "Annotations"
    subset_path = args.train_subset or annotations / "train_10pct_seed42.json"
    train = json.loads((annotations / "train.json").read_text(encoding="utf-8"))
    val = json.loads((annotations / "val.json").read_text(encoding="utf-8"))
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    categories = sorted(train["categories"], key=lambda category: category["id"])
    if [c["id"] for c in categories] != list(range(6)) or val["categories"] != train["categories"]:
        raise ValueError("Expected the same six zero-based SARDet categories in both official splits")
    train_ids = {im["id"] for im in subset["images"]}
    if not train_ids <= {im["id"] for im in train["images"]}:
        raise ValueError("Training subset contains images outside the official training split")
    # Fetch every original annotation for each selected image, including crowd flags.
    selected_train = select_images(train, train_ids)
    val_ids, strata = stratified_ids(val, args.val_images, args.seed, args.input_size)
    selected_val = select_images(val, val_ids)
    for split, data in (("train", selected_train), ("val", selected_val)):
        missing = [im["file_name"] for im in data["images"]
                   if not (root / "JPEGImages" / split / im["file_name"]).is_file()]
        if missing:
            raise FileNotFoundError(f"{split}: {len(missing)} missing images; first: {missing[0]}")
    # File names are globally assigned in this release; split-local COCO IDs repeat.
    if {im["file_name"] for im in selected_train["images"]} & {im["file_name"] for im in selected_val["images"]}:
        raise ValueError("Selected official splits contain overlapping image file names")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           cwd=Path(__file__).resolve().parents[2]).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    stats = {
        "version": "sardet100k_screen_v1",
        "generation_git_revision": revision,
        "dataset_root": str(root),
        "training_image_list_source": str(subset_path),
        "validation_seed": args.seed,
        "input_size_for_stratification": args.input_size,
        "validation_strata": strata,
        "validation_sampling": "proportional image-level class-set x median input box area x density x resolution",
        "source_dataset_labels_available": False,
        "original_train": summarize(train, args.input_size),
        "selected_train": summarize(selected_train, args.input_size),
        "original_val": summarize(val, args.input_size),
        "selected_val": summarize(selected_val, args.input_size),
    }
    # A versioned output directory keeps previous experiment annotations intact.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in (("train.json", selected_train), ("val.json", selected_val),
                       ("stats.json", stats),
                       ("image_ids.json", {"train": sorted(train_ids), "val": sorted(val_ids)})):
        (args.output_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir),
                      "train": stats["selected_train"], "val": stats["selected_val"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
