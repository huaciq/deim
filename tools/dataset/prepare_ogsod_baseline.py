"""Validate OGSOD COCO data and write derived zero-based annotations.

Original annotations and images are never modified. Requires only Python's
standard library; run on the Linux server before starting the baseline.
"""

import argparse
import json
from pathlib import Path


def prepare(source, image_root, mapping):
    data = json.loads(source.read_text(encoding="utf-8"))
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    if categories != {old: name for old, (_, name) in mapping.items()}:
        raise ValueError(f"{source}: train/validation categories differ")
    if not data["images"] or not data["annotations"]:
        raise ValueError(f"{source}: empty images or annotations")
    image_ids = set()
    paths = set()
    for image in data["images"]:
        if image["id"] in image_ids:
            raise ValueError(f"{source}: duplicate image id {image['id']}")
        image_ids.add(image["id"])
        relative = Path(image["file_name"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe image name: {relative}")
        target = image_root / relative
        if not target.is_file():
            # Some COCO exports retain a train/test prefix. Resolve against the
            # supplied split directory while keeping actual images untouched.
            candidates = [image_root / Path(*relative.parts[i:])
                          for i in range(1, len(relative.parts))]
            candidates = [path for path in candidates if path.is_file()]
            if len(candidates) != 1:
                raise FileNotFoundError(f"Cannot uniquely resolve {target}")
            target = candidates[0]
        image["file_name"] = target.relative_to(image_root).as_posix()
        paths.add(target.resolve())
    annotation_ids = set()
    for annotation in data["annotations"]:
        if annotation["id"] in annotation_ids:
            raise ValueError(f"{source}: duplicate annotation id {annotation['id']}")
        annotation_ids.add(annotation["id"])
        if annotation["image_id"] not in image_ids:
            raise ValueError(f"{source}: annotation references unknown image")
        annotation["category_id"] = mapping[annotation["category_id"]][0]
    for category in data["categories"]:
        category["id"] = mapping[category["id"]][0]
    return data, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/data2/zxy/sar/datasets/OGSOD-1.0/sar"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/home/zxy/sar/experiments/deim_dfine_s_ogsod/prepared_data"))
    args = parser.parse_args()
    root = args.dataset_root
    train_source = root / "RTDETR_COCO/train.json"
    val_source = root / "RTDETR_COCO/val.json"
    categories = json.loads(train_source.read_text(encoding="utf-8"))["categories"]
    if len(categories) != 3 or len({cat["id"] for cat in categories}) != 3:
        raise ValueError("Baseline requires exactly three unique categories")
    mapping = {cat["id"]: (index, cat["name"])
               for index, cat in enumerate(sorted(categories, key=lambda cat: cat["id"]))}
    train, train_paths = prepare(train_source, root / "images/train", mapping)
    val, val_paths = prepare(val_source, root / "images/test", mapping)
    if train_paths & val_paths:
        raise ValueError("Train/validation resolve to overlapping image paths")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {args.output_dir / "train.json": train,
               args.output_dir / "val.json": val}
    serialized = {path: json.dumps(data, ensure_ascii=False, indent=2) + "\n"
                  for path, data in outputs.items()}
    # Reruns are safe only when inputs produce identical derived annotations.
    for path, text in serialized.items():
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to replace different annotations: {path}")
    for path, text in serialized.items():
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(text)
        print(f"{path}: {len(outputs[path]['images'])} images, "
              f"{len(outputs[path]['annotations'])} annotations")
    print("Category mapping (original -> training ID, name):", mapping)
    print("Preparation passed. Original annotations are unchanged.")


if __name__ == "__main__":
    main()
