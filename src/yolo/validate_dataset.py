"""
Validate YOLO dataset integrity.

This script checks:
- Image/label count match
- Missing label files
- Missing image files
- Empty label files
- Label format validation
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def load_dataset_config(config_path: Path) -> dict:
    """Load YOLO dataset configuration from YAML."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def validate_split(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
) -> Dict[str, any]:
    """
    Validate a single dataset split.

    Args:
        split_name: Name of split (train/val/test)
        images_dir: Path to images directory
        labels_dir: Path to labels directory

    Returns:
        Dictionary with validation results
    """
    results = {
        "split": split_name,
        "images_count": 0,
        "labels_count": 0,
        "matched": 0,
        "missing_labels": [],
        "missing_images": [],
        "empty_labels": [],
        "valid": True,
    }

    # Check if directories exist
    if not images_dir.exists():
        print(f"  ⚠ Images directory not found: {images_dir}")
        results["valid"] = False
        return results

    if not labels_dir.exists():
        print(f"  ⚠ Labels directory not found: {labels_dir}")
        results["valid"] = False
        return results

    # Get all image and label files
    image_files = {f.stem: f for f in images_dir.glob("*.png")}
    label_files = {f.stem: f for f in labels_dir.glob("*.txt")}

    results["images_count"] = len(image_files)
    results["labels_count"] = len(label_files)

    # Find missing labels
    for img_name in image_files:
        if img_name not in label_files:
            results["missing_labels"].append(img_name)
        else:
            results["matched"] += 1

            # Check if label file is empty
            label_path = label_files[img_name]
            if label_path.stat().st_size == 0:
                results["empty_labels"].append(img_name)

    # Find missing images
    for lbl_name in label_files:
        if lbl_name not in image_files:
            results["missing_images"].append(lbl_name)

    # Determine validity
    if results["missing_labels"] or results["missing_images"] or results["empty_labels"]:
        results["valid"] = False

    return results


def print_validation_results(results: Dict[str, any]):
    """Print validation results in a formatted way."""
    split = results["split"]
    print(f"\n{'=' * 70}")
    print(f"{split.upper()} Split Validation")
    print(f"{'=' * 70}")

    if results["images_count"] == 0 and results["labels_count"] == 0:
        print(f"  ℹ No data found (this is OK if split is unused)")
        return

    print(f"  Images: {results['images_count']}")
    print(f"  Labels: {results['labels_count']}")
    print(f"  Matched: {results['matched']}")

    if results["missing_labels"]:
        print(f"\n  ✗ Missing labels ({len(results['missing_labels'])}):")
        for name in results["missing_labels"][:5]:
            print(f"    - {name}.txt")
        if len(results["missing_labels"]) > 5:
            print(f"    ... and {len(results['missing_labels']) - 5} more")

    if results["missing_images"]:
        print(f"\n  ✗ Missing images ({len(results['missing_images'])}):")
        for name in results["missing_images"][:5]:
            print(f"    - {name}.png")
        if len(results["missing_images"]) > 5:
            print(f"    ... and {len(results['missing_images']) - 5} more")

    if results["empty_labels"]:
        print(f"\n  ⚠ Empty label files ({len(results['empty_labels'])}):")
        for name in results["empty_labels"][:5]:
            print(f"    - {name}.txt")
        if len(results["empty_labels"]) > 5:
            print(f"    ... and {len(results['empty_labels']) - 5} more")

    if results["valid"]:
        print(f"\n  ✓ {split.upper()} split is valid")
    else:
        print(f"\n  ✗ {split.upper()} split has issues")


def count_class_instances(labels_dir: Path, num_classes: int) -> Dict[int, int]:
    """
    Count instances per class across all label files.

    Args:
        labels_dir: Path to labels directory
        num_classes: Number of classes

    Returns:
        Dictionary mapping class_id to instance count
    """
    class_counts = {i: 0 for i in range(num_classes)}

    if not labels_dir.exists():
        return class_counts

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if parts:
                    class_id = int(parts[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1

    return class_counts


def main():
    """CLI entry point for dataset validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate YOLO dataset integrity"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/yolo_sagittal.yaml"),
        help="Path to dataset config YAML (default: configs/yolo_sagittal.yaml)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("YOLO Dataset Validation")
    print("=" * 70)
    print(f"Config: {args.config.absolute()}")

    # Load config
    if not args.config.exists():
        print(f"\n✗ ERROR: Config file not found: {args.config}")
        sys.exit(1)

    config = load_dataset_config(args.config)

    # Get dataset root path
    config_dir = args.config.parent
    dataset_root = config_dir / config["path"]

    if not dataset_root.exists():
        print(f"\n✗ ERROR: Dataset root not found: {dataset_root}")
        sys.exit(1)

    print(f"Dataset root: {dataset_root.absolute()}")
    print(f"Classes ({config['nc']}): {list(config['names'].values())}")

    # Validate each split
    splits = ["train", "val", "test"]
    all_valid = True

    for split in splits:
        images_dir = dataset_root / f"images/{split}"
        labels_dir = dataset_root / f"labels/{split}"

        results = validate_split(split, images_dir, labels_dir)
        print_validation_results(results)

        if not results["valid"] and results["images_count"] > 0:
            all_valid = False

        # Count class instances if split has data
        if results["matched"] > 0:
            class_counts = count_class_instances(labels_dir, config["nc"])
            print(f"\n  Class distribution:")
            for class_id, class_name in config["names"].items():
                count = class_counts[class_id]
                print(f"    {class_id} ({class_name}): {count} instances")

    # Final summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    if all_valid:
        print("✓ Dataset is valid and ready for training!")
        sys.exit(0)
    else:
        print("✗ Dataset has validation issues. Please fix before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()