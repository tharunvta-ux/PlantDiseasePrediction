"""
Dataset Inspector Utility

This module inspects the PlantVillage dataset and prints
useful statistics before model training.

Author: Your Name
Project: Plant Disease Detection System
"""

from pathlib import Path
from collections import Counter
from PIL import Image


# =====================================================
# Dataset Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "dataset" / "raw" / "PlantVillage"

TRAIN_PATH = DATASET_PATH / "train"
VAL_PATH = DATASET_PATH / "val"


# Supported image formats
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# =====================================================
# Count Images
# =====================================================

def count_images(folder_path: Path) -> tuple[int, dict]:
    """
    Count the number of valid image files in every class.

    Returns:
        total_images (int)
        class_counts (dict)
    """

    class_counts = {}
    total_images = 0

    for class_folder in sorted(folder_path.iterdir()):

        if not class_folder.is_dir():
            continue

        image_count = sum(
            1
            for file in class_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        )

        class_counts[class_folder.name] = image_count
        total_images += image_count

    return total_images, class_counts


# =====================================================
# Analyze Images
# =====================================================

def analyze_images(folder_path: Path):
    """
    Analyze image formats, image resolutions,
    and detect corrupted images.

    Returns:
        formats
        widths
        heights
        corrupted_images
    """

    formats = Counter()

    widths = []
    heights = []

    corrupted_images = []

    for class_folder in sorted(folder_path.iterdir()):

        if not class_folder.is_dir():
            continue

        for image_path in class_folder.iterdir():

            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in VALID_EXTENSIONS
            ):
                continue

            try:

                with Image.open(image_path) as image:

                    # Count image formats
                    formats[image.format.upper()] += 1

                    width, height = image.size

                    widths.append(width)
                    heights.append(height)

            except Exception:

                corrupted_images.append(str(image_path))

    return formats, widths, heights, corrupted_images


# =====================================================
# Inspect Dataset Split
# =====================================================

def inspect_split(split_name: str, split_path: Path):
    """
    Inspect one dataset split.
    """

    print("\n")
    print("=" * 70)
    print(f"{split_name.upper()} DATASET")
    print("=" * 70)

    if not split_path.exists():
        print(f"{split_name} folder not found.\n")
        return

    total_images, class_counts = count_images(split_path)

    formats, widths, heights, corrupted = analyze_images(split_path)

    print(f"Number of Classes : {len(class_counts)}")
    print(f"Total Images      : {total_images}")

    print("\nImage Analysis")
    print("-" * 70)

    print(f"Image Formats     : {dict(formats)}")

    if widths:

        print(f"Minimum Width     : {min(widths)}")
        print(f"Maximum Width     : {max(widths)}")

        print(f"Minimum Height    : {min(heights)}")
        print(f"Maximum Height    : {max(heights)}")

        avg_width = sum(widths) / len(widths)
        avg_height = sum(heights) / len(heights)

        print(f"Average Width     : {avg_width:.2f}")
        print(f"Average Height    : {avg_height:.2f}")

    print(f"Corrupted Images  : {len(corrupted)}")

    print("\nImages Per Class")
    print("-" * 70)

    for class_name, count in class_counts.items():
        print(f"{class_name:<50} {count}")

    print()


# =====================================================
# Main
# =====================================================

def main():
    """
    Run dataset inspection.
    """

    print("\n")
    print("=" * 70)
    print("PLANTVILLAGE DATASET INSPECTION")
    print("=" * 70)

    inspect_split("Train", TRAIN_PATH)
    inspect_split("Validation", VAL_PATH)


# =====================================================
# Entry Point
# =====================================================

if __name__ == "__main__":
    main()