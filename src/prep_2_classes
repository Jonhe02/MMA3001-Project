
"""Prepare a two-class packaging defect dataset."""

from pathlib import Path
from shutil import copy2

" Change these to set directory and the name for the generated file"
ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "3 - raw_unaug"
DEST = ROOT / "data" / "4 - two_class_v2"

# Original class ID -> new class ID
CLASS_MAP = {
    1: 0,  # packaging-error
    3: 1,  # unsealed
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    """Create a new dataset without altering original data."""

    # Prevent accidental overwriting of an existing dataset.
    if DEST.exists():
        raise FileExistsError(
            f"{DEST} already exists. Review it before rerunning."
        )

    for split in ("train", "valid", "test"):
        image_dir = SOURCE / split / "images"
        label_dir = SOURCE / split / "labels"

        new_images = DEST / split / "images"
        new_labels = DEST / split / "labels"

        new_images.mkdir(parents=True, exist_ok=True)
        new_labels.mkdir(parents=True, exist_ok=True)

        kept = 0
        excluded = 0

        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label = label_dir / f"{image.stem}.txt"

            if not label.exists():
                raise FileNotFoundError(label)

            lines = [
                line.strip()
                for line in label.read_text().splitlines()
                if line.strip()
            ]

            # Conservatively exclude any image containing a
            # class outside the revised detection scope.
            if any(
                int(line.split()[0]) not in CLASS_MAP
                for line in lines
            ):
                excluded += 1
                continue

            # Preserve bounding boxes; remap class indices.
            converted = []
            for line in lines:
                parts = line.split()
                new_id = CLASS_MAP[int(parts[0])]
                converted.append(
                    " ".join([str(new_id), *parts[1:]])
                )

            copy2(image, new_images / image.name)

            (new_labels / label.name).write_text(
                "\n".join(converted) +
                ("\n" if converted else "")
            )

            kept += 1

        print(
            f"{split}: kept {kept} images, "
            f"excluded {excluded}"
        )

    (DEST / "data.yaml").write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 2\n"
        "names:\n"
        "  0: packaging-error\n"
        "  1: unsealed\n"
    )

    print(f"Two-class dataset created at: {DEST}")


if __name__ == "__main__":
    main()
