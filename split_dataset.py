import os
import shutil
import random

SOURCE_DIR = "dataset/skincancer"
TARGET_DIR = "dataset/skincancer_split"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

classes = os.listdir(SOURCE_DIR)

for cls in classes:
    cls_path = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    train_end = int(TRAIN_RATIO * len(images))
    val_end = train_end + int(VAL_RATIO * len(images))

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        split_dir = os.path.join(TARGET_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)

        for file in files:
            src = os.path.join(cls_path, file)
            dst = os.path.join(split_dir, file)
            shutil.copy(src, dst)

print("✅ Dataset split into train / val / test successfully.")
