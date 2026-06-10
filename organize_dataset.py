import os
import shutil
import pandas as pd

# Paths
BASE_DIR = "dataset"
IMG_DIRS = [
    "dataset/HAM10000_images_part_1",
    "dataset/HAM10000_images_part_2"
]
CSV_PATH = "dataset/HAM10000_metadata.csv"
OUTPUT_DIR = "dataset/skincancer"

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read CSV
df = pd.read_csv(CSV_PATH)

# Create class folders
classes = df['dx'].unique()
for cls in classes:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# Move images
moved = 0
for _, row in df.iterrows():
    image_id = row['image_id']
    label = row['dx']
    image_name = image_id + ".jpg"

    for img_dir in IMG_DIRS:
        src = os.path.join(img_dir, image_name)
        if os.path.exists(src):
            dst = os.path.join(OUTPUT_DIR, label, image_name)
            shutil.copy(src, dst)
            moved += 1
            break

print(f"✅ Organization complete. {moved} images sorted into classes.")
