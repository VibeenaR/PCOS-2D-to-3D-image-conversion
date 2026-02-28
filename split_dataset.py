import os
import shutil
import random

train_dir = "dataset/train"
val_dir = "dataset/val"

classes = ["normal", "pcos"]

split_ratio = 0.2

print("\nSTEP 1: Moving all images back to train folder...")

# Move everything from val back to train
for cls in classes:

    train_cls = os.path.join(train_dir, cls)
    val_cls = os.path.join(val_dir, cls)

    os.makedirs(train_cls, exist_ok=True)
    os.makedirs(val_cls, exist_ok=True)

    val_images = os.listdir(val_cls)

    for img in val_images:

        src = os.path.join(val_cls, img)
        dst = os.path.join(train_cls, img)

        shutil.move(src, dst)

print("All images moved back to train.")

print("\nSTEP 2: Splitting dataset properly...")

# Now split correctly
for cls in classes:

    train_cls = os.path.join(train_dir, cls)
    val_cls = os.path.join(val_dir, cls)

    images = os.listdir(train_cls)

    total = len(images)

    val_count = int(total * split_ratio)

    random.shuffle(images)

    val_images = images[:val_count]

    for img in val_images:

        src = os.path.join(train_cls, img)
        dst = os.path.join(val_cls, img)

        shutil.move(src, dst)

    print(f"\nClass: {cls}")
    print(f"Train: {len(os.listdir(train_cls))}")
    print(f"Val: {len(os.listdir(val_cls))}")

print("\nDataset reset and split completed successfully.")