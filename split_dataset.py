from pathlib import Path
import random
import shutil

#configuration

SOURCE_DIR = Path("data/flower_photos")
OUTPUT_DIR = Path("data")
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

#Random Generator

random.seed(SEED)

#Find classes

classes = sorted(
    folder.name
    for folder in SOURCE_DIR.iterdir()
    if folder.is_dir()
)

print("Classes:")
for class_name in classes:
    print(f"  {class_name}")

#create output directories

for split in ['train','val','test']:
    for class_name in classes:
        (OUTPUT_DIR/ split / class_name).mkdir(
            parents=True,
            exist_ok=True,
        )
#split each class

for class_name in classes:
    class_dir = SOURCE_DIR / class_name
    images = sorted(
        file 
        for file in class_dir.iterdir()
        if file.is_file()
    )
    random.shuffle(images)
    total  = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    print(f"\n{class_name}:")
    print(f"  total: {total}")
    print(f"  train: {len(train_images)}")
    print(f"  val:   {len(val_images)}")
    print(f"  test:  {len(test_images)}")

    # Copy files

    for image in train_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "train" / class_name / image.name,
        )

    for image in val_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "val" / class_name / image.name,
        )

    for image in test_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "test" / class_name / image.name,
        )


print("\nDataset split complete.")