from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


data_dir = Path("data/flower_photos")

classes = sorted(
    folder.name
    for folder in data_dir.iterdir()
    if folder.is_dir()
)


fig, axes = plt.subplots(
    1,
    len(classes),
    figsize=(15, 4)
)


for ax, class_name in zip(axes, classes):

    class_dir = data_dir / class_name

    image_path = next(
        (
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        ),
        None,
    )

    if image_path is None:
        raise FileNotFoundError(f"No supported image found in {class_dir}")

    image = Image.open(image_path)

    ax.imshow(image)
    ax.set_title(class_name)
    ax.axis("off")


plt.tight_layout()
plt.show()