import cv2 as cv2
import numpy as np
import os
import sys
from typing import Dict

# noinspection GrazieInspection
def delete_identical_pixels(image1_path: str, image2_path: str, output_path: str) -> None:
    """
    Сравнивает два изображения и сохраняет новое изображение, в котором остаются только отличающиеся пиксели.
    Все одинаковые пиксели становятся полностью прозрачными.

    Параметры:
    - image1_path (str): Путь к первому изображению (базовому).
    - image2_path (str): Путь ко второму изображению (сравниваемому).
    - output_path (str): Путь для сохранения результирующего PNG с альфа-каналом.

    Исключения:
    - ValueError: если одно из изображений не удалось загрузить.
    """
    img1 = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(image2_path, cv2.IMREAD_UNCHANGED)

    if img1 is None or img2 is None:
        raise ValueError("One of the images did not load properly.")

    # Приводим к одинаковому размеру
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # Берем только первые 3 канала (RGB/BGR)
    img1_rgb = img1[:, :, :3] if img1.shape[2] >= 3 else img1
    img2_rgb = img2[:, :, :3] if img2.shape[2] >= 3 else img2

    # Находим, где пиксели отличаются
    different_pixels = np.any(img1_rgb != img2_rgb, axis=2)

    # Альфа-канал: 255 где различия, 0 где одинаково
    alpha = np.where(different_pixels, 255, 0).astype(np.uint8)

    # Цветовая часть из второго изображения
    result_rgb = img2_rgb.copy()

    # Добавляем альфа-канал
    result_rgba = cv2.merge((result_rgb, alpha))

    # Сохраняем
    cv2.imwrite(output_path, result_rgba)

def parse_args() -> Dict[str, str]:
    args = sys.argv[1:]
    params = {
        "input": "input",
        "output": "output",
        "base": "base_image.png"
    }
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            if key in params:
                params[key] = value
    return params

def main():
    params = parse_args()
    input_dir = params["input"]
    output_dir = params["output"]
    base_image = params["base"]

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        output_path = os.path.join(output_dir, filename)
        delete_identical_pixels(base_image, input_path, output_path)

if __name__ == '__main__':
    main()
