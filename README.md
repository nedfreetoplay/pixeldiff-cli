# Pixel Diff CLI

Удаляет одинаковые пиксели между двумя изображениями и сохраняет 
результат в формате PNG с альфа-каналом. Идентичные пиксели 
становятся прозрачными, отличающиеся — остаются видимыми. 
Полезно для визуализации изменений между почти одинаковыми 
изображениями (например, анимации, мерцания, эффектов и т.п.).

Лично использую для создания анимированного фона в играх, 
который жрет мало памяти на диски.
Вместо того чтобы хранить полноценные изображения 
пр: 1.png, 2.png, 3.png... лучше будет создать главный фон, 
а вся анимация фона будет происходить через переключение 
картинок на втором слое.

## 📦 Установка

### 1. Клонируйте репозиторий:

```bash
git clone https://github.com/nedfreetoplay/pixeldiff-cli.git
cd pixeldiff-cli
```

### 2. Установите зависимости (в виртуальном окружении):

``` bash
python setup_venv.py
```

### 3. Активируйте виртуальное окружение:

- Windows:

```bash
.venv\Scripts\activate
```

- Linux/macOS:

``` bash
source .venv/bin/activate
```

## 🚀 Использование

Скрипт сравнивает базовое изображение с каждым файлом в папке `input` и сохраняет результат в папку `output`.

```bash
python script.py [input=path] [output=path] [base=image.png]
```
🔧 Аргументы:

| Аргумент   | Описание | Значение по умолчанию |
|------------|---|---------|
| `input`  | Папка с изображениями для сравнения	 | `input/` |
| `output` | Папка, куда сохранить результаты | `output/` |
| `base`   | 	Базовое изображение для сравнения | `base_image.png` |

## 💡 Примеры:
```bash
python script.py
```
Сравнивает `base_image.png` со всеми PNG/JPG в `input/`, результат в `output/`.
```bash
python script.py input=my_frames base=frame001.png output=diffs
```
Сравнивает `frame001.png` со всеми изображениями из `my_frames/`, сохраняет в `diffs/`.

## 📷 Пример результата

| 🟥 Исходное (base)      | 🔵 Сравниваемое (diff)   | 🟩 Результат (только отличия) |
|-------------------------|--------------------------|-------------------------------|
| ![](examples/base1.png) | ![](examples/input1.png) | ![](examples/output1.png)     |
| ![](examples/base2.png) | ![](examples/input2.png) | ![](examples/output2.png)     |
| ![](examples/base3.png) | ![](examples/input3.png) | ![](examples/output3.png)     |

## 📝 Лицензия

This project is released into the public domain under [The Unlicense](https://unlicense.org/).
