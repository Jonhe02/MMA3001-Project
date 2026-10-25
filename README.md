# MMA3001 — Pork Packaging Defect Detection

## Project overview

This repository contains a Python workflow for inspecting images of packaged pork rashers with YOLO11 object detection. Its purpose is to investigate whether a camera-based system can **locate and identify packaging defects** to support quality inspection.

The [original Roboflow dataset](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging) contains five annotated classes:

- `loose-meat`
- `packaging-error`
- `twisted-meat`
- `unsealed`
- `wrinkle`

The **final project scope uses only two classes**, `packaging-error` and `unsealed`, because the other classes had too few annotated examples for meaningful evaluation. The models return defect classes, confidence scores and bounding boxes; they do not explicitly classify packages as *safe* or *acceptable*.

## Repository structure

```text
MMA3001-Project/
├── src/                 Python scripts for data preparation, training and evaluation
├── data/                Local datasets in YOLO format (not committed to Git)
├── weights/             Pretrained YOLO weights (weight files not committed)
├── runs/                Training and validation checkpoints/plots (not committed)
├── results/             Audit reports, model comparisons and final-test results
├── docs/                Generated HTML documentation
├── tests/               Directory reserved for automated tests
├── requirements.txt     Python dependencies
├── README.md            Repository setup and usage guide
├── .gitignore           Excluded files and folders
└── LICENSE              Software licence
```

### Source scripts

| File | What it does |
| --- | --- |
| [`experiment_config.py`](src/experiment_config.py) | Maps dataset numbers to folders and standardises experiment names and model paths. |
| [`prep_2_classes.py`](src/prep_2_classes.py) | Converts the rebalanced five-class dataset into the two-class dataset by remapping labels and excluding images with out-of-scope annotations. |
| [`data_check.py`](src/data_check.py) | Checks image files, annotations, class counts and candidate duplicate images; does not modify datasets. |
| [`train.py`](src/train.py) | Fine-tunes YOLO11n or YOLO11s from pretrained weights. |
| [`evaluate.py`](src/evaluate.py) | Evaluates a trained checkpoint on validation or test data and saves metrics. |
| [`compare_models.py`](src/compare_models.py) | Compares saved YOLO11n and YOLO11s validation summaries. |
| [`optimise_inference.py`](src/optimise_inference.py) | Tests image sizes and confidence thresholds using existing checkpoints; no retraining. |
| [`test.py`](src/test.py) | Evaluates the selected final configuration on the held-out test split. |

Run all commands below **from the repository root**.

## Requirements

- **Python 3.12** (version used for development)
- Dependencies in [`requirements.txt`](requirements.txt), including the project's main libraries: **Ultralytics, PyTorch, torchvision, OpenCV, NumPy, pandas, Matplotlib and PyYAML**
- **pdoc** to regenerate the optional HTML API documentation
- **Optional NVIDIA CUDA-capable GPU** for acceleration. The supplied training and evaluation configurations use `device=0`; a CPU-only machine requires changing that setting to `device="cpu"` and may run much more slowly.

## Installation

### 1. Clone the repository

Replace the example GitHub address with this repository's actual URL:

```powershell
git clone https://github.com/Jonhe02/MMA3001-Project
cd MMA3001-Project
```

### 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For GPU training, install a **CUDA-enabled PyTorch build** compatible with your hardware. Verify CUDA availability with:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. Download and organise the dataset

Download the [Pork Rasher Error (Packaging) dataset from Roboflow](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging) in a YOLO-compatible object-detection format. The project uses these numbered folders:

| Dataset number | Local folder | Purpose |
| --- | --- | --- |
| 1 | `data/1 - five_class/` | Original augmented, five-class export |
| 2 | `data/2 - two_class/` | Earlier two-class conversion |
| 3 | `data/3 - raw_unaug/` | Five-class source export rebalanced to approximately 70% training / 15% validation / 15% testing, without additional offline augmentation |
| 4 | `data/4 - two_class_v2/` | Final two-class dataset generated from dataset 3 |

Each export should contain `train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels` and `data.yaml`.

To recreate **dataset 4**, place the matching rebalanced five-class export in dataset 3, verify the original class IDs in its `data.yaml`, then run:

```powershell
python src/prep_2_classes.py
python src/data_check.py --dataset 4
```

`prep_2_classes.py` will not overwrite an existing dataset 4 folder. Datasets are excluded from Git, so the same Roboflow export and preprocessing settings are necessary for exact reproduction.

## Usage and workflow

### 1. Train the models

```powershell
python src/train.py --dataset 4 --model n
python src/train.py --dataset 4 --model s
```

`n` selects YOLO11n (nano); `s` selects YOLO11s (small). Training checkpoints are saved under `runs/`, typically at `runs/yolo11n_2class_v2/weights/best.pt` and `runs/yolo11s_2class_v2/weights/best.pt`.

### 2. Evaluate and compare trained models

```powershell
python src/evaluate.py --dataset 4 --model n --run yolo11n_2class_v2
python src/evaluate.py --dataset 4 --model s --run yolo11s_2class_v2
python src/compare_models.py --dataset 4
```

Validation outputs are saved under `runs/`; the comparison table is saved under `results/model_comparison/`.

### 3. Investigate inference settings

```powershell
python src/optimise_inference.py --dataset 4 --models n s --sizes 320 480 640 --conf 0.10 0.25 0.40 0.60 --compute-map
```

To reproduce the refined experiment with a separate output folder:

```powershell
python src/optimise_inference.py --dataset 4 --models s --sizes 480 640 --conf 0.15 0.20 0.25 0.30 0.35 --compute-map --output-name 2class_v2_refined
```

These commands compare existing trained checkpoints on validation images. The results are saved under `results/inference_optimisation/`.

### 4. Run the selected model on the held-out test set

```powershell
python src/test.py
```

The final-test script uses YOLO11s at **640px** and **confidence 0.20**; it saves reports under `results/final_test/`. If the script in your checkout is named `final_test.py`, run `python src/final_test.py` instead.

### 5. Predict defects in a new image

After training, you can perform inference on an image without modifying the model:

```powershell
yolo predict model="runs/yolo11s_2class_v2/weights/best.pt" source="path/to/image.jpg" imgsz=640 conf=0.20 save=True
```

Replace `path/to/image.jpg` with an existing image. The Ultralytics command saves annotated predictions under its detection output directory (typically `runs/detect/predict/`). It requires the trained `best.pt` checkpoint; a missing detection is **not** proof that the package is defect-free.

For additional options, run `python src/train.py --help`, `python src/evaluate.py --help`, or the relevant script's `--help` command.

## Documentation

- [HTML source-code documentation](docs/index.html) — generated from Python docstrings; open `docs/index.html` locally in a browser.
- **Engineering report:** submitted separately for MMA3001; add a link here if it is published in the repository.

To regenerate HTML documentation in PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path + ";" + (Get-Location).Path
python -m pdoc -o docs src
```

## Software testing

Run the automated verification suite:

```powershell
python run_tests.py

**Repository note:** `data/`, `runs/`, `*.pt` and `.venv/` are excluded by `.gitignore`. Source code, HTML documentation and small reports can be tracked. To reproduce the *exact* existing predictions, obtain the matching dataset and trained checkpoint; otherwise, train the models using the commands above.
