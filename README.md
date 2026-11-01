# Egg Segmentation and Multi-Object Tracking

This repository contains the complete implementation for the Computer Vision Intern Assessment, covering foundation model segmentation (SAM), classical image processing, quantitative evaluation, and multi-object video tracking.

## Repository Structure
- `01_split_data.py`: Splits the dataset into 4:1 Dev/Eval sets while preventing sequence data leakage.
- `02_task1_sam_segmentation.py`: Segments eggs using Meta's Segment Anything Model (SAM ViT-B).
- `03_task2_classical_segmentation.py`: Classical rule-based segmentation pipeline (CIELAB/HSV thresholding, morphology, geometric priors).
- `04_task3_evaluation.py`: Evaluates classical predictions against SAM ground truth (IoU, Dice, Precision, Recall, Accuracy) and exports qualitative error plots.
- `05_task4_video_tracking.py`: Multi-object tracker assigning persistent IDs and rendering centroid trajectory trails across video frames.
- `Report.pdf`: Comprehensive assessment report including the experiment log, evaluation metrics, failure analysis, and video tracking link.

## Environment Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

```

## Weights Download

Download the SAM ViT-B checkpoint to `weights/`:

```bash
curl.exe -L -o weights/sam_vit_b_01ec64.pth [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

```

## Running the Pipeline

```bash
python 01_split_data.py
python 02_task1_sam_segmentation.py
python 03_task2_classical_segmentation.py
python 04_task3_evaluation.py
python 05_task4_video_tracking.py

```

```

### 2. Verify Your Final Directory Layout

In your VS Code file tree, verify that your project root contains:
* `01_split_data.py`
* `02_task1_sam_segmentation.py`
* `03_task2_classical_segmentation.py`
* `04_task3_evaluation.py`
* `05_task4_video_tracking.py`
* `requirements.txt`
* `.gitignore`
* `README.md`
* `Report.pdf` *(with your Google Drive tracking link and evaluation plots embedded inside)*[cite: 1, 2]
* `outputs/` *(containing `evaluation_metrics.csv`, `qualitative_plots/`, and `tracked_output.mp4`)*[cite: 1]


### 3. Stage and Push to GitHub

In your terminal or VS Code integrated terminal (`Ctrl + ~`), run:

```cmd
git init
git add requirements.txt 01_split_data.py 02_task1_sam_segmentation.py 03_task2_classical_segmentation.py 04_task3_evaluation.py 05_task4_video_tracking.py outputs/evaluation_metrics.csv outputs/qualitative_plots/*.png Report.pdf README.md .gitignore
git commit -m "Complete CV assessment: segmentation, evaluation, and video tracking"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main

```
