# Egg Segmentation and Multi-Object Tracking

This repository contains the end-to-end implementation for the Computer Vision Intern Assessment, covering foundation model segmentation (SAM ViT-B), classical image processing, quantitative evaluation, and multi-object video tracking[cite: 1].

## Repository Structure
- `01_split_data.py`: Splits the dataset into a 4:1 Dev/Eval partition while preventing sequence-level data leakage[cite: 1].
- `02_task1_sam_segmentation.py`: Generates ground-truth masks using Meta's Segment Anything Model (SAM ViT-B)[cite: 1].
- `03_task2_classical_segmentation.py`: Classical rule-based segmentation pipeline (multi-space CIELAB/HSV thresholding, morphological filtering, and geometric priors)[cite: 1].
- `04_task3_evaluation.py`: Evaluates classical predictions against SAM ground truth (IoU, Dice, Precision, Recall, Accuracy) and generates qualitative error heatmaps[cite: 1].
- `05_task4_video_tracking.py`: Multi-object tracker assigning persistent IDs and rendering centroid trajectory paths across video frames[cite: 1, 2].
- `Report.pdf`: Comprehensive assessment report containing the experiment log, quantitative tables, failure analysis, and deliverables[cite: 1, 2].
- `outputs/`: Contains `evaluation_metrics.csv`, worst-case qualitative error plots, and the generated tracking output[cite: 1].

## Setup & Reproduction Instructions

### 1. Clone the Repository
```bash
git clone [https://github.com/Aaditya-Jain-01/egg-segmentation-and-tracking.git](https://github.com/Aaditya-Jain-01/egg-segmentation-and-tracking.git)
cd egg-segmentation-and-tracking

```

### 2. Environment Setup

Create and activate a virtual environment, then install project dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

```

### 3. Model Weights Download

Create a `weights` directory and download the SAM ViT-B checkpoint (`sam_vit_b_01ec64.pth`, 375 MB):

```bash
mkdir weights
curl.exe -L -o weights/sam_vit_b_01ec64.pth [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

```

## Running the Pipeline

Execute the pipeline scripts in sequential order:

```bash
python 01_split_data.py
python 02_task1_sam_segmentation.py
python 03_task2_classical_segmentation.py
python 04_task3_evaluation.py
python 05_task4_video_tracking.py

```

## Deliverables & Results

* **Report Deliverable**: [`Report.pdf`](https://www.google.com/search?q=./Report.pdf) located at the root of this repository[cite: 2].
* **Task 4 Video Tracking Output**: [Public Google Drive Link](https://drive.google.com/file/d/1_r4a-0JjMyHaAlcRM1OdY_KvlZPC4YFP/view?usp=sharing)[cite: 2]
* **Quantitative Metrics Summary**:
* **Dev Split**: IoU: `0.3253` | Dice: `0.4183` | Precision: `0.3741` | Recall: `0.7426` | Accuracy: `0.6716`
* **Eval Split**: IoU: `0.2761` | Dice: `0.3737` | Precision: `0.3424` | Recall: `0.8163` | Accuracy: `0.5530`



## AI Tools Disclosure

In compliance with assessment requirements, AI coding assistants were utilized for boilerplate code structuring, environment configuration debugging, and documentation formatting. Pipeline architecture, classical computer vision heuristics, empirical tuning, and analytical deductions were conducted independently.




### Push the Updated `README.md` to GitHub

After saving the file in VS Code (`Ctrl + S`), run the following commands in your Command Prompt:

```cmd
git add README.md
git commit -m "Update and clean README with clone instructions and results"
git push origin main

```
