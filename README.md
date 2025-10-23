# Fruit Ripeness Classification

This repository contains the baseline experiments for Assignment 2 on training neural networks to classify fruit ripeness (unripe, ripe, rotten) from images. The repo is configured for local development and for execution on Google Colab with GPU acceleration.

## Repository Structure

- `Dataset/` – dataset description, download helper, and (ignored) storage location for downloaded data  
- `Example Scripts/` – reference code supplied with the assignment brief  
- `notebooks/` – Jupyter notebooks (baseline PyTorch training notebook added)  
- `requirements.txt` – Python dependencies for local development or Colab  
- `README.md` – project setup and usage guide

## Getting Started

1. **Create and activate a virtual environment (recommended)**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Configure Kaggle credentials**  
   - Generate `kaggle.json` from your Kaggle account settings.  
   - Place it in `~/.kaggle/kaggle.json` or the project root (the file is ignored by git).  
   - Ensure it has permissions `600` (`chmod 600 kaggle.json`).

3. **Download the dataset**  
   The dataset is too large for version control. Run the helper script, which uses `kagglehub`, to download and unpack to `Dataset/fruit_ripeness_dataset/`.
   ```bash
   python Dataset/download_dataset.py
   ```

4. **Launch JupyterLab/notebook**  
   ```bash
   jupyter lab
   ```
   The baseline notebook lives in `notebooks/baseline_pytorch.ipynb`.

## Running on Google Colab

1. Upload or clone this repository into your Colab environment.  
2. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
3. Run `Dataset/download_dataset.py` to fetch the Kaggle data inside Colab (ensure your Kaggle API token is available).  
4. Open `notebooks/baseline_pytorch.ipynb` and execute the cells. The notebook automatically detects and uses a GPU if available.

### Quick Colab Workflow

1. Open [Google Colab](https://colab.research.google.com) and choose **File → Upload Notebook…** (or **GitHub tab → paste repo URL**).  
2. If you upload the notebook directly, also upload `README.md` and `requirements.txt` or clone via `!git clone`.  
3. Run the first cell to install dependencies (this may take a few minutes).  
4. When prompted, upload your Kaggle API credentials (`kaggle.json` or similarly named). The notebook renames and stores it under `~/.kaggle/`.  
5. Execute the dataset download cell; progress bars will show copying/extraction status.  
6. Run the remaining cells sequentially to train, visualise metrics, evaluate on the test set, and optionally save checkpoints.  
7. To resume later, rerun the dependency, credential, and dataset cells before training.

## Next Steps

- Iterate on model architectures, data augmentation, and training schedules in additional notebooks.  
- Add experiment tracking (e.g., TensorBoard or Weights & Biases).  
- Implement evaluation scripts for held-out test splits once the baseline is validated.
