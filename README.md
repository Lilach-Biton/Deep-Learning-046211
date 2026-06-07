# Deep-Learning-046211

This repository contains coursework and a project from the Deep Learning course (046211). It collects the assignments (HW0–HW3) and a larger project exploring trajectory prediction using Lyft data and L5Kit.

## Repository Structure

- `HW0/` — Machine Learning Basics Refresher
	- `cancer_dataset.csv` — dataset used in HW0
	- `ece046211_hw0_205764517_316155944.ipynb`
- `HW1/` — Optimization and Automatic Differentiation
	- `ece046211_hw1_205764517_316155944.ipynb`
- `HW2/` — Multilayer NNs and Convolutional NNs
	- `ece046211_hw2_205764517_316155944.ipynb`
- `HW3/` — Sequential Tasks and Training Methods
	- `ece046211_hw3_seq_tasks_students_205764517_316155944.ipynb`
	- `data/datasets/WikiText2/wikitext-2/` — WikiText-2 splits: `wiki.train.tokens`, `wiki.valid.tokens`, `wiki.test.tokens`
- `Project/` - Course Project: Motion Prediction for Autonomous Vehicles - Lyft Level 5 Benchmark
	- `LyftTrajectoryPrediction/`
		- `full_pipeline.ipynb` — end-to-end experiment notebook
		- `README.md` — project-specific notes and usage (read first)
		- `requirements.txt` — Python dependencies for the project
		- `assets/` — visualizations, figures, and helper files
		- `data/` — expected location for raw and preprocessed project data
		- `l5kit/` — a vendored or local copy of L5Kit utilities (if present)
		- `models/`
			- `BaselineRes50Model.py` - Baseline ResNet-50 Model Class
			- `ViTDeitModel.py` - Data-efficient Image Transformer (DeiT) Model Class
			- `ViTModel.py` - Visual Transformer (ViT) Model Class
			- `configs/` — YAML model configuration files
		- `src/`
			- `trainer.py` — training loop and orchestration
			- `utils.py` — utilities for data, metrics and helpers

## Homeworks Overview

- `HW0` — Machine Learning Basics Refresher: exploratory data analysis and simple supervised learning on `cancer_dataset.csv`, covering basic metrics and model evaluation.
- `HW1` — Optimization and Automatic Differentiation: exercises in gradient computation, implementing optimization algorithms, and inspecting convergence behavior.
- `HW2` — Multilayer and Convolutional Neural Networks: building, training, and evaluating MLP and CNN architectures; covers regularization, data augmentation, and basic model design choices.
- `HW3` — Sequential Tasks and Language Modeling: sequence modelling experiments (RNN/LSTM/Transformer variants) using datasets like WikiText-2; focuses on next-token prediction, loss evaluation, and qualitative text generation.

## Project: Lyft Trajectory Prediction

The `LyftTrajectoryPrediction` project contains model implementations and a training pipeline aimed at forecasting agent trajectories using rasterized inputs and compares several deep learning architectures, starting with a ResNet-50 baseline and more advanced Transformer-based models like Vision Transformer (ViT) and Data-efficient Image Transformer (DeiT).

- `full_pipeline.ipynb` demonstrates data loading, model instantiation, training, and visualization for quick experiments.
- `models/` contains the model classes used by the notebooks and training scripts.
- `src/trainer.py` provides a reusable training loop; inspect or adapt it to run experiments from the command line or scripts.

If you use L5Kit, follow their installation and dataset preparation steps (see `Project/LyftTrajectoryPrediction/README.md` if present).

## Quick Start

1. Recommended: create and activate a Python virtual environment (tested with Python 3.8+):

	 python -m venv .venv
	 .venv\Scripts\activate

2. Install project requirements for the `LyftTrajectoryPrediction` project (if you plan to run it):

	 pip install -r Project/LyftTrajectoryPrediction/requirements.txt

3. Open the notebooks in Jupyter or VS Code and run cells interactively. Example:

	 jupyter notebook

4. Data placement notes:
	 - Small datasets used by the homework (e.g., `HW0/cancer_dataset.csv`, `HW3/data/datasets/WikiText2`) are included in this repo.
	 - The `Project/LyftTrajectoryPrediction/data/` folder is reserved for larger Lyft or L5Kit datasets and preprocessed files — these are not committed here. Place your downloaded datasets there and adjust paths in the notebooks or `src/utils.py` as needed.


