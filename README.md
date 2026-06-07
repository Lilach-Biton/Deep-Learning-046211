# Deep-Learning-046211

This repository contains coursework and project files from the Deep Learning course (046211). It includes assignments (HW0–HW3) and a final project exploring multi-agent trajectory prediction using Lyft data, L5Kit, CNNs, and Vision Transformers (ViTs).

## Repository Structure

* `HW0/` — Machine Learning Basics Refresher
    * `cancer_dataset.csv` — Dataset used in HW0
    * `ece046211_hw0.ipynb`


* `HW1/` — Optimization and Automatic Differentiation
    * `ece046211_hw1.ipynb`


* `HW2/` — Multilayer NNs and Convolutional NNs
    * `ece046211_hw2.ipynb`


* `HW3/` — Sequential Tasks and Training Methods
    * `ece046211_hw3_seq_tasks.ipynb`
    * `data/datasets/WikiText2/wikitext-2/` — WikiText-2 splits: `wiki.train.tokens`, `wiki.valid.tokens`, `wiki.test.tokens`


* `Project/` — Course Project: Motion Prediction for Autonomous Vehicles (Lyft Level 5 Benchmark)
    * `LyftTrajectoryPrediction/`
        * `full_pipeline.ipynb` — End-to-end experiment notebook
        * `README.md` — Project-specific notes and usage guidelines
        * `requirements.txt` — Python dependencies for the project
        * `assets/` — Visualizations, figures, and helper files
        * `data/` — Expected location for raw and preprocessed project data
        * `l5kit/` — Local copy of L5Kit utilities
        * `models/`
            * `BaselineRes50Model.py` — Baseline ResNet-50 model class
            * `ViTDeitModel.py` — Data-efficient Image Transformer (DeiT) model class
            * `ViTModel.py` — Vision Transformer (ViT) model class
            * `configs/` — YAML model configuration files


        * `src/`
            * `trainer.py` — Training loop and orchestration
            * `utils.py` — Utilities for data processing, metrics, and helper functions







---

## Homeworks Overview

* **HW0 — Machine Learning Basics Refresher:** Exploratory data analysis (EDA) and simple supervised learning on `cancer_dataset.csv`, covering basic performance metrics and model evaluation.
* **HW1 — Optimization and Automatic Differentiation:** Hand-rolled gradient computation, implementation of optimization algorithms, and inspection of training convergence behavior.
* **HW2 — Multilayer and Convolutional Neural Networks:** Building, training, and evaluating MLP and CNN architectures. Explores regularization techniques, data augmentation, and core model design choices.
* **HW3 — Sequential Tasks and Language Modeling:** Sequence modeling experiments using Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), and Transformer variants on the WikiText-2 dataset. Focuses on next-token prediction, loss evaluation, and qualitative text generation.

---

## Project: Lyft Trajectory Prediction

The `LyftTrajectoryPrediction` project focuses on forecasting autonomous vehicle agent trajectories using rasterized inputs. The core task is to model multi-agent future positions by leveraging semantic bird's-eye-view (BEV) map representations.

This project implements and evaluates several deep learning architectures to compare traditional inductive biases against attention-based methods:

* **Task:** Predicting multi-agent future trajectories from rasterized bird's-eye-view (BEV) inputs over a fixed time horizon.
* **CNN Baseline:** A convolutional baseline utilizing **ResNet-50** to extract spatial features from the scene rasters.
* **Attention-Based Models:** Implementing and adapting **Vision Transformers (ViT)** and **Data-efficient Image Transformers (DeiT)** to process tokenized patches of the driving environment for superior context modeling.

### Core Components

* `full_pipeline.ipynb` demonstrates data loading, model instantiation, training loops, and qualitative results visualization for quick experimentation.
* `models/` contains the modular PyTorch implementations of the architectures used in the benchmark.
* `src/trainer.py` provides a reusable, highly configurable training loop that can be adapted to run experiments from the command line or scripts.

> **Note:** If you are running the project from scratch, please follow the specific L5Kit installation and dataset preparation steps outlined in `Project/LyftTrajectoryPrediction/README.md`.

---

### Data Placement Notes

* **Homework Datasets:** Small datasets used in assignments (e.g., `HW0/cancer_dataset.csv` and `HW3/data/datasets/WikiText2`) are included directly in this repository.
* **Project Datasets:** The `Project/LyftTrajectoryPrediction/data/` folder is reserved for the larger Lyft Level 5 datasets and preprocessed cache files. These large files are ignored by git. Place your downloaded datasets there and adjust the directory paths in the configuration YAML files or `src/utils.py` as needed.