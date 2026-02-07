# 🪐 Geocentric Orbit Prediction
> **Physics-Informed Deep Learning for the Inner Solar System**

[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()
[![Physics](https://img.shields.io/badge/Physics-Relativistic-blueviolet.svg)]()
[![Precision](https://img.shields.io/badge/MAE-%3C0.005_AU-green.svg)]()

A high-precision AI model that predicts the geocentric (Earth-relative) positions of **Mercury**, **Venus**, and **Mars**. It combines classical Keplerian mechanics with a **Deep Residual Network** that learns to correct for complex N-Body gravitational perturbations and relativistic light-time delays.

---

## 📂 Project Structure

```
├── stars_utils.py              # Core Physics Engine (Barycenter, Light-Time, Featurization)
├── de421.bsp                   # NASA JPL Ephemeris Kernel (Ground Truth)
│
├── training/                   # Model Training Pipelines
│   └── train_inner_planets.py  # Automation script for Mercury, Venus, Mars
│
├── analysis/                   # Visualization & Reporting
│   ├── solar_system_viz.py     # Generates 3D Interactive Plot
│   └── generate_pdf_report.py  # Generates Data Science Reports
│
├── data/                       # Processed Datasets (CSV)
├── models/                     # Trained Keras Models (.keras)
├── visualizations/             # Output HTML 3D Plots
├── summary/                    # PDF Reports & Project Summaries
└── tests/                      # Verification Suite
```

## 🚀 Getting Started

### 1. Installation
Requires Python 3.8+.
```bash
pip install numpy pandas tensorflow skyfield plotly scikit-learn matplotlib
```

### 2. Run the Full Pipeline ("The Grand Tour")
To generate data, train models, and create visualizations for all 3 planets:
```bash
python3 training/train_inner_planets.py
```
*   **Input**: `stars_utils.py` generates physics features.
*   **Output**: Saved models in `models/` and HTML plots in `visualizations/`.

### 3. Generate Reports
To create the unified 3D simulation and PDF documents:
```bash
# Interactive 3D Plot
python3 analysis/solar_system_viz.py

# PDF Summary Report
python3 analysis/generate_full_pdf_report.py
```

## 🧠 Model Performance (Phase 16)

| Planet | MAE (AU) | Precision | Key Physics Feature |
| :--- | :--- | :--- | :--- |
| **Venus** | `0.0038` | ⭐ Extreme | **Earth Resonance Harmonics** (8:5 coupling) |
| **Mars** | `0.0040` | ⭐ Very High | **Barycentric Correction** (Removing Sun wobble) |
| **Mercury** | `0.0279` | ✅ Good | **Relativistic Light-Time** (Speed of light lag) |

## 🛠️ Tech Stack
*   **Deep Learning**: TensorFlow/Keras (Residual MLP: 256-128-64).
*   **Physics**: Skyfield (NASA JPL `de421`), General Relativity corrections.
*   **Visualization**: Plotly 3D (WebGL), Matplotlib.

## 🧪 Testing
Run the verification suite to ensure system integrity:
```bash
python3 tests/test_geocentric.py
```

---
*Created by Yasha Modi - 2026*
