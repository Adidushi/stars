# 🪐 Geocentric Orbit Prediction
> **Data Science & Physics Project: Predicting Star Locations with Neural Networks, Math, and General Relativity.**

[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()
[![Physics](https://img.shields.io/badge/Physics-Relativistic-blueviolet.svg)]()
[![Precision](https://img.shields.io/badge/MAE-%3C0.005_AU-green.svg)]()

A high-precision AI model that predicts the geocentric (Earth-relative) positions of **Mercury**, **Venus**, and **Mars**. It combines classical Keplerian mechanics with **Deep Residual Learning** to correct for complex N-Body gravitational perturbations and relativistic light-time delays.

---

## 📖 The Story (Legacy Cleanup & Enhancement)
This project began as a continuation of coursework from "Intro to Data Science" at **Tel Aviv University**. The original legacy codebase was unstructured and difficult to maintain.

As a **Math Major**, I took the initiative to:
1.  **Refactor & Restructure**: Moved from scattered scripts to a modular, professional Python package (`geocentric_project/`).
2.  **Enhance the Science**: Implemented **Gravitational Perturbation Theory** and **Relativistic Light-Time Correction** to push accuracy beyond simple regression.
3.  **Automate**: Built a full pipeline for data generation, training, and visualization.

---

## 2. The Problem Statement
*   **Context**: Traditional orbital mechanics (Kepler's Laws) are approximations that fail to account for multi-body gravitational perturbations and relativistic effects over time.
*   **Goal**: Create a lightweight, high-precision Deep Learning model to predict **Geocentric Coordinates** ($X, Y, Z$) for inner planets.
*   **Objective**: Minimize the Mean Absolute Error (MAE) without heavy numerical integration.

## 3. Methodology: Physics-Informed Residual Learning
Instead of learning "raw" coordinates, the model learns the **Residual** (Error) of the classical physics model:
$$ \text{Target} = \text{Pos}_{True} - \text{Pos}_{Kepler} $$

### Key Features (The "Secret Sauce")
*   **Relativistic Light-Time Correction**: Gravity is calculated based on where a planet *was* when its light left it ($t - \Delta t_{light}$), not where it is now.
*   **Solar System Barycenter**: All calculations reference the SSB to remove solar wobble.
*   **Harmonic Reserves**: Explicitly modeling the 8:5 Earth-Venus resonance.

## 4. Results & Metrics (Test Set 2010-2025)

| Planet | MAE (AU) | Accuracy Tier | Key Driver |
| :--- | :--- | :--- | :--- |
| **Venus** | `0.0038` | ⭐ **Extreme** | Earth Resonance captured. |
| **Mars** | `0.0040` | ⭐ **Very High** | Barycentric checks stabilized orbit. |
| **Mercury** | `0.0279` | ✅ **Good** | Relativistic effects managed. |

> **Metric**: Mean Absolute Error (MAE) of the 3D Euclidean distance in Astronomical Units (AU).

---

## 📂 Project Structure

```
├── geocentric_project/         # 🌟 The Main Project
│   ├── stars_utils.py          # Core Physics Engine
│   ├── training/               # Model Training Pipelines
│   ├── analysis/               # Visualization & Reporting
│   ├── data/                   # Processed Datasets (CSV)
│   ├── models/                 # Trained Models (.keras)
│   ├── visualizations/         # 3D Interactive Plots
│   └── summary/                # PDF Reports & Documentation
│
└── legacy_base_project/        # 🏚️ The Original Legacy Code (Archive)
```

## 🚀 Usage

### 1. Installation
```bash
pip install numpy pandas tensorflow skyfield plotly scikit-learn matplotlib
```

### 2. Run the Full Pipeline ("The Grand Tour")
To generate data, train models, and create visualizations for all 3 planets:
```bash
cd geocentric_project
python3 training/train_inner_planets.py
```

### 3. Generate Reports
To create the unified 3D simulation and PDF documents:
```bash
cd geocentric_project
# Interactive 3D Plot
python3 analysis/solar_system_viz.py

# PDF Summary Report
python3 analysis/generate_full_pdf_report.py
```

---
*Created by Yasha Modi - 2026*
