# 📄 Geocentric Project Contents

> **For Project Overview & Installation, see the [Root README](../README.md).**

This folder contains the complete source code for the prediction engine.

---

## 🛠️ File Manifest

### Core Physics
*   **`stars_utils.py`**: The Physics Brain.
    *   `generate_planetary_ephemeris_df()`: Generates training data.
    *   `add_astronomy_features()`: Adds Relativistic & Barycentric features.
    *   `get_geocentric_keplerian_xyz()`: Calculates the Keplerian baseline.
*   **`de421.bsp`**: NASA JPL Ephemeris Kernel (1900-2050).

### Pipelines (`training/`)
*   **`train_inner_planets.py`**: The "Grand Tour" automation.
    *   Run: `python3 training/train_inner_planets.py`
    *   Action: Generates data -> Trains MLP -> Saves Model -> Plots.
    *   Planets: Mercury, Venus, Mars.

### Analysis (`analysis/`)
*   **`solar_system_viz.py`**: Generates `visualizations/inner_solar_system_viz.html` (3D Plot).
*   **`generate_full_pdf_report.py`**: Generates `summary/Geocentric_Orbit_Full_Report.pdf` (Documentation).

### Artifacts (Output)
*   **`data/`**: Processed `.csv` datasets (Input + Features + Targets).
*   **`models/`**: Saved `.keras` model weights.
*   **`visualizations/`**: Interactive HTML plots.
*   **`summary/`**: Final PDF reports and text summaries.

---
*Technical Reference Only.*
