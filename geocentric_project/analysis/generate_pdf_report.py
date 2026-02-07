"""
PDF report generation for geocentric orbit prediction results.

This module creates a summary PDF report with visualizations and metrics.
"""
import sys
import os
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# Ensure we can import stars_utils
sys.path.append(os.getcwd())
import stars_utils


def safe_load_df(planet: str) -> Optional[pd.DataFrame]:
    """
    Safely load a processed planet dataset from CSV.
    
    Args:
        planet: Name of the planet (e.g., 'mars', 'venus')
        
    Returns:
        DataFrame if file exists, None otherwise
    """
    path = f'data/{planet}_processed.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"Warning: {path} not found.")
    return None


def generate_pdf_report() -> None:
    """Generate a comprehensive PDF report of model performance and visualizations."""
    os.makedirs('summary', exist_ok=True)
    output_file = "summary/Geocentric_Orbit_Report.pdf"
    print(f"📄 Generating PDF Report: {output_file}...")
    
    with PdfPages(output_file) as pdf:
        # --- PAGE 1: TITLE & SUMMARY ---
        fig = plt.figure(figsize=(11.69, 8.27)) # A4 Landscape
        fig.clf()
        
        # Title
        plt.text(0.5, 0.9, "Geocentric Orbit Prediction Project", 
                 fontsize=24, ha='center', fontweight='bold')
        plt.text(0.5, 0.85, f"Generated: {datetime.now().strftime('%Y-%m-%d')}", 
                 fontsize=14, ha='center', color='gray')
        
        # Summary Text
        summary_text = (
            "This report summarizes the performance of the High-Precision Geocentric Models.\n"
            "The models use Deep Residual Learning with Relativistic Physics features\n"
            "(Light-Time Correction + Barycentric Frame) to predict planetary positions.\n\n"
            "Metrics (Mean Absolute Error):"
        )
        plt.text(0.5, 0.70, summary_text, fontsize=12, ha='center')

        # Summary Table Data
        planets = ['mercury', 'venus', 'mars']
        table_data = []
        
        for planet in planets:
            df = safe_load_df(planet)
            if df is not None:
                # Calculate Baseline Kepler MAE (showing the residuals we're trying to predict)
                # The CSV contains Res_X, Res_Y, Res_Z which are (True - Kepler)
                # This represents the complexity the ML model needs to learn
                kepler_err = np.sqrt(df['Res_X']**2 + df['Res_Y']**2 + df['Res_Z']**2)
                mean_kepler_err = kepler_err.mean()
                
                table_data.append([
                    planet.capitalize(), 
                    f"{mean_kepler_err:.4f} AU", 
                    "Kepler Baseline Error"
                ])

        # Draw Table
        if table_data:
            table = plt.table(cellText=table_data, 
                              colLabels=["Planet", "Kepler Residual (Baseline)", "Note"],
                              loc='center', cellLoc='center', bbox=[0.2, 0.4, 0.6, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            
        plt.axis('off')
        pdf.savefig(fig)
        plt.close()

        # --- PLANET PAGES ---
        for planet in planets:
            print(f"Processing {planet}...")
            df = safe_load_df(planet)
            if df is None: continue

            df['Time_UTC'] = pd.to_datetime(df['Time_UTC'])
            # Filter to 5 years
            viz_start = df['Time_UTC'].max() - pd.DateOffset(years=5)
            viz_df = df[df['Time_UTC'] > viz_start].copy()
            viz_df = viz_df.iloc[::5] # Downsample

            fig = plt.figure(figsize=(11.69, 8.27))
            fig.suptitle(f"{planet.capitalize()} - Orbit & Physics Analysis (Last 5 Years)", fontsize=18, fontweight='bold')
            
            # Subplot 1: 3D Orbit
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot(viz_df['X_au'], viz_df['Y_au'], viz_df['Z_au'], label='True Orbit', color='blue')
            ax1.scatter([0], [0], [0], color='green', s=100, label='Earth')
            ax1.set_title(f"Geocentric Path ({planet.capitalize()})")
            ax1.set_xlabel('X (AU)')
            ax1.set_ylabel('Y (AU)')
            ax1.set_zlabel('Z (AU)')
            ax1.legend()
            
            # Subplot 2: Kepler Residuals (What the AI learned)
            # This shows the magnitude of correction the AI makes
            ax2 = fig.add_subplot(122)
            residuals = np.sqrt(viz_df['Res_X']**2 + viz_df['Res_Y']**2 + viz_df['Res_Z']**2)
            ax2.plot(viz_df['Time_UTC'], residuals, color='red', alpha=0.7)
            ax2.set_title("Keplerian Deviation (AI Target)")
            ax2.set_ylabel("Deviation (AU)")
            ax2.set_xlabel("Year")
            ax2.grid(True, linestyle='--')
            
            # Annotation
            max_err = residuals.max()
            ax2.text(viz_df['Time_UTC'].min(), max_err*0.9, 
                     f"Max Deviation: {max_err:.4f} AU\nMean Deviation: {residuals.mean():.4f} AU",
                     bbox=dict(facecolor='white', alpha=0.8))

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close()

    print(f"\n✅ PDF Report Saved: {output_file}")

if __name__ == "__main__":
    generate_pdf_report()
