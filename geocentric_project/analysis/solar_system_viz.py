
import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Ensure we can import stars_utils from parent directory or current
sys.path.append(os.getcwd())
import stars_utils

def safe_load_df(planet):
    path = f'data/{planet}_processed.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"Warning: {path} not found.")
    return None

def get_features_for_planet(df, target_planet):
    # Replicate logic from train_inner_planets.py
    features = [
        'Time_Index', 'Time_Index_2',
        f'Sin_{target_planet.capitalize()}', f'Cos_{target_planet.capitalize()}',
        'X_au_Lag1', 'Y_au_Lag1', 'Z_au_Lag1',
        'X_au_Lag2', 'Y_au_Lag2', 'Z_au_Lag2',
        'Kepler_X', 'Kepler_Y', 'Kepler_Z'
    ]
    
    potential_perturbers = ['Jupiter', 'Saturn', 'Venus', 'Earth']
    for p in potential_perturbers:
        g_col = f'Inv_Dist_{p}'
        if g_col in df.columns:
            if (df[g_col] != 0).any():
                features.append(g_col)
        
        if f'Sin_{p}' in df.columns:
            features.append(f'Sin_{p}')
            features.append(f'Cos_{p}')
            
    return features

def create_solar_system_viz():
    print("🌟 Generating Combined Solar System Visualization...")
    fig = go.Figure()
    
    # 1. Earth (Observer)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=12, color='blue', symbol='circle'),
        name='Earth (Geocentric Center)'
    ))
    
    planets = {
        'mercury': {'color': 'gray', 'dist_scale': 0.4},
        'venus': {'color': 'gold', 'dist_scale': 0.7},
        'mars': {'color': 'red', 'dist_scale': 1.5}
    }
    
    for planet, style in planets.items():
        print(f"Processing {planet}...")
        df = safe_load_df(planet)
        if df is None: continue
        
        # --- RECONSTRUCT MODEL PREDICTION ---
        features = get_features_for_planet(df, planet)
        
        # Load Model
        model_path = f'models/{planet}_geocentric.keras'
        if not os.path.exists(model_path):
            print(f"Model for {planet} not found. Skipping prediction.")
            continue
            
        model = load_model(model_path)
        
        # Prepare Data (Re-create Scalers)
        X = df[features].values
        TARGETS = ['Res_X', 'Res_Y', 'Res_Z']
        y = df[TARGETS].values
        
        # Split (Deterministic)
        split = int(len(df) * 0.8)
        X_train = X[:split]
        y_train = y[:split]
        
        # Fit Scalers
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        y_scaler = StandardScaler()
        y_scaler.fit(y_train)
        
        # Predict on FULL Dataset (for smooth viz)
        # Note: Predicting on train data is "cheating" for metrics, but fine for viz trace
        X_scaled = scaler.transform(X) 
        
        print(f"Predicting full trajectory for {planet}...")
        pred_res_scaled = model.predict(X_scaled, verbose=0)
        pred_res = y_scaler.inverse_transform(pred_res_scaled)
        
        # Add to DF
        df['Pred_X'] = df['Kepler_X'] + pred_res[:, 0]
        df['Pred_Y'] = df['Kepler_Y'] + pred_res[:, 1]
        df['Pred_Z'] = df['Kepler_Z'] + pred_res[:, 2]
        
        # --- PLOT TRACES (Filter to last 5 years for clarity) ---
        df['Time_UTC'] = pd.to_datetime(df['Time_UTC'])
        viz_start = df['Time_UTC'].max() - pd.DateOffset(years=5)
        viz_df = df[df['Time_UTC'] > viz_start].copy()
        
        # Downsample slightly for web performance
        viz_df = viz_df.iloc[::2] 
        
        # Ground Truth Trace (Faint)
        fig.add_trace(go.Scatter3d(
            x=viz_df['X_au'], y=viz_df['Y_au'], z=viz_df['Z_au'],
            mode='lines',
            line=dict(color=style['color'], width=2, dash='dot'),
            opacity=0.3,
            name=f'{planet.capitalize()} (True)'
        ))
        
        # Predicted Trace (Solid/Bright)
        fig.add_trace(go.Scatter3d(
            x=viz_df['Pred_X'], y=viz_df['Pred_Y'], z=viz_df['Pred_Z'],
            mode='lines',
            line=dict(color=style['color'], width=4),
            name=f'{planet.capitalize()} (Model)'
        ))
        
        # Add a marker for the "End" position (Current day approx)
        last = viz_df.iloc[-1]
        fig.add_trace(go.Scatter3d(
            x=[last['Pred_X']], y=[last['Pred_Y']], z=[last['Pred_Z']],
            mode='markers+text',
            marker=dict(size=6, color=style['color']),
            text=[f'{planet.capitalize()}'],
            textposition="top center",
            showlegend=False
        ))

    # Layout
    fig.update_layout(
        title="Inner Solar System: Geocentric Model Predictions (Phase 16)",
        scene=dict(
            xaxis_title='X (AU)',
            yaxis_title='Y (AU)',
            zaxis_title='Z (AU)',
            aspectmode='data' # Important to keep orbits circular
        ),
        template='plotly_dark',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    out_file = "visualizations/inner_solar_system_viz.html"
    fig.write_html(out_file)
    print(f"\n✨ Combined Visualization Saved: {out_file} ✨")

if __name__ == "__main__":
    create_solar_system_viz()
