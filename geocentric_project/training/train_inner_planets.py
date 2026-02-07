"""Training pipeline for inner planets (Mercury, Venus, Mars) geocentric prediction models."""
import sys
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Ensure we can import stars_utils from current directory
sys.path.append(os.getcwd())
import stars_utils


def run_planet_pipeline(target_planet: str) -> None:
    """
    Run the complete pipeline for a single planet: data generation, training, and visualization.
    
    Args:
        target_planet: Name of the planet (e.g., 'mercury', 'venus', 'mars')
    """
    print(f"\n🚀 STARTING PIPELINE FOR: {target_planet.upper()} 🚀")
    
    # --- 1. DATA GENERATION ---
    print(f"[{target_planet}] Generating Data...")
    start = datetime(1950, 1, 1) # Shorter range for speed, but sufficient for inner planets (short periods)
    end = datetime(2025, 1, 1)
    
    df = stars_utils.generate_planetary_ephemeris_df(
        target_planet, start, end, timedelta(days=1), ephemeris_file='de421.bsp'
    )
    
    if df.empty:
        print(f"CRITICAL ERROR: No data for {target_planet}")
        return

    df = stars_utils.add_astronomy_features(df, target_planet)
    df.bfill(inplace=True)
    
    # Save Data
    os.makedirs('data', exist_ok=True)
    csv_path = f'data/{target_planet}_processed.csv'
    df.to_csv(csv_path, index=False)
    print(f"[{target_planet}] Data saved to {csv_path}")

    # --- 2. DEFINE FEATURES DYNAMICALLY ---
    # Base Features
    features = [
        'Time_Index', 'Time_Index_2',
        f'Sin_{target_planet.capitalize()}', f'Cos_{target_planet.capitalize()}', # Self Cycle
        'X_au_Lag1', 'Y_au_Lag1', 'Z_au_Lag1',
        'X_au_Lag2', 'Y_au_Lag2', 'Z_au_Lag2',
        'Kepler_X', 'Kepler_Y', 'Kepler_Z'
    ]
    
    # Perturbers (Check what exists in DF)
    # We always include Jupiter/Saturn/Venus/Earth if available
    potential_perturbers = ['Jupiter', 'Saturn', 'Venus', 'Earth']
    for p in potential_perturbers:
        # Gravity
        g_col = f'Inv_Dist_{p}'
        if g_col in df.columns:
            # Only add if it's not all zeros (Self)
            if (df[g_col] != 0).any():
                features.append(g_col)
                
        # Cycles (Sin/Cos)
        # Note: stars_utils currently hardcodes Sin_Jupiter, Sin_Saturn, Sin_Venus. 
        # It does NOT generate Sin_Earth. 
        if f'Sin_{p}' in df.columns:
            features.append(f'Sin_{p}')
            features.append(f'Cos_{p}')

    print(f"[{target_planet}] Features Selected ({len(features)}): {features}")
    
    TARGETS = ['Res_X', 'Res_Y', 'Res_Z'] # We predict the deviation from Kepler
    
    X = df[features].values
    y = df[TARGETS].values
    
    # Split
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    
    # --- 3. TRAIN MODEL (MLP - High Precision) ---
    print(f"[{target_planet}] Training Model (High Precision)...")
    model = tf.keras.Sequential([
        layers.Input(shape=(len(features),)),
        layers.Dense(256, activation='relu'), # Widened
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='linear')
    ])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    
    # High Precision Training
    esc = callbacks.EarlyStopping(patience=50, restore_best_weights=True)
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=1000,
        batch_size=64,
        callbacks=[esc],
        verbose=0 # Quiet training to avoid massive logs, just print result
    )
    
    final_loss = history.history['val_loss'][-1]
    print(f"[{target_planet}] Training Done. Final Val Loss: {final_loss:.6f}")
    
    # Save Model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{target_planet}_geocentric.keras'
    model.save(model_path)
    print(f"[{target_planet}] Model saved to {model_path}")
    
    # --- 4. EVALUATE & VISUALIZE ---
    print(f"[{target_planet}] Visualizing...")
    pred_res = y_scaler.inverse_transform(model.predict(X_test_scaled, verbose=0))
    
    # Reconstruct
    test_df = df.iloc[split:].copy()
    test_df['Pred_X'] = test_df['Kepler_X'] + pred_res[:, 0]
    test_df['Pred_Y'] = test_df['Kepler_Y'] + pred_res[:, 1]
    test_df['Pred_Z'] = test_df['Kepler_Z'] + pred_res[:, 2]
    
    # Error
    diff = test_df[['Pred_X','Pred_Y','Pred_Z']].values - test_df[['X_au','Y_au','Z_au']].values
    mae = np.mean(np.sqrt(np.sum(diff**2, axis=1)))
    print(f"[{target_planet}] MAE: {mae:.6f} AU")
    
    # Plot
    fig = go.Figure()
    
    # Sun (for context, though it's moving in geocentric)
    # Actually, in Geocentric, Sun orbits Earth.
    # Let's just plot Earth at center.
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(color='blue', size=10), name='Earth'))
    
    # True Orbit
    viz_df = test_df.sample(2000).sort_index()
    fig.add_trace(go.Scatter3d(
        x=viz_df['X_au'], y=viz_df['Y_au'], z=viz_df['Z_au'], 
        mode='lines', line=dict(color='gray', width=3), name=f'True {target_planet}'
    ))
    
    # Predicted
    fig.add_trace(go.Scatter3d(
        x=viz_df['Pred_X'], y=viz_df['Pred_Y'], z=viz_df['Pred_Z'],
        mode='markers', marker=dict(color='orange', size=3, opacity=0.8), name='Prediction'
    ))
    
    fig.update_layout(title=f"Geocentric {target_planet.capitalize()} (MAE: {mae:.4f} AU)", template='plotly_dark')
    os.makedirs('visualizations', exist_ok=True)
    viz_path = f'visualizations/{target_planet}_viz.html'
    fig.write_html(viz_path)
    print(f"[{target_planet}] Saved {viz_path}")


if __name__ == "__main__":
    for planet in ['mercury', 'venus', 'mars']:
        run_planet_pipeline(planet)


def main():
    """Entry point for console script."""
    import sys
    
    # Allow running specific planets or all
    if len(sys.argv) > 1:
        planets = [p.lower() for p in sys.argv[1:]]
        for planet in planets:
            if planet not in ['mercury', 'venus', 'mars']:
                print(f"Warning: {planet} is not a supported inner planet. Skipping.")
                continue
            run_planet_pipeline(planet)
    else:
        # Run all inner planets by default
        for planet in ['mercury', 'venus', 'mars']:
            run_planet_pipeline(planet)

