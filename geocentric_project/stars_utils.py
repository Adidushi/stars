"""Utility file for the 'stars' project - Geocentric orbit prediction using physics-informed neural networks."""
import os
import pickle
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from skyfield.api import load, utc
from sklearn.preprocessing import StandardScaler

# --- KEPLERIAN ELEMENTS (J2000) ---
# Source: JPL (approximate mean elements)
ORBITAL_ELEMENTS = {
    'mars': {
        'a': 1.523679,   # Semi-major axis (AU)
        'e': 0.0934,     # Eccentricity
        'i': 1.850,      # Inclination (deg)
        'L': 49.558,     # Longitude of ascending node (deg)
        'w': 286.502,    # Longitude of perihelion (deg) -> omega = w - L
        'M_0': 19.412,   # Mean anomaly at J2000 (deg)
        'n': 0.524033    # Mean motion (deg/day)
    },
    'earth': {
        'a': 1.000000,
        'e': 0.0167,
        'i': 0.000,
        'L': -11.260,
        'w': 102.947,
        'M_0': 100.464,
        'n': 0.985600
    },
    'mercury': {
        'a': 0.387098,
        'e': 0.205630,
        'i': 7.005,
        'L': 48.331,
        'w': 77.456, # Longitude of Perihelion
        'M_0': 174.796,
        'n': 4.092334
    },
    'venus': {
        'a': 0.723332,
        'e': 0.006773,
        'i': 3.394,
        'L': 76.680,
        'w': 131.532, # Longitude of Perihelion
        'M_0': 50.115,
        'n': 1.602136
    }
}

def solve_kepler(M: float, e: float, tolerance: float = 1e-6, max_iterations: int = 50) -> float:
    """Iteratively solve Kepler's Equation: M = E - e*sin(E) for Eccentric Anomaly E.
    
    Args:
        M: Mean anomaly (radians)
        e: Eccentricity (0 <= e < 1)
        tolerance: Convergence threshold
        max_iterations: Maximum number of iterations
        
    Returns:
        Eccentric anomaly E (radians)
        
    Raises:
        ValueError: If eccentricity is invalid or convergence fails
    """
    if not (0 <= e < 1):
        raise ValueError(f"Eccentricity must be in range [0, 1), got {e}")
    
    E = M if e < 0.8 else np.pi
    for iteration in range(max_iterations):
        delta = E - e * np.sin(E) - M
        if abs(delta) < tolerance:
            return E
        E = E - delta / (1 - e * np.cos(E))
    
    raise ValueError(f"Kepler's equation failed to converge after {max_iterations} iterations")

def get_keplerian_xyz(target_planet: str, julian_dates: np.ndarray) -> np.ndarray:
    """
    Calculate Ideal Keplerian (Heliocentric) XYZ coordinates.
    
    Args:
        target_planet: Name of the planet (e.g., 'mars', 'earth', 'venus')
        julian_dates: Array of Julian dates (TDB)
        
    Returns:
        (N, 3) numpy array of heliocentric XYZ coordinates in AU
        
    Raises:
        ValueError: If planet is not in ORBITAL_ELEMENTS dictionary
    """
    if target_planet not in ORBITAL_ELEMENTS:
        raise ValueError(f"No Keplerian elements for {target_planet}. "
                        f"Available planets: {list(ORBITAL_ELEMENTS.keys())}")
    
    el = ORBITAL_ELEMENTS[target_planet]
    J2000 = 2451545.0
    d = julian_dates - J2000
    
    # 1. Mean Anomaly
    M_deg = el['M_0'] + el['n'] * d
    M_rad = np.radians(M_deg % 360)
    
    # 2. Solve for Eccentric Anomaly
    # Vectorized iterative solution for Kepler's equation
    E_rad = M_rad # Initialization
    for _ in range(5): # 5 iterations is usually enough for array ops
        E_rad = E_rad - (E_rad - el['e'] * np.sin(E_rad) - M_rad) / (1 - el['e'] * np.cos(E_rad))
        
    # 3. True Anomaly (v)
    # tan(v/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    sqrt_term = np.sqrt((1 + el['e']) / (1 - el['e']))
    tan_v2 = sqrt_term * np.tan(E_rad / 2)
    v_rad = 2 * np.arctan(tan_v2)
    
    # 4. Radius (r)
    # r = a * (1 - e * cos(E))
    r = el['a'] * (1 - el['e'] * np.cos(E_rad))
    
    # 5. Position in Orbital Plane (x', y')
    x_prime = r * np.cos(v_rad)
    y_prime = r * np.sin(v_rad)
    
    # 6. Rotate to Ecliptic (3D rotation)
    # Elements: 
    #   O (Omega) = Longitude of Ascending Node (L)
    #   w (omega) = Argument of Perihelion = (Longitude of Perihelion - L)
    #   i = Inclination
    
    Omega_rad = np.radians(el['L'])
    w_bar_rad = np.radians(el['w'])
    omega_rad = w_bar_rad - Omega_rad
    i_rad = np.radians(el['i'])
    
    # Constants for rotation matrix
    cos_O = np.cos(Omega_rad)
    sin_O = np.sin(Omega_rad)
    cos_w = np.cos(omega_rad)
    sin_w = np.sin(omega_rad)
    cos_i = np.cos(i_rad)
    sin_i = np.sin(i_rad)
    
    # Heliocentric coordinates (X, Y, Z)
    # X = x' (cosO cosw - sinO sinw cosi) - y' (cosO sinw + sinO cosw cosi)
    # Y = x' (sinO cosw + cosO sinw cosi) - y' (sinO sinw - cosO cosw cosi)
    # Z = x' (sinw sini) + y' (cosw sini)
    
    X = x_prime * (cos_O * cos_w - sin_O * sin_w * cos_i) - y_prime * (cos_O * sin_w + sin_O * cos_w * cos_i)
    Y = x_prime * (sin_O * cos_w + cos_O * sin_w * cos_i) - y_prime * (sin_O * sin_w - cos_O * cos_w * cos_i)
    Z = x_prime * (sin_w * sin_i) + y_prime * (cos_w * sin_i)
    
    return np.stack([X, Y, Z], axis=1)

def get_geocentric_keplerian_xyz(target_planet: str, julian_dates: np.ndarray) -> np.ndarray:
    """
    Calculate Ideal Geocentric XYZ (Keplerian Target - Keplerian Earth).
    
    Args:
        target_planet: Name of the target planet
        julian_dates: Array of Julian dates (TDB)
        
    Returns:
        (N, 3) numpy array of geocentric XYZ coordinates in AU
    """
    # 1. Get Target Heliocentric
    target_helio = get_keplerian_xyz(target_planet, julian_dates)
    
    # 2. Get Earth Heliocentric
    earth_helio = get_keplerian_xyz('earth', julian_dates)
    
    # 3. Vector Subtraction (Target - Earth)
    return target_helio - earth_helio


# Load timescale globally
ts = load.timescale()

# Define constants for feature engineering (Orbital Periods in Days)
PERIOD_YEAR = 365.25          # Earth's orbital period

# Dictionary for Sidereal Periods (Used to make feature generation dynamic)
PLANET_PERIODS = {
    'mercury': 87.97,
    'venus': 224.70,
    'earth': PERIOD_YEAR,
    'mars': 686.98,
    'jupiter': 4332.6,
    'saturn': 10759.2,
    'uranus': 30685.4,
    'neptune': 60189.0
}

class CustomEpochLogger(tf.keras.callbacks.Callback):
    """Custom callback that logs epoch progress at specified intervals.
    
    This callback reduces console output by only printing metrics every N epochs,
    making it easier to monitor long-running training sessions.
    
    Args:
        log_step: Frequency of logging (e.g., 10 means log every 10 epochs)
    """
    def __init__(self, log_step: int = 10):
        super().__init__()
        self.log_step = log_step
        if log_step < 1:
            raise ValueError(f"log_step must be >= 1, got {log_step}")

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """Prints loss and validation loss every 'log_step' epochs.
        
        Args:
            epoch: Current epoch number (0-indexed)
            logs: Dictionary containing training metrics
        """
        if logs is None:
            logs = {}
            
        if (epoch + 1) % self.log_step == 0:
            # Safely check for 'val_loss' which might not be present in the first few epochs
            val_loss = logs.get('val_loss', float('nan'))
            train_loss = logs.get('loss', float('nan'))
            total_epochs = self.params.get('epochs', 'unknown')
            print(f"Epoch {epoch + 1}/{total_epochs} - Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")


def save_scaler(scaler: StandardScaler, filepath: str):
    """
    Saves the fitted StandardScaler object to a file using pickle.
    This is essential to ensure the same scaling is used for prediction as was used for training.
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved successfully to: {filepath}")
    except Exception as e:
        print(f"ERROR: Failed to save scaler to {filepath}: {e}")

def load_scaler(filepath: str) -> Optional[StandardScaler]:
    """
    Loads a fitted StandardScaler object from a file using pickle.
    """
    if not os.path.exists(filepath):
        print(f"ERROR: Scaler file not found at {filepath}.")
        return None
    try:
        with open(filepath, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded successfully from: {filepath}")
        return scaler
    except Exception as e:
        print(f"ERROR: Failed to load scaler from {filepath}: {e}")
        return None

def generate_planetary_ephemeris_df(
    target_planet: str, 
    start_date: datetime, 
    end_date: datetime, 
    time_step: timedelta = timedelta(days=1),
    ephemeris_file: str = 'de421.bsp'
) -> pd.DataFrame:
    """
    Generates a DataFrame containing the geocentric position (RA, Dec) 
    for a given celestial body over a specified time range.
    
    Args:
        target_planet: Name of the target celestial body (e.g., 'mars', 'venus')
        start_date: Start date for data generation
        end_date: End date for data generation
        time_step: Time step between data points (default: 1 day)
        ephemeris_file: Path to JPL ephemeris file (default: 'de421.bsp')
        
    Returns:
        DataFrame with geocentric positions and auxiliary features
        
    Raises:
        ValueError: If dates are invalid or ephemeris file cannot be loaded
    """
    # Validate inputs
    if end_date <= start_date:
        raise ValueError(f"end_date ({end_date}) must be after start_date ({start_date})")
    
    if not os.path.exists(ephemeris_file):
        raise FileNotFoundError(f"Ephemeris file not found: {ephemeris_file}")
    try:
        # Load the JPL ephemeris data file and timescale
        planets = load(ephemeris_file)
        ts = load.timescale()

    except Exception as e:
        print(f"ERROR: Could not load ephemeris file '{ephemeris_file}'. Check file existence and Skyfield installation.")
        print(e)
        return pd.DataFrame()

    # Determine the correct name for Skyfield lookup
    target_name = target_planet.lower()
    
    # List of bodies whose center is explicitly listed in de421.bsp (inner planets/Moon)
    inner_bodies = ['mercury', 'venus', 'earth', 'mars', 'moon']
    
    # Use barycenter for outer planets since de421.bsp only provides barycenter data for them
    if target_name not in inner_bodies:
        skyfield_target_name = target_name + ' barycenter'
        print(f"Note: Using '{skyfield_target_name}' for target lookup.")
    else:
        skyfield_target_name = target_name

    # Get the target body and the observer (Earth)
    try:
        target = planets[skyfield_target_name]
    except KeyError:
        # Check against the derived name, which should exist if the input was valid
        print(f"ERROR: Skyfield target '{skyfield_target_name}' (derived from '{target_planet}') not found in ephemeris data.")
        return pd.DataFrame()
        
    earth = planets['earth']

    # 1. Generate a list of dates
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += time_step

    if not dates:
        print("ERROR: Time range resulted in zero data points.")
        return pd.DataFrame()

    # 2. Convert Python datetimes to Skyfield time objects
    # Skyfield requires time zone aware datetime objects (using UTC here)
    timezone_aware_dates = [d.replace(tzinfo=utc) for d in dates]
    t = ts.utc(timezone_aware_dates)

    # 3. Calculate Geocentric Position (Position as seen from Earth)
    # astrometric = position of target relative to the observer (Earth)
    astrometric = earth.at(t).observe(target)

    # Get Right Ascension, Declination, and distance
    ra, dec, distance = astrometric.radec()

    # Get the astrometric XYZ coordinates in AU
    pos_vector_au = astrometric.xyz.au # 3xN NumPy array

    # 4. Calculate Features using Relativistic Light-Time (Gravity Delay)
    # Instead of simplistic vector subtraction (Instantaneous), we use Skyfield's observe()
    # to calculate exactly how far the perturber is from the target *accounting for light speed*.
    
    # Define Perturbers (Include Earth)
    perturber_names = ['jupiter', 'saturn', 'venus', 'earth']
    
    # Dictionary to hold the relativistic distances
    rel_distances = {}
    
    for p_name in perturber_names:
        if p_name == target_name:
            rel_distances[p_name] = np.zeros(len(dates)) # Distance to self is 0
            continue
            
        # Handle naming (barycenter vs planet)
        if p_name in inner_bodies:
            p_obj = planets[p_name]
        else:
            p_obj = planets[p_name + ' barycenter']
            
        # Target observes Perturber (Includes Light Time / Gravity Propagation Delay)
        # This gives the true distance the gravity wave travelled
        obs = target.at(t).observe(p_obj)
        rel_distances[p_name] = obs.distance().au
        
    # --- HELIOCENTRIC COORDS (Keep purely for Training Targets & Reconstruction) ---
    sun = planets['sun']
    
    # Training Target: Ideal Keplerian Position (Sun-centered)
    helio_target_xyz = sun.at(t).observe(target).xyz.au
    
    # Reconstruction Data
    helio_earth_xyz = sun.at(t).observe(earth).xyz.au
    
    # Legacy Feature Support (Optional)
    jupiter = planets['jupiter barycenter']
    helio_jupiter_xyz = sun.at(t).observe(jupiter).xyz.au
    saturn = planets['saturn barycenter']
    helio_saturn_xyz = sun.at(t).observe(saturn).xyz.au
    venus = planets['venus']
    helio_venus_xyz = sun.at(t).observe(venus).xyz.au

    # 5. Create the DataFrame (Your Dataset)
    data = {
            'Time_UTC': dates,
            'Julian_Date': t.tdb,
            
            # Ground Truth Geocentric
            'RA_deg': ra.degrees,          
            'Dec_deg': dec.degrees,        
            'Distance_AU': distance.au,    
            'X_au': pos_vector_au[0],
            'Y_au': pos_vector_au[1],
            'Z_au': pos_vector_au[2],

             # Heliocentric targets
            'Helio_X_au': helio_target_xyz[0],
            'Helio_Y_au': helio_target_xyz[1],
            'Helio_Z_au': helio_target_xyz[2],

            # Perturber Distances (Relativistic) - DIRECTLY USED FOR FEATURES
            'Dist_Jupiter_AU': rel_distances['jupiter'],
            'Dist_Saturn_AU': rel_distances['saturn'],
            'Dist_Venus_AU': rel_distances['venus'],
            'Dist_Earth_AU': rel_distances['earth'],
            
            # Legacy Coordinates columns (ensuring backward compatibility)
            'Earth_Helio_X_au': helio_earth_xyz[0],
            'Earth_Helio_Y_au': helio_earth_xyz[1],
            'Earth_Helio_Z_au': helio_earth_xyz[2],
            'Jupiter_Helio_X_au': helio_jupiter_xyz[0],
            'Jupiter_Helio_Y_au': helio_jupiter_xyz[1],
            'Jupiter_Helio_Z_au': helio_jupiter_xyz[2],
            'Saturn_Helio_X_au': helio_saturn_xyz[0],
            'Saturn_Helio_Y_au': helio_saturn_xyz[1],
            'Saturn_Helio_Z_au': helio_saturn_xyz[2],
            'Venus_Helio_X_au': helio_venus_xyz[0],
            'Venus_Helio_Y_au': helio_venus_xyz[1],
            'Venus_Helio_Z_au': helio_venus_xyz[2]
        }

    df = pd.DataFrame(data)
    print(f"Dataset for {target_planet.capitalize()} created successfully with {len(df)} data points.")
    
    return df

def add_astronomy_features(df: pd.DataFrame, target_planet: str) -> pd.DataFrame:
    """
    Adds polynomial and sinusoidal features related to orbital mechanics dynamically 
    based on the target planet's period and the resulting synodic period with Earth.
    """
    
    # --- Dynamic Period Lookup ---
    target_planet_key = target_planet.lower()
    if target_planet_key not in PLANET_PERIODS:
        raise ValueError(f"Period for target planet '{target_planet}' not found in PLANET_PERIODS dictionary.")

    P_target = PLANET_PERIODS[target_planet_key]

    # Calculate Synodic Period (P_syn)
    # Formula: 1/P_syn = |1/P_Earth - 1/P_target|
    if abs(P_target - PERIOD_YEAR) < 0.1: # If target IS Earth
        # For Earth, there is no "Earth-Earth" synodic period (inf). 
        # Features should be 0 or constant.
        P_synodic = 1e9 
        print(f"Target is Earth (Synodic Period Undefined).")
    else:
        P_synodic = 1 / abs((1 / PERIOD_YEAR) - (1 / P_target))
        print(f"Calculated {target_planet}'s Synodic Period with Earth: {P_synodic:.2f} days.")

    # 1. Create Base Time Index (normalized Julian Date)
    JD_min = df.Julian_Date.min()
    df["Time_Index"] = df.Julian_Date - JD_min
    time_index = df["Time_Index"]

    # 2. Polynomial Features (Long term drift/precession)
    df["Time_Index_2"] = time_index ** 2
    df["Time_Index_3"] = time_index ** 3
   
    # 3. Heliocentric Orbital Cycles
    
    # Target Planet's Orbital Cycle (Primary Signal)
    df[f'Sin_{target_planet_key.capitalize()}'] = np.sin(2 * np.pi * time_index / P_target)
    df[f'Cos_{target_planet_key.capitalize()}'] = np.cos(2 * np.pi * time_index / P_target)
    
    # Jupiter perturbation (Major influence in solar system)
    P_jupiter = PLANET_PERIODS['jupiter']
    df['Sin_Jupiter'] = np.sin(2 * np.pi * time_index / P_jupiter)
    df['Cos_Jupiter'] = np.cos(2 * np.pi * time_index / P_jupiter)

    # Saturn Perturbation (Long term massive pull)
    P_saturn = PLANET_PERIODS['saturn']
    df['Sin_Saturn'] = np.sin(2 * np.pi * time_index / P_saturn)
    df['Cos_Saturn'] = np.cos(2 * np.pi * time_index / P_saturn)
    
    # Venus Perturbation (Short term close pull)
    P_venus = PLANET_PERIODS['venus']
    df['Sin_Venus'] = np.sin(2 * np.pi * time_index / P_venus)
    df['Cos_Venus'] = np.cos(2 * np.pi * time_index / P_venus)
    
    # Earth Perturbation (Crucial for Inner Planets)
    # Even though we are geocentric, the Earth's position relative to the Sun defines the physics frame
    P_earth = PLANET_PERIODS['earth']
    df['Sin_Earth'] = np.sin(2 * np.pi * time_index / P_earth)
    df['Cos_Earth'] = np.cos(2 * np.pi * time_index / P_earth)

    # --- EARTH SPECIFIC PHYSICS (MOON) ---
    if target_planet == 'earth':
        # Moon's Sidereal Period ~27.32 days
        P_moon = 27.32
        df['Sin_Moon'] = np.sin(2 * np.pi * time_index / P_moon)
        df['Cos_Moon'] = np.cos(2 * np.pi * time_index / P_moon)
        
        # Moon Position (perturbation force)
        print("Calculating Moon features...")
        try:
            planets = load('de421.bsp')
            earth_bary = planets['earth'] 
            moon = planets['moon']
            
            ts = load.timescale()
            t_vec = ts.tdb(jd=df['Julian_Date'].values)
            
            # Vector from Earth to Moon (Geocentric Moon)
            astrometric = earth_bary.at(t_vec).observe(moon) 
            moon_xyz = astrometric.xyz.au # Shape (3, N)
            
            # Inverse Distance Squared (Gravity ~ 1/r^2)
            dist_sq = moon_xyz[0]**2 + moon_xyz[1]**2 + moon_xyz[2]**2
            df['Inv_Dist_Moon'] = 1.0 / dist_sq
            
            print("Moon features added.")
            
        except Exception as e:
            print(f"WARNING: Could not load Moon data: {e}")
            df['Sin_Moon'] = 0
            df['Cos_Moon'] = 0
            df['Inv_Dist_Moon'] = 0

    # 4. Lag Features (Momentum/Velocity Check)
    # Using Geocentric Coords for this project
    lag_cols = ['X_au', 'Y_au', 'Z_au']
    for col in lag_cols:
        df[f'{col}_Lag1'] = df[col].shift(1)
        df[f'{col}_Lag2'] = df[col].shift(2)
        
    # 5. Physics: N-Body Gravity Features (Inverse Distance Squared)
    # We now use the Relativistic Distances pre-calculated in generate_planetary_ephemeris_df
    
    perturbers = ['Jupiter', 'Saturn', 'Venus', 'Earth']
    
    for p in perturbers:
        dist_col = f'Dist_{p}_AU'
        
        # Check if we have the relativistic distance column
        if dist_col in df.columns:
            # Add feature: 1 / r^2
            # Handle possible zero division for self (though we filtered it, better safe)
            dist_sq = df[dist_col] ** 2
            df[f'Inv_Dist_{p}'] = 1.0 / (dist_sq + 1e-9)
            
            # If self (distance=0), set to 0 manually to be clean
            if target_planet.lower() == p.lower():
                df[f'Inv_Dist_{p}'] = 0.0
                
        else:
             # Fallback to legacy calculation if column missing (shouldn't happen with new pipeline)
             # ... Logic removed for brevity as we are upgrading fully ...
             pass
    
    # Fill NaNs created by lagging (first 2 rows) using backward fill
    df.bfill(inplace=True)

    # 6. Physics: Residual Calculation (True Geocentric - Ideal Geocentric)
    print("Calculating Ideal Geocentric Keplerian Orbit...")
    kepler_geo = get_geocentric_keplerian_xyz(target_planet, df['Julian_Date'].values)
    df['Kepler_X'] = kepler_geo[:, 0]
    df['Kepler_Y'] = kepler_geo[:, 1]
    df['Kepler_Z'] = kepler_geo[:, 2]
    
    # The Residuals (Targets)
    df['Res_X'] = df['X_au'] - df['Kepler_X']
    df['Res_Y'] = df['Y_au'] - df['Kepler_Y']
    df['Res_Z'] = df['Z_au'] - df['Kepler_Z']
    
    print(f"Added dynamic features (Time Index, Polynomial, Target Cycle, Jupiter Cycle, Lags, Gravity, Residuals) to the DataFrame.")
    
    return df


# Data Analysis functions

def models_loader(directory: str, models_lis: list, target_planet: str) -> list:
    """
    Loads Keras models from a specified directory, checking both .keras and .h5 extensions.
    The model filenames are expected to be prefixed by the target planet name.
    
    Args:
        directory: The folder containing the Keras model files.
        models_lis: A list of model suffixes (e.g., ['mm0', 'mm1', 'mm2']).
        target_planet: The name of the planet the model was trained for (e.g., 'mars').

    Returns:
        A list of loaded Keras Model objects.
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Model directory not found: {directory}")
    
    models = []
    print(f"Attempting to load {len(models_lis)} models for {target_planet} from '{directory}'...")
    
    for suffix in models_lis:
        # Base filename format: '{target_planet}_position_predictor_{suffix}'
        base_filename = f'{target_planet}_position_predictor_{suffix}'
        path_keras = os.path.join(directory, f'{base_filename}.keras')
        path_h5 = os.path.join(directory, f'{base_filename}.h5')
        
        loaded = False
        model = None

        # --- 2. Try to load .keras (Preferred) ---
        if os.path.exists(path_keras):
            print(f"Attempting to load: {path_keras}")
            model_path_to_use = path_keras
            
        # --- 3. Fallback to .h5 ---
        elif os.path.exists(path_h5):
            print(f"File not found at {path_keras}. Falling back to: {path_h5}")
            model_path_to_use = path_h5
            
        else:
            print(f"ERROR: Model file not found for {target_planet} with suffix '{suffix}'. Checked both .keras and .h5 extensions.")
            continue # Skip to the next model in the list
            
        # --- 4. Load the found model ---
        try:
            # tf.keras.models.load_model handles both .keras and .h5 formats
            model = tf.keras.models.load_model(model_path_to_use)
            models.append(model)
            loaded = True
            
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load Keras model from {model_path_to_use}: {e}")
            # If loading fails (e.g., file corruption), we stop trying to load this specific model
            
    if models:
        print(f"Successfully loaded {len(models)} models.")
    else:
        print(f"No models were successfully loaded for {target_planet}.")

    return models

def xyz_to_radec(pred_au: np.ndarray) -> pd.DataFrame:
    """
    Converts predicted equatorial rectangular coordinates (X, Y, Z in AU) 
    to Right Ascension (RA) and Declination (Dec) in degrees.
    
    Args:
        pred_au: Nx3 array of XYZ coordinates in AU
        
    Returns:
        DataFrame with columns 'Predicted_RA_deg' and 'Predicted_Dec_deg'
        
    Steps:
        1. Extract Coordinates
        2. Calculate Predicted Distance (r) where r = sqrt(X^2 + Y^2 + Z^2)
        3. Calculate Declination (Dec) - The angle North/South where Dec = arcsin(Z / r) 
        4. Calculate Right Ascension (RA) - The angle East/West where RA = arctan2(Y, X)
        5. Convert Dec and RA to degrees from radians
        6. Normalize RA to the 0 to 360 degree range (adding +360 for negative value)
    """
    if pred_au.shape[1] != 3:
        raise ValueError(f"Input must have 3 columns (X, Y, Z), got shape {pred_au.shape}")

    X_pred = pred_au[:, 0]
    Y_pred = pred_au[:, 1]
    Z_pred = pred_au[:, 2]
    
    r_pred = np.sqrt(X_pred**2 + Y_pred**2 + Z_pred**2)
    
    dec_rad = np.arcsin(Z_pred / r_pred)
    dec_deg = np.degrees(dec_rad)
    
    ra_rad = np.arctan2(Y_pred, X_pred)
    ra_deg = np.degrees(ra_rad)
    
    # arctan2 returns results in the range [-180, 180]. Add 360 to negative values.
    ra_deg = np.where(ra_deg < 0, ra_deg + 360, ra_deg)
    
    # Return results as a DataFrame for easy handling
    results_df = pd.DataFrame({
        'Predicted_RA_deg': ra_deg,
        'Predicted_Dec_deg': dec_deg
    })
    
    return results_df

def get_angular_error_arcsec(df: pd.DataFrame, ts) -> np.ndarray:
    """
    Calculates the 3D positional error magnitude in AU.
    
    Args:
        df: DataFrame containing both actual and predicted XYZ coordinates
        ts: Skyfield timescale (currently unused but kept for API compatibility)
        
    Returns:
        Array of position errors in AU
    
    Note:
        Calculating true angular separation requires a full ephemeris load and 
        observer definition, which is complex. We report the magnitude of the 
        positional error vector in AU as a primary metric.
    """
    print("Calculating 3D positional error magnitude...")
    
    # Validate required columns exist
    required_cols = ['X_au', 'Y_au', 'Z_au', 'X_pred', 'Y_pred', 'Z_pred']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ground Truth Positions
    actual_xyz_au = df[['X_au', 'Y_au', 'Z_au']].values.T
    
    # Predicted Positions
    predicted_xyz_au = df[['X_pred', 'Y_pred', 'Z_pred']].values.T

    # Calculate the vector difference
    error_vector = predicted_xyz_au - actual_xyz_au
    
    # Calculate the 3D position error magnitude (Euclidean distance)
    df['Pos_Error_AU'] = np.linalg.norm(error_vector, axis=0)
    
    # Return the magnitude array
    return df['Pos_Error_AU']

def plot_error_over_time(df: pd.DataFrame, planet: str, error_col: str = 'Pos_Error_AU', 
                        title: str = '3D Position Error Over Time', ylabel: str = 'Error (AU)') -> None:
    """
    Plots the error metric over the prediction time range.
    
    Args:
        df: DataFrame containing error data
        planet: Planet name for the title
        error_col: Column name containing error values
        title: Plot title
        ylabel: Y-axis label
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14, 6))
    plt.plot(df['Time_UTC'], df[error_col], label=f'{error_col}', color='#3b82f6')
    plt.title(f'{planet.capitalize()} {title}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()