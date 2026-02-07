
import unittest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path (from tests/ -> root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import stars_utils

# ANSI Color Codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

def print_pass(msg):
    print(f"{GREEN}[PASS]{RESET} {msg}")

def print_info(msg):
    print(f"{CYAN}[INFO]{RESET} {msg}")

def print_fail(msg):
    print(f"{RED}[FAIL]{RESET} {msg}")

class TestStarsProject(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print(f"\n{YELLOW}=================================================={RESET}")
        print(f"{YELLOW}   Running Comprehensive Test Suite (v2.0)        {RESET}")
        print(f"{YELLOW}=================================================={RESET}\n")
        
        # Locate project root (tests/ -> root)
        cls.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cls.bsp_path = os.path.join(cls.root_dir, 'de421.bsp')

    def test_01_imports_and_syntax(self):
        """[Syntax] Verify critical modules can be imported."""
        print_info("Checking Dependencies...")
        try:
            import tensorflow
            import pandas
            import skyfield
            import stars_utils
            print_pass("All core modules imported successfully.")
        except ImportError as e:
            self.fail(f"Module missing: {e}")

    def test_02_project_structure(self):
        """[Location] Verify directory layout and file existence."""
        print_info("Checking Project Structure...")
        
        # Check Critical Files (Functional Structure)
        required_files = [
            'stars_utils.py',
            'training/generate_data.ipynb',
            'training/train_geocentric.ipynb',
            'analysis/evaluate_geocentric.ipynb'
        ]
        
        for f in required_files:
            if os.path.exists(os.path.join(self.root_dir, f)):
                print_pass(f"File confirmed: {f}")
            else:
                self.fail(f"Missing mandatory file: {f}")
        
        # Check Models Dir
        if os.path.isdir(os.path.join(self.root_dir, 'models')):
             print_pass("Directory confirmed: models")
        else:
             self.fail("Missing 'models' directory.")

        # Check Data Dir
        if os.path.isdir(os.path.join(self.root_dir, 'data')):
             print_pass("Directory confirmed: data")
        else:
             self.fail("Missing 'data' directory.")
        
        # Check BSP
        if os.path.exists(self.bsp_path):
             print_pass("Ephemeris kernel (de421.bsp) found.")
        else:
            self.fail(f"de421.bsp missing from project root: {self.bsp_path}")

    def test_03_data_integrity(self):
        """[Data Validation] Generate a sample dataset and check schema/types."""
        print_info("Verifying Data Generation Logic...")
        try:
            # Generate 1 week of data
            start = datetime(2025, 1, 1)
            end = datetime(2025, 1, 8)
            
            # Note: We must pass the absolute path to bsp since we might be running from tests/ dir
            df = stars_utils.generate_planetary_ephemeris_df(
                'mars', start, end, timedelta(days=1), ephemeris_file=self.bsp_path
            )
            
            # Check 1: Non-empty
            self.assertFalse(df.empty, "Generated DataFrame is empty.")
            print_pass("Data generation produced records.")
            
            # Check 2: Schema
            required_cols = ['Julian_Date', 'X_au', 'Y_au', 'Z_au', 'Helio_X_au']
            for col in required_cols:
                self.assertIn(col, df.columns)
            print_pass("Schema validation passed.")
            
            # Check 3: Data Types (No NaNs)
            self.assertFalse(df.isnull().values.any(), "Dataset contains NaN values.")
            print_pass("Data integrity (No Nulls) passed.")
            
        except Exception as e:
             self.fail(f"Data generation raised exception: {e}")

    def test_04_physics_logic(self):
        """[Physics] Verify Keplerian calculations match physical reality."""
        print_info("Verifying Physics Engine...")
        
        # Setup Time
        dates = [datetime(2000, 1, 1).replace(tzinfo=stars_utils.utc)]
        t = stars_utils.ts.from_datetimes(dates)
        jd_values = t.tdb
        
        # Mars Distance Check
        coords = stars_utils.get_geocentric_keplerian_xyz('mars', jd_values)
        r_mars = np.linalg.norm(coords)
        
        # Mars is 1.5 AU from Sun, Earth is 1.0. 
        # Geocentric distance varies ~0.5 AU (Opposition) to ~2.5 AU (Conjunction)
        if 0.3 < r_mars < 3.0:
            print_pass(f"Mars geocentric distance ({r_mars:.3f} AU) is within physical bounds.")
        else:
            self.fail(f"Mars distance {r_mars} is physically impossible!")

    def test_05_error_handling(self):
        """[Breakage] Test system resilience to invalid inputs."""
        print_info("Testing Error Handling...")
        
        # 1. Invalid Planet Name
        with self.assertRaises(ValueError):
            stars_utils.get_geocentric_keplerian_xyz('pluto', np.array([2451545.0]))
        print_pass("Caught 'Invalid Planet' error correctly.")
        
        # 2. Invalid Date Input
        try:
            # Passing a string instead of numeric array
            stars_utils.get_keplerian_xyz('mars', "tomorrow") 
            self.fail("Should have raised error for invalid date input")
        except (AttributeError, TypeError, ValueError):
             print_pass("Caught 'Invalid Date Format' error correctly.")

    def test_06_security_checks(self):
        """[Security] Verify no suspicious executables in data folders."""
        print_info("Performing Security Scan...")
        suspicious_exts = ['.exe', '.sh', '.bat', '.cmd', '.vbs']
        
        clean = True
        for dirpath, _, filenames in os.walk(self.root_dir):
            if 'env' in dirpath or '.git' in dirpath or '__pycache__' in dirpath: 
                continue
            
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in suspicious_exts:
                    print_fail(f"Suspicious executable found: {os.path.join(dirpath, f)}")
                    clean = False
        
        if clean:
            print_pass("Directory is clean of executables.")
        else:
            self.fail("Security Scan Failed: Executables found in project.")

if __name__ == '__main__':
    # Run tests with basic text runner, verbosity=0 so our custom prints shine
    unittest.main(verbosity=0)
