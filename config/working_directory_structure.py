# KMTNet ToO Pipeline - Working Directory Structure Configuration
# =============================================================
# 
# This file defines the directory structure and paths for the KMTNet ToO pipeline.
# It automatically detects the repository root directory and creates all necessary
# subdirectories relative to that location.
#
# HOW IT WORKS:
# 1. This config file is located in: <repo_root>/config/
# 2. It automatically finds the repository root by going up one directory level
# 3. All pipeline paths are created relative to the repository root
# 4. Users can run the pipeline from anywhere - paths will always be correct
#
# BENEFITS:
# - No hardcoded paths that break when moved to different systems
# - Works regardless of where the user runs the pipeline script from
# - Centralized configuration - change paths in one place
# - Automatic directory creation - no manual setup required
#
# USAGE:
# - Import this file in your pipeline scripts
# - All path variables (path_data, path_cfg, etc.) will be available
# - Call create_directories() to ensure all directories exist

import os

# STEP 1: Find the repository root directory
# This config file is in <repo_root>/config/, so we go up one level
config_dir = os.path.dirname(os.path.abspath(__file__))  # Gets <repo_root>/config/
BASE_PATH = os.path.dirname(config_dir)                  # Gets <repo_root>/

# STEP 2: Define the directory structure
# All paths are relative to the repository root directory
DIRECTORIES = {
    'data': 'data',           # Main data directory
    'config': 'config',       # Configuration files
    'catalog': 'catalog',     # Reference catalogs
    'result': 'result',       # Output results
    'raw': 'data/raw',        # Raw input images
    'scaled': 'data/scaled',  # Scaled/calibrated images
    'stack': 'data/stack',    # Stacked images
    'subt': 'data/subt',      # Subtraction results
    'tmpl': 'data/tmpl',      # Template images
    'plot': 'result/plot',    # Output plots
    'log': 'result/log'       # Log files
}

# STEP 3: Generate absolute paths
# Combine the base path with each relative directory path
PATHS = {}
for name, relative_path in DIRECTORIES.items():
    PATHS[f'path_{name}'] = os.path.join(BASE_PATH, relative_path)

# STEP 4: Create individual path variables for easy access
# These variables can be imported directly by other scripts
path_base = BASE_PATH
path_data = PATHS['path_data']
path_cfg = PATHS['path_config']
path_cat = PATHS['path_catalog']
path_raw = PATHS['path_raw']
path_scale = PATHS['path_scaled']
path_stack = PATHS['path_stack']
path_subt = PATHS['path_subt']
path_tmpl = PATHS['path_tmpl']
path_res = PATHS['path_result']
path_plot = PATHS['path_plot']
path_log = PATHS['path_log']

# STEP 5: Directory creation function
def create_directories(verbose=False):
    """
    Create all necessary directories for the pipeline.
    
    This function ensures that all required directories exist before the pipeline runs.
    It's safe to call multiple times - existing directories won't cause errors.
    
    Args:
        verbose (bool): If True, print messages for each directory created. Default: False.
    
    Returns:
        dict: Dictionary of all path variables (PATHS)
    
    Usage:
        paths = create_directories()  # Creates all directories and returns PATHS dict
        create_directories(verbose=True)  # Creates directories with verbose output
    """
    for path_name, path_value in PATHS.items():
        os.makedirs(path_value, exist_ok=True)
        if verbose:
            print(f"Created directory: {path_value}")
    
    return PATHS

if __name__ == "__main__":
    # Test the configuration
    print("KMTNet ToO Pipeline Configuration")
    print("=" * 40)
    print(f"Base path: {path_base}")
    print("\nDirectory paths:")
    for name, path in PATHS.items():
        print(f"  {name}: {path}")
    
    print("\nCreating directories...")
    create_directories()
