import pandas as pd
from shapely.geometry import Point, box
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import argparse
import sys
import os
from argparse import RawTextHelpFormatter

# --- Configuration ---
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.working_directory_structure import *
paths = create_directories()
path_cfg = paths['path_config']
PREPROCESSED_PATH = os.path.join(path_cfg, 'KS4_DR1_Depth_Polygons.pkl')
FILTERS = ['B', 'V', 'R', 'I']

# Default test coordinates
DEFAULT_RA = 179.716
DEFAULT_DEC = -40.41

def load_data(path):
    if not os.path.exists(path):
        print(f"[Error] Data file not found: {path}")
        return None
    try:
        with open(path, 'rb') as f:
            df = pickle.load(f)
        return df
    except Exception as e:
        print(f"[Error] Failed to load data: {e}")
        return None

def get_point_depths(ra, dec, df):
    """
    Queries the 5-sigma depth for a specific (RA, DEC) coordinate.
    """
    target = Point(ra, dec)
    results = {}
    
    for filt in FILTERS:
        df_filt = df[df['filter'] == filt]
        matches = df_filt[df_filt['poly_obj'].apply(lambda p: p.contains(target))]
        
        if not matches.empty:
            results[filt] = matches['depth'].max()
        else:
            results[filt] = np.nan
            
    return results

def plot_zoom_map(ra, dec, df, radius=2.0):
    """
    Visualizes the survey depth map.
    - Title: Cleaned (Filter name only).
    - Legend: Shows the exact depth at the target position.
    - Colorbar: Fixed to Global Median +/- 1 Sigma.
    """
    search_box = box(ra - radius, dec - radius, ra + radius, dec + radius)
    
    # Pre-calculate point depths for the legend labels
    point_depths = get_point_depths(ra, dec, df)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    
    print(f"[Info] Generating maps for {radius}-degree radius around RA={ra}, DEC={dec}...")

    for i, filt in enumerate(FILTERS):
        ax = axes[i]
        df_filt = df[df['filter'] == filt].copy()
        
        # --- 1. Calculate Global Statistics (Fixed Color Scale) ---
        global_median = df_filt['depth'].median()
        global_std = df_filt['depth'].std()
        if pd.isna(global_std) or global_std == 0: global_std = 0.5 
        
        vmin = global_median - global_std
        vmax = global_median + global_std
        
        # --- 2. Filter Local Data ---
        nearby_polys = df_filt[df_filt['poly_obj'].apply(lambda p: p.intersects(search_box))]
        
        # Style settings
        title_fs = 11
        label_fs = 9
        tick_fs = 8
        
        # --- 3. Plot Polygons ---
        if nearby_polys.empty:
            ax.text(0.5, 0.5, "No Data Covered", ha='center', va='center', transform=ax.transAxes)
        else:
            patches = []
            depth_values = nearby_polys['depth'].values
            
            for _, row in nearby_polys.iterrows():
                poly = row['poly_obj']
                if poly.geom_type == 'Polygon':
                    x, y = poly.exterior.xy
                    patches.append(MplPolygon(np.column_stack([x, y]), closed=True))
                elif poly.geom_type == 'MultiPolygon':
                    for sub_poly in poly.geoms:
                        x, y = sub_poly.exterior.xy
                        patches.append(MplPolygon(np.column_stack([x, y]), closed=True))

            p = PatchCollection(patches, cmap='viridis', alpha=0.7, edgecolor='none')
            p.set_array(depth_values)
            p.set_clim([vmin, vmax]) # Fixed Scale
            ax.add_collection(p)
            
            # Colorbar
            cbar = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(r'5$\sigma$ Depth (mag)', fontsize=label_fs)
            cbar.ax.tick_params(labelsize=tick_fs)

        # --- 4. Plot Target Marker & Legend ---
        target_val = point_depths[filt]
        
        if pd.isna(target_val):
            legend_label = "Target (Not Covered)"
            marker_color = 'gray'
        else:
            legend_label = f"Target ({target_val:.3f} mag)"
            marker_color = 'red'

        ax.plot(ra, dec, marker='*', color=marker_color, markersize=14, 
                markeredgecolor='white', label=legend_label, zorder=10)
        
        # Legend (Shows the value explicitly)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

        # Axis settings
        ax.set_xlim(ra + radius, ra - radius)
        ax.set_ylim(dec - radius, dec + radius)
        ax.set_xlabel("RA (deg)", fontsize=label_fs)
        ax.set_ylabel("DEC (deg)", fontsize=label_fs)
        
        # Simple Title
        ax.set_title(f"{filt}-band", fontsize=title_fs, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)
        ax.grid(True, linestyle=':', alpha=0.5)

    # Main Title
    plt.suptitle(fr"KS4 DR1 Depth Map (Center: {ra}°, {dec}°) - Fixed Scale (Median $\pm$ 1$\sigma$)", fontsize=13)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="KS4 Depth Lookup & Visualization Tool",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('--ra', type=float, help='Target RA (deg)')
    parser.add_argument('--dec', type=float, help='Target DEC (deg)')
    parser.add_argument('--plot', action='store_true', help='Show visualization')

    args = parser.parse_args()

    df = load_data(PREPROCESSED_PATH)
    if df is None: sys.exit(1)

    ra, dec = args.ra, args.dec
    if ra is None or dec is None:
        # Quick interactive fallback
        ra = float(input(f"RA (default {DEFAULT_RA}): ") or DEFAULT_RA)
        dec = float(input(f"DEC (default {DEFAULT_DEC}): ") or DEFAULT_DEC)

    print(f"\n=== Query Result (RA={ra}, DEC={dec}) ===")
    depths = get_point_depths(ra, dec, df)
    for f in FILTERS:
        print(f"  {f}-band: {depths[f]:.3f} mag" if not pd.isna(depths[f]) else f"  {f}-band: Not Covered")

    if args.plot:
        plot_zoom_map(ra, dec, df)

if __name__ == "__main__":
    main()
