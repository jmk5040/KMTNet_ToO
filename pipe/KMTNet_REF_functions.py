#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Import packages
import numpy as np
import astropy.units as u
import os, re, glob, time
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

#%%
def badpixel_clear():
    return 0
#%%
def ks4_photometry(img, mask, path_output, path_cfg, pixscale=0.4, clsstar=0.8, apertures=['FWHM', 'APER3', 'APER5', 'APER10', 'AUTO'], detimg=None, start=None):

    """
    This function is used to update the header of the input image with the photometric zeropoint and other information.
    """

    import time
    import shutil
    
    if os.path.isfile(detimg):
        dualband = fits.getheader(detimg)['FILTER']
        dualphot     = True

    mid     = time.time()
    print('='*80)
    print(f"{img} ({mid-start:.2f}sec elapsed)")
    print('='*80)
    # header
    hdul    = fits.open(img)
    hdr     = hdul[0].header
    band    = hdr['FILTER']
    field   = hdr['FIELD1']

    # SExtractor photometry
    # configs
    cfg         = path_cfg+'ks4catalog.sex'
    param       = path_cfg+'ks4catalog.param'
    conv        = path_cfg+'default.conv'
    nnw         = path_cfg+'default.nnw'
    
    # seeing check (with 2k cropped image)
    try:
        seeing     = hdr['FWHM']
    except KeyError:
        if os.path.isfile(f'crop_{img}'): os.remove(f'crop_{img}')
        os.system('imcopy {0}[4000:6000,4000:6000] crop_{0}'.format(img))
        os.system(f'sex crop_{img} -c {cfg} -PARAMETERS_NAME {path_cfg}ks4catalog_nomask.param -FILTER_NAME {conv} -STARNNW_NAME {nnw}')
        tempcat     = Table(fits.open('test.fits')[1].data)
        tempcat     = tempcat[tempcat['FWHM_IMAGE'] != 0]
        seeing      = np.median(tempcat[tempcat['CLASS_STAR']>np.median(tempcat['CLASS_STAR'])]['FWHM_IMAGE'] * pixscale)
        os.system('rm test.fits')
        os.system(f'rm crop_{img}')
    peeing      = seeing/pixscale

    # prompts
    photapers   = ','.join([str(float(ap.split('APER')[-1])/pixscale) for ap in apertures if ap != 'AUTO' and ap != 'FWHM'])
    if 'FWHM' in apertures:
        photapers  = f'{peeing},{photapers}'
    prompt_opt  = f' -PIXEL_SCALE {pixscale} -SEEING_FWHM {seeing:.2f} -PHOT_APERTURES {photapers}'
    prompt_cfg  = f' -PARAMETERS_NAME {param} -FILTER_NAME {conv} -STARNNW_NAME {nnw}'
    prompt_flg  = f' -FLAG_IMAGE {mask} -FLAG_TYPE MAX'
    prompt_chk  = f' -CHECKIMAGE_TYPE BACKGROUND_RMS -CHECKIMAGE_NAME {img.replace(".fits",".bkgrms")}'
    # photometry
    catname_single      = f'{path_output}ks4_{field}_{band}_single.fits'
    inim_single         = img
    prompt_catsingle    = f' -c {cfg} -CATALOG_NAME {catname_single}'
    prompt      = 'sex '+inim_single+prompt_catsingle+prompt_cfg+prompt_opt+prompt_flg+prompt_chk
    os.system(prompt)

    # dual mode
    if dualphot:
        catname_dual   = f'{path_output}ks4_{field}_{band}_dual.fits'
        if band == dualband:
            shutil.copy(catname_single, catname_dual)
        else:
            inim_dual      = f'{detimg},{img}'
            prompt_catdual = f' -c {cfg} -CATALOG_NAME {catname_dual}'
            prompt  = 'sex '+inim_dual+prompt_catdual+prompt_cfg+prompt_opt+prompt_flg
            os.system(prompt)

    intbl_single    = Table(fits.open(catname_single)[1].data)
    if dualphot: intbl_dual  = Table(fits.open(catname_dual)[1].data)
    
    # FWHM update
    seeing      = np.median(intbl_single[intbl_single['CLASS_STAR']>clsstar]['FWHM_IMAGE'] * pixscale)
    peeing      = seeing/pixscale
                    
    with fits.open(img, 'update') as f:
        for hdu in f:
            hdu.header['FWHM']      = (round(seeing, 3), "Median seeing of point sources [arcsec]")
            hdu.header['FLAGIMG']   = (os.path.basename(mask), "Mask image for IMAFLAGS_ISO")

    return

#%%
def zeropoint_map_calibration():
    return
#%%
def zeropoint_homogenization():
    return 

#%%
def load_map(map_category, field, band, mode, aperture=None, directory="."):
    """
    Load a 2D numpy map from a .npy file.

    Parameters:
      map_category (str): One of 'zp', 'zperr', 'align', 'alerr'.
      field (str): Field name, e.g. "1234.567-89".
      band (str): One of 'B', 'V', 'R', or 'I'.
      mode (str): Either 'single' or 'Idual'.
      aperture (str or None): For 'zp' and 'zperr', one of 'FWHM', 'APER3', 'APER5', 'APER10', or 'AUTO'. 
                              Not required for 'align' and 'alerr'.
      directory (str): Optional directory where the file is located.

    Returns:
      np.ndarray: The loaded 2D map.
      
    For 'zp' and 'zperr', the expected filename format is:
      "{map_category}_map_{field}_{band}_{mode}_{aperture}.npy"
      
    For 'align' and 'alerr', the expected filename format is:
      "{map_category}_map_{field}_{band}_{mode}.npy"
    """
    valid_categories = ['zp', 'zperr', 'align', 'alerr']
    if map_category not in valid_categories:
        raise ValueError(f"map_category must be one of {valid_categories}")

    if map_category in ['zp', 'zperr']:
        if aperture is None:
            raise ValueError("For 'zp' and 'zperr' maps, an aperture value must be provided.")
        filename = f"{map_category}_map_{field}_{band}_{mode}_{aperture}.npy"
    elif map_category in ['align', 'alerr']:
        filename = f"{map_category}_map_{field}_{band}_{mode}.npy"

    filepath = f"{directory}/{filename}"
    map_array = np.load(filepath)
    return map_array

def get_map_value_for_target(x, y, map_array, physical_max=22000):
    """
    Get the map value for a single target given its x and y coordinates.

    Parameters:
      x (float): x coordinate (e.g. from X_IMAGE).
      y (float): y coordinate (e.g. from Y_IMAGE).
      map_array (np.ndarray): 2D numpy array of shape (n_bins, n_bins).
      physical_max (int or float): Maximum physical coordinate value (default is 22000).

    Returns:
      The corresponding value from the map_array.
    """
    # Determine the number of bins along one axis.
    n_bins = map_array.shape[0]
    # Compute the bin size based on the physical coordinate range.
    bin_size = int(physical_max / n_bins)
    
    # Calculate bin indices by integer division.
    x_bin = int(x // bin_size)
    y_bin = int(y // bin_size)
    
    # Clip the indices to ensure they do not exceed the map boundaries.
    x_bin = min(x_bin, n_bins - 1)
    y_bin = min(y_bin, n_bins - 1)
    
    return map_array[x_bin, y_bin]

def assign_map_values_to_catalog(intbl, map_array, x_key='X_IMAGE', y_key='Y_IMAGE', 
                                 new_col='ZP', physical_max=22000):
    """
    Assign corresponding map values to every entry in a catalog and add a new column.

    This function estimates the proper bin size based on the physical coordinate range 
    (default 0 to physical_max) and the dimensions of the map_array.

    Parameters:
      intbl (astropy.table.Table):
            The catalog with coordinate columns.
      map_array (np.ndarray): 2D numpy array, e.g., 220x220.
      x_key (str): Name of the column with x coordinates (default 'X_IMAGE').
      y_key (str): Name of the column with y coordinates (default 'Y_IMAGE').
      new_col (str): Name of the new column to create (default 'ZP').
      physical_max (int or float): Maximum physical coordinate value (default is 22000).

    Returns:
      The modified catalog with the new column added.

    Example:
      >>> catalog = assign_map_values_to_catalog(catalog, zp_map, new_col='ZP')
    """
    # Number of bins along x (and y) inferred from the map's shape.
    n_bins = map_array.shape[0]
    # Compute the bin size based on the physical coordinate range.
    bin_size = int(physical_max / n_bins)
    # Generate bin edges from 0 to physical_max.
    bin_edges = np.arange(0, physical_max, bin_size)

    # Convert the catalog's coordinates into bin indices.
    x_bins = np.digitize(intbl[x_key], bins=bin_edges) - 1
    y_bins = np.digitize(intbl[y_key], bins=bin_edges) - 1

    # Clip the indices to ensure they fall within valid ranges.
    x_bins = np.clip(x_bins, 0, n_bins - 1)
    y_bins = np.clip(y_bins, 0, n_bins - 1)

    # Assign the corresponding map value to each entry in the catalog.
    intbl[new_col] = map_array[x_bins, y_bins]
    return intbl


#%%
    numimg  = hdr['NUMIMAGE']
        
    center  = [hdr['CRVAL1'], hdr['CRVAL2']]
    
    xscale  = hdr['NAXIS1'] * pixscale # arcsec
    yscale  = hdr['NAXIS2'] * pixscale # arcsec
    frac    = 2.1
    radius  = frac*np.mean([xscale, yscale])/3600 # searching radius in deg
            
    obses = []
    for l in range(numimg):
        obses.append(hdr[f'OBSERV{hex(l)[-1]}'])

    try:
        reftbl, bcoef, vcoef, rcoef, icoef  = GAIAXP_query(field, os.path.join(path_cat, 'gaiaxp'), nctio=obses.count('kmtc'), nsaao=obses.count('kmts'), nsso=obses.count('kmta'))
    except FileNotFoundError:
        path_ref    = f'{path_cat}apass/'
        os.makedirs(path_ref, exist_ok=True)
        try:
            reftbl  = ascii.read(f'{path_ref}apass_{field}.cat')
        except FileNotFoundError:
            frac    = 3 # >2*np.sqrt(2) due to dithering
            radius  = frac*np.mean([xscale, yscale])/3600 # searching radius in deg
            reftbl  = apass_query(center.ra.deg, center.dec.deg, radius)
            reftbl.write(f'{path_ref}apass_{field}.cat', format='ascii', overwrite=True)
        hdr['PHOTREF']  = 'APASS DR9'

    reftbl  = reftbl[reftbl[f'XP_{band}flag']==1]