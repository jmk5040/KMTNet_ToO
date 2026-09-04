#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Import packages
import time
import shutil
import numpy as np
import os, re, glob
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
#%% Import utility functions
from KMTNet_util_functions import (
    rss, apass_query, GAIAXP_query, sort_BVRI, limitmag, 
    matching, star4zp, zpcal, add_colorbar, date2MJD, 
    MJD2date, create_ldac_fits, hotpants, invert_image, 
    mask2weight, generate_snapshot, rename_convention, 
    safe_load_fits, find_longest_exposure_image, read_header,
    parse_region_bounds, mosaic_image, combine_subtracted_images,
    calculate_crosstalk_positions, build_sex_command
)
#%% KS4 Reference Image Reduction Functions
#%% KS4 Reference Image Bad Pixel Cleaning
def badpixel_clean(img, mask, outname, path_cfg, thresh=1e6):
    """
    Generate a bad-pixel corrected image using SExtractor's weighting scheme.
    
    This function performs bad pixel correction by using SExtractor with an inverted bad pixel mask as the weight map. Valid pixels are assigned a value of 1 and masked regions a value of 0. The correction employs MASK_TYPE = CORRECT and enables CLEAN = Y to interpolate over masked regions and suppress spurious detections. A bad-pixel corrected image is generated using CHECKIMAGE_TYPE = -BACKGROUND, with BACK_TYPE = MANUAL and BACK_VALUE = 0, producing an image identical to the original except for the interpolated regions.
    
    Parameters
    ----------
    img : str
        Path to the input FITS image file (stacked KMTNet image)
    mask : str
        Path to the bad pixel mask file (1 = bad pixel, 0 = good pixel)
    outname : str
        Path for the output corrected image file (cleaned KMTNet image)
    path_cfg : str
        Directory path containing SExtractor configuration files
    thresh : float, optional
        Detection threshold for SExtractor. Default is 1e6.
        
    Returns
    -------
    str
        Path to the output corrected image file
        
    Notes
    -----
    The function uses the following SExtractor configuration files:
    - kmtnet.sex: Main SExtractor configuration
    - kmtnet_imask.param: Parameter file
    - kmtnet.conv: Convolution filter
    - kmtnet.nnw: Neural network weights
    
    The corrected image is identical to the original except for interpolated
    regions where bad pixels were masked. This image is suitable for subsequent
    photometric measurements and source catalog construction.
    
    Examples
    --------
    >>> corrected_img = badpixel_clean('science.fits', 'mask.fits', 
    ...                                'corrected.fits', '/path/to/config/')
    """
    
    # configs
    cfg         = os.path.join(path_cfg, 'kmtnet.sex')
    param       = os.path.join(path_cfg, 'kmtnet_imask.param')
    conv        = os.path.join(path_cfg, 'kmtnet.conv')
    nnw         = os.path.join(path_cfg, 'kmtnet.nnw')

    # weight image
    weightname  = mask2weight(mask)
    prompt_cat  = f' -c {cfg} -CATALOG_TYPE NONE'
    prompt_cfg  = f' -PARAMETERS_NAME {param} -FILTER_NAME {conv} -STARNNW_NAME {nnw}'
    prompt_wgt  = f' -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE {weightname} -MASK_TYPE CORRECT -RESCALE_WEIGHTS Y -WEIGHT_GAIN Y -DETECT_THRESH {thresh} -ANALYSIS_THRESH {thresh} -BACK_TYPE MANUAL -BACK_VALUE 0.0'
    prompt_chk  = f' -CHECKIMAGE_TYPE -BACKGROUND -CHECKIMAGE_NAME {outname}'

    prompt  = f'sex {img} {prompt_cat} {prompt_cfg} {prompt_wgt} {prompt_chk}'
    os.system(prompt)

    # remove temporary files
    os.remove(weightname)

    # update header
    with fits.open(outname, 'update') as f:
        for hdu in f:
            hdu.header['MASKNAME']   = (os.path.basename(mask), "Bad pixel mask")
            hdu.header['MASKTYPE']   = ('CORRECT', "Bad pixel mask cleared")
    return outname    

#%% KS4 Reference Image Photometry
def ks4_photometry(img, mask, path_output, path_cfg, path_ref, path_result, pixscale=0.4, clsstar=0.8, apertures=['FWHM', 'APER3', 'APER5', 'APER10', 'AUTO'], detimg=None):
    
    from astropy.stats import sigma_clip
    from astropy.table import Table, Column

    """
    Parameters
    ----------
    img : str
        Path to the input FITS image file (stacked KMTNet image)
    mask : str
        Path to the bad pixel mask file (1 = bad pixel, 0 = good pixel)
    path_output : str
        Path for the output photometry catalog file
    path_cfg : str
        Directory path containing SExtractor configuration files
    path_ref : str
        Directory path containing GAIAXP catalog (for zero-point calibration)
    path_result : str
        Directory path for the output result files
    pixscale : float, optional
        Pixel scale of the image in arcsec/pixel. Default is 0.4.
    clsstar : float, optional
        Minimum CLASS_STAR value for point sources. Default is 0.8.
    apertures : list, optional
        List of aperture sizes to use for photometry. Default is ['FWHM', 'APER3', 'APER5', 'APER10', 'AUTO'].
    detimg : str, optional
        Path to the detection image file (if dual-band photometry is desired). Default is None.
    """

    # dualband check
    singlephot = True
    if detimg is not None and os.path.isfile(detimg):
        try:
            dualband = fits.getheader(detimg)['FILTER']
            dualphot = True
        except KeyError:
            dualphot = False

    # header check
    hdul    = fits.open(img)
    hdr     = hdul[0].header
    band    = hdr['FILTER']
    field   = hdr['FIELD1'] # KS4 specific field name (0-2748)

    # SExtractor photometry configs
    cfg         = os.path.join(path_cfg, 'kmtnet.sex')
    param       = os.path.join(path_cfg, 'kmtnet_imask.param')
    conv        = os.path.join(path_cfg, 'kmtnet.conv')
    nnw         = os.path.join(path_cfg, 'kmtnet.nnw')
    
    # seeing check (with 2k cropped image) with SExtractor
    try:
        seeing     = hdr['FWHM']
    except KeyError:
        if os.path.isfile(f'crop_{img}'): os.remove(f'crop_{img}')
        os.system('imcopy {0}[4000:6000,4000:6000] crop_{0}'.format(img))
        os.system(f'sex crop_{img} -c {cfg} -PARAMETERS_NAME {param} -FILTER_NAME {conv} -STARNNW_NAME {nnw}')
        # get seeing from the cropped image
        tempcat     = Table(fits.open('test.fits')[1].data)
        tempcat     = tempcat[tempcat['FWHM_IMAGE'] != 0]
        seeing      = np.median(tempcat[tempcat['CLASS_STAR']>np.median(tempcat['CLASS_STAR'])]['FWHM_IMAGE'] * pixscale)
        os.system('rm test.fits')
        os.system(f'rm crop_{img}')
    peeing      = seeing/pixscale

    # prompts
    # apertures: should be more flexible for 2FWHM, ... (should be more flexible)
    photapers   = ','.join([str(float(ap.split('APER')[-1])/pixscale) for ap in apertures if ap != 'AUTO' and ap != 'FWHM'])
    if 'FWHM' in apertures:
        photapers  = f'{peeing},{photapers}'
    prompt_opt  = f' -PIXEL_SCALE {pixscale} -SEEING_FWHM {seeing:.2f} -PHOT_APERTURES {photapers}'
    prompt_cfg  = f' -PARAMETERS_NAME {param} -FILTER_NAME {conv} -STARNNW_NAME {nnw}'
    prompt_flg  = f' -FLAG_IMAGE {mask} -FLAG_TYPE MAX'
    prompt_chk  = f' -CHECKIMAGE_TYPE BACKGROUND_RMS -CHECKIMAGE_NAME {img.replace(".fits",".bkgrms")}'
    # photometry catalog
    catname_single      = f'{path_output}ks4_{field}_{band}_single.fits'
    inim_single         = img
    prompt_catsingle    = f' -c {cfg} -CATALOG_NAME {catname_single}'
    prompt      = 'sex '+inim_single+prompt_catsingle+prompt_cfg+prompt_opt+prompt_flg+prompt_chk
    os.system(prompt)

    # dual mode photometry catalog
    if dualphot:
        catname_dual   = f'{path_output}ks4_{field}_{band}_dual.fits'
        if band == dualband:
            shutil.copy(catname_single, catname_dual)
        else:
            inim_dual      = f'{detimg},{img}'
            prompt_catdual = f' -c {cfg} -CATALOG_NAME {catname_dual}'
            prompt  = 'sex '+inim_dual+prompt_catdual+prompt_cfg+prompt_opt+prompt_flg
            os.system(prompt)

    intbl_single    = Table.read(catname_single)
    if dualphot: intbl_dual    = Table.read(catname_dual)

    # zero-point calibration

    # GAIAXP catalog query
    try:
        reftbl, bcoef, vcoef, rcoef, icoef  = GAIAXP_query(field, path_ref)
    except FileNotFoundError:
        print(f'{field} {band} GAIAXP catalog not found. ZP calibration skipped.')
        return catname_single
    reftbl  = reftbl[reftbl[f'GAIA_{band}flag']==1]

    # matching with the reference
    
    for i, pros in enumerate([singlephot, dualphot]): 
        
        if pros:
            rad = 2 # matching radius in arcsec
            if i==0:
                intbl   = Table.read(catname_single)
                param_matching  = dict(intbl    = intbl,
                                        reftbl   = reftbl,
                                        inra     = intbl['ALPHA_J2000'], 
                                        indec    = intbl['DELTA_J2000'],
                                        refra    = reftbl['RAJ2000'], 
                                        refdec   = reftbl['DEJ2000'],
                                        sep      = rad)
                ptbl    = matching(**param_matching)
                mode    = 'single'
            elif i==1:
                intbl   = Table.read(catname_dual)
                param_matching  = dict(intbl    = intbl,
                                        reftbl   = reftbl,
                                        inra     = intbl['ALPHA_J2000'], 
                                        indec    = intbl['DELTA_J2000'],
                                        refra    = reftbl['RAJ2000'], 
                                        refdec   = reftbl['DEJ2000'],
                                        sep      = rad)
                ptbl    = matching(**param_matching)
                mode    = 'dual'

        # Split the aperture columns
        for i, aperture in enumerate(apertures):
            if aperture == 'AUTO':
                pass
            else:
                intbl[f'MAG_{aperture}'] = Column([x[i] for x in intbl['MAG_APER']], name=f'MAG_{aperture}')
                ptbl[f'MAG_{aperture}'] = Column([x[i] for x in ptbl['MAG_APER']], name=f'MAG_{aperture}')
                intbl[f'MAGERR_{aperture}'] = Column([x[i] for x in intbl['MAGERR_APER']], name=f'MAGERR_{aperture}')
                ptbl[f'MAGERR_{aperture}'] = Column([x[i] for x in ptbl['MAGERR_APER']], name=f'MAGERR_{aperture}')

        # Initialize the columns
        for col in ['MISALIGN', 'MISALIGNERR', 'NUMASTRO', 'NUMPHOTO']:
            if col not in intbl.colnames:
                intbl[col] = [-99.] * len(intbl)  # Initialize with default value -99

            # Bin configuration
            centbins    = 200
            extbins     = 1000
            bin_edges   = np.arange(0, 22001, centbins) # 22001 = hdr['NAXIS1']+1

            # Filter the table for the 'single' mode
            if mode == 'single':
                ptbl = ptbl[(ptbl['FLAGS'] == 0) & (ptbl['IMAFLAGS_ISO'] == 0) & (ptbl['CLASS_STAR']>clsstar)]
                ptbl = ptbl[ptbl['MAGERR_AUTO']<0.05]
                ptbl = ptbl[ptbl['phot_g_mean_mag']<20]
                ptbl = ptbl[ptbl['phot_g_mean_mag']>14]
                # Precompute overlap masks
                overlap_mask_x = {
                    l: (ptbl['X_IMAGE'] >= bin_edges[l - 1] - (extbins - centbins) / 2) &
                    (ptbl['X_IMAGE'] < bin_edges[l] + (extbins - centbins) / 2)
                    for l in range(1, len(bin_edges))
                }

                overlap_mask_y = {
                    m: (ptbl['Y_IMAGE'] >= bin_edges[m - 1] - (extbins - centbins) / 2) &
                    (ptbl['Y_IMAGE'] < bin_edges[m] + (extbins - centbins) / 2)
                    for m in range(1, len(bin_edges))
                }

                # Initialize maps
                num_bins = len(bin_edges) - 1
                align_map = np.zeros((num_bins, num_bins))
                alerr_map = np.zeros((num_bins, num_bins))
                astar_map = np.zeros((num_bins, num_bins))

                for l in range(1, len(bin_edges)):
                    for m in range(1, len(bin_edges)):
                    # Ensure correct overlap masks for X and Y
                        x_in_overlap = overlap_mask_x.get(l, np.zeros(len(ptbl), dtype=bool))
                        y_in_overlap = overlap_mask_y.get(m, np.zeros(len(ptbl), dtype=bool))
                        indexes = np.where(x_in_overlap & y_in_overlap)
                        amatches = ptbl[indexes]
                        # Calculate median misalignment for the bin
                        if len(amatches) > 0:
                            misalign = np.median(amatches['sep']) * 3600  # Convert to arcseconds
                            misalerr = np.std(amatches['sep']) * 3600  # Convert to arcseconds
                        else:
                            misalign = 0  # No matches in the bin
                            misalerr = 0

                        # Assign values to the maps
                        align_map[l - 1, m - 1] = misalign
                        alerr_map[l - 1, m - 1] = misalerr
                        astar_map[l - 1, m - 1] = len(amatches)  # Number of matches in the bin
            # Assign astrometric corrections to the table
            x_bins_intbl = np.digitize(intbl['X_IMAGE'], bins=bin_edges) - 1
            y_bins_intbl = np.digitize(intbl['Y_IMAGE'], bins=bin_edges) - 1
            intbl['MISALIGN'] = align_map[x_bins_intbl, y_bins_intbl]
            intbl['NUMASTRO'] = astar_map[x_bins_intbl, y_bins_intbl]

            # Save astrometric correction maps
            np.save(f'{path_result}align_map_{field}_{band}_{mode}.npy', align_map)
            np.save(f'{path_result}alerr_map_{field}_{band}_{mode}.npy', alerr_map)
            np.save(f'{path_result}astar_map_{field}_{band}_{mode}.npy', astar_map)

            # Photometric zero-point calculation
            for aperture in apertures:
                
                refmaglower = 14
                refmagupper = 19
                flagcut = 0

                # Filter ptbl to create ctbl
                ctbl = ptbl[(ptbl['IMAFLAGS_ISO'] == 0) & (ptbl['CLASS_STAR']>clsstar)]
                ctbl = ctbl[ctbl[f'MAG_{aperture}'] != 99]
                ctbl = ctbl[ctbl['FLAGS'] <= flagcut]
                ctbl = ctbl[ctbl[f'{band}mag'] < refmagupper]
                ctbl = ctbl[ctbl[f'{band}mag'] > refmaglower]

                param_zpcal     = dict(intbl=ctbl, inmagkey=f'MAG_{aperture}', refmagkey=f'{band}mag', sigma=2.0)

                global_zp, global_zper, otbl, xtbl = zpcal(**param_zpcal)
                with fits.open(img, 'update') as f:
                    for hdu in f:
                        hdu.header[f'ZP{aperture}']   = (round(global_zp, 3), f'Photometric zero-point for {aperture} [ABmag]')
                        hdu.header[f'ZE{aperture}']   = (round(global_zper, 3), f'Photometric zero-point uncertainty for {aperture} [ABmag]')
                        hdu.header[f'ZS{aperture}']   = (len(otbl), f'The number of stars for {aperture} zero-point calculation')

                # Initialize the columns
                for col_type in ['ZP', 'ZPERR']:
                    col_name = f'{col_type}_{aperture}'
                    if col_name not in intbl.colnames:
                        intbl[col_name] = [-99.0] * len(intbl)

                # Bin configuration (400x400)
                centbins    = 400
                extbins     = 400
                bin_edges   = np.arange(0, 22001, centbins)

                # Initialize the maps
                num_bins = len(bin_edges) - 1
                zp_map = np.zeros((num_bins, num_bins))
                zperr_map = np.zeros((num_bins, num_bins))
                pstar_map = np.zeros((num_bins, num_bins))

                overlap_mask_x = {
                    l: (ctbl['X_IMAGE'] >= bin_edges[l - 1] - (extbins - centbins) / 2) &
                    (ctbl['X_IMAGE'] < bin_edges[l] + (extbins - centbins) / 2)
                    for l in range(1, len(bin_edges))
                }

                overlap_mask_y = {
                    m: (ctbl['Y_IMAGE'] >= bin_edges[m - 1] - (extbins - centbins) / 2) &
                    (ctbl['Y_IMAGE'] < bin_edges[m] + (extbins - centbins) / 2)
                    for m in range(1, len(bin_edges))
                }

                # Iterate over bins (400x400)
                for l in range(1, len(bin_edges)):
                    for m in range(1, len(bin_edges)):
                        # Combine overlap masks (400x400)
                        x_in_overlap = overlap_mask_x[l]
                        y_in_overlap = overlap_mask_y[m]
                        indexes = np.where(x_in_overlap & y_in_overlap)
                        pmatches = ctbl[indexes]

                        # Extract matches
                        align_map   = np.load(f'{path_result}align_map_{field}_{band}_{mode}.npy')
                        radius = np.max([0.5 / 3600, align_map[l - 1, m - 1]])
                        pmatches = pmatches[pmatches['sep'] < radius]

                        # Calculate statistics
                        if len(pmatches) > 0:
                            # Extract non-masked (valid) values after clipping
                            zplist = pmatches[f'{band}mag'] - pmatches[f'MAG_{aperture}']
                            zplist_clipped = sigma_clip(zplist, sigma=2, maxiters=None, cenfunc='median')
                            valid_zplist = zplist_clipped.data[~zplist_clipped.mask]

                            zp_map[l - 1, m - 1] = np.median(valid_zplist)
                            zperr_map[l - 1, m - 1] = np.std(valid_zplist)
                            pstar_map[l - 1, m - 1] = len(valid_zplist)
                        else:
                            zp_map[l - 1, m - 1] = 30
                            zperr_map[l - 1, m - 1] = 0
                            pstar_map[l - 1, m - 1] = 0
                x_bins_intbl = np.digitize(intbl['X_IMAGE'], bins=bin_edges) - 1
                y_bins_intbl = np.digitize(intbl['Y_IMAGE'], bins=bin_edges) - 1
                
                # Assign corrections to sources
                intbl[f'MAG_{aperture}'] += zp_map[x_bins_intbl, y_bins_intbl]
                intbl[f'MAGERR_{aperture}'] = np.sqrt(
                    intbl[f'MAGERR_{aperture}'] ** 2 + zperr_map[x_bins_intbl, y_bins_intbl] ** 2
                )
                intbl[f'ZP_{aperture}'] = zp_map[x_bins_intbl, y_bins_intbl]
                intbl[f'ZPERR_{aperture}'] = zperr_map[x_bins_intbl, y_bins_intbl]

                # save result map
                np.save(f'{path_result}zp_map_{field}_{band}_{mode}_{aperture}.npy', zp_map)
                np.save(f'{path_result}zperr_map_{field}_{band}_{mode}_{aperture}.npy', zperr_map)
                np.save(f'{path_result}pstar_map_{field}_{band}_{mode}_{aperture}.npy', pstar_map)

                # depth check
                if aperture=='FWHM' and mode=='single':
                    bkgrms  = fits.getdata(img.replace(".fits",".bkgrms"))
                    skysig  = np.median(bkgrms[bkgrms!=0])
                    depth   = limitmag(5, global_zp, peeing, skysig)
                    os.system(f'rm {img.replace(".fits",".bkgrms")}')

            intbl['NUMPHOTO'] = pstar_map[x_bins_intbl, y_bins_intbl]
            zpcatname   = f'{path_result}ks4_{field}_{band}_{mode}.zp.fits'

    # update header
    seeing      = np.median(intbl_single[intbl_single['CLASS_STAR']>clsstar]['FWHM_IMAGE'] * pixscale)       
    with fits.open(img, 'update') as f:
        for hdu in f:
            hdu.header['FWHM']      = (round(seeing, 3), "Median seeing of point sources [arcsec]")
            hdu.header['FLAGIMG']   = (os.path.basename(mask), "Mask image for IMAFLAGS_ISO")
            hdu.header['DEPTH5']    = (depth, "5sigma detection limiting magnitude for seeing size aperture")
            hdu.header['PHOTREF']   = ("Gaia XP", "Reference catalog used for source catalog generation")

    return zpcatname

#%% plot photometry residue and RMSE calculation
def plot_phot_residue(img, intbl, reftbl, path_plot, aperture='AUTO', clsstar=0.8, maglower=14, magupper=19, flagcut=0, magerrcut=0.05, hdr_update=False):
    """
    Plot photometry residue and calculate RMSE for KMTNet reference images.
    
    This function compares photometric measurements between KMTNet images and reference
    catalogs to assess photometric quality. It generates diagnostic plots showing
    magnitude differences, spatial distribution of residuals, and statistical metrics
    for photometric calibration validation.
    
    Parameters
    ----------
    img : str
        Path to the KMTNet FITS image file. The function extracts field and band
        information from the FITS header (FIELD1 and FILTER keywords).
    intbl : astropy.table.Table
        Input source catalog from SExtractor containing photometric measurements.
        Must include columns: ALPHA_J2000, DELTA_J2000, MAG_{aperture}, MAGERR_{aperture},
        FLAGS, IMAFLAGS_ISO, CLASS_STAR, X_IMAGE, Y_IMAGE.
    reftbl : astropy.table.Table
        Reference catalog for photometric comparison. Must include columns:
        RAJ2000, DEJ2000, {band}mag (where {band} is the filter band).
    path_plot : str
        Path to the output directory where diagnostic plots will be saved.
    aperture : str, optional
        Aperture type for photometric comparison ('AUTO', 'APER3', 'APER5', etc.).
        Default is 'AUTO'.
    clsstar : float, optional
        Minimum CLASS_STAR value for stellar source selection. Default is 0.8.
    maglower : float, optional
        Lower magnitude limit for source selection. Default is 14.
    magupper : float, optional
        Upper magnitude limit for source selection. Default is 19.
    flagcut : int, optional
        Maximum FLAGS value to accept for source selection. Default is 0.
    magerrcut : float, optional
        Maximum magnitude error to accept for source selection. Default is 0.05.
    hdr_update : bool, optional
        Whether to update the FITS header with RMSE information. Default is False.
    
    Returns
    -------
    int
        Returns 0 upon successful completion.
    
    Notes
    -----
    The function performs the following operations:
    
    1. **Source Matching**:
       - Matches sources between input and reference catalogs
       - Uses 0.5 arcsecond matching radius
       - Applies quality cuts for reliable photometric comparison
    
    2. **Quality Filtering**:
       - Magnitude range: maglower < MAG_{aperture} < magupper
       - FLAGS = flagcut (excludes sources with detection flags)
       - IMAFLAGS_ISO = 0 (excludes sources with bad pixels)
       - CLASS_STAR > clsstar (selects stellar sources)
       - MAGERR_{aperture} < magerrcut (excludes sources with large errors)
    
    3. **Photometric Analysis**:
       - Calculates magnitude differences: MAG_{aperture} - {band}mag
       - Computes median difference and RMSE using 3-sigma clipping
       - Assesses photometric precision and systematic offsets
    
    4. **Diagnostic Plot Generation**:
       - **1D Plot**: Magnitude difference vs. magnitude with error bars
       - **Histogram**: Distribution of magnitude differences
       - **2D Map**: Spatial distribution of photometric residuals
       - **Statistics**: Median difference and RMSE displayed
    
    5. **Header Updates** (if requested):
       - Adds RMSPHOT keyword with RMSE value
       - Records photometric precision for quality assessment
    
    Quality Cuts Applied
    --------------------
    - **FLAGS = flagcut**: Excludes sources with detection flags
    - **IMAFLAGS_ISO = 0**: Excludes sources with bad pixels
    - **CLASS_STAR > clsstar**: Selects stellar sources only
    - **Magnitude range**: maglower < MAG_{aperture} < magupper
    - **Magnitude error**: MAGERR_{aperture} < magerrcut
    
    Output Files
    ------------
    - **Diagnostic Plot**: `RESIDUEMAP_{field}_{band}_{aperture}.png`
    
    Header Keywords Added (if hdr_update=True)
    ------------------------------------------
    - RMSPHOT: RMSE of photometric differences (3-sigma clipped)
    
    Examples
    --------
    >>> plot_phot_residue('/path/to/image.fits', input_catalog, reference_catalog,
    ...                   '/path/to/plots/', aperture='AUTO', clsstar=0.8,
    ...                   maglower=14, magupper=19, hdr_update=True)
    
    See Also
    --------
    zeropoint_homogenization : Zero-point correction function
    KMTNet_ToO_functions.zpscale : Zero-point scaling function
    
    Notes
    -----
    This function is typically used for photometric quality assessment of reference
    images and validation of zero-point calibrations.
    """

    from astropy.stats import sigma_clip
    import matplotlib.gridspec as gridspec

    # header information
    field   = fits.getheader(img)['FIELD1']
    band    = fits.getheader(img)['FILTER']

    # reference catalog matching
    param_matching  = dict(intbl    = intbl,
            reftbl   = reftbl,
            inra     = intbl['ALPHA_J2000'], 
            indec    = intbl['DELTA_J2000'],
            refra    = reftbl['RAJ2000'], 
            refdec   = reftbl['DEJ2000'],
            sep      = 0.5)
    tester    = matching(**param_matching)
    tester  = tester[tester[f'MAG_{aperture}']>maglower]
    tester  = tester[tester[f'MAG_{aperture}']<magupper]
    tester  = tester[tester['FLAGS']==flagcut]
    tester  = tester[tester['IMAFLAGS_ISO']==0]
    tester  = tester[tester['CLASS_STAR']>clsstar]
    tester  = tester[tester[f'MAGERR_{aperture}']<magerrcut]
    
    # photometry comparison
    magdif  = tester[f'MAG_{aperture}']-tester[f'{band}mag']
    magerr  = tester[f'MAGERR_{aperture}']
    meddif  = round(np.nanmedian(magdif), 3)
    clean_magdif = magdif[np.isfinite(magdif)]
    clipped = sigma_clip(clean_magdif, sigma=3)
    rmse = round(np.sqrt(np.nanmean(clipped**2)), 3)
    
    # plot
    plt.figure(figsize=(8, 12))
    gs = gridspec.GridSpec(15, 15)
    plt.rcParams.update({'font.size': 14})

    plt.suptitle(f'KS4 & REF Photometry Comparison\nField : {field}, MAG_{aperture}, {band} band,\nMagnitude Difference RMSE : {rmse:.3f} ABmag\n5$\sigma$ Image Depth : ABmag')
    ax_1d   = plt.subplot(gs[9:15, :10])
    ax_hist = plt.subplot(gs[9:15, 10:])
    ax_2d   = plt.subplot(gs[0:8, :14])

    ax_1d.errorbar(tester[f'MAG_{aperture}'], magdif, yerr=magerr, ms=6, ls='', c='crimson', marker='o', capsize=4, capthick=1, alpha=0.1)
    ax_1d.set(xlabel=r'$m_{KS4}$ [ABmag]', ylabel=r'$m_{KS4} - m_{REF}$ [ABmag]')
    ax_1d.set_xlim(19.5,13.75)
    ax_1d.set_ylim(-1,1)
    ax_1d.grid(which='major',linestyle='-', alpha=0.5)
    
    # histograms
    uweights   = np.ones_like(magdif)/len(magdif)
    ax_hist.hist(magdif,weights=uweights, bins=np.arange(-1,1,0.05), color='crimson', orientation='horizontal',align='mid')
    ax_hist.axhline(y=meddif, color='dodgerblue', linestyle='-', alpha=0.75, label='{:6}={:6.3f}mag'.format('Median', meddif))
    ax_hist.axhline(y=meddif-rmse, color='dodgerblue', linestyle='--', alpha=0.75, label='{:6}={:6.3f}mag'.format('RMSE', rmse))
    ax_hist.axhline(y=meddif+rmse, color='dodgerblue', linestyle='--', alpha=0.75)
    ax_hist.axes.yaxis.set_ticklabels([])
    ax_hist.set(xlabel='proportion')
    ax_hist.set_ylim(-1,1)
    ax_hist.legend(fontsize=10, loc='upper right')
    ax_hist.grid(which='major',linestyle='-', alpha=0.5)
    
    plot = ax_2d.scatter(tester['X_IMAGE'], tester['Y_IMAGE'], marker='o', c=magdif, cmap='seismic', edgecolor='k', alpha=0.7, label=f'FLAG==0 ({len(tester)})\nCLASS_STAR>{clsstar}')
    add_colorbar(plot, clabel=r'$m_{KS4} - m_{REF}$ [mag]', clim=[-0.5, 0.5])
    ax_2d.legend(loc='upper right')
    ax_2d.set(xlabel='X axis [pixel]', ylabel='Y axis [pixel]')
    ax_2d.set_xlim(-1000, 24000)
    ax_2d.set_ylim(-1000, 24000)
    plt.savefig(f'{path_plot}RESIDUEMAP_{field}_{band}_{aperture}.png')
    print(f'{path_plot}RESIDUEMAP_{field}_{band}_{aperture}.png saved')
    plt.close()

    if hdr_update:
        with fits.open(img, 'update') as f:
            for hdu in f:
                hdu.header['RMSPHOT']   = (rmse, f'RMSE of PHOTREF mag - KMTN mag (3sigma clipped) for {aperture} aperture')
    
    return 0

#%% zero-point homogenization using zero-point correctionmap
def zeropoint_homogenization(img, path_map, outname, aperture='APER5', mode='single', zp_to_scale=30):
    """
    Zero-point homogenization using pre-computed zero-point correction maps.
    
    This function applies spatial zero-point corrections to KMTNet images using
    pre-computed correction maps. It homogenizes photometric zero-points across
    the entire image by applying flux scaling factors derived from zero-point
    variations, ensuring uniform photometric quality.
    
    Parameters
    ----------
    img : str
        Path to the input KMTNet FITS image file. The function extracts field
        and band information from the FITS header (FIELD1 and FILTER keywords).
    path_map : str
        Path to the directory containing zero-point correction maps:
        - zp_map_{field}_{band}_{mode}_{aperture}.npy: Zero-point correction map
        - zperr_map_{field}_{band}_{mode}_{aperture}.npy: Zero-point error map
    outname : str
        Path and filename for the output corrected image.
    aperture : str, optional
        Aperture type for zero-point correction ('APER5', 'APER3', 'AUTO', etc.).
        Default is 'APER5'.
    mode : str, optional
        Correction mode for the zero-point map ('single', 'multi', etc.).
        Default is 'single'.
    zp_to_scale : float, optional
        Target zero-point magnitude for scaling. Default is 30.0.
    
    Returns
    -------
    None
        Results are written to the output FITS file.
    
    Notes
    -----
    The function performs the following operations:
    
    1. **Map Loading**:
       - Loads zero-point correction map (zp_map) and error map (zperr_map)
       - Maps are stored as NumPy arrays with spatial zero-point variations
       - Maps are specific to field, band, mode, and aperture combination
    
    2. **Flux Scaling Calculation**:
       - Calculates zero-point difference: del_zp = zp_to_scale - zp_map
       - Converts to flux ratio: fratio = 10^(del_zp/2.5)
       - Each pixel gets individual scaling factor based on local zero-point
    
    3. **Image Processing**:
       - Upscales flux ratio map to match image dimensions
       - Uses scipy.ndimage.zoom with linear interpolation
       - Applies flux scaling to image data: data_scaled = data * fratio_upscaled
    
    4. **Header Updates**:
       - Updates MAGZERO with target zero-point
       - Adds EMAGZERO with average zero-point uncertainty
       - Updates SATURATE and UNDERSAT with scaling factors
       - Sets PHOTREF to 'GAIA DR3'
    
    5. **Output Generation**:
       - Saves corrected image with updated headers
       - Preserves original image structure and metadata
    
    Zero-point Correction Process
    -----------------------------
    - **Input**: Image with non-uniform zero-point across field
    - **Correction**: Spatial flux scaling based on zero-point variations
    - **Output**: Image with uniform zero-point (zp_to_scale)
    - **Method**: Pixel-wise flux multiplication with correction factors
    
    Aperture Types Supported
    ------------------------
    - **FWHM**: Seeing size aperture
    - **AUTO**: Kron-like aperture
    - **APER3**: 3 arcsec aperture
    - **APER5**: 5 arcsec aperture (default)
    - **APER10**: 10 arcsec aperture
    
    Header Keywords Updated
    -----------------------
    - MAGZERO: Target zero-point magnitude
    - EMAGZERO: Average zero-point uncertainty
    - SATURATE: Updated saturation level (scaled)
    - UNDERSAT: Updated undersaturation level (scaled)
    - PHOTREF: Reference catalog used ('GAIA DR3')
    
    Examples
    --------
    >>> zeropoint_homogenization('/path/to/image.fits', '/path/to/maps/',
    ...                          '/path/to/output.fits', aperture='APER5',
    ...                          mode='single', zp_to_scale=30.0)
    
    See Also
    --------
    plot_phot_residue : Photometric quality assessment function
    KMTNet_ToO_functions.zpscale : Zero-point scaling function
    
    Notes
    -----
    This function is typically used for reference image preparation, ensuring
    uniform photometric quality across the entire field of view. The correction
    maps should be pre-computed using photometric analysis of reference stars.
    """
        
    from scipy.ndimage import zoom

    hdul    = fits.open(img)
    data    = hdul[0].data
    hdr     = hdul[0].header
    field   = hdr['FIELD1']
    band    = hdr['FILTER']

    # ZP scaling process
    # ZP map load
    zp_map  = np.load(f'{path_map}zp_map_{field}_{band}_{mode}_{aperture}.npy')
    zperr_map = np.load(f'{path_map}zperr_map_{field}_{band}_{mode}_{aperture}.npy')

    # ZP scaler define
    del_zp = zp_to_scale - zp_map
    fratio = 10**(del_zp/(2.5))

    # Upscale the flux ratio map to match the image physical dimension
    upscale_factor = data.shape[0] // zp_map.shape[0]
    fratio_upscaled = zoom(fratio, upscale_factor, order=1)

    # Apply the flux ratio to the image
    data_scaled = data * fratio_upscaled.T
    data_scaled = data_scaled.astype(np.float32)
    
    # Save the corrected image
    # headers for zero-points
    zp_error = round(np.mean(zperr_map[zperr_map != 0]), 3)

    mapping = {
        'FWHM'  : 'seeing size aperture',
        'AUTO'  : 'Kron-like aperture',
        'APER3' : '3 arcsec aperture',
        'APER5' : '5 arcsec aperture',
        'APER10': '10 arcsec aperture',
    }

    apstring =  mapping.get(aperture, f'{aperture}')

    hdr['MAGZERO']  = (zp_to_scale, f"Magnitude zeropoint for {apstring}.")
    hdr['EMAGZERO'] = (zp_error, f"Zeropoint uncertainty for {apstring}.")

    hdr['SATURATE'] = int(np.mean(fratio)*hdr['SATURATE'])
    hdr['UNDERSAT'] = int(np.mean(fratio)*hdr['UNDERSAT'])
    hdr['PHOTREF'] = 'GAIA DR3'

    fits.writeto(outname, data_scaled, hdr, overwrite=True)

    return 

#%% load zero-point correction map
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

#%% get zero-point correction map value for target
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

#%% assign zero-point correction map values to catalog
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

# %% Pan-STARRS-1 Reference Image Generation with "PanStitch" (in case of no KS4 reference image)
def generate_panstarrs_reference(
    field,
    cra,
    cdec,
    path_output,
    path_cfg,
    filte='r',
    xsize=22000,
    ysize=22000,
    pixscale=0.4,
    n_grid=16,
    m_grid=16,
    margin_frac=0.0,
    slice_size=4000,
    swarp_config='kmtnet.swarp'
):
    """
    Generates a Pan-STARRS-1 reference image for a given field using PanStitch.

    This function performs the following steps:
    1. Generates a grid of coordinates to cover the desired field of view.
    2. Queries the Pan-STARRS archive to get image information.
    3. Downloads the individual image slices.
    4. Uses SWarp to stitch the slices into a single FITS image.

    Parameters:
      field (str): The name of the target field (e.g., "0123.4").
      cra (str): The center Right Ascension in hms.
      cdec (str): The center Declination in dms.
      path_output (str): The directory to save the output files.
      path_cfg (str): The directory containing configuration files (e.g., kmtnet.swarp).
      filte (str): The filter to use ('g', 'r', 'i', 'z', 'y'). Default is 'r'.
      xsize (int): The width of the final image in pixels. Default is 22000.
      ysize (int): The height of the final image in pixels. Default is 22000.
      pixscale (float): The pixel scale of the final image in arcsec/pixel. Default is 0.4.
      n_grid (int): The number of grid points along the RA axis. Default is 8.
      m_grid (int): The number of grid points along the Dec axis. Default is 8.
      margin_frac (float): The fractional margin for downloading slices. Default is 0.0.
      slice_size (int): The size of the individual download slices in pixels. Default is 6000.
      swarp_config (str): The name of the swarp configuration file. Default is 'kmtnet.swarp'.

    Returns:
      str: The path to the final stacked FITS image.
    """
    try:
        from PanStitch import (
            generate_pointings, getimages, download_images_for_pointings,
            write_images_to_swarp, degrees_to_hms_dms, run_swarp
        )
    except ImportError:
        raise ImportError(
            "PanStitch package is required for Pan-STARRS reference image generation. "
            "Install it with: pip install PanStitch"
        )

    # 1. Define paths
    path_slice = os.path.join(path_output, 'ps1_slice')
    os.makedirs(path_slice, exist_ok=True)
    path_swarp_conf = os.path.join(path_cfg, swarp_config)
    list_file_path = os.path.join(path_slice, 'images_to_stitch.txt')

    # 2. Generate the grid of coordinates to download
    print(f"Generating {n_grid}x{m_grid} grid for {field} at ({cra}, {cdec})...")
    coord = SkyCoord(cra, cdec, unit=(u.hourangle, u.deg))
    cra_deg = coord.ra.deg
    cdec_deg = coord.dec.deg
    pointings = generate_pointings(cra_deg, cdec_deg, xsize, ysize, pixscale, n=n_grid, m=m_grid, margin_frac=margin_frac)

    # 3. Get image info from Pan-STARRS
    print("Querying Pan-STARRS for image information...")
    tra = [p[0] for p in pointings]
    tdec = [p[1] for p in pointings]
    try:
        image_table = getimages(tra, tdec, filters=filte, size=slice_size)
    except:
        print(f"No PS1 image found for {field} at ({cra}, {cdec})")
        return None

    # 4. Download the actual image files
    print(f"Downloading {len(image_table)} image slices to {path_slice}...")
    image_files = download_images_for_pointings(image_table, path_slice)

    # 5. Prepare and run SWarp to stitch the images
    print("Stitching images with SWarp...")
    write_images_to_swarp(list_file_path, image_files)

    # Calculate median exposure time for the output filename
    try:
        exptime = int(np.median([fits.getheader(f)['EXPTIME'] for f in image_files]))
    except Exception as e:
        print(f"Could not determine median exposure time: {e}. Using 0.")
        exptime = 100

    path_outim = os.path.join(path_output, f'ps1.{field}.{filte.upper()}.{exptime}sec.reduced.scaled.stack.fits')
    
    run_swarp(
        list_file_path,
        path_outim,
        path_swarp_conf,
        cra,
        cdec,
        xsize=xsize,
        ysize=ysize
    )
    # remove weight image
    os.remove(path_outim.replace('.fits', '.weight.fits'))
    # update header
    with fits.open(path_outim, 'update') as f:
        for hdu in f:
            hdu.header['OBJECT']    = f'{field}'
            hdu.header['FILTER']    = filte.upper()
            hdu.header['EXPTIME']   = exptime
            hdu.header['FWHM']      = 1.5
            hdu.header['CENTRA']    = cra
            hdu.header['CENTDEC']   = cdec

    os.system(f'chmod 777 {path_outim}')
    print(f"Process completed. Stacked image saved to: {path_outim}")

    return path_outim

# %% SkyMapper Reference Image Generation with "PanStitch" (southern fields with no KS4/PS1 coverage)
def get_skymapper_image_urls(ra_deg, dec_deg, size_deg, filte):
    """
    Queries the SkyMapper SIAP service to find a FITS image URL for a given pointing.

    Parameters:
      ra_deg (float): Right Ascension of the pointing in decimal degrees.
      dec_deg (float): Declination of the pointing in decimal degrees.
      size_deg (float): The size of the cutout to request in decimal degrees.
      filte (str): The filter to use ('u', 'v', 'g', 'r', 'i', 'z').

    Returns:
      str: The first FITS download URL found, or None if no suitable image is found.
    """
    import csv
    import requests
    from io import StringIO

    base_url = "https://api.skymapper.nci.org.au/public/siap/dr4/query"
    params = {
        'POS': f'{ra_deg},{dec_deg}',
        'SIZE': f'{size_deg}',
        'FORMAT': 'image/fits',
        'BAND': filte,
        'RESPONSEFORMAT': 'CSV',
        'VERB': 1
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        # The response is CSV, parse it to find the download URL
        reader = csv.reader(StringIO(response.text))
        header = next(reader)

        # Find the index of the 'get_fits' column which contains the URL
        try:
            url_idx = header.index('get_fits')
        except ValueError:
            print("Warning: 'get_fits' column not found in SkyMapper response.")
            return None

        # Return the first URL found
        for row in reader:
            if row and len(row) > url_idx and row[url_idx].startswith('http'):
                return row[url_idx]

    except requests.exceptions.RequestException as e:
        print(f"Error querying SkyMapper API: {e}")

    return None


def generate_skymapper_reference(
    field,
    cra_hms,
    cdec_dms,
    path_output_base,
    path_cfg,
    filte='r',
    xsize=22000,
    ysize=22000,
    pixscale=0.4,
    n_grid=8,
    m_grid=8,
    margin_frac=0.0,
    swarp_config='kmtnet.swarp'
):
    """
    Generates a SkyMapper reference image for a given field using PanStitch.

    .. warning::
        The SkyMapper data service has a strict usage policy against systematic
        harvesting of large sky areas. This function generates a grid of image
        requests. Please use conservatively-sized grids (e.g., n_grid=8, m_grid=8)
        to avoid being blocked by the service.

    .. note::
        Unlike generate_panstarrs_reference(), this path has not yet been
        exercised end-to-end. It uses the PanStitch submodule API
        (PanStitch.util / .downloader / .stitching) because SkyMapper slices are
        fetched from SIAP URLs rather than from a Pan-STARRS image table.

    Parameters:
      field (str): The name of the target field (e.g., 'LMC').
      cra_hms (str): The central Right Ascension in HMS format (e.g., '05:23:34.50').
      cdec_dms (str): The central Declination in DMS format (e.g., '-69:45:22.0').
      path_output_base (str): The base directory for all outputs.
      path_cfg (str): The directory containing configuration files like the SWarp config.
      filte (str): The filter to use ('u', 'v', 'g', 'r', 'i', 'z'). Default is 'r'.
      xsize (int): The final desired image width in pixels. Default is 22000.
      ysize (int): The final desired image height in pixels. Default is 22000.
      pixscale (float): The desired pixel scale of the final image in arcsec/pixel. Default is 0.4.
      n_grid (int): The number of grid points in the RA direction. Default is 8.
      m_grid (int): The number of grid points in the Dec direction. Default is 8.
      margin_frac (float): Fractional margin for downloading slices. Default is 0.0.
      swarp_config (str): Name of the SWarp configuration file. Default is 'kmtnet.swarp'.

    Returns:
      str: The path to the final stitched FITS image, or None if no slice was found.
    """
    try:
        import PanStitch
    except ImportError:
        raise ImportError(
            "PanStitch package is required for SkyMapper reference image generation. "
            "Install it with: pip install PanStitch"
        )

    print("--- Starting SkyMapper Reference Image Generation ---")

    # --- 1. Define Paths and Parameters ---
    path_output_field = os.path.join(path_output_base, field)
    path_output_fits = os.path.join(path_output_field, filte, 'fits')
    path_output_log = os.path.join(path_output_field, filte, 'log')
    os.makedirs(path_output_fits, exist_ok=True)
    os.makedirs(path_output_log, exist_ok=True)

    # Convert center coordinates to degrees for pointing generation
    coord = SkyCoord(cra_hms, cdec_dms, unit=(u.hourangle, u.deg))
    cra_deg, cdec_deg = coord.ra.deg, coord.dec.deg

    # --- 2. Generate Pointings ---
    print(f"Generating {n_grid}x{m_grid} grid of pointings...")
    pointings = PanStitch.util.generate_pointings(
        cra_deg, cdec_deg, xsize, ysize, pixscale,
        n=n_grid, m=m_grid, margin_frac=margin_frac
    )
    tra = [p[0] for p in pointings]
    tdec = [p[1] for p in pointings]

    # --- 3. Determine Slice Size and Get Image URLs ---
    # Calculate the angular size needed for each slice download
    angular_width_arcsec = xsize * pixscale * (1 + margin_frac)
    slice_angular_size_arcsec = angular_width_arcsec / n_grid
    slice_angular_size_deg = slice_angular_size_arcsec / 3600

    print("Querying SkyMapper for image download URLs...")
    image_urls = []
    for ra, dec in zip(tra, tdec):
        url = get_skymapper_image_urls(ra, dec, slice_angular_size_deg, filte)
        if url:
            image_urls.append(url)
        else:
            print(f"Warning: Could not find an image for pointing RA={ra}, Dec={dec}")

    if not image_urls:
        print("Error: No image URLs found for any pointing. Aborting.")
        return None

    # --- 4. Download Images ---
    print(f"Downloading {len(image_urls)} image slices...")
    PanStitch.downloader.download_images_for_pointings(
        image_urls,
        output_dir=path_output_fits,
        n_processes=10
    )

    # --- 5. Run SWarp to Stitch Images ---
    print("Stitching images with SWarp...")
    output_fits_name = f'skymapper.{field}.{filte}.{xsize}x{ysize}.fits'
    output_fits_path = os.path.join(path_output_field, filte, output_fits_name)

    PanStitch.stitching.run_swarp(
        input_dir=path_output_fits,
        output_path=output_fits_path,
        config_path=os.path.join(path_cfg, swarp_config),
        center_hms=cra_hms,
        center_dms=cdec_dms,
        image_size_x=xsize,
        image_size_y=ysize,
        pixel_scale=pixscale,
        log_path=os.path.join(path_output_log, 'swarp.log')
    )

    # --- 6. Update Header ---
    # The original draft called an add_fwhm_to_header() helper that was never
    # defined; mirror what generate_panstarrs_reference() writes instead.
    print("Updating header...")
    with fits.open(output_fits_path, 'update') as f:
        for hdu in f:
            hdu.header['OBJECT']    = f'{field}'
            hdu.header['FILTER']    = filte.upper()
            hdu.header['FWHM']      = 1.5
            hdu.header['CENTRA']    = cra_hms
            hdu.header['CENTDEC']   = cdec_dms

    os.system(f'chmod 777 {output_fits_path}')
    print(f"--- SkyMapper Reference Image Generation Complete ---")
    print(f"Final image saved to: {output_fits_path}")

    return output_fits_path
