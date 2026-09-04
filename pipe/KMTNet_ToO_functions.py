#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Import packages
import warnings
import numpy as np
import astropy.units as u
import os, re, glob, time
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)
#%% Import utility functions
from KMTNet_util_functions import (
    rss, apass_query, GAIAXP_query, download_gaiaxp, sort_BVRI, limitmag, 
    matching, star4zp, zpcal, add_colorbar, date2MJD, 
    MJD2date, create_ldac_fits, hotpants, invert_image, 
    mask2weight, generate_snapshot, rename_convention, 
    safe_load_fits, find_longest_exposure_image, read_header,
    parse_region_bounds, mosaic_image, combine_subtracted_images,
    calculate_crosstalk_positions, build_sex_command
)
#%% ToOAmplifierCombine.py
def ampcom(path_data, path_cfg):
    """
    Amplifier combination and quality control for KMTNet images.
    
    This function processes KMTNet multi-extension FITS files by combining 32 amplifier
    extensions into 4 individual chip images (kk, mm, tt, nn). It performs quality
    control checks and isolates poor-quality images into separate directories.
    
    Parameters
    ----------
    path_data : str
        Path to the directory containing raw KMTNet FITS files. Files should follow
        the naming convention 'kmt*.fits'. The function will create subdirectories
        for quality control: 'badccderror/', 'badseeing/', and 'badtracking/'.
    path_cfg : str
        Path to the configuration directory containing SExtractor configuration files:
        - kmtnet.param: SExtractor parameters file
        - kmtnet.sex: SExtractor configuration file
        - kmtnet.conv: SExtractor convolution file
        - kmtnet.nnw: SExtractor neural network weights file
    
    Returns
    -------
    int
        Returns 0 upon successful completion.
    
    Notes
    -----
    The function performs the following operations:
    
    1. **Amplifier Combination**: 
       - Combines 8 amplifiers per chip into 4 chips (kk, mm, tt, nn)
       - Adds WCS information to each chip header
       - Handles both 32-extension and 4-extension input files
    
    2. **Quality Control Checks**:
       - **Bad CCD Error**: Images with sky values ≤ 50 or missing coordinates
       - **Bad Seeing**: Images with FWHM ≥ 6 arcseconds
       - **Bad Tracking**: Images with mean elongation ≥ 2.0
       - Good quality images are processed and sky-subtracted
    
    3. **Output Files**:
       - Creates individual chip files: `{serial}.{chip}.fits`
       - Generates quality log: `ToOampcom.cat`
       - Moves poor-quality images to appropriate subdirectories
    
    4. **Sky Subtraction**:
       - Calculates median sky value for each amplifier
       - Performs differential sky subtraction across amplifiers
       - Updates FITS headers with sky values and quality metrics
    
    Examples
    --------
    >>> ampcom('/path/to/raw/data/', '/path/to/config/')
    Total number of raw images = 10
    Amp combining process for frame:240101 (1/10)
    Amp combining process for frame:240102 (2/10)
    ...
    
    The function will create:
    - 240101.kk.fits, 240101.mm.fits, 240101.tt.fits, 240101.nn.fits
    - ToOampcom.cat (quality log)
    - badccderror/, badseeing/, badtracking/ (quality control directories)
    
    See Also
    --------
    KMTNet_ToO_pipeline : Main pipeline that calls this function
    """
    
    import numpy as np
    import os, shutil, re
    from pathlib import Path 
    import astropy.units as u
    from astropy.io import fits
    from astropy.table import Table
    import astropy.coordinates as coord
    from astropy.stats import sigma_clip

    if not path_data.endswith('/'):
        path_data   = path_data + '/'

    allframes = sorted(str(p) for p in Path(path_data).glob('kmt*fits'))

    os.makedirs(os.path.join(path_data, 'badccderror'), exist_ok=True)
    os.makedirs(os.path.join(path_data, 'badseeing'), exist_ok=True)
    os.makedirs(os.path.join(path_data, 'badtracking'), exist_ok=True)

    print(f'Total number of raw images = {len(allframes)}')

    # Collect the header info
    data_rows = []
    # Loop over each file and extract the header data
    for frame in allframes:

        # KMTNet image check (unimpaired)
        pattern = r'kmt[asc]\.\d{8}\.\d{6}\.fits'
        locations = {
            "kmts"  : "SAAO",
            "kmtc"  : "CTIO",
            "kmta"  : "SSO"}
        file_sizes = {
            "SAAO"  : 1361664000,
            "CTIO"  : 1361583360,
            "SSO"   : 1361583360}

        file_size = os.path.getsize(frame)
        location = locations[os.path.basename(frame).split('.')[0]]
        if file_size < file_sizes[location] or not re.match(pattern, os.path.basename(frame)):
            shutil.move(frame, os.path.join(path_data, 'badccderror', os.path.basename(frame)))
            print(f"Invalid file name {os.path.basename(frame)}. Skipping.")
            continue
        
        with fits.open(frame) as hdul:
            header = hdul[0].header
            # Collect the desired header values
            row = {
                'filename': os.path.basename(frame),
                'ra': header.get('RA', None),
                'dec': header.get('DEC', None),
                'filter': header.get('FILTER', None),
                'object': header.get('OBJECT', None),
                'exptime': header.get('EXPTIME', None),
                'date-obs': header.get('DATE-OBS', None)
            }
            data_rows.append(row)

    # Convert the list of dictionaries into an Astropy Table
    info = Table(rows=data_rows)

    name    = info['filename']
    ra      = info['ra']
    dec     = info['dec']
    filt    = info['filter']
    obj     = info['object']
    expt    = info['exptime']
    dateo   = info['date-obs']

    # Convert ra, dec from sexagesimal to decimal
    rad = coord.Angle(ra,unit=u.hour)
    radd = rad.degree
    decd = coord.Angle(dec,unit=u.deg)
    decdd = decd.degree

    # Comebine 32 amps to 4 chips
    with open(f'{path_data}ToOampcom.cat', 'w') as f:
        f.write('#name skykk skymm skytt skynn fwhmkk fwhmmm fwhmtt fwhmnn\n')

    for i in range(len(info)):

        serial  = name[i][14:20]
        print(f'Amp combining process for frame:{serial} ({i+1}/{len(name)})')

        hdul    = fits.open(f"{path_data}{name[i]}")
        extension = np.size(hdul)-1
        if extension == 32:
            chiparr = ['mm','kk','nn','tt']
            for k in range(len(chiparr)):
                fits1, hdr1 = fits.getdata(f"{path_data}{name[i]}", header=True, ext=(k*8)+1)
                fits2, hdr2 = fits.getdata(f"{path_data}{name[i]}", header=True, ext=(k*8)+2)
                fits3, hdr3 = fits.getdata(f"{path_data}{name[i]}", header=True, ext=(k*8)+3)
                fits4, hdr4 = fits.getdata(f"{path_data}{name[i]}", header=True, ext=(k*8)+4)
                fits5, hdr5 = fits.getdata(f"{path_data}{name[i]}", header=True, ext=(k*8)+5)
                fits6, hdr6 = fits.getdata(f"{path_data}{name[i]}", header=True, ext=(k*8)+6)
                fits7, hdr7 = fits.getdata(f"{path_data}{name[i]}", header=True, ext=(k*8)+7)
                fits8, hdr8 = fits.getdata(f"{path_data}{name[i]}", header=True, ext=(k*8)+8)
                tempfits=np.hstack((fits1,fits2,fits3,fits4,fits5,fits6,fits7,fits8))
                hdu = fits.ImageHDU(tempfits)
                if chiparr[k] == 'mm': hdu.header.set('dec', decdd[i] + 0.5)
                if chiparr[k] == 'mm': hdu.header.set('ra', (radd[i]/15.) + 0.03333)
                if chiparr[k] == 'kk': hdu.header.set('dec', decdd[i] + 0.5)
                if chiparr[k] == 'kk': hdu.header.set('ra', (radd[i]/15.) - 0.03333)
                if chiparr[k] == 'nn': hdu.header.set('dec', decdd[i] - 0.5)
                if chiparr[k] == 'nn': hdu.header.set('ra', (radd[i]/15.) + 0.03333)
                if chiparr[k] == 'tt': hdu.header.set('dec', decdd[i] - 0.5)
                if chiparr[k] == 'tt': hdu.header.set('ra', (radd[i]/15.) - 0.03333)
                hdu.header.set('crpix1', 4600)
                hdu.header.set('crpix2', 4600)
                hdu.header.set('cd1_1', -0.00011)
                hdu.header.set('cd2_2', 0.00011)
                hdu.header.set('radecsys', 'icrs')
                hdu.header.set('equinox', 2000.0)
                hdu.header.set('wcsdim', 2)
                if chiparr[k] == 'mm': hdu.header.set('crval1', radd[i])
                if chiparr[k] == 'mm': hdu.header.set('crval2', decdd[i])
                if chiparr[k] == 'kk': hdu.header.set('crval1', radd[i])
                if chiparr[k] == 'kk': hdu.header.set('crval2', decdd[i])
                if chiparr[k] == 'nn': hdu.header.set('crval1', radd[i])
                if chiparr[k] == 'nn': hdu.header.set('crval2', decdd[i])
                if chiparr[k] == 'tt': hdu.header.set('crval1', radd[i])
                if chiparr[k] == 'tt': hdu.header.set('crval2', decdd[i])
                hdu.writeto(f"{path_data}{serial}.{chiparr[k]}.fits", overwrite='True')
        elif extension == 4:
            fits1 = hdul[1]
            fits2 = hdul[2]
            fits3 = hdul[3]
            fits4 = hdul[4]
            fits1.writeto(f'{path_data}{serial}.mm.fits', overwrite='True')
            fits2.writeto(f'{path_data}{serial}.kk.fits', overwrite='True')
            fits3.writeto(f'{path_data}{serial}.nn.fits', overwrite='True')
            fits4.writeto(f'{path_data}{serial}.tt.fits', overwrite='True')
        
        fitskk, hdrkk = fits.getdata(f'{path_data}{serial}.kk.fits', header=True)
        fitsmm, hdrmm = fits.getdata(f'{path_data}{serial}.mm.fits', header=True)
        fitstt, hdrtt = fits.getdata(f'{path_data}{serial}.tt.fits', header=True)
        fitsnn, hdrnn = fits.getdata(f'{path_data}{serial}.nn.fits', header=True)
        fitsarr = [fitskk, fitsmm, fitstt, fitsnn]
        hdrarr = [hdrkk, hdrmm, hdrtt, hdrnn]
        
        skykk = np.median(fitskk)
        skymm = np.median(fitsmm)
        skytt = np.median(fitstt)
        skynn = np.median(fitsnn)
        
        skyarr  = [skykk, skymm, skytt, skynn]
        chiparr = ['kk','mm','tt','nn']
        fwhmarr = [99.,99.,99.,99.]
        elonarr = [99.,99.,99.,99.]
        
        for k in range(len(chiparr)):
    
            chip = chiparr[k]
            catname     = f'{path_data}{serial}.{chip}.ampcom.cat'
            param       = os.path.join(path_cfg, 'kmtnet.param')
            cfg         = os.path.join(path_cfg, 'kmtnet.sex')
            conv        = os.path.join(path_cfg, 'kmtnet.conv')
            nnw         = os.path.join(path_cfg, 'kmtnet.nnw')
    
            os.system(f'source-extractor {path_data}{serial}.{chip}.fits -c {cfg} -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME {catname} -PARAMETERS_NAME {param} -FILTER_NAME {conv} -STARNNW_NAME {nnw} -DETECT_THRESH 50 -ANALYSIS_THRESH 50')
                      
            cat = Table.read(f'{path_data}{serial}.{chip}.ampcom.cat', format ='ascii')
            magauto = np.array(cat['MAG_AUTO'])
            magerr  = np.array(cat['MAGERR_AUTO'])
            flag    = np.array(cat['FLAGS'])
            fwhm    = np.array(cat['FWHM_IMAGE'])
            elon    = np.array(cat['ELONGATION'])
            pa      = np.array(abs(cat['THETA_IMAGE']))
            fwhmcut = np.where((magauto != 99) & (magerr <= 0.005) & (flag == 0) & (fwhm <= 30))
            if len(fwhmcut[0]) != 0 :
                sigmaclip = sigma_clip(fwhm[fwhmcut], sigma=3, cenfunc=np.median, stdfunc=np.std)
                clip = np.where(sigmaclip.mask == False)
                nclip = np.where(sigmaclip.mask == True)
                fwhmarr[k] = 0.4 * np.median(fwhm[fwhmcut][clip])
            eloncut = np.where((magauto != 99) & (magerr <= 0.005) & (pa < 88))
            if len(eloncut[0]) != 0 : elonarr[k] = np.mean(elon[eloncut])
        
        if (skykk <= 50) or (skymm <= 50) or (skytt <= 50) or (skynn <= 50):
            os.system(f'mv {path_data}*{serial}* {path_data}badccderror')
        elif (radd[i] == 0.0) and (decdd[i] == 0.0):
            os.system(f'mv {path_data}*{serial}* {path_data}badccderror')
        elif max(fwhmarr) >= 6:
            os.system(f'mv {path_data}*{serial}* {path_data}badseeing')
        elif np.mean(elonarr) >= 2.0:
            os.system(f'mv {path_data}*{serial}* {path_data}badtracking')
        else:
            for k in range(len(chiparr)):
                tempfits, temphdr = fitsarr[k], hdrarr[k]
                sky = []
                for l in range(8):
                    ctempfits = tempfits[0:9232,l*1152:1152+(l*1152)]
                    bkg_value = np.median(ctempfits)
                    sky = sky + [bkg_value]
                medsky = np.median(sky)
                for l in range(8):
                    ctempfits = tempfits[0:9232,l*1152:1152+(l*1152)]
                    tempfits[0:9232,l*1152:1152+(l*1152)] = ctempfits - (sky[l] - medsky)
                temphdr['SKYVAL'] = (skyarr[k], 'Median pixel value')
                temphdr['FWHM'] = fwhmarr[k]
                temphdr['ELONG']= (elonarr[k], 'Mean elongation of the sources')
                fits.writeto(f"{path_data}{serial}.{chiparr[k]}.fits", tempfits, temphdr, overwrite=True)
            
            hdul[0].header.set('skykk', skykk)
            hdul[0].header.set('skymm', skymm)
            hdul[0].header.set('skytt', skytt)
            hdul[0].header.set('skynn', skynn)
            hdul[0].header.set('skyavg', np.mean(skyarr))
            hdul[0].header.set('fwhm1kk', fwhmarr[0])
            hdul[0].header.set('fwhm1mm', fwhmarr[1])
            hdul[0].header.set('fwhm1tt', fwhmarr[2])
            hdul[0].header.set('fwhm1nn', fwhmarr[3])
            hdul[0].header.set('fwhm1avg', np.mean(fwhmarr))
            hdul[0].header.set('elonavg', np.mean(elonarr))
            hdul.writeto(f"{path_data}{name[i]}", overwrite='True')
            hdul.close()
    
        with open(f'{path_data}ToOampcom.cat', 'a') as f:
            f.write(f'{name[i]} {skykk:7.1f} {skymm:7.1f} {skytt:7.1f} {skynn:7.1f} {fwhmarr[0]:5.1f} {fwhmarr[1]:5.1f} {fwhmarr[2]:5.1f} {fwhmarr[3]:5.1f}\n')
    
    os.system(f'chmod 777 {path_data}*')
    os.system(f'chmod 777 {path_data}badccderror/*')
    os.system(f'chmod 777 {path_data}badseeing/*')
    os.system(f'chmod 777 {path_data}badtracking/*')
    os.system(f'rm {path_data}??????.??.ampcom.cat')
    
    return 0
#%% ToOAstrometry.py
def astrom(path_data, path_cfg, path_cat, radius=1.0, ithresh=5, gridcat='kmtnet_grid.fits',
           astrom_rms_max=1e-4, neighbour_fallback=True, neighbour_max_age_days=3.0,
           gaiaxp_download=True):
    """
    Astrometric calibration of KMTNet chip images using SCAMP.
    
    This function performs astrometric calibration on KMTNet chip images (kk, mm, tt, nn)
    by running SExtractor to detect sources and SCAMP to solve for World Coordinate System
    (WCS) parameters. It uses reference catalogs (GAIA or UCAC-4) for calibration.
    
    Parameters
    ----------
    path_data : str
        Path to the directory containing KMTNet chip images. Files should follow
        the naming convention '{serial}.{chip}.fits' where chip is one of [kk, mm, tt, nn].
    path_cfg : str
        Path to the configuration directory containing:
        - kmtnet.param: SExtractor parameters file
        - kmtnet.sex: SExtractor configuration file
        - kmtnet.conv: SExtractor convolution file
        - kmtnet.nnw: SExtractor neural network weights file
        - kmtnet.scamp: SCAMP configuration file
        - ahead/: Directory containing aheader files for each observatory
    path_cat : str
        Path to the catalog directory containing reference catalogs:
        - gaiaxp/: GAIA reference catalogs
        - kmtnet_grid.fits: KMTNet field grid catalog
    radius : float, optional
        Search radius in degrees for reference catalog matching. Default is 1.0.
    ithresh : int, optional
        Initial detection threshold for SExtractor. Default is 5.
    gridcat : str, optional
        Name of the KMTNet grid catalog file. Default is 'kmtnet_grid.fits'.
    gaiaxp_download : bool, optional
        If True (default), when a field has no precomputed local Gaia-XP
        catalogue the function attempts to download the Gaia DR3 synthetic
        Johnson-Kron-Cousins BVRI catalogue (GSPC) for that field and caches it
        under ``path_cat/gaiaxp/``. If the download is unavailable, the frame
        falls back to UCAC-4 (network); should that also fail/time out, the
        frame is skipped after a single attempt instead of looping on repeated
        network queries.
    
    Returns
    -------
    int
        Returns 0 upon successful completion.
    
    Notes
    -----
    The function performs the following operations:
    
    1. **Source Detection**:
       - Runs SExtractor on each chip image to detect sources
       - Creates source catalogs with positions and photometry
       - Uses adaptive thresholding for optimal source detection
    
    2. **Astrometric Solution**:
       - Matches detected sources with reference catalog (GAIA or UCAC-4)
       - Uses SCAMP to solve for WCS parameters (plate scale, rotation, distortion)
       - Applies TPV (Tangent Plane) projection for accurate coordinate transformation
    
    3. **Quality Control**:
       - Iteratively adjusts detection threshold if RMS errors are too high
       - Maximum 3 iterations with increasing threshold
       - Rejects solutions with RMS > 1e-4 arcseconds
    
    4. **Header Updates**:
       - Updates FITS headers with astrometric solution
       - Adds WCS keywords (CRVAL1, CRVAL2, CD matrix)
       - Preserves original observation metadata
    
    5. **Observatory-Specific Processing**:
       - SSO (Australia): Uses kmtnet_global_sso.ahead files
       - SAAO (South Africa): Uses kmtnet_global.ahead files  
       - CTIO (Chile): Uses kmtnet_global_ctio.ahead files
    
    Examples
    --------
    >>> astrom('/path/to/chip/images/', '/path/to/config/', '/path/to/catalogs/')
    Astrometry process for frame:240101, thresh:5 (1/10)
    Astrometry process for frame:240102, thresh:5 (2/10)
    ...
    
    The function will create:
    - {serial}.{chip}.astrom.cat (SExtractor source catalogs)
    - {serial}.{chip}.astrom.head (SCAMP astrometric headers)
    - Updated FITS files with WCS information
    
    See Also
    --------
    ampcom : Amplifier combination function that creates input chip images
    KMTNet_ToO_pipeline : Main pipeline that calls this function
    """

    import os
    from pathlib import Path
    import astropy.units as u
    from astropy.io import fits
    from astropy.table import Table
    from astropy.coordinates import Angle
    from astropy.coordinates import SkyCoord

    if not path_data.endswith('/'):
        path_data   = path_data + '/'
    # The aheader path below is built by string concatenation (f'{path_cfg}ahead/...'),
    # so a trailing separator must be guaranteed even when the path config omits it.
    if not path_cfg.endswith('/'):
        path_cfg    = path_cfg + '/'

    import time

    # ------------------------------------------------------------------ #
    # Neighbour-ahead recovery helpers.
    # When the static (global) initial-guess header fails to converge, SCAMP
    # is retried seeded with the most-recent successful solution of the same
    # observatory+chip (its CD/CRPIX/PV terms, while the frame keeps its own
    # CRVAL). The seed is persisted under config/ahead/lastgood/ so it can
    # rescue sibling frames within a run and across recent runs.
    # ------------------------------------------------------------------ #
    _AHEAD_KEEP = ('EQUINOX', 'RADESYS', 'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
                   'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')

    def _cache_lastgood_ahead(head_path, out_path):
        """Distil a solved SCAMP header into a reusable .ahead seed (no CRVAL)."""
        with open(head_path, 'r', encoding='latin-1') as fh:
            txt = fh.read().encode('ascii', 'ignore').decode('ascii')
        src = fits.Header.fromstring(txt, sep='\n')
        seed = fits.Header()
        for kk in _AHEAD_KEEP:
            if kk in src:
                seed[kk] = src[kk]
        for kk in src:
            if kk.startswith('PV1_') or kk.startswith('PV2_'):
                seed[kk] = src[kk]
        if 'CD1_1' not in seed:        # nothing usable -> do not cache
            return False
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        seed.totextfile(out_path, overwrite=True)
        return True

    def _ahead_is_fresh(path, max_age_days):
        if not os.path.isfile(path):
            return False
        if max_age_days is None:
            return True
        return (time.time() - os.path.getmtime(path)) <= max_age_days * 86400.0

    fits_files = sorted(Path(path_data).glob('kmt*.fits'))

    # Initialize a list to hold all the header information
    data_rows = []

    # Loop over each file and extract the header data
    for fits_file in fits_files:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            # Collect the desired header values, providing defaults if not found
            row = {
                'name': fits_file.name,
                'ra': header.get('RA', ''),
                'dec': header.get('DEC', ''),
                'filter': header.get('FILTER', ''),
                'object': header.get('OBJECT', ''),
                'exptime': header.get('EXPTIME', 0),
                'date-obs': header.get('DATE-OBS', ''),
                'observat': header.get('OBSERVAT', ''),
                'secz': header.get('SECZ', 0),
                'skykk': header.get('SKYKK', 0),
                'skymm': header.get('SKYMM', 0),
                'skytt': header.get('SKYTT', 0),
                'skynn': header.get('SKYNN', 0),
                'fwhm1kk': header.get('FWHM1KK', 0),
                'fwhm1mm': header.get('FWHM1MM', 0),
                'fwhm1tt': header.get('FWHM1TT', 0),
                'fwhm1nn': header.get('FWHM1NN', 0),
            }
            data_rows.append(row)

    # Convert the list of dictionaries into an Astropy Table
    info = Table(rows=data_rows)
    info.write(os.path.join(path_data, 'ToOastrom.txt'), format='ascii', overwrite=True)

    # Define each variable from the columns
    name    = info['name']
    ra      = info['ra']
    dec     = info['dec']
    filt    = info['filter']
    obj     = info['object']
    expt    = info['exptime']
    dateo   = info['date-obs']
    observatory = info['observat']
    secz    = info['secz']
    skykk   = info['skykk']
    skymm   = info['skymm']
    skytt   = info['skytt']
    skynn   = info['skynn']
    fwhm1kk = info['fwhm1kk']
    fwhm1mm = info['fwhm1mm']
    fwhm1tt = info['fwhm1tt']
    fwhm1nn = info['fwhm1nn']

    # Convert RA and Dec into degrees if necessary
    radd = Angle(ra, unit=u.hourangle).degree
    decdd = Angle(dec, unit=u.deg).degree

    chiparr = ['kk','mm','tt','nn']

    # Fields whose on-demand Gaia-XP download has already been attempted this run
    # (so a missing/failing field is not re-queried for every chip and frame).
    attempted_gaiaxp = set()

    for i in range(len(info)):

        serial  = name[i][14:20]
        print(f'Astrometry process for frame:{serial}, thresh:{ithresh} ({i+1}/{len(name)})')

        if name[i][3] == 'a': gap = 53 # SSO
        else: gap = 51
        
        for k, chip in enumerate(chiparr):

            with fits.open(f"{path_data}{serial}.{chip}.fits", mode='update') as hd:
                header = hd[0].header
                header['CRVAL1'] = radd[i]  # Update RA
                header['CRVAL2'] = decdd[i]  # Update Dec
                hd.flush()  # Write changes to the file
            
            catname     = f'{path_data}{serial}.{chip}.astrom.cat'
            param       = os.path.join(path_cfg, 'kmtnet.param')
            cfg         = os.path.join(path_cfg, 'kmtnet.sex')
            conv        = os.path.join(path_cfg, 'kmtnet.conv')
            nnw         = os.path.join(path_cfg, 'kmtnet.nnw')
            outhdr      = f'{path_data}{serial}.{chip}.astrom.head'

            # Observatory-specific static (global) initial-guess header.
            site = name[i][3]
            if   site == 'a': default_ahead = f'{path_cfg}ahead/kmtnet_global_sso.{chip}.ahead'   # SSO  (Australia)
            elif site == 's': default_ahead = f'{path_cfg}ahead/kmtnet_global.{chip}.ahead'        # SAAO (South Africa)
            else:             default_ahead = f'{path_cfg}ahead/kmtnet_global_ctio.{chip}.ahead'   # CTIO (Chile)

            # Reference catalogue: prefer the local Gaia-XP catalogue (works
            # offline and is better centred on the Gaia frame than UCAC-4). When
            # a field has no precomputed catalogue (e.g. custom ToO grid fields),
            # try to download the Gaia synthetic JKC BVRI catalogue on demand.
            # Only if that is unavailable do we fall back to UCAC-4 (network);
            # a frame left with no usable reference catalogue is skipped after a
            # single attempt rather than looped over repeated network timeouts.
            centcoord   = SkyCoord(ra[i], dec[i], unit=(u.hourangle, u.deg))
            try:
                kmtgrid     = Table.read(os.path.join(path_cfg, gridcat), format='fits')
            except Exception:
                kmtgrid     = Table.read(os.path.join(path_cfg, gridcat), format='ascii')
            kmtcoord    = SkyCoord(kmtgrid['ra[deg]'], kmtgrid['dec[deg]'], unit='deg')
            trgt_field  = kmtgrid[centcoord.separation(kmtcoord).argmin()]
            fieldid     = str(trgt_field["field_name1"]).zfill(4)
            # Grid (tile) centre from kmtnet_grid.fits for the matched field; the
            # downloaded catalogue is centred here (not on the frame pointing).
            fieldcoord  = SkyCoord(trgt_field['ra[deg]'], trgt_field['dec[deg]'], unit='deg')
            gaiacat     = os.path.join(path_cat, 'gaiaxp', f'gaiaxp_{fieldid}.fits')

            if (not os.path.exists(gaiacat)) and gaiaxp_download and (fieldid not in attempted_gaiaxp):
                attempted_gaiaxp.add(fieldid)
                print(f'Local Gaia-XP catalogue not found for field {fieldid}; '
                      f'attempting download from the Gaia archive...')
                # Cone radius 1.5 deg fully encloses the 2x2 deg KMTNet tile
                # (corner distance sqrt(2)=1.414 deg) centred on the grid field.
                download_gaiaxp(fieldid, os.path.join(path_cat, 'gaiaxp'),
                                center=fieldcoord, radius=max(radius, 1.5))

            ref_is_local = os.path.exists(gaiacat)
            if ref_is_local:
                gaialdac = gaiacat.replace('.fits', '_ldac.fits')
                if not os.path.exists(gaialdac):
                    create_ldac_fits(gaiacat, gaialdac, center=centcoord, radius=max(radius, 1.5))
                refargs = f'-ASTREF_CATALOG FILE -ASTREFCAT_NAME {gaialdac}'
            else:
                print(f'No local/downloadable Gaia-XP catalogue for field {fieldid}; '
                      f'falling back to UCAC-4 (requires network).')
                refargs = '-ASTREF_CATALOG UCAC-4'

            # Recovery ladder:
            #   (1) static global ahead, raising DETECT_THRESH a few times;
            #   (2) if still failing, retry seeded with the most-recent good
            #       same-site/same-chip solution (the "neighbour" ahead).
            lastgood_ahead = os.path.join(path_cfg, 'ahead', 'lastgood', f'{site}.{chip}.ahead')
            current_ahead  = default_ahead
            thresh, repeat, tried_neighbour = ithresh, 0, False

            while True:

                sexcom   = f'source-extractor {path_data}{serial}.{chip}.fits -c {cfg} -CATALOG_NAME {catname} -PARAMETERS_NAME {param} -FILTER_NAME {conv} -STARNNW_NAME {nnw} -CATALOG_TYPE FITS_LDAC -HEADER_SUFFIX NONE -DETECT_THRESH {thresh} -ANALYSIS_THRESH {thresh} -SATUR_LEVEL 60000.0'
                scampcom = f'scamp {catname} -c {os.path.join(path_cfg, "kmtnet.scamp")} {refargs} -POSITION_MAXERR 20.0 -CROSSID_RADIUS 5.0 -DISTORT_DEGREES 3 -PROJECTION_TYPE TPV -AHEADER_GLOBAL {current_ahead} -STABILITY_TYPE INSTRUMENT'

                os.system(sexcom)
                if os.path.exists(outhdr):
                    os.remove(outhdr)  # drop any previous solution
                os.system(scampcom)

                rms1 = float(read_header(outhdr).get('ASTRRMS1', 0) or 0)
                rms2 = float(read_header(outhdr).get('ASTRRMS2', 0) or 0)
                solved = os.path.exists(outhdr) and (0 < rms1 <= astrom_rms_max) and (0 < rms2 <= astrom_rms_max)

                if solved:
                    # Persist this good solution as a seed for sibling frames.
                    try:
                        _cache_lastgood_ahead(outhdr, lastgood_ahead)
                    except Exception as e:
                        print(f'(could not cache last-good ahead for {site}.{chip}: {e})')
                    break

                # With a network reference (UCAC-4) there is no point looping:
                # raising DETECT_THRESH or seeding a neighbour just re-queries the
                # (possibly timing-out) server. Skip the frame after one attempt.
                if not ref_is_local:
                    print(f'*** {serial}.{chip}: no usable reference catalogue '
                          f'(Gaia-XP unavailable, UCAC-4 query failed/timed out); skipping. ***')
                    break

                if repeat < 3:
                    print(f'Warning: poor astrometry for {serial}.{chip} (rms={rms1:.2e},{rms2:.2e}); raising threshold (iteration {repeat + 1}).')
                    thresh += 10
                    repeat += 1
                    continue

                if neighbour_fallback and (not tried_neighbour) and (current_ahead != lastgood_ahead) \
                        and _ahead_is_fresh(lastgood_ahead, neighbour_max_age_days):
                    print(f'*** {serial}.{chip}: global ahead failed; retrying with most-recent good neighbour seed ({lastgood_ahead}). ***')
                    current_ahead = lastgood_ahead
                    thresh, repeat, tried_neighbour = ithresh, 0, True
                    continue

                print(f'*** {serial}.{chip}: astrometry did not converge (rms={rms1:.2e},{rms2:.2e}). ***')
                break

        skyarr  = [skykk[i], skymm[i], skytt[i], skynn[i]]
        fwhmarr = [fwhm1kk[i], fwhm1mm[i], fwhm1tt[i], fwhm1nn[i]]

        for k in range(len(chiparr)):
            chip = chiparr[k]
            tempsky = skyarr[k]
            tempfwhm = fwhmarr[k]
            
            inhdr = f"{path_data}{serial}.{chip}.astrom.head"
            if os.path.exists(inhdr):
                # SCAMP headers can contain non-ASCII bytes (e.g. in COMMENT/
                # HISTORY cards); read tolerantly and rewrite clean ASCII so the
                # downstream fits.Header.fromtextfile() never chokes.
                with open(inhdr, 'r', encoding='latin-1') as f:
                    lines = f.readlines()

                lines[1] = lines[1][0:37] + '\n'
                with open(inhdr, 'w', encoding='ascii') as f:
                    for line in lines[0:gap]:
                        f.write(line.encode('ascii', 'ignore').decode('ascii'))
                
                hdr     = fits.getheader(f"{path_data}{serial}.{chip}.fits")
                hdu     = hdr.fromtextfile(inhdr)
                hdu     = hdr[0:7] + hdu
                hdu.append(('filter',filt[i]), end=True)
                hdu.append(('object',obj[i]), end=True)
                hdu.append(('exptime',expt[i]), end=True)
                hdu.append(('date-obs',dateo[i]), end=True)
                hdu.append(('observat', observatory[i]), end=True)
                hdu.append(('centra', ra[i]), end=True)
                hdu.append(('centdec', dec[i]), end=True)
                hdu.append(('centsecz', secz[i]), end=True)
                hdu.append(('skyval', tempsky), end=True)
                hdu.append(('fwhm', tempfwhm), end=True)

                fits.PrimaryHDU(data=fits.getdata(f'{path_data}{serial}.{chip}.fits'), header=hdu).writeto(f'{path_data}{serial}.{chip}.fits', overwrite=True)

    os.system(f'chmod 777 {path_data}*')
    os.system(f'rm {path_data}??????.??.astrom.cat')
    
    return 0
#%% ToOAstrometryQA.py
def qatest(fname, configdir, gridcat, refcatdir, refcatname='GAIAXP', divnum=8, crreject=True, bleedreject=True, weightmap=True, imtype='chip',
           qa_edge_ring=1, qa_max_edge_bad=2):
    """
    Quality Assurance (QA) test for astrometric calibration of KMTNet images.
    
    This function performs comprehensive quality assurance testing on individual KMTNet
    images (chip or stacked) by evaluating astrometric accuracy, detecting artifacts,
    and assessing overall image quality. It generates bad-pixel masks, matches sources
    with reference catalogs, and updates FITS headers with QA metrics.
    
    Parameters
    ----------
    fname : str
        Path to the input FITS image file. Can be either a chip image (e.g., 
        '062889.nn.fits') or a stacked image (e.g., 'TOO_0578.stack.fits').
    configdir : str
        Path to the configuration directory containing:
        - kmtnet.sex: SExtractor configuration file
        - kmtnet.param: SExtractor parameters file
        - kmtnet_imask.param: SExtractor parameters for mask analysis
        - kmtnet.conv: SExtractor convolution file
        - kmtnet.nnw: SExtractor neural network weights file
        - kmtnet_grid.fits: KMTNet field grid catalog
        - badastrom.txt: Log file for bad astrometric solutions
    gridcat : str
        Name of the KMTNet grid catalog file (e.g., 'kmtnet_grid.fits').
    refcatdir : str
        Path to the reference catalog directory containing:
        - gaiaxp/: GAIA reference catalogs
        - gaiaedr3/: GAIA EDR3 reference catalogs
    refcatname : str, optional
        Name of the reference catalog to use ('GAIAXP', 'gaiaedr3'). 
        Default is 'GAIAXP'.
    divnum : int, optional
        Number of divisions for image segmentation analysis (divnum x divnum grid).
        Default is 8 (creates 8x8 = 64 sections).
    crreject : bool, optional
        Whether to perform cosmic ray rejection and create bad-pixel masks.
        Default is True.
    bleedreject : bool, optional
        Whether to detect and mask pixel bleeding artifacts.
        Default is True.
    weightmap : bool, optional
        Whether to create and use weight maps for photometry.
        Default is True.
    imtype : str, optional
        Type of input image: 'chip' for individual chip images or 'stack' for 
        stacked images. Default is 'chip'.
    
    Returns
    -------
    None
        Results are written to FITS headers and log files.
    
    Notes
    -----
    The function performs the following operations:
    
    1. **Bad-Pixel Mask Generation** (for chip images):
       - **Cosmic Ray Detection**: Uses astroscrappy to identify cosmic rays
       - **Cross-talk Masking**: Detects amplifier cross-talk artifacts
       - **Pixel Bleeding**: Identifies and masks bleeding patterns from saturated pixels
       - **Weight Map Creation**: Generates weight maps for photometric analysis
    
    2. **Source Detection and Analysis**:
       - Runs SExtractor to detect sources in the image
       - Applies bad-pixel masks to exclude contaminated regions
       - Performs quality checks (minimum source count, tracking issues)
    
    3. **Reference Catalog Matching**:
       - Loads appropriate reference catalog (GAIA or GAIA EDR3)
       - Matches detected sources with reference catalog positions
       - Calculates astrometric offsets and misalignments
    
    4. **Quality Assessment**:
       - **Sectional Analysis**: Divides image into grid sections for detailed QA
       - **Good/Bad Classification**: Each section classified as 'good', 'bad', or 'empty'
       - **Overall QA Result**: Determines if image passes quality standards
    
    5. **Header Updates**:
       - Adds comprehensive QA information to FITS headers
       - Logs bad astrometric solutions to badastrom.txt
       - Includes version information and processing details
    
    Quality Criteria
    ----------------
    - **Good Section**: Detection ratio > 0.6 and RMS alignment < 0.5 arcsec
    - **Bad Section**: Detection ratio ≤ 0.6 or RMS alignment ≥ 0.5 arcsec
    - **Overall QA Pass**: ≤ 2 bad sections, ≤ 10 empty sections, median offset < 0.4 arcsec
    
    Header Keywords Added
    ---------------------
    - QAREFCAT: Reference catalog used for QA
    - QAALNNUM: Number of objects used for QA
    - QAALNRMS: RMS of misalignment with reference catalog [arcsec]
    - QAALNSTD: Uncertainty of misalignment [arcsec]
    - QANSECT: Total number of sections analyzed
    - QAGDSECT: Number of good sections
    - QABDSECT: Encoded positions of bad sections
    - QABADAMP: True if bad amplifier detected
    - QARESULT: True if QA passes overall quality standards
    
    Examples
    --------
    >>> # QA test for a chip image
    >>> qatest('/path/to/062889.nn.fits', '/path/to/config/', 'kmtnet_grid.fits', 
    ...        '/path/to/catalogs/', 'GAIAXP', divnum=8, crreject=True, 
    ...        bleedreject=True, weightmap=True, imtype='chip')
    
    >>> # QA test for a stacked image
    >>> qatest('/path/to/TOO_0578.stack.fits', '/path/to/config/', 'kmtnet_grid.fits',
    ...        '/path/to/catalogs/', 'GAIAXP', divnum=8, crreject=False,
    ...        bleedreject=False, weightmap=True, imtype='stack')
    
    See Also
    --------
    astrom : Astrometric calibration function
    ampcom : Amplifier combination function
    KMTNet_ToO_pipeline : Main pipeline that calls this function
    
    Warning
    -------
    Cosmic ray rejection should be performed before background subtraction
    for optimal results.
    """

    __version__ = '1.4.0' 

    # ====== IMPORTS ========================================================
    import os
    import warnings
    import numpy as np
    import pandas as pd
    try:
        import astroscrappy as cr
    except ImportError:
        raise ImportError(
            "astroscrappy package is required for cosmic ray rejection. "
            "Install it with: pip install astroscrappy"
        )
    import astropy.units as u
    from datetime import date as dt
    from astropy.table import Table
    from astropy.io import fits,ascii
    import astropy.coordinates as coord
    from astroquery.vizier import Vizier
    from astropy.stats import sigma_clip
    from numpy import std,sqrt,mean,median
    from astropy.coordinates import SkyCoord

    warnings.simplefilter('ignore', UserWarning)

    # Normalise the config directory so that the many `configdir + 'file'`
    # string concatenations below resolve correctly whether or not the caller
    # passed a trailing separator (the centralised path config does not add one).
    if not configdir.endswith(os.sep):
        configdir += os.sep

    # ====== CLASSES ==========================================================

    class Cmd :

        def ln(fpath) :
            import os
            os.system(f'ln -s {fpath}')

        def cp(fpath) :
            import os 
            os.system(f'cp {fpath} ./')

        def cp_all(path) :
            import os
            all_fpath = os.path.join(path, '*')
            os.system(f'cp -r {all_fpath} ./')

        def rm(file) :
            import os
            os.system(f'rm {file}')

        def uln(file) :
            import os
            os.system(f'unlink {file}')

    class Msg :
        '''
        All messages used in QATEST
        '''

        def err(fname, err) :
            msg = { 
                'trackerr' : 'tracking issue!',
                'fpatherr' : 'wrong directory!',
                'pointerr' : 'out of coverage!'
            }
            badastromtxt    = os.path.join(configdir, 'badastrom.txt')
            if os.path.exists(badastromtxt):
                with open(badastromtxt, 'a') as f :
                    f.write(f'{err} {fname} \n')
            else:
                with open(badastromtxt, 'w') as f :
                    f.write(f'{err} {fname} \n')

            print(f'*** {fname} {msg[err]} ***')

        def start(fname) :
            print('='*79 + '\n' + 
                  ' '*int((63-len(fname))/2) + f'running qatest({os.path.basename(fname)})\n' + 
                  '='*79)

        def end() :
            print('> All done')

        def crmap():
            print('> Creating Cosmic ray reduction map...')
        
        def bleedmap():
            print('> Creating bleeding pattern reduction map...')

        def run_sex() :
            print('> Running Source Extractor...')

        def match_src() :
            print('> Matching sources...')

        def load_refcat(local=True) :
            refcat_loc = 'local server' if local==True else 'vizier'
            print(f'> Loading reference catalog from {refcat_loc}...')

        def anlz_gbmap() :
            print('> Testing astrometric Quality Assurace...')

        def qaresult(fastrom) :
            print(f'> QA result : {fastrom}')

        def hdr_update() :
            print('> Updating header...')

    class RndVal :

        def mean(ls) :
            import numpy as np
            from numpy import mean
            return round(float(np.mean(ls)), 5)

        def std(ls) :
            import numpy as np
            from numpy import std
            return round(float(np.std(ls)), 5)

        def rms(ls) :
            import numpy as np
            from numpy import sqrt
            return(round(float(sqrt(mean(ls**2))), 5))
    # ====== NESTED FUNCTIONS =================================================
    def bleed_masking(datapath,  bleeding_thres=50000, BI_thres=500, detect_thres=0.4, CL=6, flagval=4, data=None):
        """
        Mask saturation bleeding trails.

        Vectorised re-implementation (Jeong 2026): instead of the original
        O(N_seed x trail_length) per-column Python scan, every saturated column
        is processed once with a directional, gap-tolerant flood that is fully
        expressed with NumPy. A pixel is flagged when, travelling in the bleed
        direction from a saturated seed, the running count of consecutive
        below-`signal_thres` pixels has not yet reached `CL` (i.e. the trail is
        still alive). This reproduces the intent of the original routine
        (saturated core + bleed trail until it fades) while running hundreds of
        times faster. `BI_thres` is retained for signature compatibility.

        `data` lets the caller hand in the already-loaded image array so the
        ~340 MB chip is not read from disk a second time.
        """

        import os
        import numpy as np
        from astropy.stats import sigma_clipped_stats
        from astropy.io import fits

        Msg.bleedmap()
        if data is None:
            data = fits.getdata(datapath)
        leny, lenx = data.shape
        bleeding_mask = np.zeros((leny, lenx), dtype=np.int16)
        # Sky level/scatter are estimated on a strided subsample: the median and
        # sigma of the background are insensitive to it but it is ~10x cheaper
        # than clipping all ~85M pixels.
        _, med, sig = sigma_clipped_stats(data[::4, ::4], maxiters=3)
        signal_thres = med + detect_thres * sig

        chip    = os.path.basename(datapath).split(".")[1]

        # Determine the direction of masking based on the chip type
        if chip in ['kk', 'nn']:
            direction = 'downward'
        elif chip in ['mm', 'tt']:
            direction = 'upward'
        else:
            raise ValueError("Unknown chip type")

        sat_full = data > bleeding_thres
        sig_full = data > signal_thres
        # Only columns that actually contain saturated pixels can bleed.
        cols = np.where(sat_full.any(axis=0))[0]
        idx = np.arange(leny)

        def _trail(sat_col, sig_col):
            # Travel in +index order; for 'downward' chips bleeding runs toward
            # decreasing row, so the column is reversed before/after the scan.
            if direction == 'downward':
                sat_col = sat_col[::-1]
                sig_col = sig_col[::-1]
            faint = ~(sat_col | sig_col)
            # Length of the current run of consecutive faint pixels.
            cs = np.cumsum(faint)
            reset = np.where(~faint, cs, 0)
            run = cs - np.maximum.accumulate(reset)
            term = run == CL                       # trail dies here
            last_seed = np.maximum.accumulate(np.where(sat_col, idx, -1))
            last_term = np.maximum.accumulate(np.where(term, idx, -1))
            last_term_before = np.empty(leny, dtype=np.int64)
            last_term_before[0] = -1
            last_term_before[1:] = last_term[:-1]
            active = (last_seed >= 0) & (last_seed > last_term_before)
            return active[::-1] if direction == 'downward' else active

        for i in cols:
            mask_col = _trail(sat_full[:, i], sig_full[:, i])
            if mask_col.any():
                bleeding_mask[mask_col, i] = flagval

        return bleeding_mask
    # -------------------------------------------------------------------------
    def crmap(fname, ction=False, bleedreject=False) :

        """
        generating a badpixel mask
        1. cosmic-ray
        2. crosstalk
        3. bleeding pattern
        """

        Msg.crmap()
        crmapname = fname.replace('.fits', '.mask.fits')
        # Cosmic-ray masking
        data, hdr = fits.getdata(fname, header=True)
        try:
            c1, c2 = cr.detect_cosmics(
                data,
                gain    = 1.0,
                readnoise= 10,
                sigclip = 4.5,
                sigfrac = 0.3,
                objlim  = 5.0,
                niter   = 2,
                cleantype= 'medmask',
                fsmode  = 'median',
                verbose = False
            )
        except Exception as e:
            # A cosmic-ray failure must not abort the whole frame: fall back to
            # an empty CR layer and keep the crosstalk/bleed masking.
            print(f'  ! cosmic-ray rejection failed for {os.path.basename(fname)} ({e}); continuing without CR mask.')
            c1 = np.zeros_like(data, dtype=bool)

        # Crosstalk masking
        saturation_limit = 56000
        mask    = np.where(data > saturation_limit, 1, 0)
        bound   = np.arange(0, int(hdr['NAXIS1'])+1, int(hdr['NAXIS1']/8))

        amp1  = mask[:,bound[0]:bound[1]]
        amp2  = mask[:,bound[1]:bound[2]]
        amp3  = mask[:,bound[2]:bound[3]]
        amp4  = mask[:,bound[3]:bound[4]]
        amp5  = mask[:,bound[4]:bound[5]]
        amp6  = mask[:,bound[5]:bound[6]]
        amp7  = mask[:,bound[6]:bound[7]]
        amp8  = mask[:,bound[7]:bound[8]]
        
        ct1     = amp1 + amp3 + np.flip(amp5, axis=1)# + np.flip(amp7, axis=1) # Xtalk weaker while propagating through amps
        ct2     = amp2 + amp4 + np.flip(amp6, axis=1)# + np.flip(amp8, axis=1)
        ct3     = amp1 + amp3 + np.flip(amp5, axis=1) + np.flip(amp7, axis=1)
        ct4     = amp2 + amp4 + np.flip(amp6, axis=1) + np.flip(amp8, axis=1)
        ct5     = np.flip(amp1, axis=1) + np.flip(amp3, axis=1) + amp5 + amp7
        ct6     = np.flip(amp2, axis=1) + np.flip(amp4, axis=1) + amp6 + amp8
        if ction == True:
            ct7     = np.full_like(amp7, 4)
        else:
            ct7     = np.flip(amp3, axis=1) + amp5 + amp7# + np.flip(amp1, axis=1)
        ct8     = np.flip(amp4, axis=1) + amp6 + amp8# + np.flip(amp2, axis=1)
        Xtalk   = 1*c1+np.concatenate((ct1, ct2, ct3, ct4, ct5, ct6, ct7, ct8), axis=1)
        
        if bleedreject:
            bpMask  = bleed_masking(fname, data=data)
            Xtalk   = np.add(Xtalk, bpMask)

        fits.PrimaryHDU(data=Xtalk.astype(np.int16), header=fits.getheader(fname)).writeto(crmapname, overwrite=True)
    # -------------------------------------------------------------------------
    def mask2int16(crmap):
        
        if not os.path.isfile(crmap):
            print(f'No Bad Pixel Mask Exist: {crmap}.')
            return None
        # Open the FITS file using a context manager
        with fits.open(crmap) as mask:
            # Convert pixel values to integers and create a new FITS file
            crdata  = mask[0].data
            if crdata.dtype == np.int16:
                return None
            else:
                mdata   = np.ceil(crdata).astype(np.int16)
                header  = mask[0].header
                fits.PrimaryHDU(mdata, header).writeto(crmap, overwrite=True)
        # Set file permissions using os.chmod
        os.chmod(crmap, 0o777)
        return None
    # -------------------------------------------------------------------------
    def mask2weight(crmap):
        
        if not os.path.isfile(crmap):
            print(f'No Bad Pixel Mask Exist: {crmap}.')
            return None

        mask_hdul = fits.open(crmap)
        mask_data = mask_hdul[0].data

        # Invert the mask values: 0 becomes 1 and 1 becomes 0
        mask_data[mask_data != 0] = 1
        weight_data = 1 - mask_data

        # Save the new data to a FITS file
        weight_hdul = fits.PrimaryHDU(weight_data)
        msuffix     = crmap.split('.')[-2]
        weight_name = crmap.replace(f'.{msuffix}.','.weight.')
        weight_hdul.writeto(weight_name, overwrite=True)

        # Close the original weight file to free resources
        mask_hdul.close()
        os.chmod(weight_name, 0o777)

        return weight_name
    # -------------------------------------------------------------------------
    def modify_config(fname, crmap, config_in, config_out, param='kmtnet.param'):
        """
        Program updated by Mankeun Jeong (2023.04.25.)
        Modifies a Source Extractor configuration file, updating specific parameters with custom values.

        Parameters:
        -----------
        fname : str
            Name of output configuration file, e.g., 'TOO_0503.061-76.I.20220323.CTIO.120sec.scaled.stack.fits'
        config_in : str
            Path to the default configuration file, e.g., 'default.sex'.
        config_out : str
            Path to the updated configuration file, e.g., f'{fname}.sex'.

        Note:
        -----
        Edit the `new_params` dictionary inside the function to change parameter values or add new ones.
        """
        new_params = {
            "CATALOG_NAME"      : f"{fname}.cat",
            "CATALOG_TYPE"      : "ASCII_HEAD",
            "PARAMETERS_NAME"   : param,
            "DETECT_THRESH"     : "3.0",
            "ANALYSIS_THRESH"   : "3.0",
            "FLAG_IMAGE"        : crmap,
            "FLAG_TYPE"         : "MAX",
            "PHOT_APERTURES"    : "12.5", # 5"
            "SATUR_LEVEL"       : "150000.0",
            "SATUR_KEY"         : "SATURATE",
            "PIXEL_SCALE"       : 0.4,
            "SEEING_FWHM"       : fits.getheader(fname)['FWHM'],
            "BACK_SIZE"         : 256,
            "BACKPHOTO_TYPE"    : "LOCAL",
            "MEMORY_BUFSIZE"    : 2048
            # Add more parameters to update as needed
        }

        with open(config_in, "r") as infile:
            lines = infile.readlines()

        with open(config_out, "w") as outfile:
            for line in lines:
                modified_line = line
                for param, value in new_params.items():
                    if line.startswith(param):
                        modified_line = f"{param}\t{value}\n"
                        break
                outfile.write(modified_line)

        return
    # -------------------------------------------------------------------------
    def load_incat(fname, incatname, configdir, flagname, checktype=None, weightphot=True) :
        '''
        Creates input catalog for target file after reading file name

        # Input Format :
        - fname : name of target file
        - incatname : name of '.cat' file
        
        # Output Format :
        - data : input catalog for target file
        '''

        Msg.run_sex()
        # configs
        cfg         = configdir+'kmtnet.sex'
        param       = configdir+'kmtnet_imask.param'
        conv        = configdir+'kmtnet.conv'
        nnw         = configdir+'kmtnet.nnw'
        prompt_cfg  = f'-c {cfg} -PARAMETERS_NAME {param} -FILTER_NAME {conv} -STARNNW_NAME {nnw}'
        prompt_cat  = f'-CATALOG_NAME {incatname} -CATALOG_TYPE ASCII_HEAD -DETECT_THRESH 5 -ANALYSIS_THRESH 5'
        try:
            prompt_opt  = f'-SEEING_FWHM {fits.getheader(fname)["FWHM"]}'
        except KeyError:
            prompt_opt  = ''
        prompt_flg  = f'-FLAG_IMAGE {flagname} -FLAG_TYPE MAX'
        # check image
        if checktype == None:
            prompt_chk = ''
        else:
            if checktype == '-BACKGROUND':
                checkname   = fname.replace(".fits", ".mbkg")
            elif checktype == 'BACKGROUND_RMS':
                checkname   = fname.replace(".fits", ".bkgrms")
            prompt_chk = f'-CHECKIMAGE_TYPE {checktype} -CHECKIMAGE_NAME {checkname}'
        # weight image
        weightname = mask2weight(flagname)
        if weightphot and weightname is not None:
            prompt_wgt = f' -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE {weightname} -RESCALE_WEIGHTS Y -WEIGHT_GAIN Y'
        else:
            prompt_wgt = ''

        prompt  = f'source-extractor {fname} {prompt_cfg} {prompt_cat} {prompt_opt} {prompt_flg} {prompt_chk} {prompt_wgt}'
        os.system(prompt)
        if weightphot and os.path.exists(weightname):
            os.remove(weightname)
        data = ascii.read(incatname)
        data = data[np.where(data['FLAGS'] == 0)] 
        data = data[np.where(data['IMAFLAGS_ISO'] == 0)]
        data['KRON_RADIUS_A'] = list(
            data['KRON_RADIUS'][i]*data['A_IMAGE'][i] 
            for i in range(len(data))
        )

        return data
    # -------------------------------------------------------------------------
    def load_refcat(fname, configdir, refcatdir, refcatname='gaiaxp', srchsz=1.0, gridcat='kmtnet_grid.fits') :
        '''
        Loads reference catalog for target file after reading file name

        # Input Format :
        - fname : name of target file
        - refcatdir='/data4/kmtntoo/cat/'
        - refcatname : name of reference catalog (e.g. 'GAIA EDR3') 

        # Output Format :
        - data : input catalog for target file
        '''

        fheader     = fits.getheader(fname)
        centcoord   = SkyCoord(fheader['CENTRA'], fheader['CENTDEC'], unit=(u.hourangle, u.deg))
        radd, decdd = centcoord.ra.deg, centcoord.dec.deg
        try:
            kmtgrid     = Table.read(os.path.join(configdir, gridcat), format='fits')
        except:
            kmtgrid     = Table.read(os.path.join(configdir, gridcat), format='ascii')
        kmtcoord    = SkyCoord(kmtgrid['ra[deg]'], kmtgrid['dec[deg]'], unit='deg')
        trgt_field  = kmtgrid[centcoord.separation(kmtcoord).argmin()]
        
        if centcoord.separation(kmtcoord).min().value > srchsz:
            print(f'Image pointing is out of {gridcat}. Check the coord in the header.')
            return None
        
        trgt     = f'{refcatname}_{str(trgt_field["field_name1"]).zfill(4)}.fits'

        if os.path.isfile(os.path.join(refcatdir, refcatname, trgt)) :
            Msg.load_refcat(local=True)
            refcat = pd.concat([
                pd.DataFrame(
                    np.array(
                        fits.open(os.path.join(refcatdir, refcatname, trgt))[1].data
                    ).byteswap().newbyteorder()
                ) 
            ])
        else :
            refcatname = 'gaiaedr3'
            trgt    = f'{refcatname}_{str(trgt_field["field_name1"]).zfill(4)}.fits'
            if os.path.isfile(os.path.join(refcatdir, refcatname, trgt)):
                refcat = pd.concat([
                    pd.DataFrame(
                        np.array(
                            fits.open(os.path.join(refcatdir, refcatname, trgt))[1].data
                        ).byteswap().newbyteorder()
                    )
                ])
            else:
                Msg.load_refcat(local=False)
                v = Vizier(columns = ['RAJ2000', 'DEJ2000'])
                v.ROW_LIMIT = -1 # no row limit
                v.TIMEOUT = 500
                result = v.query_region(
                    coord.SkyCoord(ra = radd, 
                                dec = decdd, 
                                unit = (u.deg, u.deg), 
                                frame = 'icrs'
                                ), 
                    width = 2.4*u.deg, 
                    catalog = ['GAIA EDR3'],
                    cache = False)
                refcat = result[0].to_pandas()
                os.makedirs(os.path.join(refcatdir, refcatname), exist_ok=True)
                Table.from_pandas(refcat).write(os.path.join(refcatdir, refcatname, trgt), overwrite=True)

        return refcat
    # -------------------------------------------------------------------------
    def cutclip(mref, cutnum, clipnum) :
        mrefcut = mref[mref['sep'] < cutnum]

        sigmaclip = sigma_clip(
            mrefcut['sep'], 
            sigma = clipnum, 
            maxiters = None, 
            cenfunc = median, 
            stdfunc = std)
        clip = np.where(sigmaclip.mask == False)
        csep = mrefcut['sep'].iloc[clip]*3600

        return mrefcut, csep
    # -------------------------------------------------------------------------
    def anlz_gbmap(mref, mrefcut, csep, divnum) :

        Msg.anlz_gbmap()

        df_sect = pd.DataFrame(
            columns = [
                'dtect num', 
                'rmsalign', 
                'alignstd', 
                'astrometry', 
                'dtct ratio'
            ]
        )
        for j in range(divnum) :
            ymin = 0.5 + j*9232/divnum
            ymax = 0.5 + (j+1)*9232/divnum
            for i in range(divnum) :
                xmin = 0.5 + i*9216/divnum
                xmax = 0.5 + (i+1)*9216/divnum

                def step(i, j) :
                    return ((i >= xmin) & 
                            (i <= xmax) & 
                            (j >= ymin) & 
                            (j <= ymax))

                sep0 = mref[
                    step(mref['XWIN_IMAGE'], 
                         mref['YWIN_IMAGE']
                        )
                ]['sep']*3600
                sep = mrefcut[
                    step(
                        mrefcut['XWIN_IMAGE'], 
                        mrefcut['YWIN_IMAGE']
                    )
                ]['sep']*3600

                if len(sep0) == 0 : #if no objects are detected
                    dtctRatio = 1
                    rmsalign = 0
                    alignstd = 0
                    sect_astrom = 'empty'
                else : 
                    if len(sep) == 0 : #if all objects are bad
                        dtctRatio = 0
                        rmsalign = 99
                        alignstd = 99
                        sect_astrom = 'bad'
                    else : 
                        dtctRatio = len(sep)/len(sep0)
                        rmsalign  = sqrt(mean((sigma_clip(sep, sigma=3))**2))
                        # rmsalign  = sqrt(mean((sep)**2))
                        alignstd  = std(sep)
                        sect_astrom = ('good' if (dtctRatio > 0.6 and 
                                                  rmsalign < 0.5) 
                                       else 'bad')
                df_sect.loc[8*j + i] = [len(sep), 
                                        rmsalign, 
                                        alignstd, 
                                        sect_astrom, 
                                        dtctRatio]

        # ---- Analysis report -------------------------------------------------
        gbmap_row = list(df_sect['astrometry'])
        bad_sect = [i for i, x in enumerate(gbmap_row) if x == 'bad']
        empty_sect = [i for i, x in enumerate(gbmap_row) if x == 'empty']

        # Pass/fail is decided ONLY from the sections lying on the outermost
        # `qa_edge_ring` ring(s) of the divnum x divnum grid. Interior sections are
        # almost always well solved (dense star coverage, well-constrained central
        # WCS), so they carry little diagnostic value and are deliberately ignored.
        # A poor astrometric/registration solution shows up first at the field
        # edges and corners, so QA scrutiny is focused there: the chip is rejected
        # once `qa_max_edge_bad` or more EDGE sections are flagged bad. The full
        # good/bad section map is still stored in the header for diagnostics.
        def _is_edge(k):
            i, j = k % divnum, k // divnum
            return (i < qa_edge_ring or i >= divnum - qa_edge_ring or
                    j < qa_edge_ring or j >= divnum - qa_edge_ring)
        edge_bad = [k for k in bad_sect if _is_edge(k)]

        if len(csep) > 0:
            med_off = float(median(csep))
            rms_off = float(sqrt(mean(csep**2)))
        else:
            med_off, rms_off = 99.0, 99.0

        good_solution = (len(edge_bad) < qa_max_edge_bad)
        fastrom = 'good' if good_solution else 'bad'
        badamp_ls = [[j+i*8 for i in range(8)] for j in range(8)]
        badamp_bl = True if empty_sect in badamp_ls else False
        result = [fname, fastrom, bad_sect]

        print(f'  QA: n_match={len(csep)}, median_off={med_off:.3f}", rms={rms_off:.3f}", '
              f'good_sect={gbmap_row.count("good")}/{divnum**2}, '
              f'edge_bad={len(edge_bad)}/{len([1 for k in range(divnum**2) if _is_edge(k)])} '
              f'(ring={qa_edge_ring}, interior ignored)')
        Msg.qaresult(fastrom)

        if fastrom == 'bad' :
            
            badastromtxt    = os.path.join(configdir, 'badastrom.txt')
            if os.path.exists(badastromtxt):
                with open(badastromtxt, 'a') as f :
                    f.write(f'badastrom {fname} \n')
            else:
                with open(badastromtxt, 'w') as f :
                    f.write(f'badastrom {fname} \n')

        return [gbmap_row, 
                bad_sect, 
                empty_sect, 
                fastrom, 
                badamp_bl, 
                result]
    # -------------------------------------------------------------------------
    def offset(mref) :
        try: df_refobj = SkyCoord(mref['RA'], mref['DEC'], unit=u.deg)
        except KeyError: df_refobj = SkyCoord(mref['RAJ2000'], mref['DEJ2000'], unit=u.deg)
        df_inpobj = SkyCoord(mref['ALPHA_J2000'], mref['DELTA_J2000'], unit=u.deg)

        dra, ddec = df_refobj.spherical_offsets_to(df_inpobj)
        dra = dra.to(u.arcsec)/u.arcsec
        ddec = ddec.to(u.arcsec)/u.arcsec
        sep = mref['sep'] * 3600

        hdr_dict = {
        'dra_mean' : RndVal.mean(dra),
        'dra_std' : RndVal.std(dra),
        'ddec_mean' : RndVal.mean(ddec),
        'ddec_std' : RndVal.std(ddec),
        'sep_mean' : RndVal.mean(sep),
        'sep_std' : RndVal.std(sep),
        'alnrms' : RndVal.rms(sep)
        }

        return hdr_dict
    # -------------------------------------------------------------------------
    def hdr_update(anlz_rprt, __version__, csep, refcatname, divnum, absfpath) :

        def encrypt(inp_ls) :
            zero_list = [0]*64
            for i in inp_ls :
                zero_list[i] += 1

            key   = '0123456789ABCDEF'
            ls    = []
            parts = [list(map(str, zero_list[i:i+4])) for i in range(0, 64, 4)]
            for part in parts :
                binnum = int(''.join(part),2)
                ls.append(key[binnum])

            return ''.join(ls)

        Msg.hdr_update()

        gbmap_row, bad_sect, empty_sect, fastrom, badamp_bl, result = anlz_rprt
        curdate = dt.today().strftime('%Y-%m-%d')
        fixed = 'Quality Assurance (QA) by QATEST version'
        hdrcmthist = {
            'HISTORY' : f'  {fixed} {__version__} ({curdate})',
            'COMMENT' : '  2022 JSH', 
        }

        hdrupdate = {
            'REASTROM': (
                False,
                'True if reastrometry done'
            ),
            'QAREFCAT': (
                refcatname.upper(), 
                'Reference Catalog used for QA'
            ),
            'QAALNNUM': (
                len(csep), 
                'Number of objects for QA [integer]'
            ),
            'QAALNRMS': (
                float(format(sqrt(mean(csep**2)), '.5f')), 
                'RMS of misalignment with QAREFCAT [arcsec]'
            ),
            'QAALNSTD': (
                float(format(std(csep), '.5f')), 
                'Uncertainty of misalignment [arcsec]'
            ),
            'QANSECT' : (
                divnum**2, 
                'Total num of divided sections for QA [integer]'
            ),
            'QAGDSECT': (
                gbmap_row.count('good'), 
                'Number of sections classified as good [integer]'
            ),
            'QABDSECT': (
                encrypt(bad_sect), 
                'Position of sections classified as bad'
            ),
            'QABADAMP': (
                badamp_bl, 
                'True if bad AMP exists'
            ),
            'QARESULT': (
                fastrom == 'good', 
                'True if QA is good'
            ),
            'MWEIGHT': (
                weightmap, 
                'True if MAP_WEIGHT applied'
            )
        }

        with fits.open(absfpath, 'update') as hdul:
        #with fits.open(f'dummydir/{fname}', 'update') as hdul:
            hdr = hdul[0].header
            for cmthist in hdrcmthist.keys() :
                hdr.insert(len(hdr), (cmthist, hdrcmthist[cmthist]))
            hdr.insert(
                len(hdr), 
                ('CCDNAME', fname.split('.')[1][0], 'Name of CCD')
            )
            for hdrkey in hdrupdate.keys() :
                hdr[hdrkey] = hdrupdate[hdrkey]
    # -------------------------------------------------------------------------
    def hdr_update_for_stack(hdr_dict, absfpath, refcatname, editmode=True) :

        Msg.hdr_update()

        hdrupdate = {
            'ASTREF': (
                refcatname.upper(),
                'Reference catalog used for astrometry QA'
            ),
            'DRAMEAN': (
                hdr_dict['dra_mean'],
                'Mean of RA direction offset [arcsec]'
            ),
            'DRASTD': (
                hdr_dict['dra_std'],
                'Uncertainty of RA offset [arcsec]'
            ),
            'DDECMEAN': (
                hdr_dict['ddec_mean'],
                'Mean of DEC direction offset [arcsec]'
            ),
            'DDECSTD': (
                hdr_dict['ddec_std'],
                'Uncertainty of DEC offset [arcsec]'
            ),
            'ALNMEAN': (
                hdr_dict['sep_mean'],
                'Mean of misalignment [arcsec]'
            ),
            'ALNSTD': (
                hdr_dict['sep_std'],
                'Uncertainty of misalignment [arcsec]'
            ),
            'ALNRMS': (
                hdr_dict['alnrms'], 
                'RMS of misalignment with QAREFCAT [arcsec]'
            ),
        }

        trgtpath = fname if editmode else absfpath

        with fits.open(fname, 'update') as hdul:
            hdr = hdul[0].header
            for hdrkey in hdrupdate.keys() :
                hdr[hdrkey] = hdrupdate[hdrkey]
    # ======= MAIN ============================================================

    Msg.start(fname)
    if os.path.isfile(fname) == True :
        if imtype == 'chip':
            # 1. cosmic-ray mask generation
            if crreject == True:
                if fits.getheader(fname)['OBSERVAT']=='CTIO' and os.path.basename(fname).split('.')[1] == 'nn':
                    crmap(fname, ction=True, bleedreject=bleedreject)
                else:
                    crmap(fname, ction=False, bleedreject=bleedreject)
            # 2. astrometry quality assurance
            # 2.1. SExtractor run
            data = load_incat(fname, incatname=fname.replace(".fits", ".astromqa.cat"), configdir=configdir, flagname=fname.replace(".fits", ".mask.fits"), checktype='-BACKGROUND', weightphot=weightmap)
            # 2.2. tracking issue check
            if len(data) < 50:
                Msg.err(fname, 'trackerr')
            else:
                trkerr = data[np.where(data['KRON_RADIUS_A'] > 200)]        
                if len(trkerr) < 50 :
                    # 2.3. reference catalog query
                    refcat  = load_refcat(fname, configdir=configdir, refcatdir=refcatdir, refcatname=refcatname, srchsz=1.0, gridcat=gridcat)
                    # 2.4. pointing issue check
                    if refcat is not None:
                        # 2.5. match the sources with the reference
                        reftbl  = Table.from_pandas(refcat)
                        # Determine the column names for RA and DEC in the input table
                        inra_col = 'ALPHA_J2000' if 'ALPHA_J2000' in data.colnames else 'RAJ2000'
                        indec_col = 'DELTA_J2000' if 'DELTA_J2000' in data.colnames else 'DEJ2000'
                        # Determine the column names for RA and DEC in the reference table
                        refra_col = 'RA' if 'RA' in reftbl.colnames else 'RAJ2000'
                        refdec_col = 'DEC' if 'DEC' in reftbl.colnames else 'DEJ2000'
                        # Define matching parameters
                        param_matching = dict(intbl   = data,
                                            reftbl  = reftbl,
                                            inra    = data[inra_col], 
                                            indec   = data[indec_col],
                                            refra   = reftbl[refra_col], 
                                            refdec  = reftbl[refdec_col],
                                            sep     = 2)
                        mref    = matching(**param_matching).to_pandas()
                        # 2.6. cliping unmatched catalog (cutnum: match radius threshold [arcsec], clipnum: sigma threshold)
                        mrefcut, csep   = cutclip(mref, cutnum=2/3600, clipnum=3)
                        # 2.7. [Main] analyzing good & bad sections.
                        # divnum: divide the image into 8x8 segments for analysis
                        # in the function, image size is customized for KMTNet chip images
                        anlz_rprt       = anlz_gbmap(mref, mrefcut, csep, divnum)
                        # 2.8. header updates
                        hdr_update(anlz_rprt, 
                                __version__, 
                                csep, 
                                refcatname, 
                                divnum, 
                                fname)
                    else:
                        Msg.err(fname, 'pointerr')
                else : 
                    Msg.err(fname, 'trackerr')
            
        elif imtype =='stack':
            fmask = fname.replace('.stack.', '.mstack.')
            inconfigname= f'{fname}.sex'
            incatname   = f'{fname}.cat'
            if os.path.isfile(fmask):
                mask2int16(fmask) # BITPIX = 16
                modify_config(fname, fmask, f"{configdir}kmtnet.sex", inconfigname, f'{configdir}kmtnet.param') # fname.sex generation
            else: # no cr reduction map
                modify_config(fname, f"{configdir}flag.fits", f"{configdir}kmtnet.sex", inconfigname, f'{configdir}kmtnet.param')
            data    = load_incat(fname, incatname, configdir, flagname=fname.replace(".stack.", ".mstack."), checktype='BACKGROUND_RMS', weightphot=weightmap)
            refcat  = load_refcat(fname, configdir=configdir, refcatdir=refcatdir, refcatname=refcatname, srchsz=1.0, gridcat=gridcat)
            if refcat is not None:
                reftbl  = Table.from_pandas(refcat)
                inra_col = 'ALPHA_J2000' if 'ALPHA_J2000' in data.colnames else 'RAJ2000'
                indec_col = 'DELTA_J2000' if 'DELTA_J2000' in data.colnames else 'DEJ2000'
                # Determine the column names for RA and DEC in the reference table
                refra_col = 'RA' if 'RA' in reftbl.colnames else 'RAJ2000'
                refdec_col = 'DEC' if 'DEC' in reftbl.colnames else 'DEJ2000'
                param_matching = dict(intbl   = data,
                                    reftbl  = reftbl,
                                    inra    = data[inra_col], 
                                    indec   = data[indec_col],
                                    refra   = reftbl[refra_col], 
                                    refdec  = reftbl[refdec_col],
                                    sep     = 2)
                mref    = matching(**param_matching).to_pandas()
                mrefcut = mref[mref['sep'] < 2]
                sigmaclip   = sigma_clip(
                    mrefcut['sep'], 
                    sigma   = 3, 
                    maxiters= None, 
                    cenfunc = median, 
                    stdfunc = std)
                clip        = np.where(sigmaclip.mask == False)
                mrefcutclip = mrefcut.iloc[clip]
                hdr_dict    = offset(mrefcutclip)
                hdr_update_for_stack(hdr_dict, fname, refcatname, editmode=True)
            else:
                Msg.err(fname, 'pointerr')
        else:
            print('Specify the type of the input image correctly: "chip" or "stack"?')
    else : 
        Msg.err(fname, 'fpatherr')
    Msg.end()
#%% ToOZeroPointScaler.py
def zpscale(img, path_output, path_cfg, path_cat, path_plot, mode='1DLINEAR', magkey='AUTO', zpscaled=30.0, pixscale=0.4, gain=1, figure=False, start=None, gridcat='kmtnet_grid.cat'):
    """
    Zero-point calibration and homogenization for KMTNet chip images.
    
    This function performs photometric calibration and homogenization of KMTNet chip images
    by estimating zero-points for each of the eight amplifier regions and correcting
    geometric tendencies to achieve uniform photometric quality across the entire image.
    
    Parameters
    ----------
    img : str
        Path to the input KMTNet chip image FITS file (e.g., '057488.kk.fits').
    path_output : str
        Path to the output directory where scaled images will be saved.
    path_cfg : str
        Path to the configuration directory containing:
        - kmtnet.sex: SExtractor configuration file
        - kmtnet_imask.param: SExtractor parameters file
        - kmtnet.conv: SExtractor convolution file
        - kmtnet.nnw: SExtractor neural network weights file
        - kmtnet_grid.cat: KMTNet field grid catalog
    path_cat : str
        Path to the reference catalog directory containing:
        - gaiaxp/: GAIA XP reference catalogs
        - apass/: APASS reference catalogs
    path_plot : str
        Path to the output directory for diagnostic plots.
    mode : str, optional
        Scaling mode for zero-point correction:
        - '1DLINEAR': Linear fitting along Y-axis for each amplifier
        - '2DPOLYNOMIAL': 2D polynomial fitting across X,Y plane for each amplifier
        Default is '1DLINEAR'.
    magkey : str, optional
        Magnitude type to use for photometry ('AUTO', 'APER', etc.).
        Default is 'AUTO'.
    zpscaled : float, optional
        Target zero-point magnitude for scaling. Default is 30.0.
    pixscale : float, optional
        Pixel scale in arcseconds per pixel. Default is 0.4.
    gain : float, optional
        Detector gain for photometry. Default is 1.0.
    figure : bool, optional
        Whether to generate diagnostic plots. Default is False.
    start : float, optional
        Start time for elapsed time calculation. Default is None.
    gridcat : str, optional
        Name of the KMTNet grid catalog file. Default is 'kmtnet_grid.cat'.
    
    Returns
    -------
    str
        Filename of the output scaled image.
    
    Notes
    -----
    The function performs the following operations:
    
    1. **Quality Assurance Check**:
       - Verifies that astrometric QA has been completed (QARESULT=True)
       - Checks image pointing against KMTNet field grid
       - Validates image coordinates and field assignment
    
    2. **Reference Catalog Query**:
       - Loads appropriate reference catalog (GAIA XP or APASS)
       - Matches image pointing to KMTNet field grid
       - Downloads reference catalog if not available locally
    
    3. **Photometric Analysis**:
       - Runs SExtractor to detect sources and measure photometry
       - Applies bad-pixel masks and weight maps if available
       - Performs source cleaning and quality filtering
    
    4. **Amplifier-wise Zero-point Scaling**:
       - **1D Linear Mode**: Fits linear trend along Y-axis for each amplifier
       - **2D Polynomial Mode**: Fits 2D polynomial surface across X,Y plane
       - Uses robust regression (Huber loss) for outlier rejection
       - Calculates flux scaling factors for each pixel
    
    5. **Image Processing**:
       - Applies zero-point scaling corrections to image data
       - Removes background and pattern noise
       - Updates FITS headers with scaling parameters
    
    6. **Quality Control**:
       - Tracks failed amplifier fittings
       - Updates header with scaling results and quality flags
       - Generates diagnostic plots if requested
    
    Scaling Methods
    ---------------
    - **1D Linear**: Corrects Y-axis zero-point variations using linear regression
    - **2D Polynomial**: Corrects both X and Y zero-point variations using polynomial fitting
    
    Quality Criteria
    ---------------
    - **Good Amplifier**: Successful fitting with sufficient reference stars
    - **Bad Amplifier**: Failed fitting or insufficient reference stars
    - **Overall QA**: Passes if majority of amplifiers are successfully fitted
    
    Header Keywords Added
    ---------------------
    - PHOTREF: Reference catalog used for photometry
    - FWHM: Updated seeing value from photometric analysis
    - SLOPE1-8: Linear slope for each amplifier
    - OFFSET1-8: Linear offset for each amplifier
    - FITSTAR1-8: Number of stars used for fitting each amplifier
    - BADAMP: Binary flag indicating failed amplifiers
    - PHOTQA: Overall photometric quality assessment
    - MAGZERO: Target zero-point magnitude
    - SATURATE: Updated saturation level after scaling
    - SCALEFIT: Scaling method used
    
    Examples
    --------
    >>> # 1D Linear scaling
    >>> zpscale('/path/to/057488.kk.fits', '/path/to/output/', '/path/to/config/',
    ...         '/path/to/catalogs/', '/path/to/plots/', mode='1DLINEAR',
    ...         magkey='AUTO', zpscaled=30.0, figure=True)
    
    >>> # 2D Polynomial scaling
    >>> zpscale('/path/to/057488.kk.fits', '/path/to/output/', '/path/to/config/',
    ...         '/path/to/catalogs/', '/path/to/plots/', mode='2DPOLYNOMIAL',
    ...         magkey='AUTO', zpscaled=30.0, figure=True)
    
    See Also
    --------
    ampcom : Amplifier combination function
    astrom : Astrometric calibration function
    qatest : Quality assurance function
    KMTNet_ToO_pipeline : Main pipeline that calls this function
    
    Requirements
    ------------
    - Input image must have completed astrometric QA (QARESULT=True)
    - Reference catalogs must be available in path_cat directory
    - SExtractor configuration files must be present in path_cfg directory
    """
    
    import os
    import time
    import scipy
    import shutil
    import numpy as np
    import astropy.units as u
    from astropy.io import fits
    from astropy.io import ascii
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from astropy.coordinates import SkyCoord
    from astropy.modeling import models, fitting
    from sklearn.linear_model import HuberRegressor
    
    # 0. Mode checker
    mode = mode.upper()
    if mode not in ['1DLINEAR', '2DPOLYNOMIAL']:
        raise ValueError("Mode must be either '1DLINEAR' or '2DPOLYNOMIAL'")
    
    if not path_output.endswith('/'):
        path_output   = path_output + '/'
        
    # 1. Basic analysis
    
    # 1.1. Image header
    file    = fits.open(img)
    serial  = os.path.basename(img).split('.')[0]
    hdr     = fits.getheader(img)
    # data    = fits.getdata(img)
    
    try:
        chip    = hdr['CCDNAME'].upper()
    except:
        chip    = os.path.basename(img).split('.')[-2][0].upper() # 000000.kk.fits
    band    = hdr['FILTER']
    obs     = hdr['OBSERVAT']
    
    xaxis   = hdr['NAXIS1']
    yaxis   = hdr['NAXIS2']
    
    obj     = hdr['OBJECT'].split('_')[0] # e.g. S190425z, TOO_0503, etc. 
    center  = SkyCoord(hdr['CENTRA'], hdr['CENTDEC'], unit=(u.hourangle, u.deg))
    
    # 1.2. Preprocess completeness check
    try:
        if hdr['QARESULT']==False: 
            print(f'Bad QA in astrometry for {img}.'); return None
    except KeyError:
        print(f'No QARESULT for {img}. Run astromqa in advance.'); return None

    # 1.3. Coordinates check (ks4 predetermined tiles)
    try:
        ks4cat  = ascii.read(os.path.join(path_cfg, gridcat))
    except:
        ks4cat  = Table(fits.open(os.path.join(path_cfg, gridcat))[1].data)
    ks4flds = SkyCoord(ks4cat['ra[deg]'], ks4cat['dec[deg]'], unit='deg')
    field   = str(ks4cat[center.separation(ks4flds).argmin()]['field_name1']).zfill(4)
    radec   = ks4cat[ks4cat['field_name1']==int(field)]['field_name2'][0]
    if center.separation(ks4flds).min().value > 1.0:
        print(f'Image pointing is out of {gridcat}. Check the coord in the header.'); return None

    # Basic analysis done. Checking an intermediate time.
    mid     = time.time()
    print('='*60)
    if start == None:
        print(f"ZPSCALPRO: {serial}_{field}.{radec}_{band}_{chip}_{obs} ({mid:.2f}sec elapsed)")
    else:
        print(f"ZPSCALPRO: {serial}_{field}.{radec}_{band}_{chip}_{obs} ({mid-start:.2f}sec elapsed)")
    print('='*60)
    
    # 2. Reference catalog query
    # This is basically prepared for KS4 fields.
    try:
        path_ref    = os.path.join(path_cat, 'gaiaxp')
        reftbl  = GAIAXP_query(field, path_ref)
        hdr['PHOTREF']  = 'GAIA XP'
    except FileNotFoundError:
        path_ref    = f'{path_cat}apass/'
        os.makedirs(path_ref, exist_ok=True)
        try:
            reftbl  = ascii.read(f'{path_ref}apass_{field}.cat')
        except FileNotFoundError:
            xscale  = xaxis * pixscale # arcsec
            yscale  = yaxis * pixscale # arcsec
            frac    = 3 # >2*np.sqrt(2) due to dithering
            radius  = frac*np.mean([xscale, yscale])/3600 # searching radius in deg
            reftbl  = apass_query(center.ra.deg, center.dec.deg, radius)
            reftbl.write(f'{path_ref}apass_{field}.cat', format='ascii', overwrite=True)
        hdr['PHOTREF']  = 'APASS DR9'
    
    # 3. Photometry: SExtractor run
    
    # 3.1. FWHM check. Utilizing values from previous quality check --> this should be in the header (important)
    try:
        seeing = float(hdr[f'FWHM'])
    except KeyError:
        seeing  = 2.0
    
    # 3.2. Configurations
    param       = os.path.join(path_cfg, 'kmtnet_imask.param')
    cfg         = os.path.join(path_cfg, 'kmtnet.sex')
    conv        = os.path.join(path_cfg, 'kmtnet.conv')
    nnw         = os.path.join(path_cfg, 'kmtnet.nnw')
    
    catname     = img.replace('.fits', '.zpscal.cat')
    mbkgname     = img.replace('.fits', '.mbkg')
    # resname     = img.replace('.fits', '.res')
    thres       = 5     # detection threshold
    bkgsize     = 256   # background size (global)

    # 3.3. Command line args
    inim_single = img
    prompt_cat  = f' -CATALOG_NAME {catname} -CATALOG_TYPE ASCII_HEAD'
    prompt_aper = f' -PHOT_APERTURES {14/pixscale}' # 14" same with APASS
    prompt_cfg  = f' -c {cfg} -PARAMETERS_NAME {param} -FILTER_NAME {conv} -STARNNW_NAME {nnw}'
    prompt_opt  = f' -GAIN {gain:.2f} -PIXEL_SCALE {pixscale:.2f} -SEEING_FWHM {seeing:.2f} -SATUR_LEVEL 57000'
    prompt_chk  = f' -CHECKIMAGE_TYPE -BACKGROUND -CHECKIMAGE_NAME {mbkgname}' # -CHECKIMAGE_TYPE -OBJECT -CHECKIMAGE_NAME {resname}
    prompt_flg  = f' -FLAG_IMAGE {img.replace(".fits", ".mask.fits")}'
    prompt_bkg  = f' -BACK_SIZE {bkgsize} -DETECT_THRESH {thres}'
    if os.path.exists(img.replace(".fits", ".weight.fits")):
        prompt_wgt  = f' -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE {img.replace(".fits", ".weight.fits")} -RESCALE_WEIGHTS Y -WEIGHT_GAIN Y'
    else:
        prompt_wgt = ''
    prompt      = 'source-extractor '+inim_single+prompt_cfg+prompt_aper+prompt_opt+prompt_cat+prompt_chk+prompt_flg+prompt_bkg+prompt_wgt
    if os.path.exists(img.replace(".fits", ".astromqa.cat")) and os.path.exists(mbkgname):
        shutil.move(img.replace(".fits", ".astromqa.cat"), catname)
    else:
        os.system(prompt)
    
    # 4. Output analysis 
    
    # 4.1. FWHM re-check
    intbl   = ascii.read(catname)
    stbl    = intbl[intbl['CLASS_STAR']>np.median(intbl['CLASS_STAR'])]
    stbl    = stbl[stbl['FLAGS']==0]
    stbl    = stbl[stbl[f'MAGERR_{magkey}']<0.05]
    seeing  = np.median(stbl['FWHM_IMAGE'])*0.4
    hdr['FWHM'] = round(seeing,3)
    
    # 4.2. Electronic pattern noise
    """ Only For SAAO Images (2020.11~2022.12)
    residue = fits.getdata(resname)
    bound   = np.arange(0, int(hdr['NAXIS1'])+1, int(hdr['NAXIS1']/8))

    c1  = residue[:,bound[0]:bound[1]]
    c2  = residue[:,bound[1]:bound[2]]
    c3  = residue[:,bound[2]:bound[3]]
    c4  = residue[:,bound[3]:bound[4]]
    c5  = residue[:,bound[4]:bound[5]]
    c6  = residue[:,bound[5]:bound[6]]
    c7  = residue[:,bound[6]:bound[7]]
    c8  = residue[:,bound[7]:bound[8]]

    pattern    = np.median(np.dstack([np.flip(c1, axis=1), np.flip(c2, axis=1), np.flip(c3, axis=1), np.flip(c4, axis=1), c5, c6, c7, c8]), -1)
    """
    
    # 5. Zero-point scaling
    
    badamp  = 0 
    imscale = np.ones((yaxis, xaxis), dtype=np.float32)  
    
    for i in range(8):

        # 5.1. Amp division
        # i=7
        xps     = int(xaxis/8*(i))
        xpe     = int(xaxis/8*(i+1))
    
        amptbl  = intbl[intbl['X_IMAGE']>xps]
        amptbl  = amptbl[amptbl['X_IMAGE']<xpe]

        # 5.2. Matching with the reference
        thres   = 0
        while 1:
            if 'QAALNRMS' not in hdr: rad   = 0.5
            else: rad = 0.2+hdr['QAALNRMS']*thres
            param_matching  = dict( intbl   = amptbl,
                                    reftbl  = reftbl,
                                    inra    = amptbl['ALPHA_J2000'], 
                                    indec   = amptbl['DELTA_J2000'],
                                    refra   = reftbl['RAJ2000'], 
                                    refdec  = reftbl['DEJ2000'],
                                    sep     = rad)
            mtbl    = matching(**param_matching)
            thres += 1
            if len(mtbl) > 300 or thres > 5 or 'QAALNRMS' not in hdr:
                break
        print(f'MATCHING RADIUS: {rad:.2f} arcsec (NUM OF STARS = {len(mtbl)})')
        mtbl    = mtbl[mtbl[f'MAG_{magkey}']!=99]

        # 5.3. Source cleaning
        cmtbl   = mtbl[mtbl['FLAGS']<4]
        cmtbl   = cmtbl[cmtbl['IMAFLAGS_ISO']==0]
        cmtbl   = cmtbl[~np.isnan(cmtbl[f'{band}mag'])]
        cmtbl   = cmtbl[cmtbl[f'MAGERR_{magkey}']<0.05]
        cmtbl   = cmtbl[cmtbl['CLASS_STAR']>np.median(amptbl['CLASS_STAR'])]
        
        # 5.4. Zeropoint for each sources (2d distribution)
        magdif  = cmtbl[f'MAG_{magkey}'] - cmtbl[f'{band}mag']
        magerr  = rss([cmtbl[f'MAGERR_{magkey}'], cmtbl[f'e_{band}mag']])
      
        # 5.5. Zeropoint tendency fitting (1DLinear)
        if mode == '1DLINEAR':
            def linfun(x, a, b): return a*x+b
            
                # 5.5.1. Huber-loss regression fitting
            try:
                X = np.array(cmtbl['Y_IMAGE']).reshape(-1,1)
                Y = np.array(magdif)
                # Y = np.array(magdif-zp)
                huber = HuberRegressor().fit(X,Y)
                m = huber.coef_[0]
                b = huber.intercept_
                s = huber.score(X,Y)
                hdr['SLOPE{}'.format(i+1)]  = float('{:.3e}'.format(m))
                hdr['OFFSET{}'.format(i+1)] = round(b, 3)
                hdr[f'FITSTAR{i+1}']        = len(X)
            except:
                s = -1
            
            # 5.5.2. Scipy curve fit in case poor regression fitting
            if s < -0.1 or len(cmtbl) < 3:
                try:
                    popt, pcov   = curve_fit(linfun, cmtbl['Y_IMAGE'], magdif)
                    hdr['SLOPE{}'.format(i+1)]  = float('{:.3e}'.format(popt[0]))
                    hdr['OFFSET{}'.format(i+1)] = round(popt[1], 3)
                    m, b    = popt[0], popt[1]
                    if np.sqrt(np.diag(pcov))[0] != np.inf:
                        hdr[f'FITSTAR{i+1}']    = len(X)
                    else:
                        hdr[f'FITSTAR{i+1}']    = len(X)
                        badamp  += 1*10**i
                        continue
                except (ValueError, TypeError, scipy.optimize.OptimizeWarning):
                    popt, pcov   = np.array([0, -30], dtype=np.float32), np.zeros([2, 2], dtype=np.float32)
                    hdr['SLOPE{}'.format(i+1)]  = float('{:.3e}'.format(popt[0]))
                    hdr['OFFSET{}'.format(i+1)] = round(popt[1], 3)
                    hdr[f'FITSTAR{i+1}']    = len(X)
                    badamp  += 1*10**i
                    continue
            
            print(f'AMP{i} Flux Scaling: {m:.2e} * y + {b:.2f}')
            # 5.5.3. Scaling factor array composite
            for j in range(yaxis):
                del_zp = zpscaled + linfun(j, m, b)
                fratio = round(10**(del_zp/(2.5)), 4)
                for k in np.arange(xps, xpe):
                    imscale[j][int(k)] = fratio
            
            # 5.5.3. Fitting the result plot 
            if figure: 
                plt.figure(figsize=(7,4))
                plt.errorbar(cmtbl['Y_IMAGE'], magdif, magerr, ms=6, marker='s', ls='', c='dodgerblue', capsize=4, capthick=1, alpha=0.5, zorder=0)
                plt.plot(X, m*X+b,color='blue', zorder=1, label=f'Linear fit (slope={m:.2e}, const={b:.2f})')
                plt.xlim(0, xaxis)
                plt.legend(loc='upper right')
                plt.title(f'Zeropoint Tendency \n{radec}-field, {band}-band, {chip}-chip, {i+1}-amp\n Stars={len(cmtbl)}, $R^2$={s:.3f}')
                plt.xlabel('Y_IMAGE [PIXEL]')
                plt.ylabel(rf'$\Delta {band}_{{KMTN-REF}}$ [ABmag]')
                plt.ylim(-30,-28)
                plt.savefig(f'{path_plot}{os.path.basename(img)}_{i+1}AMP_zptend.png')
                plt.close()
                
        # 5.6. Zeropoint tendency fitting (2DPolynomial)
        elif mode == '2DPOLYNOMIAL':
                        
            cmtbl['magdif'] = magdif
            mask = np.ones(len(cmtbl), dtype=bool)  # start with all points
            max_iter = 5
            sigma_threshold = 3.0
            if len(cmtbl) > 0:
                for iteration in range(max_iter):
                    # Fit a 2D polynomial of degree 2 using only the points not rejected.
                    p_init = models.Polynomial2D(degree=3)
                    fitter = fitting.LevMarLSQFitter()
                    p_fit = fitter(p_init, cmtbl['X_IMAGE'][mask], cmtbl['Y_IMAGE'][mask], cmtbl['magdif'][mask])
                    
                    # Evaluate the fitted model at all catalog positions
                    zp_fit_all = p_fit(cmtbl['X_IMAGE'], cmtbl['Y_IMAGE'])
                    
                    # Compute residuals
                    residuals = cmtbl['magdif'] - zp_fit_all
                    
                    # Calculate standard deviation using only the currently accepted data
                    std_res = np.std(residuals[mask])
                    
                    # Update the mask: keep only points within the sigma threshold
                    new_mask = np.abs(residuals) < sigma_threshold * std_res
                    print(f"Iteration {iteration+1}: {np.sum(new_mask)} out of {len(cmtbl)} points kept")
                    
                    # Check for convergence (mask no longer changes)
                    if np.all(new_mask == mask):
                        print(f"Convergence reached after {iteration+1} iterations.")
                        break
                    
                    mask = new_mask
                x_grid = np.arange(xps, xpe)  # x indices: 0 to 1151
                y_grid = np.arange(9232)  # y indices: 0 to 9231

                # Create a meshgrid using 'xy' indexing.
                X, Y = np.meshgrid(x_grid, y_grid, indexing='xy')

                # Evaluate the final fitted model on the grid to produce the zero-point map.
                zp_map = p_fit(X, Y)
                zp_to_scale = 30.0

                # Compute the difference needed to correct the zero-point.
                del_zp = zp_to_scale + zp_map

                # Compute the flux ratio.
                fratio = 10**(del_zp / 2.5)

                imscale.T[xps:xpe] = fratio.T
                if figure:
                    # Calculate R2
                    zp_fit_values = p_fit(cmtbl['X_IMAGE'], cmtbl['Y_IMAGE'])
                    residuals = magdif - zp_fit_values
                    SS_res = np.sum(residuals**2)
                    SS_tot = np.sum((magdif - np.mean(magdif))**2)
                    R_squared = 1 - (SS_res / SS_tot)
                    # plot
                    plt.figure(figsize=(6, 6))
                    plt.suptitle(f'Zeropoint Tendency \n{radec}-field, {band}-band, {chip}-chip, {i+1}-amp\n Stars={len(cmtbl)}, $R^2$={R_squared:.3f}')
                    plt.subplot(121)
                    plt.imshow(zp_map, origin='lower', aspect='auto', cmap='RdBu')
                    plt.xlabel('X_IMAGE [PIXEL]')
                    plt.ylabel('Y_IMAGE [PIXEL]')
                    plt.clim(min(magdif), max(magdif))
                    plt.subplot(122)
                    plt.scatter(cmtbl['X_IMAGE']%1152, cmtbl['Y_IMAGE'], c=magdif, s=10, cmap='RdBu', alpha=1.0)
                    plt.clim(min(magdif), max(magdif))
                    plt.colorbar(label=rf'$\Delta {band}_{{KMTN-REF}}$ [ABmag]')
                    plt.xlabel('X_IMAGE [PIXEL]')
                    plt.tight_layout()
                    plt.show()
            else:
                imscale.T[xps:xpe] = 1.0
                badamp  += 1*10**i
    
    # 6. Pattern noise reduction, background subtraction
    mbkgdata        = fits.getdata(mbkgname)
    # if obs == 'SAAO':
    #     flippattern     = np.flip(pattern, axis=1)
    #     noisedata       = np.concatenate((flippattern, flippattern, flippattern, flippattern, pattern, pattern, pattern, pattern), axis=1)
    #     file[0].data    = imscale * (data - bkgdata - noisedata)
    # else:
    file[0].data    = imscale * (mbkgdata)
        
    # 7. Header update
    hdr['FIELD1']   = field
    hdr['FIELD2']   = radec
    hdr['BADAMP']   = str(badamp).zfill(8)[::-1]
    hdr.comments['BADAMP'] = 'Amps flag (0:okay, 1:fitting fail)'
    if badamp == 0:
        hdr['PHOTQA']   = True
    else:
        hdr['PHOTQA']   = False
    hdr['MAGZERO']  = zpscaled
    hdr.comments['MAGZERO'] = f'ABmagnitude system, {magkey} aperture'
    hdr['SATURATE']     = 60000 * np.median(imscale)
    hdr.comments['SATURATE']    = 'Saturation values after zero-point scaling'
    hdr['SCALEFIT']     = mode
    
    # 8. Ouput file save
    newname     = f'{obj}_{field}.{radec}.{band}.{hdr["DATE-OBS"].split("T")[0].replace("-","")}.{obs}.{serial}.{os.path.basename(img).split(".")[1]}.scaled.fits'
    fits.writeto(f'{path_output}{newname}', file[0].data, hdr, overwrite=True)

    os.system(f'rm {mbkgname}')
    # os.system(f'rm {resname}')
    os.system(f'chmod 777 {path_output}{newname}')

    return newname

#%% ToOBadPixelMask.py
def BPM_update(img, path_cfg):
    """
    Bad Pixel Map (BPM) combining and updating function for KMTNet images.
    
    This function combines cosmic ray masks, bad pixel maps, and bad amplifier
    information to create a comprehensive bad pixel mask for KMTNet images.
    It integrates multiple types of pixel defects and updates FITS headers
    with quality assessment information.
    
    Parameters
    ----------
    img : str
        Path to the input KMTNet scaled image FITS file. The function expects
        the corresponding cosmic ray mask file to exist with the same name but
        with '.mask.fits' extension instead of '.scaled.fits'.
    path_cfg : str
        Path to the configuration directory containing:
        - badpixelmap/: Directory with observatory-specific bad pixel maps
        - BPM files: Named as '{OBSERVAT}_BPM.{chip*2}.fits'
    
    Returns
    -------
    None
        Results are written to the cosmic ray mask file and input image headers.
    
    Notes
    -----
    The function performs the following operations:
    
    1. **File Validation**:
       - Checks for existence of cosmic ray mask file (.mask.fits)
       - Verifies bad pixel map file exists for the observatory and chip
       - Reads observatory and chip information from image header
    
    2. **Bad Pixel Map Integration**:
       - Loads cosmic ray mask from previous processing
       - Loads observatory-specific bad pixel map
       - Applies edge masking to exclude dithered regions
       - Combines multiple types of pixel defects
    
    3. **Bad Amplifier Masking**:
       - Reads BADAMP header keyword to identify failed amplifiers
       - Masks entire amplifier regions that failed photometric fitting
       - Updates mask with bad amplifier information
    
    4. **Mask Combination**:
       - Combines cosmic ray mask, bad pixel map, and bad amplifier mask
       - Creates comprehensive bad pixel mask with different flag values
       - Updates mask file with combined information
    
    5. **Header Updates**:
       - Adds mask type descriptions to mask file header
       - Updates original image with quality assessment
       - Records bad pixel map information in headers
    
    Mask Flag Values
    ---------------
    - **1**: Cosmic ray affected pixels
    - **2**: Cross-talk affected pixels  
    - **4**: Pixel bleeding affected pixels
    - **8**: CCD bad pixels (from observatory-specific BPM)
    - **16**: Bad amplifier regions (entire amplifier masked)
    
    Quality Assessment
    ------------------
    - **PHOTQA = True**: If fewer than 2 amplifiers are bad
    - **PHOTQA = False**: If 2 or more amplifiers are bad
    
    File Requirements
    -----------------
    - Input image must have BADAMP header keyword from zpscale function
    - Cosmic ray mask file (.mask.fits) must exist
    - Observatory-specific bad pixel map must exist in path_cfg/badpixelmap/
    
    Examples
    --------
    >>> BPM_update('/path/to/S240422ed_0749.kk.scaled.fits', '/path/to/config/')
    CR mask and BPM exist for S240422ed_0749.kk.scaled.fits.
    
    The function will:
    - Combine cosmic ray mask with bad pixel map
    - Mask bad amplifier regions
    - Update both mask file and original image headers
    
    See Also
    --------
    zpscale : Zero-point scaling function that creates BADAMP information
    qatest : Quality assurance function that creates cosmic ray masks
    KMTNet_ToO_pipeline : Main pipeline that calls this function
    
    Notes
    -----
    This function is typically called after zpscale to integrate bad pixel
    information from multiple sources into a single comprehensive mask.
    """
    
    import os
    import numpy as np
    from astropy.io import fits

    # Load the FITS file and read the header and data (if required)
    with fits.open(img) as f:
        hdr = f[0].header
        obs = hdr['OBSERVAT']
        chip = hdr['CCDNAME'].lower()
        badamp = hdr['BADAMP']
        
    msk = img.replace(".scaled.", ".mask.")
    bpm = f'{path_cfg}badpixelmap/{obs}_BPM.{chip*2}.fits'

    if not (os.path.exists(msk) and os.path.exists(bpm)):
        print(f'CR mask or BPM does not exist for {os.path.basename(img)}.')
        return
    
    print(f'CR mask and BPM exist for {os.path.basename(img)}.')
    
    mdata = fits.getdata(msk)
    bpmdata = fits.getdata(bpm)
    bpmdata[940:-940, 460:-460] = 0 # bad pixels will be dithered out
    bpmdata[bpmdata != 0] = 8
    
    # Vectorize the masking process for bad amps
    naxis1 = hdr['NAXIS1']
    bound = np.arange(0, naxis1 + 1, naxis1 // 8)
    bads = 0
    for i, amp in enumerate(badamp):
        if amp == '1':
            mdata[:, bound[i]:bound[i+1]] = 16
            bads += 1
    
    mbpmdata = mdata + bpmdata
    fits.PrimaryHDU(data=mbpmdata.astype(np.int16), header=fits.getheader(msk)).writeto(msk, overwrite=True)

    with fits.open(msk, 'update') as m:
        for hdu in m:
            hdu.header['CRMASK'] = (1, 'Cosmic-ray marked as 1')
            hdu.header['XTMASK'] = (2, 'Crosstalk region marked as 2')
            hdu.header['BLDMASK'] = (4, 'Bleeding pattern marked as 4')
            hdu.header['BPMASK'] = (8, 'CCD badpixel marked as 8')
            hdu.header['BAMPMASK'] = (16, 'Bad amplifier marked as 16')
            hdu.header['BADMAP'] = (os.path.basename(bpm), 'Badpixels map')

    # Update header information on the original image
    with fits.open(img, 'update') as f:
        for hdu in f:
            if bads < 2:
                hdu.header['PHOTQA'] = (True, 'Badpixels and bad amps masked out')
            else:
                hdu.header['PHOTQA'] = (False, 'Badpixels and bad amps masked out')
            hdu.header['BADMAP'] = (os.path.basename(bpm), 'Badpixels map')
    
    return
#%% ToOImageStackter.py
def stacking(filename_convention, path_input, path_output, path_cfg, path_ref, combinetype='MEDIAN', start=None, gridcat='kmtnet_grid.cat'):
    """
    Image stacking function for KMTNet chip images using SWarp.
    
    This function collects complete sets of four KMTNet chip images (kk, mm, tt, nn) and their
    corresponding mask images, then stacks them using SWarp to create final stacked images.
    It performs quality control, coordinate alignment, and generates both science and mask stacks.
    
    Parameters
    ----------
    filename_convention : str
        Regular expression pattern to match filenames. Should capture groups for:
        field, radec, band, date, site, serial, chip, and type (scaled|mask).
        Example: r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<band>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<serial>\d{6})\.(?P<chip>\w+)\.(?P<type>scaled|mask)\.fits"
    path_input : str
        Path to the input directory containing scaled and mask FITS files.
    path_output : str
        Path where the output stacked images will be saved.
    path_cfg : str
        Path to the configuration directory containing:
        - kmtnet.swarp: SWarp configuration file
        - mask.swarp: SWarp configuration for mask stacking
        - kmtnet_grid.cat: KMTNet field grid catalog
    path_ref : str
        Path to the reference/template images directory for coordinate alignment.
    combinetype : str, optional
        Type of pixel combination method to use in SWarp:
        - 'MEDIAN': Median combination (default)
        - 'WEIGHTED': Weighted average combination
        Default is 'MEDIAN'.
    start : float, optional
        Start time of the operation for performance measurement. Default is None.
    gridcat : str, optional
        Filename of the KMTNet field grid catalog. Default is 'kmtnet_grid.cat'.
    
    Returns
    -------
    int
        Total number of stacked image sets processed.
    
    Notes
    -----
    The function performs the following operations:
    
    1. **File Collection and Organization**:
       - Scans input directory for scaled and mask FITS files
       - Groups files by field, band, and observatory
       - Identifies complete sets of four chip images (kk, mm, tt, nn)
    
    2. **Quality Control**:
       - Checks for QARESULT=True (astrometric QA passed)
       - Verifies PHOTQA=True (photometric QA passed)
       - Handles special cases for CTIO nn chip with BADAMP='00000010'
       - Filters out images that don't meet quality standards
    
    3. **Complete Set Identification**:
       - Ensures all four chips (kk, mm, tt, nn) are available
       - Groups images by dither position (same serial number)
       - Only processes complete sets with all four chips
    
    4. **Coordinate Alignment**:
       - Uses reference/template images for coordinate alignment
       - Falls back to field grid catalog if reference images unavailable
       - Determines center coordinates for SWarp alignment
    
    5. **Image Stacking**:
       - Creates input file lists for SWarp
       - Runs SWarp with specified combination type
       - Generates both science and weight images
       - Applies coordinate transformation and resampling
    
    6. **Mask Stacking**:
       - Updates mask files with WCS information
       - Stacks mask images using SWarp
       - Creates corresponding mask stack files
    
    7. **Header Updates**:
       - Updates stacked image headers with observation information
       - Records dither information, exposure times, and quality metrics
       - Adds coordinate and field information
    
    Quality Requirements
    --------------------
    - **QARESULT=True**: Astrometric quality assurance must pass
    - **PHOTQA=True**: Photometric quality assurance must pass
    - **Complete Sets**: All four chips (kk, mm, tt, nn) must be available
    - **Special Case**: CTIO nn chip with BADAMP='00000010' is accepted
    
    Output Files
    ------------
    - **Science Stack**: `{obj}_{field}.{radec}.{band}.{date}.{obs}.{exptime}sec.stack.fits`
    - **Mask Stack**: `{obj}_{field}.{radec}.{band}.{date}.{obs}.{exptime}sec.mstack.fits`
    
    Header Keywords Added
    ---------------------
    - OBJECT: Object name and field identifier
    - FIELD1: Field number
    - FIELD2: Field coordinates
    - FILTER: Filter band
    - NUMDITH: Number of dither positions
    - NUMIMAGE: Number of images stacked
    - EXPTIME: Total exposure time
    - DATE-OBS: Average observation date
    - MEANMJD: Average observation date in MJD
    - FWHM: Average seeing
    - MAGZERO: Photometric zero-point
    - CENTRA/CENTDEC: Center coordinates
    - SATURATE/UNDERSAT: Saturation levels
    
    Examples
    --------
    >>> filename_convention = r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<band>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<serial>\d{6})\.(?P<chip>\w+)\.(?P<type>scaled|mask)\.fits"
    >>> stacking(filename_convention, '/path/to/scaled/', '/path/to/stack/', 
    ...          '/path/to/config/', '/path/to/template/', combinetype='MEDIAN')
    
    See Also
    --------
    zpscale : Zero-point scaling function that creates input images
    BPM_update : Bad pixel map function that creates mask images
    KMTNet_ToO_pipeline : Main pipeline that calls this function
    
    Requirements
    ------------
    - Input images must have completed astrometric and photometric QA
    - SWarp configuration files must be present in path_cfg directory
    - Reference/template images should be available for coordinate alignment
    """
    import numpy as np
    import os, re, time, glob
    import astropy.units as u
    from astropy.wcs import WCS
    from astropy.io import fits, ascii
    from astropy.coordinates import SkyCoord

    if not path_input.endswith('/'):
        path_input   = path_input + '/'
    if not path_output.endswith('/'):
        path_output   = path_output + '/'

    # os.chdir(path_input)
    
    regex = re.compile(filename_convention)
    
    all_files   = sorted(glob.glob(f'{path_input}*.fits'))
    sfits   = [file for file in all_files if regex.match(os.path.basename(file)) and regex.match(os.path.basename(file)).group("type") == "scaled"]
    mfits   = [file for file in all_files if regex.match(os.path.basename(file)) and regex.match(os.path.basename(file)).group("type") == "mask"]

    try: fields  = sorted(list(set([fits.getheader(f)['FIELD1'] for f in sfits]))) # \w+_0000
    except KeyError: fields  = sorted(list(set([os.path.basename(f).split('.')[0].split('_')[1] for f in sfits]))) # \w+_0000
    bands   = sorted(list(set([os.path.basename(b).split('.')[2] for b in sfits])))
    observs = sorted(list(set([os.path.basename(b).split('.')[4] for b in sfits])))
    total   = 0

    for observ in observs:

        for field in fields:
            
            try:
                ks4cat  = ascii.read(os.path.join(path_cfg, gridcat))
            except:
                ks4cat  = Table(fits.open(os.path.join(path_cfg, gridcat))[1].data)
            radec   = ks4cat[ks4cat['field_name1']==int(field)]['field_name2'][0]
            
            for band in bands:

                # center coordinate (should be fixed for dual-mode photometry)
                try:
                    centerimgs      = [f for f in sorted(glob.glob(f'{path_ref}{field}.{radec}/*{band}*scaled.stack.fits')) if 'mask' not in f] # reference images
                    centerimg       = max(centerimgs, key=lambda f: int(re.compile(r'\d+(?=sec)').search(f).group())) # longest exposure time
                    centra, centdec = fits.getheader(centerimg)['CENTRA'], fits.getheader(centerimg)['CENTDEC']
                except:
                    print("The reference/template image not exist.")
                    fieldinfo   = ks4cat[ks4cat['field_name1']==int(field)]
                    centcoord   = SkyCoord(ra=round(fieldinfo['ra[deg]'][0],5)*u.degree, dec=round(fieldinfo['dec[deg]'][0],5)*u.degree)
                    centra, centdec = centcoord.to_string('hmsdms').replace('h',':').replace('m',':').replace('s','').replace('d',':').split(' ')
        
                # clean quality chip images
                allist  = []
                for im in [f for f in sfits if field in os.path.basename(f).split('.')[0] and os.path.basename(f).split('.')[2]==band and os.path.basename(f).split('.')[4]==observ]:
                # for im in sorted(glob.glob(f'{field}*.{band}.*scaled.fits')):
                    hdr     = fits.getheader(im)
                    if 'QARESULT' not in hdr or 'BADAMP' not in hdr:
                        print(f'{im}: Either astromqa or zpscalepro is incomplete.')
                    elif hdr['QARESULT']==True:
                        if hdr['PHOTQA']==True:
                            allist.append(im)
                        elif hdr['BADAMP']=='00000010':
                            if hdr['OBSERVAT']=='CTIO' and hdr['CCDNAME']=='n':
                                allist.append(im)
                        else:
                            print(f'{im} BADAMP={hdr["BADAMP"]}')
                    else:
                        print(f'{im} QARESULT={hdr["QARESULT"]}')

                # k,m,t,n full-frame chip sets
                klist = [i for i in allist if 'kk' in os.path.basename(i)]
                mlist = [i for i in allist if 'mm' in os.path.basename(i)]
                tlist = [i for i in allist if 'tt' in os.path.basename(i)]
                nlist = [i for i in allist if 'nn' in os.path.basename(i)]
                
                fullframes  = list({idn.split('.kk.scaled.')[0] for idn in klist} & {idn.split('.mm.scaled.')[0] for idn in mlist} & {idn.split('.nn.scaled.')[0] for idn in nlist} & {idn.split('.tt.scaled.')[0] for idn in tlist})
                dither  = len(fullframes)
                # checker
                if dither == 0: continue
                total  += dither
                
                # time check
                mid     = time.time()
                print('='*70)
                if start == None: print(f'Stacking Proess: {field}-{band}-band: {mid:.2f}sec passed')
                else: print(f'Stacking Proess: {field}-{band}-band: {mid-start:.2f}sec passed')
                print('='*70)

                # list input
                imlist  = []
                for ff in fullframes:
                    imgs    = glob.glob(f'{ff}*scaled.fits')
                    imlist.extend(imgs)
                imlist  = sorted(imlist)
                f = open(f'{path_input}diths.list', 'w')
                for j in imlist:
                    f.write(j+'\n')
                f.close()
                
                # each frame observation info
                try: obj = hdr['OBJECT']
                except: obj = 'TOO'
                obs     = [os.path.basename(img).split('.')[4] for img in imlist if 'kk' in img]
                dateobs = [fits.getheader(img)['DATE-OBS'] for img in imlist if 'kk' in img]
                meanobs = np.mean([date2MJD(date) for date in dateobs])
                meandate= MJD2date(meanobs)
                saturat = [fits.getheader(img)['SATURATE'] for img in imlist]
                fwhm    = np.mean([float(fits.getheader(img)[f'FWHM']) for img in imlist])
                exptime = float(fits.getheader(imlist[0])['EXPTIME']) * dither
                # stacking with swarp
                stack   = os.path.join(path_output, f'{obj}_{field}.{radec}.{band}.{meandate.split("T")[0].replace("-","")}.{observ}.{round(exptime)}sec.stack.fits')
                weight  = stack.replace(".stack.fits", ".weight.fits")
                if combinetype == 'WEIGHTED':
                    for mfit in mfits:
                        wdata = (fits.getdata(mfit) == 0).astype(int)
                        fits.writeto(mfit.replace('.mask.fits', '.scaled.weight.fits'), wdata, overwrite=True)
                    os.system(f'swarp @{path_input}diths.list -c {path_cfg}kmtnet.swarp -IMAGEOUT_NAME {stack} -WEIGHTOUT_NAME {weight} -CENTER {centra},{centdec} -IMAGE_SIZE 22000,22000 -WEIGHT_TYPE MAP_WEIGHT -COMBINE_TYPE {combinetype} -BLANK_BADPIXELS Y -INTERPOLATE Y') 
                else:
                    os.system(f'swarp @{path_input}diths.list -c {path_cfg}kmtnet.swarp -IMAGEOUT_NAME {stack} -WEIGHTOUT_NAME {weight} -CENTER {centra},{centdec} -IMAGE_SIZE 22000,22000 -COMBINE_TYPE {combinetype}') 
                os.system(f'rm {weight}')
                os.system(f'rm {path_input}diths.list')
                
                # header updates
                with fits.open(stack, 'update') as f:
                    for hdu in f:
                
                        hdu.header['OBJECT']    = f'{obj}_{field}'
                        hdu.header['FIELD1']    = field
                        hdu.header['FIELD2']    = radec
                        hdu.header['FILTER']    = band
                        hdu.header['NUMDITH']   = dither
                        hdu.header['NUMIMAGE']  = dither
                        hdu.header['EXPTIME']   = exptime
                        hdu.header['DATE-OBS']  = (meandate, 'Average DATE-OBS of images stacked')
                        hdu.header['MEANMJD']   = (meanobs, 'Average DATE-OBS of images in MJD')
                        hdu.header['FWHM']      = round(fwhm, 2)
                        for i in range(dither):
                            hdu.header[f'OBSERV{hex(i)[-1]}']    = obs[i]
                            hdu.header[f'IMAGE{hex(i)[-1]}']     = [os.path.basename(k) for k in imlist if 'kk' in k][i].replace('.kk','')
                            hdu.header[f'DATEOBS{hex(i)[-1]}']   = dateobs[i]
                        hdu.header['MAGZERO']   = 30.0
                        hdu.header['CENTRA']    = centra
                        hdu.header['CENTDEC']   = centdec
                        hdu.header['SATURATE']  = round(np.mean(saturat))
                        hdu.header['UNDERSAT']  = round(np.min(saturat))
            
                os.system(f'chmod 777 {stack}')
                
                # 3.5. WCS info update for masks
                masks  = [f for f in mfits if os.path.basename(f).split('.')[0].split('_')[1]==field and os.path.basename(f).split('.')[2]==band and os.path.basename(f).split('.')[4]==observ]
                if len(masks) != 0:
                    f = open(f'{path_input}mdiths.list', 'w')
                    for mask in masks:
                        im  = mask.replace('mask', 'scaled')
                        hdu = fits.PrimaryHDU(fits.getdata(mask), header=fits.getheader(im)+WCS(im).to_header())
                        hdu.writeto(mask, overwrite=True)
                        f.write(mask+'\n')
                    f.close()

                    # 3.7. SWarp for mask
                    mstack = stack.replace('.stack.fits', '.mstack.fits')
                    mweight = weight.replace('.weight.fits', '.mweight.fits')
                    os.system(f'swarp @{path_input}mdiths.list -c {path_cfg}mask.swarp -IMAGEOUT_NAME {mstack} -WEIGHTOUT_NAME {mweight} -CENTER {centra},{centdec} -IMAGE_SIZE 22000,22000')
                    os.system(f'rm {mweight}')
                    os.system(f'rm {path_input}mdiths.list')
                else:
                    mstack = f'{path_cfg}ks4.empty.mask.fits'
                
                # 3.8. header update for mask
                with fits.open(mstack, 'update') as f:
                    for hdu in f:
                
                        hdu.header['OBJECT']    = f'{obj}_{field}'
                        hdu.header['FIELD1']    = field
                        hdu.header['FIELD2']    = radec
                        hdu.header['FILTER']    = band
                        hdu.header['NUMDITH']   = dither
                        hdu.header['NUMIMAGE']  = dither
                        hdu.header['CENTRA']    = centra
                        hdu.header['CENTDEC']   = centdec

                os.system(f'chmod 777 {mstack}')
    
    return total
#%% ToOsourcecatalog.py
def catalogmaker(cat, path_output, path_cat, flagcut=0, pixscale=0.4, clsstar=0.8, refmaglower=14, refmagupper=17, apertures=['APER', 'AUTO'], figure=False, path_plot='./', start=None):
    """
    Source catalog generation and zero-point calibration for stacked images.
    
    This function processes SExtractor output catalogs from stacked images to generate
    calibrated source catalogs with zero-point corrections. It does not perform source
    detection (which is done by the qatest function), but rather calibrates existing
    photometric measurements against reference catalogs and calculates image depth.
    
    Parameters
    ----------
    cat : str
        Path to the SExtractor output catalog file (.cat) from stacked image.
        Expected naming convention: {obj}_{field}.{radec}.{band}.{date}.{site}.{exptime}sec.stack.fits.cat
    path_output : str
        Path to the output directory where calibrated catalogs will be saved.
    path_cat : str
        Path to the reference catalog directory containing:
        - gaiaxp/: GAIA XP reference catalogs
        - apass/: APASS reference catalogs
    flagcut : int, optional
        Maximum FLAGS value to accept for source selection. Default is 0.
    pixscale : float, optional
        Pixel scale in arcseconds per pixel. Default is 0.4.
    clsstar : float, optional
        Minimum CLASS_STAR value for stellar source selection. Default is 0.8.
    refmaglower : float, optional
        Lower magnitude limit for reference catalog matching. Default is 14.
    refmagupper : float, optional
        Upper magnitude limit for reference catalog matching. Default is 17.
    apertures : list, optional
        List of aperture types to calibrate ('APER', 'AUTO', etc.). Default is ['APER', 'AUTO'].
    figure : bool, optional
        Whether to generate diagnostic plots. Default is False.
    path_plot : str, optional
        Path to the output directory for diagnostic plots. Default is './'.
    start : float, optional
        Start time for elapsed time calculation. Default is None.
    
    Returns
    -------
    int
        Returns 0 upon successful completion.
    
    Notes
    -----
    The function performs the following operations:
    
    1. **Catalog Input and Validation**:
       - Reads SExtractor output catalog from stacked image
       - Validates that corresponding FITS image exists
       - Extracts field, band, and observation information from filename
    
    2. **Reference Catalog Loading**:
       - Loads appropriate reference catalog (GAIA XP or APASS)
       - Matches field information to reference catalog
       - Determines photometric reference system
    
    3. **Source Matching**:
       - Matches detected sources with reference catalog
       - Uses adaptive matching radius based on astrometric quality
       - Performs quality cuts on matched sources
    
    4. **Zero-point Calibration**:
       - Calculates zero-point for each aperture type
       - Applies zero-point corrections to source magnitudes
       - Estimates zero-point uncertainty and quality metrics
    
    5. **Image Depth Calculation**:
       - Calculates 5-sigma detection limiting magnitude
       - Uses background RMS map if available
       - Falls back to error curve fitting if RMS map unavailable
    
    6. **Quality Assessment**:
       - Calculates photometric precision (RMSE)
       - Assesses magnitude difference statistics
       - Generates quality metrics for catalog validation
    
    7. **Output Generation**:
       - Saves calibrated source catalog with zero-point corrections
       - Updates FITS image headers with photometric information
       - Generates diagnostic plots if requested
    
    Quality Cuts Applied
    --------------------
    - **FLAGS ≤ flagcut**: Excludes sources with detection flags
    - **CLASS_STAR ≥ clsstar**: Selects stellar sources only
    - **Reference magnitude range**: refmaglower ≤ mag ≤ refmagupper
    - **Magnitude error limits**: Both input and reference errors ≤ 0.05 mag
    
    Output Files
    ------------
    - **Calibrated Catalog**: `{basename}.zp.cat` (zero-point corrected catalog)
    - **Diagnostic Plot**: `{field}_{band}_{aperture}_phot.png` (if figure=True)
    
    Header Keywords Added
    ---------------------
    - PHOTREF: Reference catalog used for calibration
    - FWHM: Median seeing of point sources [arcsec]
    - MAGZERO: Photometric zero-point for MAG_AUTO [ABmag]
    - ZEROERR: Standard deviation of zero-point [ABmag]
    - ZPSTAR: Number of stars used for zero-point calculation
    - RMSPHOT: RMSE of reference vs. KMTNet magnitudes
    - DEPTH5: 5-sigma detection limiting magnitude
    - MATCHRAD: Matching radius with reference catalog [arcsec]
    - CLSSTAR: CLASS_STAR cut used for analysis
    
    Examples
    --------
    >>> catalogmaker('/path/to/G331903-12-1_9012.020-31.R.20231108.CTIO.600sec.stack.fits.cat',
    ...              '/path/to/output/', '/path/to/catalogs/', flagcut=0, clsstar=0.8,
    ...              refmaglower=14, refmagupper=17, apertures=['APER', 'AUTO'], figure=True)
    
    See Also
    --------
    qatest : Quality assurance function that performs source detection
    stacking : Image stacking function that creates input stacked images
    KMTNet_ToO_pipeline : Main pipeline that calls this function
    
    Notes
    -----
    This function is typically called after qatest has performed source detection
    on stacked images. It focuses on photometric calibration rather than source
    detection, making it complementary to the qatest function.
    """

    import time
    import numpy as np
    import os, sys, re
    from astropy.io import fits
    from astropy.io import ascii
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from astropy.stats import sigma_clip
    import matplotlib.gridspec as gridspec

    # basic info
    field   = os.path.basename(cat).split('.')[0].split('_')[-1]      # 0000
    radec   = os.path.basename(cat).split('.')[1]                     # 000-00
    band    = os.path.basename(cat).split('.')[2]                     # B|V|R|I
    date    = os.path.basename(cat).split('.')[3]                     # 20000000
    site    = os.path.basename(cat).split('.')[4]                     # SAAO|SSO|CTIO
    exptime = int(os.path.basename(cat).split('.')[5].split('sec')[0])
    img     = cat.replace('.cat', '')
    
    # original image check
    if not os.path.isfile(img):
        print(f'Warning: No FITS image in the directory. Cannot update the header.')
        return 0
    else:
        hdr     = fits.getheader(img)
    
    # time check
    mid     = time.time()
    print('='*80)
    if start == None:
        print(f"{os.path.basename(img)} ({mid:.2f}sec elapsed)")
    else:
        print(f"{os.path.basename(img)} ({mid-start:.2f}sec elapsed)")
    print('='*80)
        
    # 2. reference table query
    try:
        path_ref    = os.path.join(path_cat, 'gaiaxp')
        reftbl  = GAIAXP_query(field, path_ref)
        photref = 'GAIA XP'
    except FileNotFoundError: # either gaia or apass catalog downloaded from zpscalepro
        path_ref    = os.path.join(path_cat, 'apass')
        reftbl  = ascii.read(f'{path_ref}/apass_{field}.cat')
        photref = 'APASS DR9'
    
    # catalog file read
    intbl   = ascii.read(cat)
    seeing  = round(np.median(intbl[intbl['CLASS_STAR']>clsstar]['FWHM_IMAGE'] * pixscale), 3)
    
    # matching with the reference
    thres   = 0
    while 1:
        if 'ALNRMS' not in hdr: rad     = 0.5
        else: rad = 0.2 + hdr['ALNRMS']*thres
        param_matching  = dict(intbl    = intbl,
                                reftbl   = reftbl,
                                inra     = intbl['ALPHA_J2000'], 
                                indec    = intbl['DELTA_J2000'],
                                refra    = reftbl['RAJ2000'], 
                                refdec   = reftbl['DEJ2000'],
                                sep      = rad)
        mtbl    = matching(**param_matching)
        thres  += 1
        if len(mtbl) > 5000 or thres >= 5 or 'ALNRMS' not in hdr:
            break

    # zero-point calculation
    for aperture in apertures:
        
        mtbl            = mtbl[mtbl[f'MAG_{aperture}']!=99]
        inmagkey        = 'MAG_{}'.format(aperture)
        inmagerkey      = 'MAGERR_{}'.format(aperture)
        refmagkey       = '{}mag'.format(band)
        refmagerkey     = 'e_{}mag'.format(band)

        refmagerupper   = 0.05
        inmagerupper    = 0.05

        param_st4zp     = dict(intbl=mtbl,
                                inmagerkey=inmagkey,
                                refmagkey=refmagkey, refmagerkey=refmagerkey,
                                refmaglower=refmaglower, refmagupper=refmagupper,
                                refmagerupper=refmagerupper, inmagerupper=inmagerupper,
                                flagcut=flagcut)

        param_zpcal     = dict(intbl=star4zp(**param_st4zp),
                                inmagkey=inmagkey,
                                refmagkey=refmagkey,
                                sigma=2.0)

        zp, zper, otbl, xtbl = zpcal(**param_zpcal)
        intbl[f'MAG_{aperture}']    = np.array([min(intbl[f'MAG_{aperture}'][i]+round(zp, 3),99.0) for i in range(len(intbl))])
        intbl[f'MAGERR_{aperture}'] = rss([intbl[f'MAGERR_{aperture}'], round(zper, 3)])

        if aperture == 'AUTO':
            mtbl = mtbl[mtbl['FLAGS']<=flagcut] # flag cut
            mtbl = mtbl[mtbl['CLASS_STAR']>clsstar] # stellarity cut
            mtbl = mtbl[mtbl['{}mag'.format(band)] < refmagupper] #mag cut
            mtbl = mtbl[mtbl['{}mag'.format(band)] > refmaglower]
            
            magdif  = mtbl[f'MAG_{aperture}'] + zp - mtbl['{}mag'.format(band)]
            magerr  = rss([mtbl[f'MAGERR_{aperture}'], zper])
            
            meddif  = round(np.median(magdif), 3)
            rmse    = round(np.sqrt(np.mean(sigma_clip(magdif,3)**2)), 3)

    # save zp applied magnitude catalog
    zpcatname   = f'{path_output}{os.path.basename(cat).replace(".cat",".zp.cat")}'
    intbl.write(zpcatname, format='ascii', overwrite=True)
    os.system(f'chmod 777 {zpcatname}')
    
    # depth check
    if os.path.exists(img.replace(".fits",".bkgrms")):
        bkgrms  = fits.getdata(img.replace(".fits",".bkgrms"))
        skysig  = np.median(bkgrms[bkgrms!=0])
        # skysig  = np.median(bkgrms)
        depth   = limitmag(5, zp, 5/0.4, skysig) # aperture = 5"
        os.remove(img.replace(".fits",".bkgrms"))
    else: # in case bkgrms map is unavailable, we fit the error curve and estimate the point source magnitude whose error is approximately 0.2mag.
        mincut  = 0.03
        while 1:
            # defining inputs
            point_sources = intbl[(intbl["FLAGS"] < 4) & (intbl["CLASS_STAR"] >= clsstar)]
            point_sources = point_sources[(point_sources["MAGERR_APER"]>=mincut) & (point_sources["MAGERR_APER"]<=0.5)]
            mag_error_threshold = 0.2 # supposed that snr = 5 magnitude would have error ~ 0.2
            
            # fitting function
            def exponential_function(x, a, b):
                return a * np.exp(b * x)
            
            # fitting
            try:
                weights = 1 / point_sources['MAGERR_APER'] # because there are few points err>0.03, we'll give them more weights
                params, _ = curve_fit(exponential_function, point_sources['MAG_APER'], point_sources['MAGERR_APER'], sigma=weights)
                depth   = round(np.log(mag_error_threshold / params[0]) / params[1], 2)
                
                # checking good of fitness
                y_pred = exponential_function(point_sources['MAG_APER'], params[0], params[1])
                residuals = point_sources['MAGERR_APER'] - y_pred
                ss_total = np.sum((point_sources['MAGERR_APER'] - np.mean(point_sources['MAGERR_APER']))**2)
                ss_residual = np.sum(residuals**2)
                r_squared = 1 - (ss_residual / ss_total)
                
                # recursive process
                if not np.isnan(depth) and depth < 30 and r_squared > 0.8: # in case exponent underestimated, depth would be exaggerated
                    break
                elif mincut > 0.2: # fail to good fit.. should check with caution
                    depth = -99
                    break
                else:
                    mincut += 0.005
            except:
                depth = -99
                break

    if figure:
        plt.figure(figsize=(8, 12))
        gs = gridspec.GridSpec(15, 15)
        plt.rcParams.update({'font.size': 14})

        plt.suptitle(f'ToO & Reference (GAIA) Photometry Comparison\nField : {field}, MAG_{aperture}, {band} band\nMagnitude Difference RMSE : {rmse:.3f} ABmag\n5$\sigma$ Image Depth : {depth} ABmag')
        ax_1d   = plt.subplot(gs[9:15, :10])
        ax_hist = plt.subplot(gs[9:15, 10:])
        ax_2d   = plt.subplot(gs[0:8, :14])

        # 8.1. magnitude differences
        ax_1d.scatter(mtbl['MAG_{}'.format(aperture)]+zp, magdif, c='crimson', marker='o', alpha=0.1, label='Field Stars')
        ax_1d.errorbar(mtbl['MAG_{}'.format(aperture)]+zp, magdif, yerr=magerr, ms=6, ls='', c='crimson', marker='o', capsize=4, capthick=1, alpha=0.1)
        ax_1d.set(xlabel=r'$m_{ToO}$ [ABmag]', ylabel=r'$m_{ToO} - m_{Reference}$ [ABmag]')
        ax_1d.set_xlim(16.5,13.75)
        ax_1d.set_ylim(-1,1)
        ax_1d.grid(which='major',linestyle='-', alpha=0.5)
        
        # 8.2. histograms
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
        
        # 8.3 plot: precision map
        plot = ax_2d.scatter(mtbl['X_IMAGE'], mtbl['Y_IMAGE'], marker='o', c=mtbl['MAG_{}'.format(aperture)] + zp - mtbl['{}mag'.format(band)], cmap='seismic', alpha=0.7, label='FLAG=0 ({})\nCLASS_STAR>0.8'.format(len(mtbl)))
        add_colorbar(plot, clabel=r'$m_{ToO} - m_{Reference}$ [mag]', clim=[-0.5, 0.5])
        ax_2d.legend(loc='upper right')
        ax_2d.set(xlabel='X axis [pixel]', ylabel='Y axis [pixel]')
        ax_2d.set_xlim(-1000, 24000)
        ax_2d.set_ylim(-1000, 24000)
        plt.savefig(f'{path_plot}{field}_{band}_{aperture}_phot.png')
        plt.close()

    with fits.open(img, 'update') as f:
        for hdu in f:
            hdu.header['PHOTREF']   = (photref, f'Reference catalog used for source catalog generation')
            hdu.header['FWHM']      = (round(seeing, 2), f'Median seeing of point sources [arcsec]]')
            hdu.header['MAGZERO']   = (round(zp, 3), 'Photometric zero-point for MAG_AUTO [ABmag]')
            hdu.header['ZEROERR']   = (round(zper, 3), 'Standard deviation of MAG_ZERO [ABmag]')
            hdu.header['ZPSTAR']    = (len(otbl), 'The number of stars for MAG_ZERO calculation')
            hdu.header['RMSPHOT']   = (rmse, 'RMSE of PHOTREF mag - KMTN mag (3sigma clipped)')
            hdu.header['DEPTH5']    = (depth, '5sigma detection limiting magnitude')
            hdu.header['MATCHRAD']  = (rad, 'Matching radius with PHOTREF [arcsec]')
            hdu.header['CLSSTAR']   = (clsstar, 'CLASS_STAR cut used for analysis')

    return 0
#%% ToOTransientSearch.py
def subtraction(sciimg, path_ref, path_cat, path_refcat, path_output, path_config, div_col=4, div_row=4, pixscale=0.4, ncore=1, detect=1.5, cutsize=1.0, twoway_subt=False, psf_analysis=False, reuse_dia=False, known_obj=None):
    """
    Image subtraction and transient candidate detection for KMTNet stacked images.
    
    This function performs difference image analysis (DIA) by subtracting reference images
    from science images to detect transient candidates. It uses HOTPANTs for image
    subtraction, applies comprehensive source flagging to filter artifacts, and generates
    cutout images for potential transient candidates.
    
    Parameters
    ----------
    sciimg : str
        Path to the science stacked image FITS file. Expected naming convention:
        {obj}_{field}.{radec}.{band}.{date}.{site}.{exptime}sec.stack.fits
    path_ref : str
        Path to the reference image directory containing:
        - ks4*{band}*.stack.fits: KMTNet reference images
        - ps1*{band}*.stack.fits: Pan-STARRS reference images
    path_cat : str
        Path to the science image catalog directory containing:
        - {basename}.zp.cat: Zero-point calibrated source catalogs
    path_refcat : str
        Path to the reference catalog directory containing:
        - ks4_{field}.{radec}*{band}*.zp.fits: Reference image catalogs
    path_output : str
        Path to the output directory where subtraction results will be saved.
    path_config : str
        Path to the configuration directory containing:
        - kmtnet.sex: SExtractor configuration file
        - kmtnet_imask.param: SExtractor parameters file
        - kmtnet.nnw: SExtractor neural network weights file
        - kmtnet.conv: SExtractor convolution file
        - kmtnet.psfex: PSFEx configuration file (if psf_analysis=True)
    div_col : int, optional
        Number of columns for image subdivision in HOTPANTs. Default is 4.
    div_row : int, optional
        Number of rows for image subdivision in HOTPANTs. Default is 4.
    pixscale : float, optional
        Pixel scale in arcseconds per pixel. Default is 0.4.
    ncore : int, optional
        Number of CPU cores for parallel processing. Default is 1.
    detect : float, optional
        Detection threshold for SExtractor. Default is 1.5.
    cutsize : float, optional
        Size of cutout images in arcseconds. Default is 1.0.
    twoway_subt : bool, optional
        Whether to perform subtraction in both directions (science-ref and ref-science).
        Default is False.
    psf_analysis : bool, optional
        Whether to perform PSF analysis using PSFEx. Default is False.
    
    Returns
    -------
    int
        Returns 0 upon successful completion.
    
    Notes
    -----
    The function performs the following operations:
    
    1. **Image Set Preparation**:
       - Loads science stacked image and validates existence
       - Searches for reference images (KMTNet, Pan-STARRS, or generates Pan-STARRS)
       - Creates mask image combining science and reference masks
       - Validates complete image set (science, reference, mask)
    
    2. **Source Catalog Preparation**:
       - Loads science image photometric catalog (.zp.cat)
       - Attempts to load reference image catalog for matching
       - Creates stamp catalog for HOTPANTs subtraction
       - Applies quality cuts (FLAGS=0, CLASS_STAR>0.8, magnitude range)
    
    3. **Image Subtraction**:
       - Determines convolution direction based on seeing comparison
       - Runs HOTPANTs for image subtraction with subdivision
       - Supports both single-direction and two-way subtraction
       - Generates convolved and subtracted images
    
    4. **Source Detection**:
       - Runs SExtractor on subtracted image to detect sources
       - Optionally performs PSF analysis with PSFEx
       - Creates inverted image for artifact detection
       - Detects sources in inverted image for comparison
    
    5. **Comprehensive Source Flagging**:
       - **Flag 0**: Asteroid/moving object matching using Skybot
       - **Flag 1**: Inverted image detections (artifacts around sources)
       - **Flag 2**: SExtractor flags and bad pixel masking
       - **Flag 3**: High ellipticity compared to other sources
       - **Flag 4**: Unusual FWHM compared to other sources
       - **Flag 5**: Weird background values (2-sigma outliers)
       - **Flag 6**: Low signal-to-noise ratio (SNR < 5)
       - **Flag 7**: Crosstalk contamination from bright sources
       - **Flag 8**: HOTPANTs chi-squared values (poor subtraction regions)
       - **Flag 9**: PSFEx analysis (if enabled)
    
    6. **Transient Candidate Selection**:
       - Combines all flags to identify clean transient candidates
       - Excludes flagged sources from final candidate list
       - Generates transient candidate catalog
       - Creates summary log with flagging statistics
    
    7. **Snapshot Generation**:
       - Generates cutout images for each transient candidate
       - Supports parallel processing for multiple candidates
       - Creates science, reference, and subtraction cutouts
       - Saves snapshots in organized directory structure
    
    Quality Cuts Applied
    --------------------
    - **FLAGS = 0**: Excludes sources with detection flags
    - **CLASS_STAR > 0.8**: Selects stellar sources only
    - **Magnitude range**: 14 < MAG_AUTO < 20
    - **SNR threshold**: SNR_WIN > 5 (for flagging)
    - **Ellipticity**: ratio_elong < 4 (for flagging)
    - **FWHM**: 0.5 < ratio_seeing < 2.5 (for flagging)
    
    Output Files
    ------------
    - **Calibrated Science**: `{basename}_Calib.fits`
    - **Reference Image**: `{basename}_REF.fits`
    - **Mask Image**: `{basename}_MASK.fits`
    - **Convolved Image**: `{basename}_hcCONV.fits`
    - **Subtracted Image**: `{basename}_hdCalib.fits`
    - **Source Catalog**: `{basename}_hdCalib.cat`
    - **Transient Catalog**: `{basename}_hdCalib.transient.cat`
    - **Summary Log**: `{basename}_hdCalib.summary.txt`
    - **Snapshots**: `snap/` directory with cutout images
    
    Flagging System
    ---------------
    The function uses a comprehensive 10-flag system to identify and filter artifacts:
    
    - **Flag 0**: Solar system objects (asteroids, comets)
    - **Flag 1**: Subtraction artifacts (detected in inverted image)
    - **Flag 2**: Bad pixels and detection flags
    - **Flag 3**: Non-stellar morphology (high ellipticity)
    - **Flag 4**: Unusual seeing (too sharp or too broad)
    - **Flag 5**: Background anomalies
    - **Flag 6**: Low signal-to-noise ratio
    - **Flag 7**: Crosstalk contamination
    - **Flag 8**: Poor subtraction quality regions
    - **Flag 9**: PSF analysis failures
    
    Examples
    --------
    >>> subtraction('/path/to/TOO_0578_0578.316-78.I.20250831.SAAO.480sec.stack.fits',
    ...             '/path/to/reference/', '/path/to/catalogs/', '/path/to/refcatalogs/',
    ...             '/path/to/output/', '/path/to/config/', div_col=4, div_row=4,
    ...             ncore=4, detect=1.5, cutsize=1.0, twoway_subt=False, psf_analysis=False)
    
    See Also
    --------
    catalogmaker : Source catalog generation function
    stacking : Image stacking function that creates input stacked images
    KMTNet_ToO_pipeline : Main pipeline that calls this function
    
    Notes
    -----
    This function is the core of the transient detection pipeline, combining
    sophisticated image subtraction with comprehensive artifact filtering to
    identify genuine transient candidates while minimizing false positives.
    """

    import shutil
    import numpy as np
    import multiprocessing
    import re, os, glob, copy
    import astropy.units as u
    from astropy.wcs import WCS
    from itertools import repeat
    from functools import partial
    from astropy.time import Time
    from astropy.table import Table
    from astropy.io import fits, ascii
    from astroquery.imcce import Skybot
    from astropy.coordinates import SkyCoord

    # basic info
    field   = sciimg.split('.')[0].split('_')[-1]      # 0000
    radec   = sciimg.split('.')[1]                     # 000-00
    band    = sciimg.split('.')[2]                     # B|V|R|I
    date    = sciimg.split('.')[3]                     # 20000000
    site    = sciimg.split('.')[4]                     # SAAO|SSO|CTIO
    exptime = int(sciimg.split('.')[5].split('sec')[0])

    # images to process
    # science image
    if not os.path.isfile(sciimg):
        sciimg  = None
        print("Check if the science image is in the working directory.")
    else:
        print(f"Science image: \n{os.path.basename(sciimg)}")

    # reference image
    pattern_ks4 = os.path.join(path_ref, f'{field}.{radec}', f'ks4*{band}*.stack.fits')
    refimg = find_longest_exposure_image(pattern_ks4)

    if refimg is None and int(radec[-3:]) >= -30: 
        # PS1 reference image is available for above declination -30
        pattern_ps1 = os.path.join(path_ref, f'{field}.{radec}', f'ps1*{band}*.stack.fits')
        refimg = find_longest_exposure_image(pattern_ps1)
        if refimg is None:
            try:
                from KMTNet_REF_functions import generate_panstarrs_reference
                refimg = generate_panstarrs_reference(
                    field=f"{field}.{radec}",
                    cra=fits.getheader(sciimg)['CENTRA'],
                    cdec=fits.getheader(sciimg)['CENTDEC'],
                    path_output=os.path.join(path_ref, f"{field}.{radec}"),
                    path_cfg=path_config,
                    filte=band
                )
            except:
                print(f"Failed to generate Pan-STARRS reference image for {field}.{radec} {band}-band")
        
    if refimg:
        print(f"Reference image: \n{os.path.basename(refimg)}")
        # mask image
        ref_shape = fits.getdata(refimg).shape
        maskimg = sciimg.replace(".stack.", ".mask.")  # Define the output filename
        # if os.path.exists(maskimg):
        #     print('The mask image already exists.')
        # else:
        fits.PrimaryHDU(data=(safe_load_fits(sciimg.replace(".stack.", ".mstack."), shape=ref_shape)
                            +safe_load_fits(refimg.replace(".scaled.", ".crmap."), shape=ref_shape)
                            +safe_load_fits(refimg.replace(".scaled.", ".bleed."), shape=ref_shape)
                            +np.where(fits.getdata(refimg)==0,1,0)+np.where(fits.getdata(sciimg)==0,1,0)).astype(np.uint8), 
                        header=fits.getheader(sciimg)).writeto(maskimg, overwrite=True)
        print(f"Mask image: \n{os.path.basename(maskimg)}")
    else:
        print(f'No suitable reference image found. Checked patterns: \n{pattern_ks4}\n{pattern_ps1}')

    if (sciimg is None) or (refimg is None) or (not os.path.isfile(maskimg)):
        print('The image set of science, referernce and mask images is not ready.')
        return 0

    # science image photometric catalog --> HOTPANTs subtraction stamps
    # science image catalog
    scicat  = ascii.read(f'{path_cat}{os.path.basename(sciimg)}.zp.cat') # mandatory
    scicat  = scicat[scicat['FLAGS']==0]
    scicat  = scicat[scicat['CLASS_STAR'] > 0.8]
    scicat  = scicat[(scicat['MAG_AUTO'] > 14) & (scicat['MAG_AUTO'] < 20)]
    try: # HOTPANTs Stamp File Generation by Matching the Source Catalogs
        # reference image catalog
        refband = re.search(r'\.(B|V|R|I)\.', refimg).group(1)
        refcat  = Table(fits.open(glob.glob(f'{path_refcat}{field}.{radec}/ks4_{field}.{radec}*{refband}*.zp.fits')[0])[1].data)
        refcat  = refcat[refcat['FLAGS']==0]
        refcat  = refcat[refcat['CLASS_STAR'] > 0.9]
        refcat  = refcat[(refcat['MAG_AUTO'] > 14) & (refcat['MAG_AUTO'] < 20)]
        # matching
        param_matching  = dict(intbl     = scicat,
                                reftbl   = refcat,
                                inra     = np.array(scicat['ALPHA_J2000']), 
                                indec    = np.array(scicat['DELTA_J2000']),
                                refra    = np.array(refcat['ALPHA_J2000']), 
                                refdec   = np.array(refcat['DELTA_J2000']),
                                sep      = 2.0)
        mtbl    = matching(**param_matching)
        stamp   = copy.deepcopy(mtbl)
        if len(stamp)<200: stamp = None
    except:
        print(f"Check if the catalog is in the {path_refcat} directory.")
        stamp   = copy.deepcopy(scicat)
        try: 
            stamp   = stamp[stamp['SNR_WIN']>20] 
            if len(stamp)<200: stamp = None
        except KeyError: 
            stamp_sorted = stamp[np.argsort(stamp['MAGERR_AUTO'])]
            stamp = stamp_sorted[:5000]

    # sci HDU
    scihdu = fits.open(sciimg)[0]
    sciwcs = WCS(scihdu.header)
    scidat = scihdu.data
    scihdr = scihdu.header
    try:
        magautozero = scihdr['MAGZERO']
    except KeyError:
        magautozero = 30
        
    # Image Subtraction Process
    """ nrx = 4, nry = 4
    +----+----+----+----+
    | 12 | 13 | 14 | 15 |
    +----+----+----+----+
    | 08 | 09 | 10 | 11 |
    +----+----+----+----+
    | 04 | 05 | 06 | 07 |
    +----+----+----+----+
    | 00 | 01 | 02 | 03 |
    +----+----+----+----+
    """
    # Output Naming Convention
    shutil.copy(sciimg, os.path.join(path_output, os.path.basename(rename_convention(sciimg, "Calib"))))
    shutil.copy(refimg, os.path.join(path_output, os.path.basename(rename_convention(refimg, "REF"))))
    shutil.copy(maskimg, os.path.join(path_output, os.path.basename(rename_convention(maskimg, "MASK"))))
    # IMAGES in the output directory
    SCIIMG      = os.path.join(path_output, os.path.basename(rename_convention(sciimg, "Calib")))
    REFIMG      = os.path.join(path_output, os.path.basename(rename_convention(refimg, "REF")))
    CONV_SCIIMG = os.path.join(path_output, os.path.basename(rename_convention(sciimg, "hcCalib")))
    CONV_REFIMG = os.path.join(path_output, os.path.basename(rename_convention(refimg, "hcREF")))
    SUBTIMG     = os.path.join(path_output, os.path.basename(rename_convention(sciimg, "hdCalib")))
    MASKIMG     = os.path.join(path_output, os.path.basename(rename_convention(maskimg, "MASK")))
    
    #   FWHM check
    try:
        ref_seeing  = float(fits.getheader(REFIMG)['FWHM'])
        sci_seeing  = float(fits.getheader(SCIIMG)['FWHM'])
        if ref_seeing < sci_seeing:
            convdir = 't'
            CONVIMG = CONV_REFIMG
        else: # when the template/reference image has poorer image quality.
            convdir = 'i'
            CONVIMG = CONV_SCIIMG
            shutil.copy(REFIMG, CONV_REFIMG)
    except KeyError:
        convdir     = 't'
        CONVIMG = CONV_REFIMG
    #   HOTPANTs Running
    # `reuse_dia` lets a re-run skip the (very slow) HOTPANTS pass when the
    # difference and convolution images from a previous run are already present.
    # This is purely opt-in (default False keeps the original always-recompute
    # behaviour) and is meant for resuming after a downstream failure without
    # repeating ~30 min of image differencing.
    if reuse_dia and os.path.isfile(SUBTIMG) and os.path.isfile(CONVIMG):
        print('Reusing existing HOTPANTS difference/convolution images (reuse_dia=True).')
    elif not twoway_subt:
        hotpants(inim=SCIIMG, refim=REFIMG, outim=SUBTIMG, inmsk=MASKIMG, refmsk=MASKIMG, convim=CONVIMG, stamp=stamp, nrx=div_col, nry=div_row, convdir=convdir)
    else: # subtraction in both direction "i" and "t"
        # convdir = 't'
        outname_t = SUBTIMG.replace(os.path.basename(SUBTIMG).split('.')[0], "hdCalib_t")
        conname_t = CONVIMG.replace(os.path.basename(CONVIMG).split('.')[0], "hcCONV_t")
        hotpants(inim=SCIIMG, refim=REFIMG, outim=outname_t, inmsk=MASKIMG, refmsk=MASKIMG, convim=conname_t, stamp=stamp, nrx=div_col, nry=div_row, convdir="t")
        # convdir = 'i'
        outname_i = SUBTIMG.replace(os.path.basename(SUBTIMG).split('.')[0], "hdCalib_i")
        conname_i = CONVIMG.replace(os.path.basename(CONVIMG).split('.')[0], "hcCONV_i")
        hotpants(inim=SCIIMG, refim=REFIMG, outim=outname_i, inmsk=MASKIMG, refmsk=MASKIMG, convim=conname_i, stamp=stamp, nrx=div_col, nry=div_row, convdir="i")
        # mosaic the better sections
        combine_subtracted_images(header_i_path=outname_i, header_t_path=outname_t, conv2i_path=outname_i, conv2t_path=outname_t, template_i_path=conname_i, template_t_path=conname_t, output_path=SUBTIMG, template_output_path=CONVIMG, div_col=div_col, div_row=div_row)
    
    #   Readiness of the DIA
    if (os.path.isfile(SCIIMG) and os.path.isfile(REFIMG) and os.path.isfile(CONVIMG) and os.path.isfile(MASKIMG) and os.path.isfile(SUBTIMG)):
        print('All images are ready for different image analysis.')
    #------------------------------------------------------------
    #    Photometry
    #------------------------------------------------------------
    
    # configuration files
    conf_sex    = os.path.join(path_config, 'kmtnet.sex')
    conf_param  = os.path.join(path_config, 'kmtnet_imask.param')
    conf_nnw    = os.path.join(path_config, 'kmtnet.nnw')
    conf_conv   = os.path.join(path_config, 'kmtnet.conv')
    
    WEIGHTIMG   = mask2weight(MASKIMG)

    if psf_analysis==True:
        os.system(build_sex_command(
            SUBTIMG, conf_sex, os.path.join(path_config, 'kmtnet.param'), conf_conv, conf_nnw, detect=20, fwhm=fits.getheader(SCIIMG).get("FWHM"), mask=MASKIMG, weight=WEIGHTIMG, extra_args={"CATALOG_TYPE": "FITS_LDAC"}))
        if os.path.isfile(SCIIMG.replace('.fits', '.cat')):
            os.system(f'psfex {SCIIMG.replace(".fits", ".cat")} -c {path_config}kmtnet.psfex')
            os.system(build_sex_command(
                SUBTIMG, conf_sex, {os.path.join(path_config, "kmtnet_psf.param")}, conf_conv, conf_nnw, detect, fwhm=fits.getheader(SCIIMG).get("FWHM"), mask=MASKIMG, weight=WEIGHTIMG, extra_args={"PSF_NAME": SCIIMG.replace(".fits", ".psf")}
            ))
            subtbl      = ascii.read(SUBTIMG.replace(".fits", ".psf.cat"))
    else:
        # The output catalogue is consumed below with `ascii.read`, so SExtractor
        # must emit ASCII_HEAD. The shared `kmtnet.sex` config defaults to
        # CATALOG_TYPE=FITS_LDAC (needed by the PSFEx branch / catalogmaker), which
        # would otherwise produce a binary FITS table and make `ascii.read` fail with
        # a UTF-8 decode error. Override it explicitly, matching the convention used
        # everywhere else an ASCII catalogue is read back.
        _subcat = SUBTIMG.replace(".fits", ".cat")
        if not (reuse_dia and os.path.isfile(_subcat)):
            os.system(build_sex_command(SUBTIMG, conf_sex, conf_param, conf_conv, conf_nnw, detect, fwhm=fits.getheader(SCIIMG).get("FWHM"), mask=MASKIMG, weight=WEIGHTIMG, extra_args={"-CATALOG_TYPE": "ASCII_HEAD"}))
        subtbl      = ascii.read(_subcat)
    INV_SUBTIMG = SUBTIMG.replace("hd", "invhd")
    _invcat = INV_SUBTIMG.replace(".fits", ".cat")
    if not (reuse_dia and os.path.isfile(_invcat)):
        invert_image(inim=SUBTIMG, outim=INV_SUBTIMG)
        os.system(build_sex_command(INV_SUBTIMG, conf_sex, conf_param, conf_conv, conf_nnw, detect, fwhm=fits.getheader(SCIIMG).get("FWHM"), mask=MASKIMG, weight=WEIGHTIMG, extra_args={"-CATALOG_TYPE": "ASCII_HEAD"}))
    invsubtbl   = ascii.read(_invcat)

    print(f"# Number of sources: {len(subtbl)}")
    subtbl['inim']  = SCIIMG
    subtbl['hcim']  = CONV_REFIMG
    subtbl['hdim']  = SUBTIMG
    subtbl['mask']  = MASKIMG
    subtbl.meta['SEEING']   = np.median(scicat['FWHM_IMAGE']*0.4)
    subtbl['ratio_seeing']  = subtbl['FWHM_WORLD']/np.median(scicat['FWHM_WORLD'])
    subtbl.meta['ELLIPTICITY']  = np.median(scicat['ELLIPTICITY'])
    scicat['ELONGATION'] = 1 / (1-scicat['ELLIPTICITY'])
    # ELONGATION (=A/B) is not requested in kmtnet_imask.param, so derive it from the
    # ELLIPTICITY column (=1-B/A) instead of indexing a missing column. Identity:
    # 1/(1-ELLIPTICITY) = 1/(B/A) = A/B = ELONGATION.
    subtbl['ELONGATION'] = 1 / (1-subtbl['ELLIPTICITY'])
    subtbl['ratio_ellip']   = subtbl['ELLIPTICITY']/np.median(scicat['ELLIPTICITY'])
    subtbl['ratio_elong']   = subtbl['ELONGATION']/np.median(scicat['ELONGATION'])
    subtbl['MAG_AUTO']      = subtbl['MAG_AUTO'] + magautozero
    invsubtbl['MAG_AUTO']   = invsubtbl['MAG_AUTO'] + magautozero

    w = WCS(SUBTIMG)
    #    Positional information
    c_cent = w.pixel_to_world(scihdr['NAXIS1']/2, scihdr['NAXIS2']/2)
    c_sub = SkyCoord(subtbl['ALPHA_J2000'], subtbl['DELTA_J2000'], unit=u.deg)

    flagnumbers = np.arange(10)
    #    Generate flag columns
    for num in flagnumbers:
        subtbl[f'flag_{num}'] = False
    epoch = Time(fits.getheader(SCIIMG)['DATE-OBS'], format='isot')
    #    
    if "CTIO" in SCIIMG:
        location="807"
    elif "SSO" in SCIIMG:
        location="Q60"
    elif "SAAO" in SCIIMG:
        location="M22"
        
    #------------------------------------------------------------
    #    flag 0: Asteroid/Moving Object Matching
    #------------------------------------------------------------
    sep     = 5.0       # flag0: matching radius [arcsec]
    fovval  = 1.0*60    # flag0: solar object searching radius[arcmin]
    max_retries = 5
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            # Your network request here
            sbtbl = Skybot.cone_search(c_cent, fovval*u.arcmin, epoch, location=location)
            c_sb = SkyCoord(sbtbl['RA'], sbtbl['DEC'])
            sbtbl['sep'] = c_cent.separation(c_sb).to(u.arcmin)
            #    Skybot matching
            indx_sb, sep_sb, _ = c_sub.match_to_catalog_sky(c_sb)
            subtbl['flag_0'][(sep_sb.arcsec<sep)] = True
            break  # If the request was successful, exit the loop
        except RuntimeError as e:
            if "No solar system object was found" in str(e):
                print(f"No solar system objects found in the FOV for {epoch}. Continuing without flagging.")
                break  # Exit the loop, as retrying won't change the outcome
            else:
                print(f"Unexpected RuntimeError encountered: {e}. Skipping this function.")
                # raise  # Re-raise the exception for any other RuntimeError
        except ConnectionError as e:
            print(f"Connection failed on attempt {attempt+1} of {max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # Wait for a bit before retrying
            else:
                print("Final attempt failed. Skipping this function due to connection issues.")
        except Exception as e:
            # The IMCCE SkyBoT VO response format can drift out of sync with the
            # installed astroquery version (e.g. a KeyError: 'RA(h)' raised inside
            # astroquery's own parser). Asteroid flagging is an OPTIONAL refinement,
            # so any unexpected failure here must not abort transient detection:
            # leave flag_0 all-False and carry on.
            print(f"Skybot asteroid query failed ({type(e).__name__}: {e}). Skipping flag_0.")
            break
                # raise  # Re-raise the exception if the final attempt fails
    #------------------------------------------------------------
    #    flag 1: Inverted Image Detections (Artifacts Around the Source)
    #------------------------------------------------------------
    if len(invsubtbl)>0:
        #    Coordinate
        # invsubtbl = invsubtbl[invsubtbl['SNR_WIN']>20]
        c_invhd = SkyCoord(invsubtbl['ALPHA_J2000'], invsubtbl['DELTA_J2000'], unit=u.deg)
        #    Matching with inverted images
        indx_invhd, sep_invhd, _ = c_sub.match_to_catalog_sky(c_invhd)
        # subtbl['flag_1'][(sep_invhd.arcsec<subtbl['FWHM_IMAGE']) & (np.abs((subtbl['MAG_AUTO'] - invsubtbl[indx_invhd]['MAG_AUTO'])) <= 1)] = True
        subtbl['flag_1'][(sep_invhd.arcsec<subtbl['FWHM_IMAGE'])] = True
    else:
        print('Inverted subtraction image has no source. ==> pass flag1')
        pass
    #------------------------------------------------------------
    #    flag 2: Source Extractor Flag (Bad Pixel Masking) 
    #------------------------------------------------------------
    flagcut = 4         # flag2
    subtbl['flag_2'][(subtbl['FLAGS'] >= flagcut) | 
                     (subtbl['IMAFLAGS_ISO'] > 0)
                     ] = True
    #------------------------------------------------------------
    #    flag 3: High Ellipiticity Compared to Other Sources
    #------------------------------------------------------------
    elongcut= 4         # flag3
    subtbl['flag_3'][(subtbl['ratio_elong'] > elongcut)] = True
    #------------------------------------------------------------
    #    flag 4: Small/Large FWHM Compared to Other Sources
    #------------------------------------------------------------
    fwhmcut = [0.5,2.5] # flag4
    subtbl['flag_4'][
                    (subtbl['ratio_seeing']<fwhmcut[0]) |
                    (subtbl['ratio_seeing']>fwhmcut[1])
                    ] = True
    #------------------------------------------------------------
    #    flag 5: Weird Background Values
    #------------------------------------------------------------
    # backcut = 50        # flag5
    subtbl['flag_5'][
                    (subtbl['BACKGROUND']<np.median(subtbl['BACKGROUND'])-2*np.std(subtbl['BACKGROUND'])) |
                    (subtbl['BACKGROUND']>np.median(subtbl['BACKGROUND'])+2*np.std(subtbl['BACKGROUND']))
                    ] = True
    #------------------------------------------------------------
    #    flag 6: Too Low SNR ==> ???
    #------------------------------------------------------------
    snrcut  = 5         # flag6
    subtbl['flag_6'][(subtbl['SNR_WIN']<snrcut)] = True
    #------------------------------------------------------------
    #    flag 7: Crosstalk Contamination
    #------------------------------------------------------------
    path_single = path_output.replace('subt', 'scaled')
    subtbl['flag_7'] = False

    try:
        fullcat = ascii.read(f'{path_cat}{os.path.basename(sciimg)}.zp.cat')
        fullcat_coords = SkyCoord(ra=fullcat['ALPHA_J2000'], dec=fullcat['DELTA_J2000'], unit='deg')
    except Exception as e:
        print(f"Warning: Could not load or process fullcat for crosstalk check: {e}")
        fullcat = None

    if fullcat is not None:
        wcs_list = []
        dither_files = []
        num_dith = int(scihdr.get('NUMDITH', 0))
        for l in range(num_dith):
            # The current stacking step records per-dither provenance as
            # IMAGE0/IMAGE1/... (the scaled exposure base name, chip stripped) and
            # does NOT write the legacy per-chip FILE0001.. keywords this block was
            # originally indexing. Indexing a missing keyword returned None and made
            # os.path.join crash, aborting the whole subtraction. Rebuild each
            # single-chip path from the IMAGE# base name, fall back to the legacy
            # FILE#### keyword when present, and skip anything that is missing so the
            # (optional) crosstalk check degrades gracefully instead of failing.
            dith_base = scihdr.get(f'IMAGE{hex(l)[-1]}')
            for n, chip in enumerate(['kk', 'mm', 'tt', 'nn']):
                single_name = scihdr.get(f'FILE{str(l*4+n+1).zfill(4)}')
                if single_name is None and dith_base is not None:
                    single_name = dith_base.replace('.scaled.fits', f'.{chip}.scaled.fits')
                if single_name is None:
                    continue
                single = os.path.join(path_single, single_name)
                if os.path.isfile(single):
                    dither_files.append(single)
                    wcs_list.append(WCS(fits.getheader(single)))

        subtbl_coords = SkyCoord(ra=subtbl['ALPHA_J2000'].data, dec=subtbl['DELTA_J2000'].data, unit='deg')

        all_xtalk_coords = []
        for wcs in wcs_list:
            x, y = wcs.wcs_world2pix(subtbl_coords.ra, subtbl_coords.dec, 0)
            
            # Create a mask for sources within the image boundaries
            valid_mask = (x > 0) & (x < 9216) & (y > 0) & (y < 9232)
            
            if np.any(valid_mask):
                valid_x = x[valid_mask]
                valid_y = y[valid_mask]
                
                # Calculate crosstalk positions for valid sources
                xtalk_positions_x = np.array([calculate_crosstalk_positions(px) for px in valid_x])
                
                # For each source, we have multiple crosstalk x positions, but the same y
                num_xtalk_per_source = xtalk_positions_x.shape[1]
                xtalk_y = np.repeat(valid_y, num_xtalk_per_source)
                
                # Convert all crosstalk pixel coordinates to world coordinates
                xtalk_sky_coords = wcs.wcs_pix2world(xtalk_positions_x.flatten(), xtalk_y, 0)
                
                # Store xtalk coords with the original index of the source in subtbl
                original_indices = np.where(valid_mask)[0]
                repeated_indices = np.repeat(original_indices, num_xtalk_per_source)
                
                all_xtalk_coords.append(np.column_stack((xtalk_sky_coords[0], xtalk_sky_coords[1], repeated_indices)))

        if all_xtalk_coords:
            stacked_xtalk_coords = np.vstack(all_xtalk_coords)
            xtalk_skycoord = SkyCoord(ra=stacked_xtalk_coords[:, 0], dec=stacked_xtalk_coords[:, 1], unit='deg')
            
            # Match all crosstalk positions to the full catalog at once
            idx, d2d, _ = xtalk_skycoord.match_to_catalog_sky(fullcat_coords)

            max_sep = 1.0 * u.arcsec
            match_mask = d2d < max_sep
            
            # Filter for bright matches
            bright_mask = fullcat['MAG_AUTO'][idx[match_mask]] < 15
            
            # Get the original indices of sources that have a bright crosstalk match
            culprit_indices = stacked_xtalk_coords[match_mask][bright_mask][:, 2].astype(int)
            
            if len(culprit_indices) > 0:
                print(f"Found {len(np.unique(culprit_indices))} sources with potential crosstalk contamination.")
                # Flag the sources in subtbl
                subtbl['flag_7'][np.unique(culprit_indices)] = True
        else:
            print("No valid sources found on individual images to check for crosstalk.")

    #------------------------------------------------------------
    #    flag 8: HOTPANTs Chi2 Value
    #------------------------------------------------------------
    subthdr     = fits.getheader(SUBTIMG)
    hotpants_chi2 = []
    for i in range(div_col*div_row):
        hotpants_chi2.append(float(subthdr[f'X2NRM{str(i).zfill(2)}']))
    hotpants_chi2_3sigma = np.median(hotpants_chi2) + 3*np.std(hotpants_chi2)
    for i in range(div_col*div_row):
        if hotpants_chi2[i] > hotpants_chi2_3sigma or hotpants_chi2[i] > 1000:
            badregion   = subthdr[f'REGION{str(i).zfill(2)}']
            parts = badregion.strip('[]').split(',')
            xmin, xmax = map(int, parts[0].split(':'))
            ymin, ymax = map(int, parts[1].split(':'))
            indx_nosci = np.where(
                (subtbl['X_IMAGE'] >= xmin) & (subtbl['X_IMAGE'] <= xmax) &
                (subtbl['Y_IMAGE'] >= ymin) & (subtbl['Y_IMAGE'] <= ymax)
            )
            subtbl['flag_8'][indx_nosci] = True
    #------------------------------------------------------------
    #    flag 9: PSFEx Analysis
    #------------------------------------------------------------
    if psf_analysis == True:
        subtbl['flag_9'][(np.abs(subtbl['SPREAD_MODEL'])>1)] = True
    #------------------------------------------------------------
    #    Final flag
    #------------------------------------------------------------
    # TODO: Force unflag sources already reported as transients or located near putative host galaxies.
    subtbl['flag'] = False
    flag    = subtbl['flag']
    n_all   = len(subtbl)
    for n in flagnumbers:
        tmptbl  = subtbl[subtbl[f'flag_{n}']==True] 
        print(f'flag=={n} : {len(tmptbl)} {int(100*len(tmptbl)/n_all)}%')
        flag    = flag + subtbl[f'flag_{n}']
    subtbl['flag']  = flag
    # generating snapshots for asteroids
    indx_sb         = np.where(subtbl['flag_0']==True)
    subtbl['flag'][indx_sb] = False
    #------------------------------------------------------------
    #    Known-object override (GW host candidates / known transients)
    #------------------------------------------------------------
    # A user-supplied CSV lists targets we always want to inspect. Any detected
    # source matching a target within the matching radius gets a forced snapshot
    # regardless of its artifact flags, and the target name is propagated to the
    # snapshot header. The CSV must have 'Name', 'RA', 'Dec' columns and may carry
    # an optional per-row 'radius' (arcsec); when absent, `known_obj_radius` is used.
    known_obj_radius = 2.0          # default matching radius [arcsec]
    subtbl['known_match']  = False
    subtbl['known_target'] = np.array([''] * len(subtbl), dtype='U64')
    if known_obj is not None:
        try:
            ktab = ascii.read(known_obj, format='csv')
            cmap = {c.lower(): c for c in ktab.colnames}
            name_col = cmap.get('name')
            ra_col   = cmap.get('ra')
            dec_col  = cmap.get('dec', cmap.get('decl', cmap.get('de')))
            rad_col  = cmap.get('radius', cmap.get('rad', cmap.get('matching_radius')))
            if (name_col is None) or (ra_col is None) or (dec_col is None):
                raise ValueError("CSV must contain 'Name', 'RA' and 'Dec' columns")
            # RA/Dec may be decimal degrees or sexagesimal (HMS/DMS).
            try:
                c_known = SkyCoord(ra=np.array(ktab[ra_col], dtype=float)*u.deg,
                                   dec=np.array(ktab[dec_col], dtype=float)*u.deg)
            except (ValueError, TypeError):
                c_known = SkyCoord(ra=ktab[ra_col], dec=ktab[dec_col], unit=(u.hourangle, u.deg))
            if rad_col is not None:
                radii = np.array(ktab[rad_col], dtype=float)
            else:
                radii = np.full(len(ktab), known_obj_radius)
            names = np.array([str(x) for x in ktab[name_col]])
            c_all = SkyCoord(subtbl['ALPHA_J2000'], subtbl['DELTA_J2000'], unit='deg')
            kidx, ksep, _ = c_all.match_to_catalog_sky(c_known)
            kmatched = ksep.arcsec < radii[kidx]
            subtbl['known_match'] = kmatched
            subtbl['known_target'][kmatched] = names[kidx[kmatched]]
            n_forced = int(np.count_nonzero(kmatched & (subtbl['flag'] == True)))
            print(f'Known-object override: {len(ktab)} target(s), '
                  f'{int(kmatched.sum())} matched detection(s) '
                  f'({n_forced} otherwise-flagged forced into snapshots).')
        except Exception as e:
            print(f'*** known-object matching skipped ({type(e).__name__}: {e}). ***')
    #    Transient Catalog
    trtbl   = subtbl[subtbl['flag']==False]
    transient_cat   = SUBTIMG.replace('.fits', '.transient.cat')
    print('-'*60)
    print(f'Filtered sources\t: {len(trtbl)} ({100*len(trtbl)/len(subtbl):1.3f})%')
    subtbl.write(transient_cat, format='ascii.tab', overwrite=True)
    #------------------------------------------------------------
    #    Log
    #------------------------------------------------------------
    logname     = transient_cat.replace("cat", "summary.txt")
    f = open(logname, 'w')
    for n in flagnumbers:
        tmptbl = subtbl[subtbl[f'flag_{n}']==True] 
        line = f'flag=={n}: {len(tmptbl)} {int(100*len(tmptbl)/n_all)}%\n'
        f.write(line)
    line = f'Filtered sources\t: {len(trtbl)} ({100*len(trtbl)/len(subtbl):1.3f})%'
    f.write(line)
    f.close()

    # ------------------------------------------------------------
    #     Snapshot maker
    # ------------------------------------------------------------
    # Snapshots cover the flag-passing transient candidates PLUS any known-object
    # matches that are forced through regardless of their flags.
    snaptbl = subtbl[(subtbl['flag'] == False) | (subtbl['known_match'] == True)]
    n_forced_snap = int(np.count_nonzero((subtbl['flag'] == True) & (subtbl['known_match'] == True)))
    print(f"#\tSnapshot maker ({len(snaptbl)}; {n_forced_snap} forced by known-object match)")
    if len(snaptbl) > 0:
        rows = [snaptbl[i] for i in range(len(snaptbl))]
        outdir = os.path.join(path_output, 'snap')
        if ncore == 1:
            for row in rows:
                generate_snapshot(row, cutsize=cutsize, pixscale=pixscale, outdir=outdir)
        else:
            with multiprocessing.Pool(processes=ncore) as pool:
                func = partial(generate_snapshot, cutsize=cutsize, pixscale=pixscale, outdir=outdir)
                results = pool.map(func, rows)
    else:
        print('No transient candidates.')
    print("All Done")

    return 0

#%% ToO Result Checker
def csv_to_ds9reg(csv_path, region_path='default', wcs_option=True, color='green', size=50, shape='circle', label_column=None):
    """
    Converts a CSV file with source locations into a DS9 region file with customizable shapes and optional labels.
    
    Args:
    - csv_path: Path to the input CSV file.
    - region_path: Path to the output DS9 region file (defaults to replacing '.csv' with '.reg').
    - wcs_option: If True, use WCS for conversion if a corresponding FITS file exists.
    - color: Color of the regions in the DS9 region file (default is 'green').
    - size: Size of the region (radius for circles, semi-major axis for ellipses, width for boxes, etc.).
    - shape: Shape of the region ('circle', 'ellipse', 'box', 'point'). Default is 'circle'.
    - label_column: Column name in the CSV to use as the label for each region (default is None, meaning no labels).
    """
    
    import csv
    import os
    from astropy.io import fits
    from astropy.wcs import WCS
    
    if region_path == 'default':
        region_path = csv_path.replace('.csv', '.reg')
    
    image_path = csv_path.replace('.csv', '.fits')
    
    wcs = None
    # Check if corresponding FITS file exists and wcs_option is True
    if wcs_option and os.path.exists(image_path):
        with fits.open(image_path) as hdul:
            wcs = WCS(hdul[0].header)
    
    with open(csv_path, newline='') as csvfile, open(region_path, 'w') as regionfile:
        reader = csv.DictReader(csvfile)
        
        # Write the header for DS9 region file
        regionfile.write("# Region file format: DS9 version 4.1\n")
        regionfile.write(f"global color={color} dashlist=8 3 width=2 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        
        if wcs is not None:
            regionfile.write("fk5\n")  # Use fk5 for celestial coordinates
        else:
            regionfile.write("physical\n")  # Use physical for pixel coordinates
        
        # Write each source as a region with the specified shape and optional label
        for row in reader:
            if wcs is not None:
                # Convert pixel coordinates to RA and DEC if WCS is available
                ra, dec = wcs.pixel_to_world_values(float(row['X_IMAGE']), float(row['Y_IMAGE']))
            else:
                # Use pixel coordinates directly
                ra = row['X_IMAGE']
                dec = row['Y_IMAGE']
            
            # Create the label if label_column is provided
            label = f"# text={{{row[label_column]}}}" if label_column else ""
            
            # Generate the region based on the chosen shape
            if shape == 'circle':
                regionfile.write(f"circle({ra},{dec},{size}){label}\n")
            elif shape == 'ellipse':
                regionfile.write(f"ellipse({ra},{dec},{size},{size/2},0){label}\n")  # Semi-major axis is size, semi-minor axis is half size
            elif shape == 'box':
                regionfile.write(f"box({ra},{dec},{size},{size},{0}){label}\n")  # Width and height as size
            else:
                regionfile.write(f"point({ra},{dec}) # point={shape}{label}\n")  # You can replace 'cross' with 'diamond', 'box', etc.
