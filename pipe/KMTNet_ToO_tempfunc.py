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
#%% General Functions
def sqsum(numlist):
    S   = 0
    for i in range(len(numlist)):
        S   += numlist[i]**2
    sqS     = np.sqrt(S)
    return sqS
#------------------------------------------------------------
def apass_query(ra, dec, radius=1.0): # unit=(deg, deg, arcsec)

    import astropy.coordinates as coord
    from astroquery.vizier import Vizier 

    """
    APASS QUERY
    INPUT   :   RA [deg], Dec [deg], radius
    OUTPUT  :   QUERY TABLE
    #   Vega    : B, V
    #   AB      : g, r, i
    #   Vega - AB Magnitude Conversion (Blanton+07)
    #   B       : m_AB - m_Vega =-0.09
    #   V       : m_AB - m_Vega = 0.02
    #   R       : m_AB - m_Vega = 0.21
    #   I       : m_AB - m_Vega = 0.45
    """
    Vizier.ROW_LIMIT    = -1
    query       = Vizier.query_region(coord.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'), width=str(radius*60)+'m', catalog=["APASS9"])
    dum         = query[0]
    colnames    = dum.colnames
    
    for col in colnames:
        indx    = np.where(dum[col].mask == False)
        dum     = dum[indx]
        
    querycat    = Table()
    querycat['NUMBER']  = dum['recno']
    querycat['RAJ2000'] = dum['RAJ2000']
    querycat['DEJ2000'] = dum['DEJ2000']
    querycat['Numb_obs']= dum['nobs']
    querycat['Numb_img']= dum['mobs']
    querycat['B-V']     = dum['B-V']+ (-0.09 - 0.02)
    querycat['e_B-V']   = dum['e_B-V']
    # querycat['Bmag']    = dum['Bmag']   - 0.09  # [Vega] to [AB]
    querycat['Bmag']    = dum['Bmag'] - 0.09 - 0.27 * (dum['B-V'] - (0.09 + 0.02))  # Park+19
    querycat['e_Bmag']  = dum['e_Bmag']
    querycat['Vmag']    = dum['Vmag']   + 0.02  # [Vega] to [AB]
    querycat['e_Vmag']  = dum['e_Vmag']
    querycat['Rmag']    = dum['r_mag'] - 0.0576 - 0.3718 * (dum['r_mag'] - dum['i_mag'] - 0.2589) # Blanton+07, sigma = 0.0072
    querycat['e_Rmag']  = dum['e_r_mag']
    # querycat['Imag']    = dum['r_mag'] - 1.2444 * (dum['r_mag'] - dum['i_mag']) - 0.3820 + 0.45 # Lupton+05, sigma = 0.0078
    querycat['Imag']    = dum['i_mag'] # Park+19
    querycat['e_Imag']  = dum['e_i_mag']

    return querycat
#------------------------------------------------------------
def GAIAXP_query(field, path_ref):
    """
    field = '0000'
    path_ref = '/data4/kmtntoo/cat/gaiaxp/'
    """
    field = field.split('.')[0]
    refcat  = Table(fits.open(os.path.join(path_ref, f'gaiaxp_{field}.fits'))[1].data)
    refcat.rename_columns(['RA', 'DEC'], ['RAJ2000', 'DEJ2000'])
    refcat['Bmag'] = refcat['XP_B'] - 0.30 * (refcat['XP_B']-refcat['XP_V'])
    refcat['Rmag'] = refcat['XP_R'] - 0.04 * (refcat['XP_V']-refcat['XP_R'])
    refcat.rename_columns(['XP_eB'], ['e_Bmag'])
    refcat.rename_columns(['XP_V', 'XP_eV'], ['Vmag', 'e_Vmag'])
    refcat.rename_columns(['XP_eR'], ['e_Rmag'])
    refcat.rename_columns(['XP_I', 'XP_eI'], ['Imag', 'e_Imag'])

    return refcat
#------------------------------------------------------------
def sort_BVRI(imlist):
    newlist     = []
    ks4ftr  = ['B', 'V', 'R', 'I']
    for ftr in ks4ftr:
        for i in range(len(imlist)):
            if ftr in imlist[i]:
                newlist.append(imlist[i])
    return newlist
#------------------------------------------------------------
def limitmag(N, zp, aper, skysigma): # 3? 5?, zp, diameter [pixel], skysigma

    R           = float(aper)/2.                # to radius
    braket      = N*skysigma*np.sqrt(np.pi*(R**2))
    upperlimit  = float(zp)-2.5*np.log10(braket)

    return round(upperlimit, 3)
#------------------------------------------------------------
def matching(intbl, reftbl, inra, indec, refra, refdec, sep=2.0):
    """
    MATCHING TWO CATALOG WITH RA, Dec COORD. WITH python
    INPUT   :   SE catalog, SDSS catalog file name, sepertation [arcsec]
    OUTPUT  :   MATCED CATALOG FILE & TABLE
    """

    incoord     = SkyCoord(inra, indec, unit=(u.deg, u.deg))
    refcoord    = SkyCoord(refra, refdec, unit=(u.deg, u.deg))

    #   INDEX FOR REF.TABLE
    indx, d2d, d3d  = incoord.match_to_catalog_sky(refcoord)
    mreftbl         = reftbl[indx]
    mreftbl['sep']  = d2d
    mergetbl        = intbl
    for col in mreftbl.colnames:
        mergetbl[col]    = mreftbl[col]
    indx_sep        = np.where(mergetbl['sep']*3600.<sep)
    mtbl            = mergetbl[indx_sep]
    #mtbl.write(mergename, format='ascii', overwrite=True)
    return mtbl
#------------------------------------------------------------
def star4zp(intbl, inmagerkey, refmagkey, refmagerkey, refmaglower=14., refmagupper=17., refmagerupper=0.05, inmagerupper=0.1, flagcut=0):
    """
    SELECT STARS FOR USING ZEROPOINT CALCULATION
    INPUT   :   TABLE, IMAGE MAG.ERR KEYWORD, REF.MAG. KEYWORD, REF.MAG.ERR KEYWORD
    OUTPUT  :   NEW TABLE
    """
    indx    = np.where( (intbl['FLAGS'] <= flagcut) & 
                        (intbl[refmagkey] < refmagupper) & 
                        (intbl[refmagkey] > refmaglower) & 
                        (intbl[refmagerkey] < refmagerupper) &
                        (intbl[inmagerkey] < inmagerupper) 
                        )
    indx0   = np.where( (intbl['FLAGS'] <= flagcut) )
    indx2   = np.where( (intbl[refmagkey] < refmagupper) & 
                        (intbl[refmagkey] > refmaglower) & 
                        (intbl[refmagerkey] < refmagerupper) 
                        )
    indx3   = np.where( (intbl[inmagerkey] < inmagerupper) )
    newtbl  = intbl[indx]
    comment = '-'*60+'\n' \
            + 'ALL\t\t\t\t: '+str(len(intbl))+'\n' \
            + '-'*60+'\n' \
            + 'FLAG(<={})\t\t\t: '.format(flagcut)+str(len(indx0[0]))+'\n' \
            + refmagkey+' REF. MAGCUT ('+str(refmaglower)+'-'+str(refmagupper)+')'+'\t\t: '+str(len(indx2[0]))+'\n' \
            + refmagerkey+' REF. MAGERR CUT < '+str(refmagerupper)+'\n' \
            + inmagerkey+' OF IMAGE CUT < '+str(inmagerupper)+'\t: '+str(len(indx3[0]))+'\n' \
            + '-'*60+'\n' \
            + 'TOTAL #\t\t\t\t: '+str(len(indx[0]))+'\n' \
            + '-'*60
    print(comment)
    return newtbl
#------------------------------------------------------------
def zpcal(intbl, inmagkey, inmagerkey, refmagkey, refmagerkey, sigma=2.0):
    """
    ZERO POINT CALCULATION
    3 SIGMA CLIPPING (MEDIAN)
    """
    from astropy.stats import sigma_clip

    #    REMOVE BLANK ROW (=99)    
    indx_avail      = np.where( (intbl[inmagkey] != 99) & (intbl[refmagkey] != 99) )
    intbl           = intbl[indx_avail]
    zplist          = np.copy(intbl[refmagkey] - intbl[inmagkey])
    intbl['zp']     = zplist
    #    SIGMA CLIPPING
    zplist_clip     = sigma_clip(zplist, sigma=sigma, maxiters=None, cenfunc=np.median, copy=False)
    indx_alive      = np.where( zplist_clip.mask == False )
    indx_exile      = np.where( zplist_clip.mask == True )
    #    RE-DEF. ZP LIST AND INDEXING CLIPPED & NON-CLIPPED
    intbl_alive     = intbl[indx_alive]
    intbl_exile     = intbl[indx_exile]
    #    ZP & ZP ERR. CALC.
    zp              = np.median(np.copy(intbl_alive['zp']))
    zper            = np.std(np.copy(intbl_alive['zp']))
    return zp, zper, intbl_alive, intbl_exile
#------------------------------------------------------------
def add_colorbar(mappable, clabel, clim):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(clabel)
    cbar.mappable.set_clim(clim[0], clim[1])
    plt.sca(last_axes)
#------------------------------------------------------------
def date2MJD(dateobs):

    from astropy.time import Time

    return Time(dateobs, format='isot', scale='utc').mjd
#------------------------------------------------------------
def MJD2date(mjd):
    
    from datetime import datetime, timedelta
    
    jd = mjd + 2400000.5
    delta = jd - 2440587.5
    timestamp = timedelta(days=delta)
    date = datetime.utcfromtimestamp(timestamp.total_seconds())
    return date.strftime("%Y-%m-%dT%H:%M:%S")
#------------------------------------------------------------
def create_ldac_fits(input_fits, output_ldac, centrakey='X_WORLD', centdeckey='Y_WORLD', magkey='MAG', center=None, radius=None):
    with fits.open(input_fits) as hdul:
        data_table = Table(hdul[1].data)
        # Rename columns
        try:
            data_table.rename_column('RA', centrakey)
            data_table.rename_column('DEC', centdeckey)
        except:
            data_table.rename_column('ALPHAJ2000', centrakey)
            data_table.rename_column('DELTAJ2000', centdeckey)
        
        data_table['ERRA_WORLD'] = np.full(len(data_table), 0.0001)  # default small error in degrees
        data_table['ERRB_WORLD'] = np.full(len(data_table), 0.0001)
        
        try:
            data_table[magkey] = data_table['XP_V']  # default magnitude
            data_table = data_table[data_table['XP_Vflag']==1] # gaia XP flag
            data_table = data_table[np.abs(data_table['cstar'])<0.05] # star-like
        except:
            data_table[magkey] = np.full(len(data_table), 16)  # default magnitude

        # Filter the table if center and radius are provided
        if center is not None and radius is not None:
            catalog_coords = SkyCoord(ra=data_table[centrakey]*u.degree, dec=data_table[centdeckey]*u.degree)
            separation = center.separation(catalog_coords)
            data_table = data_table[separation < radius * u.degree]

        ldac_hdulist = fits.HDUList()
        primary_hdu = fits.PrimaryHDU()
        ldac_hdulist.append(primary_hdu)

        header_hdu = fits.ImageHDU()
        header_hdu.header.extend(hdul[1].header, unique=True)
        header_hdu.header['EXTNAME'] = 'LDAC_IMHEAD'
        header_hdu.header['TDIM1'] = '(1024)'  # Example, adjust based on your data structure

        ldac_hdulist.append(header_hdu)

        bintable_hdu = fits.BinTableHDU(data_table)
        bintable_hdu.header['EXTNAME'] = 'LDAC_OBJECTS'
        ldac_hdulist.append(bintable_hdu)

        ldac_hdulist.writeto(output_ldac, overwrite=True)
#--------------for trasient searching------------------------
def trim(inim, position, size, outim='trim.fits'):

    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D

    # Load the image and the WCS
    hdu = fits.open(inim)[0]
    wcs = WCS(hdu.header)
    # Make the cutout, including the WCS
    cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs)
    # Put the cutout image in the FITS HDU
    hdu.data = cutout.data
    # Update the FITS header with the cutout WCS
    hdu.header.update(cutout.wcs.to_header())
    # Write the cutout to a new FITS file
    hdu.writeto(outim, overwrite=True)
#------------------------------------------------------------
def hotpants(inim, refim, inmsk, refmsk, convdir='t', outim='hd.fits', convim='hc.fits', nrx=4, nry=4, stamp=None):
    '''
    inim : Science image
    refim : Reference image
    convdir: convolution direction ('t' for reference, 'i' for science)
    [-c  toconvolve]  : force convolution on (t)emplate or (i)mage (undef)
    [-n  normalize]   : normalize to (t)emplate, (i)mage, or (u)nconvolved (t)
    '''
    if stamp is None:
        com = f'hotpants -c {convdir} -n {convdir} -iu 100000000 -il -100000 -tu 100000000 -tl -100000 -v 0 -inim {inim} -tmplim {refim} -imi {inmsk} -tmi {refmsk} -outim {outim} -oci {convim} -nrx {nrx} -nry {nry}'
    else:
        stampname   = outim.replace('.fits', '.stamp')
        with open(stampname, "w") as f:
            for s in stamp:
                f.write(f"{s['X_IMAGE']} {s['Y_IMAGE']} \n")
        com = f'hotpants -c {convdir} -n {convdir} -iu 100000000 -il -100000 -tu 100000000 -tl -100000 -v 0 -inim {inim} -tmplim {refim} -imi {inmsk} -tmi {refmsk} -outim {outim} -oci {convim} -ssf {stampname} -nrx {nrx} -nry {nry}'
    print(com)
    os.system(com)
#------------------------------------------------------------
def invert_image(inim, outim):
    data, hdr = fits.getdata(inim, header=True)
    invdata = data*(-1)
    fits.writeto(outim, invdata, header=hdr, overwrite=True)
#------------------------------------------------------------
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
#------------------------------------------------------------
def calculate_crosstalk_positions(x_position):
    # Define the width of one section
    section_width = 1152
    # Calculate which section the bright star is in
    star_section = (x_position // section_width) + (1 if x_position % section_width != 0 else 0)
    # Determine the sections where crosstalk will appear based on the star’s section
    if star_section % 2 == 0:  # Star is in an even section
        crosstalk_sections = [2, 4, 6, 8]
    else:  # Star is in an odd section
        crosstalk_sections = [1, 3, 5, 7]
    # Calculate the flipped x position within its section
    distance_in_section = x_position - section_width * (star_section - 1)
    flipped_x = section_width - distance_in_section
    # Calculate the crosstalk positions
    crosstalk_positions = []
    flipped_sections = [5, 6, 7, 8] if star_section in [1, 2, 3, 4] else [1, 2, 3, 4]
    for section in crosstalk_sections:
        if section in flipped_sections:
            crosstalk_x = (section - 1) * section_width + flipped_x
        else:
            crosstalk_x = (section - 1) * section_width + distance_in_section
        crosstalk_positions.append(crosstalk_x)
    return crosstalk_positions
#------------------------------------------------------------
def sexcom(inim, conf_sex, conf_param, conf_conv, conf_nnw, det_thres, detectiondual=None):
    outcat = inim.replace('fits', 'cat')
    if detectiondual == None:
        sexcom = f'sex {inim} -c {conf_sex} -CATALOG_NAME {outcat} -PARAMETERS_NAME {conf_param} -FILTER_NAME {conf_conv} -STARNNW_NAME {conf_nnw} -DETECT_THRESH {det_thres}'
    else:
        sexcom = f'sex {detectiondual},{inim} -c {conf_sex} -CATALOG_NAME {outcat} -PARAMETERS_NAME {conf_param} -FILTER_NAME {conf_conv} -STARNNW_NAME {conf_nnw} -DETECT_THRESH {det_thres}'

    return sexcom
#------------------------------------------------------------
def generate_snapshot(trtbl, i, cutsize=2.0, pixscale=0.4):

    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D

    #    Images
    n = trtbl['NUMBER'][i].item()
    inim, hcim, hdim = trtbl['inim'][i], trtbl['hcim'][i], trtbl['hdim'][i]
    #    Poistion of transient candidate
    tra = trtbl['ALPHA_J2000'][i].item()
    tdec = trtbl['DELTA_J2000'][i].item()
    ximg = trtbl['X_IMAGE'][i].item()
    yimg = trtbl['Y_IMAGE'][i].item()
    position = SkyCoord(tra, tdec, frame='icrs', unit='deg')
    #    Seeing
    seeing = trtbl.meta['SEEING']
    # peeing = trtbl['peeing'][i].item()
    size = u.Quantity((cutsize, cutsize), u.arcmin)

    for image, kind in zip([inim, hcim, hdim], ['new', 'ref', 'sub']):
        hdu = fits.open(image)[0]
        wcs = WCS(hdu.header)
        peeing = seeing/pixscale
        # peeing = hdu.header['PEEING']
        # Make the cutout, including the WCS
        cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs, mode='partial', fill_value=0) # padding with 0
        data =     cutout.data
        # Put the cutout image in the FITS HDU
        hdu.data = cutout.data
        # Update the FITS header with the cutout WCS
        hdu.header.update(cutout.wcs.to_header())
        hdu.header['TRANRA']    = (tra, "transient candidate center RA")
        hdu.header['TRANDEC']   = (tdec, "transient candiate center DEC")
        hdu.header['XIMAGE']    = (ximg, "transient candidate X pixel location")
        hdu.header['YIMAGE']   = (yimg, "transient candiate Y pixel location")
        hdu.header['TRIM']      = (inim.split('.')[-2], "trimmed section")
        hdu.header['MAGAUTO']   = (trtbl['MAG_AUTO'][i], "transient candidate MAG_AUTO")
        hdu.header['SNR']       = (trtbl['SNR_WIN'][i], "transient candidate SNR")
        hdu.header['SEEING']    = (trtbl['FWHM_IMAGE'][i] * 0.4, "transient candidate FWHM")
        hdu.header['ELLIP']     = (trtbl['ELLIPTICITY'][i], "transient candidate ellipticity")
        hdu.header['ELONG']     = (trtbl['ELONGATION'][i], "transient candidate elongation")
        hdu.header['CLSSTAR']   = (trtbl['CLASS_STAR'][i], "transient candidate CLASS_STAR")
        hdu.header['ASTEROID']  = (trtbl['flag_0'][i], "moving object matched within 5arcsec")
        hdu.header['IMAFLAG']   = (trtbl['IMAFLAGS_ISO'][i], "Mask image flags")
        # Write the cutout to a new FITS file
        outim = f'{os.path.splitext(hdim)[0]}.{n:0>6}.{kind}{os.path.splitext(hdim)[1]}'
        # outpng = f'{os.path.splitext(hdim)[0]}.{n:0>6}.{kind}.png'
        #    Save postage stamp *.png & *.fits
        hdu.writeto(outim, overwrite=True)
        # plot_snapshot(data, wcs, peeing, outpng, save=True)
#------------------------------------------------------------
def plot_snapshot(data, wcs, peeing, outpng, save=True):
    
    from matplotlib.patches import Circle
    from astropy.visualization import ZScaleInterval
    from astropy.visualization.stretch import LinearStretch

    plt.close('all')
    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(1, 1))
    fig.set_size_inches(1. * data.shape[0] / data.shape[1], 1, forward = False)
    x = 720 / fig.dpi
    y = 720 / fig.dpi
    fig.set_figwidth(x)
    fig.set_figheight(y)
    #    No axes
    # ax = plt.subplot(projection=wcs)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    #    Sci
    data[np.isnan(data)] = 0.0
    transform = LinearStretch()+ZScaleInterval()
    bdata = transform(data)
    # pylab.subplot(131)
    ax.imshow(bdata, cmap="gray", origin="lower")

    #    Circle
    circle = Circle(
        (data.shape[0]/2., data.shape[1]/2.),
        2*peeing,
        edgecolor='yellow',
        lw=3,
        facecolor=None,
        fill=False
    )

    ax.add_patch(circle)

    #    RA, Dec direction
    ra0, dec0 = wcs.all_pix2world(0, 0, 1)
    ra1, dec1 = wcs.all_pix2world(data.shape[0], data.shape[1], 1)
    if ra0>ra1:
        pass
    elif ra0<ra1:
        ax.invert_xaxis()
    if dec0>dec1:
        ax.invert_yaxis()
    elif dec0<dec1:
        pass
    #    Save or not?
    if save:
        plt.savefig(outpng, dpi=100,)
    else:
        pass
#------------------------------------------------------------
def rename_convention(inim, prefix='Calib'):
    hdr = fits.getheader(inim)
    """
    Special name for splited KMTNet images
    """
    try:
        observat    = hdr['OBSERV0']
    except:
        observat    = hdr['OBSERVAT']
    obs         = f'KMTNet_{observat}'
    obj         = hdr['OBJECT']
    exptime     = hdr['EXPTIME']
    filte       = hdr['FILTER']
    try:
        dateobs     = hdr['DATE-OBS'].replace('-', '').replace(':', '').replace('T', '-')
    except:
        dateobs     = hdr['DATE'].replace('-', '').replace(':', '').replace('T', '-') # processed date
    #       New name
    newim = f"{os.path.dirname(inim)}/{prefix}.{obs}.{obj}.{dateobs}.{filte}.{exptime:g}.stack.fits"
    return newim
#------------------------------------------------------------
def safe_load_fits(filename, shape=None):
    if os.path.exists(filename):
        # Return the data if file exists
        return fits.getdata(filename).astype(int)
    else:
        # Return an array of zeros if file is missing
        # Use the provided shape, if available; otherwise, default to a shape
        if shape is not None:
            return np.zeros(shape, dtype=int)
        else:
            # Handle the case where shape is not known (for the first file)
            raise FileNotFoundError(f"File {filename} not found and no shape provided for fallback.")
#------------------------------------------------------------
def find_longest_exposure_image(pattern):
    """
    Finds the image file with the longest exposure time based on the filename pattern.
    
    Args:
    - pattern (str): The glob pattern used to match the files.
    
    Returns:
    - str: The filename of the image with the longest exposure time, or 'None' if no images are found.
    """
    # Use glob to find files matching the pattern
    files = [f for f in sorted(glob.glob(pattern)) if "mask" not in f and "crmap" not in f]
    
    # Try to find the file with the longest exposure time
    try:
        return max(files, key=lambda f: int(re.search(r'\d+(?=sec)', f).group()))
    except ValueError:
        # Return None if no files are found or if there's an issue parsing the exposure time
        return None
#------------------------------------------------------------
def read_header(filename):
    """
    Reads a .head file and extracts key-value pairs from it.
    
    Args:
        filename (str): Path to the .head file.
        
    Returns:
        dict: A dictionary containing key-value pairs from the header file.
    """
    header_dict = {}

    # Open the .head file and read it line by line
    with open(filename, 'r') as header_file:
        for line in header_file:
            # Check if the line contains a key-value pair
            if '=' in line:
                # Split the line at the '=' sign to separate the key and value
                key, value_comment = line.split('=', 1)
                key = key.strip()  # Clean up any extra spaces
                
                # Separate the value from the comment, if present
                if '/' in value_comment:
                    value, comment = value_comment.split('/', 1)
                    value = value.strip()  # Clean up value
                    comment = comment.strip()  # Clean up comment
                else:
                    value = value_comment.strip()
                    comment = None

                # Store the key-value pair in the dictionary
                header_dict[key] = value

    return header_dict
#------------------------------------------------------------
def combine_subtracted_images(conv2i_path, conv2t_path, output_path, div_col=4, div_row=4):
    # Open both fits files
    with fits.open(conv2i_path) as hdul_i, fits.open(conv2t_path) as hdul_t:
        # Get the image data for both
        image_data_i = hdul_i[0].data
        image_data_t = hdul_t[0].data
        header_i = hdul_i[0].header.copy()  # Make a copy of the norm2i header to start with
        header_t = hdul_t[0].header

        # Initialize an empty array for the combined output image
        combined_image = np.zeros_like(image_data_i)

        # Loop over the 16 regions (assuming 4x4 grid = 16 regions)
        for region in range(div_col*div_row):
            # Get the X2NRM value from the header for each region in both images
            x2nrm_i = float(header_i[f'X2NRM{region:02d}'])
            x2nrm_t = float(header_t[f'X2NRM{region:02d}'])

            # Get the region bounds from the REGION keyword
            region_bounds = header_i[f'REGION{region:02d}']
            # Adjust for 0-indexing in Python
            x_start, x_end, y_start, y_end = [int(i)-1 for i in region_bounds.replace('[', '').replace(']', '').split(',')[0].split(':') + region_bounds.replace('[', '').replace(']', '').split(',')[1].split(':')]
            
            # Choose the region with better quality (smaller X2NRM value)
            if x2nrm_i < x2nrm_t:
                print(f'chi2_i = {x2nrm_i}, chi2_t = {x2nrm_t}, selecting section normalized to the science image')
                # Use the region from image1 (norm2i)
                combined_image[y_start:y_end, x_start:x_end] = image_data_i[y_start:y_end, x_start:x_end]
                header_i[f'CONVD{region:02d}'] = 'i'  # Mark region as coming from norm2i
            else:
                # Use the region from image2 (norm2t)
                print(f'chi2_i = {x2nrm_i}, chi2_t = {x2nrm_t}, selecting section normalized to the template image')
                combined_image[y_start:y_end, x_start:x_end] = image_data_t[y_start:y_end, x_start:x_end]
                header_i[f'CONVD{region:02d}'] = 't'  # Mark region as coming from norm2t

            # Update the header for that region with values from the selected image (if from norm2t, copy X2NRM and other relevant keywords)
            if x2nrm_t < x2nrm_i:
                header_i[f'X2NRM{region:02d}']  = header_t[f'X2NRM{region:02d}']
                header_i[f'KSUM{region:02d}']   = header_t[f'KSUM{region:02d}']
                header_i[f'CONVOL{region:02d}'] = header_t[f'CONVOL{region:02d}']
                header_i[f'SSSIG{region:02d}']  = header_t[f'SSSIG{region:02d}']
                header_i[f'SSSCAT{region:02d}'] = header_t[f'SSSCAT{region:02d}']
                header_i[f'FSIG{region:02d}']   = header_t[f'FSIG{region:02d}']
                header_i[f'FSCAT{region:02d}']  = header_t[f'FSCAT{region:02d}']
                header_i[f'NX2NRM{region:02d}'] = header_t[f'NX2NRM{region:02d}']

        # Write the combined image to a new FITS file
        fits.writeto(output_path, combined_image, header=header_i, overwrite=True)

    print(f"Combined image saved to {output_path}")
#------------------------------------------------------------
def psfexxml(xmlfile):
    """
    INPUT   :   .xml
    OUTPUT  :   FWHM    [pixel]
    """
    votable     = parse(xmlfile)
    table       = votable.get_first_table()
    data        = table.array
    #   EXTRACT FWHM [pixel]
    fwhm        = data['FWHM_Mean'][0]
    fwhm        = round(fwhm, 3)
    return fwhm

def psfex(inim, pixscale, path_psf, path_cfg):
    """
    OUTPUT: fwhm, psf model
    """
    #   FILE CHECK
    prese_conf    = path_cfg+'prepsfex.sex'
    prese_param   = path_cfg+'prepsfex.param'
    psfex_conf    = path_cfg+'default.psfex'
    psfex_conv    = path_cfg+'default.conv'
    
    try:
        comment = '\nPSFex START\n' \
                + 'IMAGE\t\t: '+inim+'\n' \
                + 'PRE_CONFIG\t: '+prese_conf+'\n' \
                + 'PRE_PARAM\t: '+prese_param+'\n' \
                + 'CONFIG\t\t: '+psfex_conf+'\n' \
                + 'CONV  \t\t: '+psfex_conv
        print(comment)
    except:
        comment = 'CHECK prese/prese_param/psfex_conf/psfex_conv OR OTHERS.'
        print(comment)

    #   FILE NAME
    cat     = path_psf+inim[:-5]+'.cat'
    xml     = path_psf+inim[:-5]+'.xml'
    #   OPTION
    presecom1   = "-c {} {}".format(prese_conf, inim)
    presecom2   = " -CATALOG_NAME {}".format(cat)
    presecom3   = " -FILTER_NAME {} -PARAMETERS_NAME {}".format(psfex_conv, prese_param)
    #   COMMAND
    presecom    = "sex {}{}{}".format(presecom1, presecom2, presecom3)
    os.system(presecom)
    psfexcom    = "psfex -c {} {}".format(psfex_conf, cat)
    os.system(psfexcom) 

    os.system('mv psfex.xml {}'.format(xml))
    os.system('mv snap_{} {}'.format(inim, path_psf))
    os.system('rm {}'.format(cat))

    #   FWHM [pixel], FWHM [arcsec]
    fwhm_pix    = psfexxml(xml)
    fwhm_arcsec = round(fwhm_pix*pixscale, 3)
    comment     = '\n' \
                + 'FILE NAME'+'\t'+': '+inim+'\n' \
                + 'FWHM value'+'\t'+': '+str(fwhm_pix)+'\t'+'[pixel]'+'\n' \
                + '\t'+'\t'+': '+str(fwhm_arcsec)+'\t'+'[arcsec]'+'\n'
    print(comment)
    return fwhm_arcsec, fwhm_pix
#%% ToOAmplifierCombine.py
def ampcom(path_data, path_cfg):
    
    import os
    import numpy as np
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
    
            os.system(f'sex {path_data}{serial}.{chip}.fits -c {cfg} -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME {catname} -PARAMETERS_NAME {param} -FILTER_NAME {conv} -STARNNW_NAME {nnw} -DETECT_THRESH 50 -ANALYSIS_THRESH 50')
                      
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
def astrom(path_data, path_cfg, path_cat, radius=1.0, ithresh=5):
    """
    date = '240918_SSO'
    path_data = f'/data4/kmtntoo/tutorial/data/raw/{date}/'
    path_cfg = '/data4/kmtntoo/tutorial/config/'
    path_cat = '/data4/kmtntoo/tutorial/catalog/'
    radius = 1.0
    ithresh = 10
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
            
            thresh = ithresh
            repeat = 0

            while 1:

                sexcom  = f'sex {path_data}{serial}.{chip}.fits -c {cfg} -CATALOG_NAME {catname} -PARAMETERS_NAME {param} -FILTER_NAME {conv} -STARNNW_NAME {nnw} -CATALOG_TYPE FITS_LDAC -HEADER_SUFFIX NONE -DETECT_THRESH {thresh} -ANALYSIS_THRESH {thresh} -SATUR_LEVEL 60000.0'
                
                if name[i][3] == 'a': ahead = f'{path_cfg}ahead/kmtnet_global_sso.{chip}.ahead' #Austrailia
                if name[i][3] == 's': ahead = f'{path_cfg}ahead/kmtnet_global.{chip}.ahead' #South Africa
                if name[i][3] == 'c': ahead = f'{path_cfg}ahead/kmtnet_global_ctio.{chip}.ahead' #Chile
                gridcat = os.path.join(path_cfg, "kmtnet_grid.fits")

                centcoord   = SkyCoord(ra[i], dec[i], unit=(u.hourangle, u.deg))
                try:
                    kmtgrid     = Table.read(gridcat, format='fits')
                except:
                    kmtgrid     = Table.read(gridcat, format='ascii')
                kmtcoord    = SkyCoord(kmtgrid['ra[deg]'], kmtgrid['dec[deg]'], unit='deg')
                trgt_field  = kmtgrid[centcoord.separation(kmtcoord).argmin()]
                
                # load the reference catalog (GAIA)
                
                gaiacat = os.path.join(path_cat, 'gaiaxp', f'gaiaxp_{str(trgt_field["field_name1"]).zfill(4)}.fits')
                if os.path.exists(gaiacat) and centcoord.separation(kmtcoord).min().value < 1.0: # center matched with the grid in 0.1 deg
                    gaialdac     = gaiacat.replace(".fits", "_ldac.fits")
                    if not os.path.exists(gaialdac):
                        create_ldac_fits(gaiacat, gaialdac, center=centcoord, radius=radius)

                    scampcom = f'scamp {catname} -c {os.path.join(path_cfg, "kmtnet.scamp")} -ASTREF_CATALOG FILE -ASTREFCAT_NAME {gaialdac} -POSITION_MAXERR 20.0 -CROSSID_RADIUS 5.0 -DISTORT_DEGREES 3 -PROJECTION_TYPE TPV -AHEADER_GLOBAL {ahead} -STABILITY_TYPE INSTRUMENT'
                else:
                    print(f"Closest Separation: {centcoord.separation(kmtcoord).min().value:.2f} deg")
                    scampcom = f'scamp {catname} -c {os.path.join(path_cfg, "kmtnet.scamp")} -ASTREF_CATALOG UCAC-4 -POSITION_MAXERR 20.0 -CROSSID_RADIUS 5.0 -DISTORT_DEGREES 3 -PROJECTION_TYPE TPV -AHEADER_GLOBAL {ahead} -STABILITY_TYPE INSTRUMENT'
                
                # Run SExtractor and SCAMP
                print(sexcom)
                os.system(sexcom)
                print(scampcom)
                outhdr  = f'{path_data}{serial}.{chip}.astrom.head'
                if os.path.exists(outhdr):
                    os.remove(outhdr)
                os.system(scampcom)
                if float(read_header(outhdr).get('ASTRRMS1')) > 1e-4 and repeat <= 3: 
                    print(f'Warning: High RMS for {path_data}{serial}.{chip}.fits (iteration: {repeat})')
                    thresh += 10
                    repeat += 1
                else:
                    thresh = ithresh
                    repeat = 0
                    break

        skyarr  = [skykk[i], skymm[i], skytt[i], skynn[i]]
        fwhmarr = [fwhm1kk[i], fwhm1mm[i], fwhm1tt[i], fwhm1nn[i]]

        for k in range(len(chiparr)):
            chip = chiparr[k]
            tempsky = skyarr[k]
            tempfwhm = fwhmarr[k]
            
            inhdr = f"{path_data}{serial}.{chip}.astrom.head"
            if os.path.exists(inhdr):
                f=open(inhdr,'r')
                lines=f.readlines()
                f.close()

                f=open(inhdr,'w')
                lines[1]=lines[1][0:37]+'\n'
                for line in lines[0:gap] : f.write(line)
                f.close()
                
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
def qatest(fname, configdir, gridcat, refcatdir, refcatname='GAIAXP', divnum=8, crreject=True, bleedreject=True, weightmap=True, imtype='chip') :
    """
    QATEST ver 1.3.2

    # Input Format : 
    os.chdir(f'/data4/kmtntoo/tutorial/pipe/')
    
    fname       = '/data4/kmtntoo/tutorial/data/raw/240916_SAAO/044493.nn.fits'
    configdir   = '/data4/kmtntoo/tutorial/config/'
    gridcat     = 'kmtnet_grid.fits'
    refcatdir   = '/data4/kmtntoo/tutorial/catalog/'
    refcatname  = 'gaiaxp'
    divnum      = 8
    crreject    = True
    bleedreject = True
    weightmap   = True
    imtype      = 'chip'

    fname       = '/data4/kmtntoo/tutorial/data/stack/240916_SAAO/S240915b_2025.108-48.R.20240916.SAAO.960sec.stack.fits'
    configdir   = '/data4/kmtntoo/tutorial/config/'
    gridcat     = 'kmtnet_grid.fits'
    refcatdir   = '/data4/kmtntoo/tutorial/catalog/'
    refcatname  = 'gaiaxp'
    divnum      = 8
    crreject    = False
    bleedreject = False
    weightmap   = True
    imtype      = 'stack'

    Caution: cr rejection should be carried out before background subtraction.
    # Ouput Format : 
    ## Updates FITS header with new QA (Quality Assurance) info :
    HISTORY   Quality Assurance (QA) by QATEST version 1.3.0 (2022-04-29)
    COMMENT   2022 JSH
    CCDNAME =                 / Name of CCD
    REASTROM=                 / True if reastrometry done
    QAREFCAT=                 / Reference Catalog used for QA
    QAALNNUM=                 / Number of objects for QA [integer]
    QAALNRMS=                 / RMS of misalignment with QAREFCAT [arcsec]
    QAALNSTD=                 / Uncertainty of misalignment [arcsec]
    QANSECT =                 / Total num of divided sections for QA [integer]
    QAGDSECT=                 / Number of sections classified as good [integer]
    QABDSECT=                 / Positions of sections classified as bad
    QABADAMP=                 / True if bad AMP exists
    QARESULT=                 / True if QA is good 
    """

    __version__ = '1.4.0' 

    # ====== IMPORTS ========================================================
    import os
    import warnings
    import numpy as np
    import pandas as pd
    import astroscrappy as cr
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
    def bleed_masking(datapath, skysigcut=1.5, convergence=5, bpidx_thres=3000, saturate=52000):
        
        import numpy as np
        from astropy.stats import sigma_clipped_stats
        from astropy.io import fits
        
        Msg.bleedmap()
        hdul    = fits.open(datapath)
        data    = hdul[0].data
        hdr     = hdul[0].header
        leny, lenx = data.shape
        bpMask = np.zeros_like(data)
        _, med, sig = sigma_clipped_stats(data)
        signal_thres = med + skysigcut * sig

        chip    = os.path.basename(datapath).split(".")[1]
        # Determine the direction of masking based on the chip type
        if chip in ['kk', 'nn']:
            direction = 'downward'
        elif chip in ['mm', 'tt']:
            direction = 'upward'
        else:
            raise ValueError("Unknown chip type")

        for i in range(lenx):
            col = data[:, i]
            sat_indices = np.where(col > saturate)[0]

            for y_idx in sat_indices:
                if bpMask[y_idx, i] == 8:
                    continue  # Skip already marked pixels

                # Calculate the sum of pixel values in the vicinity to determine if there is actual bleeding
                if direction == 'downward':
                    ystart = max(y_idx - 40, 0)
                    yend = max(y_idx - 20, 0)
                else:  # 'upward'
                    ystart = min(y_idx + 20, leny - 1)
                    yend = min(y_idx + 40, leny - 1)

                if ystart < yend:
                    bpidx = np.sum(data[ystart:yend+1, i])
                    ylength = yend - ystart + 1
                    if bpidx - med * ylength > bpidx_thres:
                        bpMask[y_idx, i] = 8  # Set mask only if the condition is met
                        revert_pixel = 0

                        # Define scanning range based on direction
                        range_start, range_end, step = (y_idx, -1, -1) if direction == 'downward' else (y_idx, leny, 1)

                        # Scan through the column in the specified direction
                        for j in range(range_start, range_end, step):
                            if col[j] > signal_thres:
                                bpMask[j, i] = 8
                                revert_pixel = 0
                            else:
                                bpMask[j, i] = 8
                                revert_pixel += 1

                            # Stop marking when enough consecutive small values are found
                            if revert_pixel >= convergence:
                                if direction == 'downward':
                                    end_idx = max(j - convergence, 0)
                                else:
                                    end_idx = min(j + convergence, leny)
                                bpMask[j:end_idx, i] = 0
                                break
        # Save the mask
        # fits.PrimaryHDU(data=bpMask, header=hdr).writeto(datapath.replace('.fits', '.bmask.fits'), overwrite=True)
        return bpMask
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
        c1,c2=cr.detect_cosmics(
            data,
            gain    = 1.0,
            readnoise= 10,
            sigclip = 4.5,
            sigfrac = 0.3,
            objlim  = 5.0,
            niter   = 2,
            cleantype= 'medmask',
            fsmode  = 'median',
            verbose = True
        )

        # Cross-talk masking
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
            bpMask  = bleed_masking(fname)
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
        prompt_flg  = f'-FLAG_IMAGE {flagname}'
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
        if weightphot:
            weightname = mask2weight(flagname)
            prompt_wgt = f' -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE {weightname} -RESCALE_WEIGHTS Y -WEIGHT_GAIN Y'
        else:
            prompt_wgt = ''

        prompt  = f'sex {fname} {prompt_cfg} {prompt_cat} {prompt_opt} {prompt_flg} {prompt_chk} {prompt_wgt}'
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
            Table.from_pandas(refcat).write(os.path.join(refcatdir, refcatname, trgt))

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
    def anlz_gbmap(mref, mrefcut, divnum) :

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
                ]['sep']
                sep = mrefcut[
                    step(
                        mrefcut['XWIN_IMAGE'], 
                        mrefcut['YWIN_IMAGE']
                    )
                ]['sep']

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
                        rmsalign  = sqrt(mean((sep)**2))
                        alignstd  = std(sep)
                        sect_astrom = ('good' if (dtctRatio > 0.6 and 
                                                  rmsalign < 0.5) 
                                       else 'bad')

                df_sect.loc[8*j + i] = [len(sep), 
                                        rmsalign, 
                                        alignstd, 
                                        sect_astrom, 
                                        dtctRatio]

        # Analysis report
        gbmap_row = list(df_sect['astrometry'])
        bad_sect = [i for i, x in enumerate(gbmap_row) if x == 'bad']
        empty_sect = [i for i, x in enumerate(gbmap_row) if x == 'empty']
        fastrom = 'good' if (gbmap_row.count('bad') <= 2) and (gbmap_row.count('empty') <= 10) else 'bad'
        badamp_ls = [[j+i*8 for i in range(8)] for j in range(8)]
        badamp_bl = True if empty_sect in badamp_ls else False
        result = [fname, fastrom, bad_sect]

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
                        anlz_rprt       = anlz_gbmap(mref, mrefcut, divnum)
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
                param_matching  = dict(intbl   = data,
                                    reftbl  = reftbl,
                                    inra    = data['ALPHA_J2000'], 
                                    indec   = data['DELTA_J2000'],
                                    refra   = reftbl['RA'], 
                                    refdec  = reftbl['DEC'],
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
    
    """_summary_
    Description
        The zeropoint is scaled to ensure uniform photometric quality of KMTNet images. 
        The discrete levels of photometric zeropoints of each amplifier are corrected.
        If there is recurring electronic pattern noise (e.g. SAAO), it will be also removed.
        Methods
        (1) 1DLinear: Each amp's Y-axis zp tendency compensation by linear fitting 
        (2) 2DPolynomial: Each amp's X,Y plane zp tendency compensation by polynomial fitting
        The scaling process is basically done amp by amp. This is because the zp is discrete for each amp at least slightly.
    Args:
        date        = '240915_CTIO'
        img         = f'/data4/kmtntoo/tutorial/data/raw/{date}/025862.kk.fits'
        path_output = f'/data4/kmtntoo/tutorial/data/scaled/{date}/'
        path_cfg    = '/data4/kmtntoo/tutorial/config/'
        path_cat    = '/data4/kmtntoo/tutorial/catalog/'
        path_plot   = '/data4/kmtntoo/tutorial/result/plot/'
        mode        = '1DLINEAR'
        # mode        = '2DPOLYNOMIAL'
        magkey      = 'AUTO'
        zpscaled    = 30.0
        pixscale    = 0.4
        gain        = 1.0
        figure      = False
        start       = None
        gridcat     = 'kmtnet_grid.cat'
    Returns:
        _type_: _description_
        
    What should be prepared in advance:
        ToOampcop.cat in the working directory including FWHM data
        SMSS or APASS reference catalog in pathcat/{survey}
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
    if os.path.exists(img):
        file    = fits.open(img)
        serial  = os.path.basename(img).split('.')[0]
        hdr     = fits.getheader(img)
    else:
        raise FileNotFoundError(f'No such file or directory: {img}')
    
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
        ks4cat  = Table.read(os.path.join(path_cfg, gridcat))
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
    prompt      = 'sex '+inim_single+prompt_cfg+prompt_aper+prompt_opt+prompt_cat+prompt_chk+prompt_flg+prompt_bkg+prompt_wgt
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
        magerr  = sqsum([cmtbl[f'MAGERR_{magkey}'], cmtbl[f'e_{band}mag']])
      
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
            
            print(f'AMP{i+1} Flux Scaling: {m:.2e} * y + {b:.2f}')
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
    img = '/data4/kmtntoo/tutorial/data/scaled/240423_CTIO/S240422ed_0749.121-30.R.20240423.CTIO.052188.kk.scaled.fits'
    path_cfg = '/data4/kmtntoo/tutorial/config/'
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
    bpmdata[bpmdata != 0] = 16
    naxis1 = hdr['NAXIS1']
    bound = np.arange(0, naxis1 + 1, naxis1 // 8)

    # Vectorize the masking process for bad amps
    bads = 0
    for i, amp in enumerate(badamp):
        if amp == '1':
            mdata[:, bound[i]:bound[i+1]] = 4
            bads += 1
    
    mbpmdata = mdata + bpmdata
    fits.PrimaryHDU(data=mbpmdata.astype(np.int16), header=fits.getheader(msk)).writeto(msk, overwrite=True)

    with fits.open(msk, 'update') as m:
        for hdu in m:
            hdu.header['CRMASK'] = (1, 'Cosmic-ray marked as 1')
            hdu.header['XTMASK'] = (2, 'Crosstalk region marked as 2')
            hdu.header['BAMPMASK'] = (4, 'Bad amplifier marked as 4')
            hdu.header['BLDMASK'] = (8, 'Bleeding pattern marked as 8')
            hdu.header['BPMASK'] = (16, 'CCD badpixel marked as 16')
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
    Descriptions:
        Coadds astronomical images using SWarp, applying masks and updating header information.

    Parameters:
        filename_convention (str): Regular expression pattern to match filenames against.
        path_input (str): Path to the input directory containing FITS files.
        path_output (str): Path where the output stacked images will be saved.
        path_cfg (str): Path to the configuration directory for SWarp and other tools.
        path_ref (str): Path to the reference or template images.
        combinetype (str, optional): Type of pixel combination method to use in SWarp. Defaults to 'MEDIAN'.
        start (float, optional): Start time of the operation for performance measurement. Defaults to None.
        gridcat (str, optional): Filename of the predescribed coordinate grid file for KMTNet pointings. Defaults to 'kmtnet_grid.cat'.

    Returns:
        int: Total number of stacked image sets processed.

    date     = '220328_DWF'
    filename_convention = r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<band>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<serial>\d{6})\.(?P<chip>\w+)\.(?P<type>scaled|mask)\.fits"

    path_input  = f'/data4/kmtntoo/tutorial/data/scaled/{date}/'
    path_output = f'/data4/kmtntoo/tutorial/data/stack/{date}/'
    path_cfg    = '/data4/kmtntoo/tutorial/config/'
    path_ref    = '/data4/kmtntoo/tutorial/data/template/'
    start       = time.time()
    combinetype = 'MEDIAN'
    gridcat     = 'kmtnet_grid.cat'
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
            
            ks4cat  = ascii.read(os.path.join(path_cfg, gridcat))
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
    Input: SExtractor output catalog from qa4stackpro
    Output: mag zero-point, FWHM, 5sigma depth and ref matched catalog
    cat = r"(?P<field>\w+_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<filter>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<exptime>\d+sec)\.(?P<type>stack)\.fits\.cat"
    
    date        = '240918_SAAO'
    cat = f'/data4/kmtntoo/tutorial/data/stack/{date}/S240915b_2147.109-50.R.20240918.SAAO.960sec.stack.fits.cat'
    path_output = f'/data4/kmtntoo/tutorial/data/stack/{date}/'
    path_cat    = f'/data4/kmtntoo/tutorial/catalog/'
    flagcut     = 0
    pixscale    = 0.4
    clsstar     = 0.8
    refmaglower = 14
    refmagupper = 18
    apertures   = ['APER', 'AUTO']
    figure      = True
    path_plot   = './'
    start       = None
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
        reftbl  = ascii.read(f'{path_ref}apass_{field}.cat')
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
                                inmagkey=inmagkey, inmagerkey=inmagerkey,
                                refmagkey=refmagkey, refmagerkey=refmagerkey,
                                sigma=2.0)

        zp, zper, otbl, xtbl = zpcal(**param_zpcal)
        intbl[f'MAG_{aperture}']    = np.array([min(intbl[f'MAG_{aperture}'][i]+round(zp, 3),99.0) for i in range(len(intbl))])
        intbl[f'MAGERR_{aperture}'] = sqsum([intbl[f'MAGERR_{aperture}'], round(zper, 3)])

        if aperture == 'AUTO':
            mtbl = mtbl[mtbl['FLAGS']<=flagcut] # flag cut
            mtbl = mtbl[mtbl['CLASS_STAR']>clsstar] # stellarity cut
            mtbl = mtbl[mtbl['{}mag'.format(band)] < refmagupper] #mag cut
            mtbl = mtbl[mtbl['{}mag'.format(band)] > refmaglower]
            
            magdif  = mtbl[f'MAG_{aperture}'] + zp - mtbl['{}mag'.format(band)]
            magerr  = sqsum([mtbl[f'MAGERR_{aperture}'], zper])
            
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
def subtraction(sciimg, path_ref, path_cat, path_refcat, path_output, path_config, div_col=4, div_row=4, pixscale=0.4, ncore=1, detect=1.5, cutsize=1.0, psf=True, align=False):
    """
    Input: Photometric catalog from KMTNet_ToO.catalogmaker()
    Input: Stacked KMTNet image after astrometry & photometry (Science & Reference)
    Output: science, reference and subtraction snapshot images for transient candidates
    sciimg = r"(?P<field>\w+_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<filter>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<exptime>\d+sec)\.(?P<type>stack)\.fits"

    Args:
    date        = '190816_SSO'
    simg        = 'G331903-10-2_9028.021-31.R.20190815.SSO.480sec.stack.fits'
    sciimg      = f'/data4/kmtntoo/tutorial/data/stack/{date}/{simg}'
    path_ref    = '/data4/kmtntoo/tutorial/data/template/'
    path_cat    = f'/data4/kmtntoo/tutorial/data/stack/{date}/'
    path_refcat = '/data4/kmtntoo/tutorial/data/template/'
    path_output = f'/data4/kmtntoo/tutorial/data/subt/{date}/'
    path_config = '/data4/kmtntoo/tutorial/config/'
    div_col     = 4
    div_row     = 4
    pixscale    = 0.4
    ncore       = 4
    detect      = 1.5
    cutsize     = 1.0
    align       = False
    """

    import shutil
    import numpy as np
    import multiprocessing
    import re, os, glob, copy
    import astropy.units as u
    from astropy.wcs import WCS
    from itertools import repeat
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
    pattern_ks4 = os.path.join(path_ref, f'{field}.{radec}', f'ks4*{band}*.scaled.stack.fits')
    refimg = find_longest_exposure_image(pattern_ks4)

    if refimg is None:
        pattern_ps1 = os.path.join(path_ref, f'{field}.{radec}', f'ps1*{band}*.scaled.stack.fits')
        refimg = find_longest_exposure_image(pattern_ps1)
        
    if refimg:
        print(f"Reference image: \n{os.path.basename(refimg)}")
    else:
        print(f'No suitable reference image found. Checked patterns: \n{pattern_ks4}\n{pattern_ps1}')

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
        print(f"Check if the catalog is in the {path_cat} directory.")
        stamp   = copy.deepcopy(scicat)
        stamp   = stamp[stamp['SNR_WIN']>20]
        if len(stamp)<200: stamp = None

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
            # shutil.copy(REFIMG, CONV_REFIMG)
    except KeyError:
        convdir     = 't'
        CONVIMG = CONV_REFIMG
    #   HOTPANTs Running
    hotpants(inim=SCIIMG, refim=REFIMG, outim=SUBTIMG, inmsk=MASKIMG, refmsk=MASKIMG, convim=CONVIMG, stamp=stamp, nrx=div_col, nry=div_row, convdir=convdir)
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
    try:
        fwhmcom     = f' -SEEING_FWHM {fits.getheader(SCIIMG)["FWHM"]}'
    except:
        fwhmcom     = ''
    maskcom     = f' -FLAG_IMAGE {MASKIMG} -FLAG_TYPE OR'
    WEIGHTIMG   = mask2weight(MASKIMG)
    weightcom   = f' -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE {WEIGHTIMG} -RESCALE_WEIGHTS Y -WEIGHT_GAIN Y'
    #    Photometry on the subtracted image & inverted subt image
    if psf==True:
        conf_param  = os.path.join(path_config, 'kmtnet.param')
        os.system(sexcom(SCIIMG, conf_sex, conf_param, conf_conv, conf_nnw, det_thres=20)+maskcom+fwhmcom+' -CATALOG_TYPE FITS_LDAC')
        if os.path.isfile(SCIIMG.replace('.fits', '.cat')):
            os.system(f'psfex {SCIIMG.replace(".fits", ".cat")} -c {path_config}kmtnet.psfex')
            conf_param  = os.path.join(path_config, 'kmtnet_psf.param')
            os.system(sexcom(SUBTIMG, conf_sex, conf_param, conf_conv, conf_nnw, det_thres=detect)+maskcom+fwhmcom+weightcom+f' -PSF_NAME {SCIIMG.replace(".fits", ".psf")}')

    else:
        os.system(sexcom(SUBTIMG, conf_sex, conf_param, conf_conv, conf_nnw, det_thres=detect)+maskcom+fwhmcom+weightcom)#, detectiondual=trim_SCIIMG))
    INV_SUBTIMG = SUBTIMG.replace("hd", "invhd")
    invert_image(inim=SUBTIMG, outim=INV_SUBTIMG)
    os.system(sexcom(INV_SUBTIMG, conf_sex, conf_param, conf_conv, conf_nnw, det_thres=detect)+maskcom+fwhmcom+weightcom)#, detectiondual=trim_SCIIMG))




    subtbl      = ascii.read(SUBTIMG.replace(".fits", ".psf.cat"))
    subtbl      = ascii.read(SUBTIMG.replace(".fits", ".cat"))
    invsubtbl   = ascii.read(INV_SUBTIMG.replace(".fits", ".cat"))

    print(f"# Number of sources: {len(subtbl)}")
    subtbl['inim']  = SCIIMG
    subtbl['hcim']  = CONVIMG
    subtbl['hdim']  = SUBTIMG
    subtbl['mask']  = MASKIMG
    subtbl.meta['SEEING']   = np.median(scicat['FWHM_IMAGE']*0.4)
    subtbl['ratio_seeing']  = subtbl['FWHM_WORLD']/np.median(scicat['FWHM_WORLD'])
    subtbl.meta['ELLIPTICITY']  = np.median(scicat['ELLIPTICITY'])
    scicat['ELONGATION'] = 1 / (1-scicat['ELLIPTICITY'])
    subtbl['ratio_ellip']   = subtbl['ELLIPTICITY']/np.median(scicat['ELLIPTICITY'])
    subtbl['ratio_elong']   = subtbl['ELONGATION']/np.median(scicat['ELONGATION'])
    subtbl['MAG_AUTO']      = subtbl['MAG_AUTO'] + magautozero
    invsubtbl['MAG_AUTO']   = invsubtbl['MAG_AUTO'] + magautozero

    w = WCS(SUBTIMG)
    #    Positional information
    c_cent = w.pixel_to_world(scihdr['NAXIS1']/2, scihdr['NAXIS2']/2)
    c_sub = SkyCoord(subtbl['ALPHA_J2000'], subtbl['DELTA_J2000'], unit=u.deg)

    flagnumbers = np.arange(9)
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
                raise  # Re-raise the exception for any other RuntimeError
        except ConnectionError as e:
            print(f"Connection failed on attempt {attempt+1} of {max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # Wait for a bit before retrying
            else:
                raise  # Re-raise the exception if the final attempt fails
    #------------------------------------------------------------
    #    flag 1: Inverted Image Detections (Artifacts Around the Source)
    #------------------------------------------------------------
    if len(invsubtbl)>0:
        #    Coordinate
        # invsubtbl = invsubtbl[invsubtbl['SNR_WIN']>20]
        c_invhd = SkyCoord(invsubtbl['ALPHA_J2000'], invsubtbl['DELTA_J2000'], unit=u.deg)
        #    Matching with inverted images
        indx_invhd, sep_invhd, _ = c_sub.match_to_catalog_sky(c_invhd)
        subtbl['flag_1'][(sep_invhd.arcsec<subtbl['FWHM_IMAGE']) & (np.abs((subtbl['MAG_AUTO'] - invsubtbl[indx_invhd]['MAG_AUTO'])) <= 1)] = True
        # subtbl['flag_1'][(sep_invhd.arcsec<subtbl['FWHM_IMAGE'])] = True
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
    #    flag 6: Too Low SNR
    #------------------------------------------------------------
    snrcut  = 5         # flag6
    subtbl['flag_6'][(subtbl['SNR_WIN']<snrcut)] = True
    #------------------------------------------------------------
    #    flag 7
    #------------------------------------------------------------
    data = fits.getdata(SUBTIMG)
    peeing = subtbl.meta['SEEING']/pixscale
    skyval = np.median(subtbl['BACKGROUND'])
    skysig = np.std(subtbl['BACKGROUND'])

    subtbl['n_bad'] = 0
    subtbl['ratio_bad'] = 0.0
    subtbl['n_null'] = 0
    
    #    Fraction
    f = 0.1
    for i, (tx, ty, bkg) in enumerate(zip(subtbl['X_IMAGE'], subtbl['Y_IMAGE'], subtbl['BACKGROUND'])):

        #    Snapshot
        tsize = peeing
        y0, y1 = int(ty-tsize), int(ty+tsize)
        x0, x1 = int(tx-tsize), int(tx+tsize)
        cdata = data[y0:y1, x0:x1]
        crt = bkg - skysig*25
        cutline = cdata.size*f
        nbad = len(cdata[cdata<crt])
        try:
            ratiobad = nbad/cdata.size
        except:
            ratiobad = -99.0
        nnull = len(np.where(cdata == 1e-30)[0])
        #    Dipole
        if nbad > cutline or nnull != 0:
            subtbl['flag_7'][i] = True

        subtbl['n_bad'][i] = nbad
        subtbl['ratio_bad'][i] = ratiobad
        subtbl['n_null'][i] = nnull

    #------------------------------------------------------------
    #    flag 8: HOTPANTs Chi2 Value
    #------------------------------------------------------------
    subthdr     = fits.getheader(SUBTIMG)
    for i in range(div_col*div_row):
        if float(subthdr[f'X2NRM{str(i).zfill(2)}']) > 1000:
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
    #    flag 9: GLADE+ Galaxy Matching
    #------------------------------------------------------------
    #------------------------------------------------------------
    #    flag 10: R/B Classification from Machine Learning
    #------------------------------------------------------------
    #------------------------------------------------------------
    #    Final flag
    #------------------------------------------------------------
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
    
    # print(f"#\tSnapshot maker ({len(trtbl)})")
    # if len(trtbl) > 0:
    #     if ncore == 1:
    #         for i in range(len(trtbl)):
    #             generate_snapshot(trtbl, i, cutsize)
    #     #    Multi Thread
    #     else:
    #         with multiprocessing.Pool(processes=ncore) as pool:
    #             results = pool.starmap(generate_snapshot, zip(repeat(trtbl), np.arange(len(trtbl)), repeat(cutsize)))
    # else:
    #     print('No transient candidates.')
        
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
            label = f"text={{{row[label_column]}}}" if label_column else ""
            
            # Generate the region based on the chosen shape
            if shape == 'circle':
                regionfile.write(f"circle({ra},{dec},{size}) #{label}\n")
            elif shape == 'ellipse':
                regionfile.write(f"ellipse({ra},{dec},{size},{size/2},0) #{label}\n")  # Semi-major axis is size, semi-minor axis is half size
            elif shape == 'box':
                regionfile.write(f"box({ra},{dec},{size},{size},{0}) #{label}\n")  # Width and height as size
            else:
                regionfile.write(f"point({ra},{dec}) # point={shape} {label}\n")  # You can replace 'cross' with 'diamond', 'box', etc.
#%%
def csv_to_ds9reg(input_path, region_path='default', wcs_option=True, color='green', size=50, shape='circle', label_column=None, x_column='X_IMAGE', y_column='Y_IMAGE'):
    """
    Converts a CSV or ASCII file with source locations into a DS9 region file with customizable shapes and optional labels.
    
    Args:
    - input_path: Path to the input file (CSV or ASCII format).
    - region_path: Path to the output DS9 region file (defaults to replacing '.csv' or '.cat' with '.reg').
    - wcs_option: If True, use WCS for conversion if a corresponding FITS file exists.
    - color: Color of the regions in the DS9 region file (default is 'green').
    - size: Size of the region (radius for circles, semi-major axis for ellipses, width for boxes, etc.).
    - shape: Shape of the region ('circle', 'ellipse', 'box', 'point'). Default is 'circle'.
    - label_column: Column name in the input file to use as the label for each region (default is None, meaning no labels).
    - x_column: Column name for X-coordinate in the input file (default is 'X_IMAGE').
    - y_column: Column name for Y-coordinate in the input file (default is 'Y_IMAGE').
    """
    
    import os
    import csv
    from astropy.wcs import WCS
    from astropy.io import fits, ascii

    # Determine the output region file path
    if region_path == 'default':
        region_path = input_path.replace('.csv', '.reg').replace('.cat', '.reg')
    
    image_path = input_path.replace('.csv', '.fits').replace('.cat', '.fits')
    
    wcs = None
    # Check if corresponding FITS file exists and wcs_option is True
    if wcs_option and os.path.exists(image_path):
        with fits.open(image_path) as hdul:
            wcs = WCS(hdul[0].header)
    
    # Detect if the file is CSV or ASCII based on extension or content
    is_csv = input_path.endswith('.csv')
    if is_csv:
        # Read as CSV
        reader = csv.DictReader(open(input_path, newline=''))
    else:
        # Read as ASCII (SExtractor .cat format)
        table = ascii.read(input_path, format='ascii')
        reader = table  # astropy Table structure

    with open(region_path, 'w') as regionfile:
        # Write the header for DS9 region file
        regionfile.write("# Region file format: DS9 version 4.1\n")
        regionfile.write(f"global color={color} dashlist=8 3 width=2 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        
        if wcs is not None:
            regionfile.write("fk5\n")  # Use fk5 for celestial coordinates
        else:
            regionfile.write("physical\n")  # Use physical for pixel coordinates

        # Write each source as a region with the specified shape and optional label
        for row in reader:
            if is_csv:
                # For CSV: read directly from dictionary-like rows
                x = float(row[x_column])
                y = float(row[y_column])
                label = f"text={{{row[label_column]}}}" if label_column and label_column in row else ""
            else:
                # For ASCII (e.g., SExtractor .cat file)
                x = float(row[x_column])
                y = float(row[y_column])
                label = f"text={{{row[label_column]}}}" if label_column and label_column in row.colnames else ""
            
            if wcs is not None:
                # Convert pixel coordinates to RA and DEC if WCS is available
                ra, dec = wcs.pixel_to_world_values(x, y)
            else:
                # Use pixel coordinates directly
                ra, dec = x, y

            # Generate the region based on the chosen shape
            if shape == 'circle':
                regionfile.write(f"circle({ra},{dec},{size}) #{label}\n")
            elif shape == 'ellipse':
                regionfile.write(f"ellipse({ra},{dec},{size},{size/2},0) #{label}\n")  # Semi-major axis is size, semi-minor axis is half size
            elif shape == 'box':
                regionfile.write(f"box({ra},{dec},{size},{size},{0}) #{label}\n")  # Width and height as size
            else:
                regionfile.write(f"point({ra},{dec}) # point={shape} {label}\n")  # You can replace 'cross' with 'diamond', 'box', etc.
#%%
subtbl.write(f'{path_output}all_detection.csv', format='csv', overwrite=True)
csv_to_ds9reg(f'{path_output}all_detection.csv', color='black', shape='box', size=25, label_column='NUMBER')
# injected sources
injected = ascii.read(f'{path_output}injected_PSF.csv')
injected['MAG_APER'] = np.round(injected['MAG_APER'], 2)
injected.write(f'{path_output}injected_PSF.csv', format='csv', overwrite=True)
csv_to_ds9reg(f'{path_output}injected_PSF.csv', color='green', shape='box', label_column='MAG_APER')
# rb classification results (ri+ngi+gd)
tranrb = ascii.read(f'{path_output}hdCalib.KMTNet_SSO.G331903_9027.20190815-164806.R.480.stack.transientRB_ri+ngi+gd.cat')
rb05 = tranrb[tranrb['prob']>0.5]
rb05['prob'] = np.round(rb05['prob'], 2)
rb05.write(f'{path_output}RB05_ri+ngi+gd.csv', format='csv', overwrite=True)
csv_to_ds9reg(f'{path_output}RB05_ri+ngi+gd.csv', color='red', shape='ellipse', label_column='prob', size=50)
# rb classification results (ri)
tranri = ascii.read(f'{path_output}hdCalib.KMTNet_SSO.G331903_9027.20190815-164806.R.480.stack.transientRB_ri.cat')
tranrb['prob_ri'] = tranri['prob']
ri05 = tranri[tranri['prob']>0.5]
ri05['prob'] = np.round(ri05['prob'], 2)
ri05.write(f'{path_output}RB05_ri.csv', format='csv', overwrite=True)
csv_to_ds9reg(f'{path_output}RB05_ri.csv', color='white', shape='ellipse', size=100, label_column='prob')
# flags
flag1 = tranrb[tranrb['flag_1']=='True']
flag2 = tranrb[tranrb['flag_2']=='True']
flag3 = tranrb[tranrb['flag_3']=='True']
flag4 = tranrb[tranrb['flag_4']=='True']
flag5 = tranrb[tranrb['flag_5']=='True']
flag6 = tranrb[tranrb['flag_6']=='True']
flag7 = tranrb[tranrb['flag_7']=='True']
flag8 = tranrb[tranrb['flag_8']=='True']
flag1.write(f'{path_output}flag1.csv', format='csv', overwrite=True)
flag2.write(f'{path_output}flag2.csv', format='csv', overwrite=True)
flag3.write(f'{path_output}flag3.csv', format='csv', overwrite=True)
flag4.write(f'{path_output}flag4.csv', format='csv', overwrite=True)
flag5.write(f'{path_output}flag5.csv', format='csv', overwrite=True)
flag6.write(f'{path_output}flag6.csv', format='csv', overwrite=True)
flag7.write(f'{path_output}flag7.csv', format='csv', overwrite=True)
flag8.write(f'{path_output}flag8.csv', format='csv', overwrite=True)
csv_to_ds9reg(f'{path_output}flag1.csv', color='pink', shape='cross')   # inverted detection
csv_to_ds9reg(f'{path_output}flag2.csv', color='#456', shape='x')       # SEx flag
csv_to_ds9reg(f'{path_output}flag3.csv', color='#789', shape='diamond') # ellipticity
csv_to_ds9reg(f'{path_output}flag4.csv', color='#abc', shape='arrow')   # FWHM
csv_to_ds9reg(f'{path_output}flag5.csv', color='#af6', shape='arrow')   # background
csv_to_ds9reg(f'{path_output}flag6.csv', color='#def', shape='circle', size=10) # SNR
csv_to_ds9reg(f'{path_output}flag8.csv', color='#fed', shape='x')       # bad pixel value
csv_to_ds9reg(f'{path_output}flag7.csv', color='#cba', shape='box', size=10) # HOTPANTS chi2
# unfiltered sources
candtbl = subtbl[subtbl['flag']==False]
candtbl.write(f'{path_output}Unfiltered.csv', format='csv', overwrite=True)
csv_to_ds9reg(f'{path_output}Unfiltered.csv', color='yellow')
# %%
from astropy.wcs import WCS
from astropy.table import hstack
hdulist = fits.open(SCIIMG)
wcs = WCS(hdulist[0].header)
ras     = []
decs    = []
for c in injected:
    ra, dec = wcs.all_pix2world(c["X_IMAGE"], c["Y_IMAGE"], 1)
    coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
    ras.append(coord.ra.value)
    decs.append(coord.dec.value)
    print(f'{coord.ra.to_string(unit="deg", sep=":")} {coord.dec.to_string(sep=":")}')
injected    = hstack([injected, Table([ras,decs], names=['RA','DEC']) ])
#%%
subtbl['prob_rng'] = tranrb['prob']
subtbl['prob_ri'] = tranri['prob']
injectbl    = injected[injected['MAG_APER']<21] # PSF injected sources (bright enough to be detected)
c_sub = SkyCoord(subtbl['ALPHA_J2000'], subtbl['DELTA_J2000'], unit=u.deg)
c_inj = SkyCoord(injectbl['RA'], injectbl['DEC'], unit=u.deg)
indx_inj, sep_inj, _ = c_sub.match_to_catalog_sky(c_inj)
indx_sub, sep_sub, _ = c_inj.match_to_catalog_sky(c_sub)
recov = subtbl[(sep_inj.arcsec<2)] # Unfiltered && Real
recov = recov[recov['flag']==False] # Unfiltered && Real
depos = subtbl[(sep_inj.arcsec>2)] # Unfiltered && Bogus
depos = depos[depos['flag']==False] # Unfiltered && Bogus
missed = subtbl[(sep_inj.arcsec<2)] # Filtered && Real
missed = missed[missed['flag']==True] # Filtered && Real
dump = subtbl[(sep_inj.arcsec>2)] # Filtered && Bogus
dump = dump[dump['flag']==True] # Filtered && Bogus
print(f"Recovery Rate: {len(recov)} / {len(injectbl)} = {len(recov)/len(injectbl):.2f}")
print(f"Recovery Rate: {len(recov)} / {len(subtbl[(sep_inj.arcsec<2)])} = {len(recov)/len(subtbl[(sep_inj.arcsec<2)]):.2f}")
flagdescripts = [
    "asteroid",
    "inverted ",
    "flagging",
    "ellipticity",
    "large fwhm",ㅂ
    "background",
    "low SNR",
    "extreme value",
    "HOTPANTS chi2",
]
for i in range(9):
    print(f"{flagdescripts[i]}:\t {len(missed[missed[f'flag_{i}']])}")
    # %%
print(f'{len(recov)} \t {len(depos)} \t {100*len(recov)/len(trtbl):.1f}%')
print(f'{len(missed)} \t {len(dump)} \t {100*len(missed)/(len(missed)+len(dump)):.1f}%')
print(f'{100*len(recov)/len(subtbl[(sep_inj.arcsec<2)]):.1f}% \t {100*len(depos)/(len(depos)+len(dump)):.1f}% \t {len(subtbl)}')

# %%
for table in [recov, depos, missed, dump]:
    plt.figure(figsize=(6, 3))
    # Calculate the histograms without density normalization
    riweight = 100*np.ones_like(table['prob_ri'])/len(table['prob_ri'])
    plt.hist(table['prob_ri'], bins=np.arange(0, 1.1, 0.1), fill=False, edgecolor='crimson', hatch='/', linewidth=2, label=f'Model_RI ({100*len(table[table["prob_ri"]>0.5])/len(table):.1f}%)', weights=riweight)
    plt.hist(table['prob_rng'], bins=np.arange(0, 1.1, 0.1), fill=False, edgecolor='dodgerblue', hatch='\\', linewidth=2, label=f'Model_RI+NG+GD ({100*len(table[table["prob_rng"]>0.5])/len(table):.1f}%)', weights=riweight)
    # optval = np.maximum(table['prob_ri'], table['prob_rng'])
    # plt.hist(optval, bins=np.arange(0, 1.1, 0.1), fill=False, edgecolor='forestgreen', hatch='xx', linewidth=2, label=f'Selecting Higher Prob ({100*len(optval[optval>0.5])/len(table):.1f}%)', weights=riweight)
    plt.axvline(0.5, linewidth=2, ls='--', c='k')
    plt.xlabel('R/B Score')
    plt.ylabel('Density (%)')
    plt.ylim(0,100)
    plt.xlim(0,1)
    plt.legend()
    plt.show()
# %%
os.chdir(path_output)
q1 = recov[recov['prob_rng']<0.2]
q1 = recov[recov['prob_ri']<0.2]
q2 = depos[depos['prob_rng']>0.8]
q3 = missed[missed['prob_rng']<0.2]
q4 = dump[dump['prob_rng']>0.8]

os.system('mkdir q1 q2 q3 q4')
qdirs = 'q1 q2 q3 q4'.split()
#%%
for i, qtable in enumerate([q1, q2, q3, q4]):
    print(i)
    for q in qtable:
        os.system(f'cp *.{q["NUMBER"]:06d}.???.fits {qdirs[i]}')
# %%
q1['prob_rng'] = np.round(q1['prob_rng'], 2)
q1['prob_ri'] = np.round(q1['prob_ri'], 2)
q2['prob_rng'] = np.round(q2['prob_rng'], 2)
q3['prob_rng'] = np.round(q3['prob_rng'], 2)
q4['prob_rng'] = np.round(q4['prob_rng'], 2)
q1.write(f'{path_output}real_unfilter.csv', format='csv', overwrite=True)
csv_to_ds9reg(f'{path_output}real_unfilter.csv', color='green', shape='box', label_column='prob_rng')   # inverted detection
csv_to_ds9reg(f'{path_output}real_unfilter.csv', color='pink', shape='x', label_column='prob_ri')   # inverted detection
q2.write(f'{path_output}bogus_unfilter.csv', format='csv', overwrite=True)
csv_to_ds9reg(f'{path_output}bogus_unfilter.csv', color='orange', shape='diamond', label_column='prob_rng')   # inverted detection
q3.write(f'{path_output}real_filter.csv', format='csv', overwrite=True)
csv_to_ds9reg(f'{path_output}real_filter.csv', color='blue', label_column='prob_rng')   # inverted detection
q4.write(f'{path_output}bogus_filter.csv', format='csv', overwrite=True)
csv_to_ds9reg(f'{path_output}bogus_filter.csv', color='red', label_column='prob_rng')   # inverted detection
# %%

real = vstack([recov, missed])
bogus = vstack([depos, dump])
plt.hist(bogus['SPREAD_MODEL'], bins=np.arange(-100,100,1))
plt.hist(real['SPREAD_MODEL'], bins=np.arange(-100,100,1))
plt.xlim(-100,100)
# plt.semilogx()
#%%, bins=np.arange(0, 1.1, 0.1), fill=False, edgecolor='crimson', hatch='/', linewidth=2, label=f'Model_RI ({100*len(table[table["prob_ri"]>0.5])/len(table):.1f}%)', weights=riweight)
plt.hist(table['prob_rng'], bins=np.arange(0, 1.1, 0.1), fill=False, edgecolor='dodgerblue', hatch='\\', linewidth=2, label=f'Model_RI+NG+GD ({100*len(table[table["prob_rng"]>0.5])/len(table):.1f}%)', weights=riweight)
#%%
# configuration files
    conf_sex    = os.path.join(path_config, 'kmtnet.sex')
    conf_param  = os.path.join(path_config, 'kmtnet_imask.param')
    conf_nnw    = os.path.join(path_config, 'kmtnet.nnw')
    conf_conv   = os.path.join(path_config, 'kmtnet.conv')
    try:
        fwhmcom     = f' -SEEING_FWHM {fits.getheader(SCIIMG)["FWHM"]}'
    except:
        fwhmcom     = ''
    maskcom     = f' -FLAG_IMAGE {MASKIMG} -FLAG_TYPE OR'
    #    Photometry on the subtracted image & inverted subt image
    INV_SUBTIMG = SUBTIMG.replace("hd", "invhd")
    invert_image(inim=SUBTIMG, outim=INV_SUBTIMG)
    WEIGHTIMG   = mask2weight(MASKIMG)
    weightcom   = f' -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE {WEIGHTIMG} -RESCALE_WEIGHTS Y -WEIGHT_GAIN Y'
    os.system(sexcom(SUBTIMG, conf_sex, conf_param, conf_conv, conf_nnw, det_thres=detect)+maskcom+fwhmcom+weightcom)#, detectiondual=trim_SCIIMG))
    os.system(sexcom(INV_SUBTIMG, conf_sex, conf_param, conf_conv, conf_nnw, det_thres=detect)+maskcom+fwhmcom+weightcom)#, detectiondual=trim_SCIIMG))

    subtbl      = ascii.read(SUBTIMG.replace(".fits", ".cat"))
    invsubtbl   = ascii.read(INV_SUBTIMG.replace(".fits", ".cat"))

    print(f"# Number of sources: {len(subtbl)}")
    subtbl['inim']  = SCIIMG
    subtbl['hcim']  = CONVIMG
    subtbl['hdim']  = SUBTIMG
    subtbl['mask']  = MASKIMG
    subtbl.meta['SEEING']   = np.median(scicat['FWHM_IMAGE']*0.4)
    subtbl['ratio_seeing']  = subtbl['FWHM_WORLD']/np.median(scicat['FWHM_WORLD'])
    subtbl.meta['ELLIPTICITY']  = np.median(scicat['ELLIPTICITY'])
    scicat['ELONGATION'] = 1 / (1-scicat['ELLIPTICITY'])
    subtbl['ratio_ellip']   = subtbl['ELLIPTICITY']/np.median(scicat['ELLIPTICITY'])
    subtbl['ratio_elong']   = subtbl['ELONGATION']/np.median(scicat['ELONGATION'])
    subtbl['MAG_AUTO']      = subtbl['MAG_AUTO'] + magautozero
    invsubtbl['MAG_AUTO']   = invsubtbl['MAG_AUTO'] + magautozero

    w = WCS(SUBTIMG)
    #    Positional information
    c_cent = w.pixel_to_world(scihdr['NAXIS1']/2, scihdr['NAXIS2']/2)
    c_sub = SkyCoord(subtbl['ALPHA_J2000'], subtbl['DELTA_J2000'], unit=u.deg)

    flagnumbers = np.arange(9)
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
                raise  # Re-raise the exception for any other RuntimeError
        except ConnectionError as e:
            print(f"Connection failed on attempt {attempt+1} of {max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # Wait for a bit before retrying
            else:
                raise  # Re-raise the exception if the final attempt fails
    #------------------------------------------------------------
    #    flag 1: Inverted Image Detections (Artifacts Around the Source)
    #------------------------------------------------------------
    if len(invsubtbl)>0:
        #    Coordinate
        invsubtbl = invsubtbl[invsubtbl['SNR_WIN']>20]
        c_invhd = SkyCoord(invsubtbl['ALPHA_J2000'], invsubtbl['DELTA_J2000'], unit=u.deg)
        #    Matching with inverted images
        indx_invhd, sep_invhd, _ = c_sub.match_to_catalog_sky(c_invhd)
        # subtbl['flag_1'][(sep_invhd.arcsec<subtbl.meta['SEEING']) & (np.abs((subtbl['MAG_AUTO'] - invsubtbl[indx_invhd]['MAG_AUTO'])) <= 1)] = True
        subtbl['flag_1'][(sep_invhd.arcsec<subtbl['FWHM_IMAGE'])] = True
        # subtbl['flag_1'][(sep_invhd.arcsec<subtbl.meta['SEEING']*2)] = True
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
    #    flag 6: Too Low SNR
    #------------------------------------------------------------
    snrcut  = 5         # flag6
    subtbl['flag_6'][(subtbl['SNR_WIN']<snrcut)] = True
    #------------------------------------------------------------
    #    flag 7
    #------------------------------------------------------------
    data = fits.getdata(SUBTIMG)
    peeing = subtbl.meta['SEEING']/pixscale
    skyval = np.median(subtbl['BACKGROUND'])
    skysig = np.std(subtbl['BACKGROUND'])

    subtbl['n_bad'] = 0
    subtbl['ratio_bad'] = 0.0
    subtbl['n_null'] = 0
    
    #    Fraction
    f = 0.1
    for i, (tx, ty, bkg) in enumerate(zip(subtbl['X_IMAGE'], subtbl['Y_IMAGE'], subtbl['BACKGROUND'])):

        #    Snapshot
        tsize = peeing
        y0, y1 = int(ty-tsize), int(ty+tsize)
        x0, x1 = int(tx-tsize), int(tx+tsize)
        cdata = data[y0:y1, x0:x1]
        crt = bkg - skysig*25
        cutline = cdata.size*f
        nbad = len(cdata[cdata<crt])
        try:
            ratiobad = nbad/cdata.size
        except:
            ratiobad = -99.0
        nnull = len(np.where(cdata == 1e-30)[0])
        #    Dipole
        if nbad > cutline or nnull != 0:
            subtbl['flag_7'][i] = True

        subtbl['n_bad'][i] = nbad
        subtbl['ratio_bad'][i] = ratiobad
        subtbl['n_null'][i] = nnull

    #------------------------------------------------------------
    #    flag 8: HOTPANTs Chi2 Value
    #------------------------------------------------------------
    subthdr     = fits.getheader(SUBTIMG)
    for i in range(div_col*div_row):
        if float(subthdr[f'X2NRM{str(i).zfill(2)}']) > 1000:
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
    #    flag 9: GLADE+ Galaxy Matching
    #------------------------------------------------------------
    #------------------------------------------------------------
    #    flag 10: PSFEx Analysis
    #------------------------------------------------------------
    #------------------------------------------------------------
    #    Final flag
    #------------------------------------------------------------
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
    #    Transient Catalog
    trtbl   = subtbl[subtbl['flag']==False]
    transient_cat   = SUBTIMG.replace('.fits', '.transient.cat')
    print('-'*60)
    print(f'Filtered sources\t: {len(trtbl)} ({100*len(trtbl)/len(subtbl):1.3f})%')


#%%
transient = """
GECKO24c
GECKO24d
GECKO24e
GECKO24f
GECKO24g
""".split()
imgs = """
hdCalib.KMTNet_SAAO.0910.20240424-184605.R.480.stack.trim_0x0.051570.new.fits
hdCalib.KMTNet_SAAO.0910.20240424-184605.R.480.stack.trim_0x0.036995.new.fits
hdCalib.KMTNet_SAAO.1358.20240424-183202.R.480.stack.trim_0x0.066092.new.fits
hdCalib.KMTNet_SAAO.1062.20240424-204109.R.480.stack.trim_0x0.052081.new.fits
hdCalib.KMTNet_SAAO.1063.20240425-191242.R.480.stack.trim_0x0.023087.new.fits
""".split()

date = '20240424_SAAO'
path_snap = f'/data4/kmtntoo/subt/{date}/'

# %%
for img in imgs:
    event = 'S240422ed'
    obs = img.split('.')[1].split('_')[-1]
    band = img.split('.')[4]
    path_snap = f'/data6/GECKO/{event}/KMTN_{obs}/{band}/transients/'
    if os.path.exists(f'{path_snap}{img}'):
        hdr = fits.getheader(f'{path_snap}{img}')
        print(hdr['DATE-OBS'])
# %%
site = 'SSO'
os.chdir(f'/data6/GECKO/S240422ed/KMTN_{site}/R/transients/')
ra = 122.27530
dec = -29.77469
# %%
trancats = sorted(glob.glob('*0750*transient.cat'))
# %%
from astropy.coordinates import SkyCoord
from astropy import units as u

for trancat in trancats:
    tran = ascii.read(trancat)
    # Convert the RA and DEC columns of the catalog to SkyCoord
    tran_coords = SkyCoord(ra=tran['ALPHA_J2000'] * u.deg, dec=tran['DELTA_J2000'] * u.deg)

    # Find matches within the tolerance
    tolerance = 1.0 * u.arcsec
    target_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    separation = tran_coords.separation(target_coord)

    matched_rows = tran[separation < tolerance]

    if len(matched_rows) > 0:
        print(f"Matches found in {trancat}:")
        print(matched_rows)
    else:
        print(f"No matches found in {trancat}.")
# %%
# %%
import pandas as pd
from io import StringIO
data = """
FRB010331.081-64.I.20220323.CTIO.scaled.stack.fits 81.75839175901 -64.86140555544
FRB010331.081-64.V.20220323.CTIO.scaled.stack.fits 81.75749223895 -64.86090626457
FRB010331.082-65.I.20220328.SAAO.scaled.stack.fits 82.26329964146 -65.36653181216
FRB010331.082-65.V.20220331.CTIO.scaled.stack.fits 82.44788142500 -65.29622563612
FRB010331.082-65.I.20220331.SAAO.scaled.stack.fits 82.26994139048 -65.36800331353
FRB010331.082-65.V.20220331.SAAO.scaled.stack.fits 82.26907002917 -65.36827347199
FRB010331.082-65.I.20220401.SAAO.scaled.stack.fits 82.27426954483 -65.36482747150
FRB010331.082-65.V.20220401.SAAO.scaled.stack.fits 82.27296294193 -65.36503613225
FRB010331.082-65.I.20220403.SAAO.scaled.stack.fits 82.27547612270 -65.36730315883
FRB010331.082-65.I.20220417.SSO.scaled.stack.fits  82.26166037881 -65.36903325182
FRB010331.082-65.V.20220417.SSO.scaled.stack.fits  82.26058987015 -65.36914699997
DWF4hr.062-54.I.20220323.CTIO.scaled.stack.fits       62.57801340547 -54.92157601021
DWF4hr.062-54.V.20220323.CTIO.scaled.stack.fits       62.57766388083 -54.92218232564
DWF4hr.062-55.I.20220330.SSO.scaled.stack.fits        63.08464763620 -55.43529871581
DWF4hr.062-55.V.20220330.SSO.scaled.stack.fits        63.08431634962 -55.43547939768
DWF4hr.062-55.V.20220331.CTIO.scaled.stack.fits       63.21068287165 -55.41942168230
DWF4hr.062-55.I.20220331.SSO.scaled.stack.fits        63.08585947641 -55.43475875444
DWF4hr.063-55.V.20220331.SSO.scaled.stack.fits        63.08555799650 -55.43492466335
DWF4hr.062-55.V.20220401.SSO.scaled.stack.fits        63.08662507370 -55.43644849512
DWF4hr.063-55.I.20220401.SSO.scaled.stack.fits        63.08860468986 -55.43620550560
DWF4hr.062-55.I.20220402.SSO.scaled.stack.fits        63.08521004578 -55.43577193496
DWF4hr.063-55.V.20220402.SSO.scaled.stack.fits        63.08506016462 -55.43594093414
DWF4hr.063-55.I.20220403.SAAO.scaled.stack.fits       63.08960652124 -55.42882458696
DWF4hr.062-55.I.20220404.SSO.scaled.stack.fits        63.08421097009 -55.43502714464
DWF4hr.063-55.V.20220404.SSO.scaled.stack.fits        63.08423034829 -55.43520409611
DWF4hr.062-55.V.20220405.SSO.scaled.stack.fits        63.08375582303 -55.43541392954
DWF4hr.063-55.I.20220405.SSO.scaled.stack.fits        63.08373072404 -55.43527845250
DWF4hr.062-55.V.20220410.SSO.scaled.stack.fits        63.08201731612 -55.43372821899
DWF4hr.063-55.I.20220410.SSO.scaled.stack.fits        63.08233126697 -55.43353306747
DWF4hr.062-55.I.20220411.SSO.scaled.stack.fits        63.02325146308 -55.43478029683
DWF4hr.063-55.V.20220411.SSO.scaled.stack.fits        63.08456670315 -55.43473626242
DWF4hr.062-55.I.20220417.SSO.scaled.stack.fits        63.02570132353 -55.49346187863
DWF4hr.062-55.V.20220417.SSO.scaled.stack.fits        63.08406565303 -55.49352114521
"""
# Convert the string into a pandas DataFrame
df = pd.read_csv(StringIO(data), delim_whitespace=True, header=None, names=["Filename", "RA", "DEC"])

# Display the DataFrame
print(df)
# %%
ks4cat = ascii.read(f'{path_cfg}kmtnet_grid.cat')
ks4cat = ascii.read(f'{path_cfg}ToO_grid.cat')
ks4flds = SkyCoord(ks4cat['ra[deg]'], ks4cat['dec[deg]'], unit='deg')

for _, row in df.iterrows(): 
    center = SkyCoord(row['RA'], row['DEC'], unit=(u.deg, u.deg))  # Assuming RA/DEC are in degrees
    separations = center.separation(ks4flds)
    min_sep = separations.min()
    closest_index = separations.argmin()
    
    # Fetch the field and associated information
    field = str(ks4cat[closest_index]['field_name1']).zfill(4)
    radec = ks4cat[ks4cat['field_name1'] == int(field)]['field_name2'][0]
    
    if min_sep.value < 3:  # 5 arcminutes
        print(f"{row['Filename']} corresponds to field {field}-{radec} with separation {min_sep:.2f}")
    else:
        pass
        # print(f"{row['Filename']} is out of gridcat. Check the coord in the header.")
# %%
toocat = ascii.read(f'{path_cfg}ToO_grid.cat')
ks4cat = ascii.read(f'{path_cfg}kmtnet_grid.cat')
ks4flds = SkyCoord(toocat[31]['ra[deg]'], toocat[31]['dec[deg]'], unit='deg')
# ks4flds = SkyCoord(ks4cat['ra[deg]'], ks4cat['dec[deg]'], unit='deg')

for row in ks4cat[:3000]: 
    center = SkyCoord(row['ra[deg]'], row['dec[deg]'], unit=(u.deg, u.deg))  # Assuming RA/DEC are in degrees
    separations = center.separation(ks4flds)
    min_sep = separations.min()
    closest_index = separations.argmin()
    
    # Fetch the field and associated information
    field = str(row['field_name1']).zfill(4)
    radec = ks4cat[ks4cat['field_name1'] == int(field)]['field_name2'][0]
    
    if min_sep.value < 3:  # 5 arcminutes
        print(f"9031 corresponds to field {field}-{radec} with separation {min_sep:.2f}")
    else:
        pass
        # print(f"{row['Filename']} is out of gridcat. Check the coord in the header.")

# %%
imlist = """
xkmts.20250105.064993.fits
xkmts.20250105.064994.fits
xkmts.20250105.064995.fits
xkmts.20250105.064996.fits
xkmts.20250105.064997.fits
xkmts.20250105.064998.fits
xkmts.20250105.064999.fits
xkmts.20250105.065000.fits
xkmts.20250105.065001.fits
xkmts.20250105.065002.fits
xkmts.20250105.065003.fits
xkmts.20250105.065004.fits
xkmts.20250105.065005.fits
xkmts.20250105.065006.fits
xkmts.20250105.065007.fits
xkmts.20250105.065008.fits
xkmts.20250105.065009.fits
xkmts.20250105.065010.fits
xkmts.20250105.065011.fits
xkmts.20250105.065012.fits
xkmts.20250105.065013.fits
xkmts.20250105.065014.fits
xkmts.20250105.065015.fits
xkmts.20250105.065016.fits
xkmts.20250105.065017.fits
xkmts.20250105.065018.fits
xkmts.20250105.065019.fits
xkmts.20250105.065020.fits
xkmts.20250105.065021.fits
xkmts.20250105.065022.fits
xkmts.20250105.065023.fits
xkmts.20250105.065024.fits
xkmts.20250105.065025.fits
xkmts.20250105.065026.fits
xkmts.20250106.065228.fits
xkmts.20250106.065229.fits
xkmts.20250106.065230.fits
xkmts.20250106.065231.fits
xkmts.20250106.065232.fits
xkmts.20250106.065233.fits
xkmts.20250106.065234.fits
xkmts.20250106.065235.fits
xkmts.20250106.065236.fits
xkmts.20250106.065237.fits
xkmts.20250106.065238.fits
xkmts.20250106.065239.fits
xkmts.20250106.065240.fits
xkmts.20250106.065241.fits
xkmts.20250106.065242.fits
xkmts.20250106.065243.fits
xkmts.20250106.065244.fits
xkmts.20250106.065245.fits
xkmts.20250106.065246.fits
xkmts.20250106.065247.fits
xkmts.20250106.065248.fits
xkmts.20250108.000000.fits
xkmts.20250108.000001.fits
xkmts.20250108.000002.fits
xkmts.20250108.000003.fits
xkmts.20250108.000004.fits
xkmts.20250108.000005.fits
xkmts.20250108.000006.fits
xkmts.20250108.000007.fits
xkmts.20250108.000008.fits
xkmts.20250108.000009.fits
xkmts.20250108.000010.fits
xkmts.20250108.000011.fits
xkmts.20250108.000012.fits
xkmts.20250108.065533.fits
xkmts.20250108.065534.fits
xkmts.20250108.065535.fits
""".split()

# %%
import os, shutil

# List of image filenames to search

# Adjust filenames to match the expected format
imgs = [img.replace('xkmts', 'kmts') for img in imlist]

# Define the root directory and subdirectories
path_data = '/data5/ks4/data/dwf/'
path_dest = '/data4/kmtntoo/tutorial/data/raw/'
dates = '20250105 20250106 20250108'.split()
dirs = [os.path.join(path_data, date) for date in dates]

# Function to search for files
def find_files(dirs, filenames):
    found_files = {}
    for dir_path in dirs:
        for root, _, files in os.walk(dir_path):  # Traverse each directory
            for filename in filenames:
                if filename in files:  # Check if the file exists in the current directory
                    found_files[filename] = os.path.join(root, filename)
    return found_files

# Perform the search
found_files = find_files(dirs, imgs)

# Print the results
for img, location in found_files.items():
    date = location.split('/')[5]
    dateobs = f'{date[2:]}_SAAO'
    print(f"{img} found at {location}")
    try:
        shutil.move(location, os.path.join(path_dest, dateobs))
    except:
        pass

# Handle files not found
not_found = set(imgs) - set(found_files.keys())
if sorted(not_found):
    print("Files not found:")
    for img in not_found:
        print(f"{img}")
# %%
import numpy as np
from astropy.io import fits

os.chdir('/data4/kmtntoo/tutorial/config/badpixelmap/')
# Read the data and header from your image
masks = sorted(glob.glob('*fits'))
for mname in masks:
    data, hdr = fits.getdata(mname, header=True)
    ny, nx = data.shape

    # Create a Boolean mask that is True for pixels in the edge areas:
    # - For the X axis: columns 0 to 459 and columns nx-460 to nx-1
    # - For the Y axis: rows 0 to 939 and rows ny-940 to ny-1
    mask = np.zeros((ny, nx), dtype=bool)
    mask[:, :460] = True
    mask[:, -460:] = True
    mask[:940, :] = True
    mask[-940:, :] = True

    # Create a new array with only the edge areas (all others set to 0)
    data_edge = np.zeros_like(data)
    data_edge[mask] = data[mask]

    # Save the new image
    fits.writeto(mname, data_edge, hdr, overwrite=True)

# %%
