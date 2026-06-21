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
#%% General Functions
def rss(numlist):
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
def zpcal(intbl, inmagkey, refmagkey, sigma=2.0):
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
    
    from datetime import datetime, timedelta, timezone
    
    jd = mjd + 2400000.5
    delta = jd - 2440587.5
    timestamp = timedelta(days=delta)
    date = datetime.fromtimestamp(timestamp.total_seconds(), timezone.utc)
    return date.strftime("%Y-%m-%dT%H:%M:%S")
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
def create_ldac_fits(input_fits, output_ldac, centrakey='X_WORLD', centdeckey='Y_WORLD', magkey='MAG', center=None, radius=None):
    with fits.open(input_fits) as hdul:
        data_table = Table(hdul[1].data)
        # Rename columns
        possible_keys = [
            ('RA', 'DEC'),
            ('ALPHAJ2000', 'DELTAJ2000'),
            ('RAJ2000', 'DEJ2000'),
        ]

        # Attempt to rename using the first matching pair
        for ra_key, dec_key in possible_keys:
            if ra_key in data_table.colnames and dec_key in data_table.colnames:
                data_table.rename_column(ra_key, centrakey)
                data_table.rename_column(dec_key, centdeckey)
                break
        else:
            raise KeyError("No matching RA/DEC column names found.")
        
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
def hotpants(inim, refim, inmsk, refmsk, convdir='t', normdir='i', outim='hd.fits', convim='hc.fits', nrx=4, nry=4, stamp=None):
    '''
    inim : Science image
    refim : Reference image
    convdir: convolution direction ('t' for reference, 'i' for science)
    [-c  toconvolve]  : force convolution on (t)emplate or (i)mage (undef)
    [-n  normalize]   : normalize to (t)emplate, (i)mage, or (u)nconvolved (t)
    '''
    if stamp is None:
        com = f'hotpants -c {convdir} -n {normdir} -iu 100000000 -il -100000 -tu 100000000 -tl -100000 -v 0 -inim {inim} -tmplim {refim} -imi {inmsk} -tmi {refmsk} -outim {outim} -oci {convim} -nrx {nrx} -nry {nry}'
    else:
        stampname   = outim.replace('.fits', '.stamp')
        with open(stampname, "w") as f:
            for s in stamp:
                f.write(f"{s['X_IMAGE']} {s['Y_IMAGE']} \n")
        com = f'hotpants -c {convdir} -n {normdir} -iu 100000000 -il -100000 -tu 100000000 -tl -100000 -v 0 -inim {inim} -tmplim {refim} -imi {inmsk} -tmi {refmsk} -outim {outim} -oci {convim} -ssf {stampname} -nrx {nrx} -nry {nry}'
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
def build_sex_command(
        image, conf_sex, conf_param, conf_conv, conf_nnw, detect,
        fwhm=None, mask=None, weight=None, extra_args=None
    ):
        args = [
            "source-extractor", image,
            "-c", conf_sex,
            "-CATALOG_NAME", image.replace("fits", "cat"),
            "-PARAMETERS_NAME", conf_param,
            "-FILTER_NAME", conf_conv,
            "-STARNNW_NAME", conf_nnw,
            "-DETECT_THRESH", str(detect),
        ]
        if fwhm is not None:
            args += ["-SEEING_FWHM", str(fwhm)]
        if mask is not None:
            args += ["-FLAG_IMAGE", mask, "-FLAG_TYPE", "MAX"]
        if weight is not None:
            args += ["-WEIGHT_TYPE", "MAP_WEIGHT", "-WEIGHT_IMAGE", weight, "-RESCALE_WEIGHTS", "Y", "-WEIGHT_GAIN", "Y"]
        if extra_args:
            for key, value in extra_args.items():
                args += [key, str(value)]
        return " ".join(args)
#------------------------------------------------------------
def generate_snapshot(row, cutsize=2.0, pixscale=0.4, outdir=None):
    """
    Generate postage stamp cutouts for a single transient candidate.
    `row` should be a Table row or dict with required keys.
    """
    from pathlib import Path
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D

    # Setup output directory
    outdir = Path(outdir) if outdir else Path.cwd()
    outdir.mkdir(parents=True, exist_ok=True)

    # Data extraction
    n = row['NUMBER'].item() if hasattr(row['NUMBER'], 'item') else row['NUMBER']
    inim, hcim, hdim = row['inim'], row['hcim'], row['hdim']
    tra, tdec = (row['ALPHA_J2000'].item() if hasattr(row['ALPHA_J2000'], 'item') else row['ALPHA_J2000'],
                 row['DELTA_J2000'].item() if hasattr(row['DELTA_J2000'], 'item') else row['DELTA_J2000'])
    ximg, yimg = (row['X_IMAGE'].item() if hasattr(row['X_IMAGE'], 'item') else row['X_IMAGE'],
                  row['Y_IMAGE'].item() if hasattr(row['Y_IMAGE'], 'item') else row['Y_IMAGE'])
    position = SkyCoord(ra=tra, dec=tdec, frame='icrs', unit='deg')
    size = u.Quantity((cutsize, cutsize), u.arcmin)

    # Optional known-object provenance (present only when a target CSV was used).
    try:
        known_match = bool(row['known_match'])
    except (KeyError, IndexError, ValueError):
        known_match = False
    try:
        known_target = str(row['known_target']).strip()
    except (KeyError, IndexError, ValueError):
        known_target = ''

    for image, kind in zip([inim, hcim, hdim], ['new', 'ref', 'sub']):
        with fits.open(image) as hdul:
            hdu = hdul[0]
            wcs = WCS(hdu.header)
            cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs, mode='partial', fill_value=0)
            hdu.data = cutout.data
            hdu.header.update(cutout.wcs.to_header())
            metadata = {
                'TRANRA': (tra, "transient candidate center RA"),
                'TRANDEC': (tdec, "transient candidate center DEC"),
                'XIMAGE': (ximg, "transient candidate X pixel location"),
                'YIMAGE': (yimg, "transient candidate Y pixel location"),
                'TRIM': (inim.split('.')[-2], "trimmed section"),
                'MAGAUTO': (row['MAG_AUTO'], "transient candidate MAG_AUTO"),
                'SNR': (row['SNR_WIN'], "transient candidate SNR"),
                'SEEING': (row['FWHM_IMAGE'] * pixscale, "transient candidate FWHM"),
                'ELLIP': (row['ELLIPTICITY'], "transient candidate ellipticity"),
                'ELONG': (row['ELONGATION'], "transient candidate elongation"),
                'CLSSTAR': (row['CLASS_STAR'], "transient candidate CLASS_STAR"),
                'ASTEROID': (row['flag_0'], "moving object matched within 5arcsec"),
                'IMAFLAG': (row['IMAFLAGS_ISO'], "Mask image flags"),
                'KNOWNOBJ': (known_match, "snapshot forced by known-object (CSV) match"),
                'TARGET': (known_target, "matched known-object name"),
            }
            for key, value in metadata.items():
                hdu.header[key] = value
            outim = outdir / f"{Path(hdim).stem}.{n:06d}.{kind}{Path(hdim).suffix}"

            try:
                hdu.writeto(outim, overwrite=True)
            except Exception as e:
                print(f"Error writing file {outim}: {e}")
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

    # Check whether the files have proper BITPIX
    for f in files:
        hdr = fits.getheader(f)
        if hdr['BITPIX'] != -32:
            print(f"File {os.path.basename(f)} has BITPIX {hdr['BITPIX']}, not -32")
            files.remove(f)
    
    # Try to find the file with the longest exposure time
    try:
        refimg = max(files, key=lambda f: int(re.search(r'\d+(?=sec)', f).group()))
        print(f"Reference image: {os.path.basename(refimg)}")
        return refimg
    except ValueError:
        # Return None if no files are found or if there's an issue parsing the exposure time
        print(f"No files found or issue parsing the exposure time")
        return None
    return refimg
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
def parse_region_bounds(region_str):
    """
    Parse region bounds from the header and adjust for Python 0-indexing.
    """
    bounds = region_str.replace('[', '').replace(']', '').split(',')
    x_start, x_end = map(lambda x: int(x) - 1, bounds[0].split(':'))
    y_start, y_end = map(lambda x: int(x) - 1, bounds[1].split(':'))
    return x_start, x_end, y_start, y_end
#------------------------------------------------------------
def mosaic_image(data_i, data_t, header_i, header_t, div_col, div_row):
    """
    Mosaic two images based on chi2 values and header information.
    """
    combined_image = np.zeros_like(data_i)
    for region in range(div_col * div_row):
        # Extract chi2 values and region bounds
        try:
            x2nrm_i = float(header_i[f'X2NRM{region:02d}'])
            x2nrm_t = float(header_t[f'X2NRM{region:02d}'])
            region_bounds = header_i[f'REGION{region:02d}']
        except KeyError as e:
            raise ValueError(f"Missing required header keyword: {e}")
        
        x_start, x_end, y_start, y_end = parse_region_bounds(region_bounds)
        
        # Select the better region based on chi2
        if x2nrm_i < x2nrm_t:
            print(f'Region {region}: chi2_i = {x2nrm_i}, chi2_t = {x2nrm_t}, selecting science image.')
            combined_image[y_start:y_end, x_start:x_end] = data_i[y_start:y_end, x_start:x_end]
            header_i[f'CONVD{region:02d}'] = 'i'
        else:
            print(f'Region {region}: chi2_i = {x2nrm_i}, chi2_t = {x2nrm_t}, selecting template image.')
            combined_image[y_start:y_end, x_start:x_end] = data_t[y_start:y_end, x_start:x_end]
            header_i[f'CONVD{region:02d}'] = 't'
            # Update header information from the template image
            for key in ['X2NRM', 'KSUM', 'CONVOL', 'SSSIG', 'SSSCAT', 'FSIG', 'FSCAT', 'NX2NRM']:
                header_i[f'{key}{region:02d}'] = header_t.get(f'{key}{region:02d}', None)
    return combined_image
#------------------------------------------------------------
def combine_subtracted_images(
    header_i_path, header_t_path, 
    conv2i_path, conv2t_path, 
    template_i_path, template_t_path, 
    output_path, template_output_path, 
    div_col=4, div_row=4
):
    """
    Combine subtracted and template images into mosaics based on chi2 values.
    """
    # Open the headers
    with fits.open(header_i_path) as hdul_header_i, fits.open(header_t_path) as hdul_header_t:
        header_i = hdul_header_i[0].header.copy()
        header_t = hdul_header_t[0].header

    # Open the subtracted images
    with fits.open(conv2i_path) as hdul_conv2i, fits.open(conv2t_path) as hdul_conv2t:
        image_data_i = hdul_conv2i[0].data
        image_data_t = hdul_conv2t[0].data

    # Validate image shapes
    if image_data_i.shape != image_data_t.shape:
        raise ValueError("Input subtracted images must have the same dimensions.")

    # Open the template images
    with fits.open(template_i_path) as hdul_templ_i, fits.open(template_t_path) as hdul_templ_t:
        template_data_i = hdul_templ_i[0].data
        template_data_t = hdul_templ_t[0].data

    # Validate template shapes
    if template_data_i.shape != template_data_t.shape:
        raise ValueError("Input template images must have the same dimensions.")

    # Mosaic the subtracted and template images
    print("Mosaicing subtracted images...")
    combined_image = mosaic_image(image_data_i, image_data_t, header_i, header_t, div_col, div_row)
    
    print("Mosaicing template images...")
    combined_template = mosaic_image(template_data_i, template_data_t, header_i, header_t, div_col, div_row)

    # Save the combined images
    fits.writeto(output_path, combined_image, header=header_i, overwrite=True)
    fits.writeto(template_output_path, combined_template, header=header_i, overwrite=True)

    print(f"Combined subtracted image saved to {output_path}")
    print(f"Combined template image saved to {template_output_path}")