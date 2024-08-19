#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% packages
import time
import glob
import copy
import scipy
import shutil
import os, sys, re
import numpy as np
import pandas as pd
from tqdm import tqdm
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io import ascii
import matplotlib.pyplot as plt
import astropy.coordinates as coord
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip
from astroquery.vizier import Vizier 
from astropy.io.votable import parse
import matplotlib.gridspec as gridspec
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, vstack, hstack
from sklearn.linear_model import HuberRegressor
from astropy.visualization import ImageNormalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import ZScaleInterval, LinearStretch

#%% define functions
def sqsum(numlist):
    S   = 0
    for i in range(len(numlist)):
        S   += numlist[i]**2
    sqS     = np.sqrt(S)
    return sqS

def apass_query(ra, dec, radius=1.0): # unit=(deg, deg, arcsec)
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

def GAIAXP_query(field, path_ref):
    """
    field = '0000'
    path_ref = '/data4/kmtntoo/cat/gaiaxp/'
    """
    field = field.split('.')[0]
    refcat  = Table(fits.open(os.path.join(path_ref, f'zpcat_{field}.fits'))[1].data)
    refcat.rename_columns(['RA', 'DEC'], ['RAJ2000', 'DEJ2000'])
    refcat['Bmag'] = refcat['XP_B'] - 0.30 * (refcat['XP_B']-refcat['XP_V'])
    refcat['Rmag'] = refcat['XP_R'] - 0.04 * (refcat['XP_V']-refcat['XP_R'])
    refcat.rename_columns(['XP_eB'], ['e_Bmag'])
    refcat.rename_columns(['XP_V', 'XP_eV'], ['Vmag', 'e_Vmag'])
    refcat.rename_columns(['XP_eR'], ['e_Rmag'])
    refcat.rename_columns(['XP_I', 'XP_eI'], ['Imag', 'e_Imag'])

    return refcat

def sort_BVRI(imlist):
        newlist     = []
        ks4ftr  = ['B', 'V', 'R', 'I']
        for ftr in ks4ftr:
            for i in range(len(imlist)):
                if ftr in imlist[i]:
                    newlist.append(imlist[i])
        return newlist

def psfexxml(xmlfile):
    """
    INPUT   :   .xml
    OUTPUT  :   FWHM    [pixel]
    """
    votable     = parse(xmlfile)
    table       = votable.get_first_table()
    data        = table.array
    #   EXTRACT FWHM [pixel]
    fwhm        = data['FWHM_MEAN'][0]
    fwhm        = round(fwhm, 3)
    return fwhm

def psfex(inim, pixscale, path_psf, path_cfg):
    """
    OUTPUT: fwhm, psf model
    """
    #   FILE CHECK
    prese_conf  = os.path.join(path_cfg, 'prepsfex.sex')
    prese_param = os.path.join(path_cfg, 'prepsfex.param')
    psfex_conf  = os.path.join(path_cfg, 'default.psfex')
    psfex_conv  = os.path.join(path_cfg, 'default.conv')
    psfex_nnw   = os.path.join(path_cfg, 'default.nnw')
    
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
    cat     = inim.replace('.fits', '.cat')
    xml     = inim.replace('.fits', '.xml')
    #   OPTION
    presecom1   = f"{inim} -c {prese_conf}"
    presecom2   = f" -CATALOG_NAME {cat},prepsfex.cat -PARAMETERS_NAME {prese_param}"
    presecom3   = f" -FILTER_NAME {psfex_conv} -STARNNW_NAME {psfex_nnw}"
    #   COMMAND
    presecom    = f"sex {presecom1}{presecom2}{presecom3}"
    os.system(presecom)
    psfexcom    = f"psfex -c {psfex_conf} {cat}"
    os.system(psfexcom) 

    os.system(f'mv psfex.xml {xml}')
    os.system(f'mv snap_{inim.split("/")[-1]} {path_psf}')
    os.system(f'mv {inim.replace(".fits", ".psf")} {path_psf}')
    # os.system(f'rm {cat}')

    #   FWHM [pixel], FWHM [arcsec]
    fwhm_pix    = psfexxml(xml)
    fwhm_arcsec = round(fwhm_pix*pixscale, 3)
    comment     = '\n' \
                + 'FILE NAME'+'\t'+': '+inim+'\n' \
                + 'FWHM value'+'\t'+': '+str(fwhm_pix)+'\t'+'[pixel]'+'\n' \
                + '\t'+'\t'+': '+str(fwhm_arcsec)+'\t'+'[arcsec]'+'\n'
    print(comment)
    os.system(f'mv {xml} {path_psf}')

    return fwhm_arcsec, fwhm_pix

def limitmag(N, zp, aper, skysigma): # 3? 5?, zp, diameter [pixel], skysigma

    R           = float(aper)/2.                # to radius
    braket      = N*skysigma*np.sqrt(np.pi*(R**2))
    upperlimit  = float(zp)-2.5*np.log10(braket)

    return round(upperlimit, 3)

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

def zpcal(intbl, inmagkey, inmagerkey, refmagkey, refmagerkey, sigma=2.0):
    """
    ZERO POINT CALCULATION
    3 SIGMA CLIPPING (MEDIAN)
    """
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

def zpplot(outname, otbl, xtbl, inmagkey, inmagerkey, refmagkey, refmagerkey, zp, zper, savepath):

    """From Gregory paek's phot.py"""
    #   FILE NAME
    plt.close('all')
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 16})
    plt.axhline(zp, linewidth=1, linestyle='--', color='gray', label= 'ZP {}'.format(str(round(zp, 3))) )
    plt.axhline(zp+zper, linewidth=1, linestyle='-', color='gray', alpha=0.5, label='ZPER {}'.format(str(round(zper, 3))) )
    plt.axhline(zp-zper, linewidth=1, linestyle='-', color='gray', alpha=0.5 )#, label=str(round(zp, 3)) )
    plt.fill_between([np.min(otbl[refmagkey])-0.05, np.max(otbl[refmagkey])+0.05],
                     zp-zper, zp+zper,
                     color='silver', alpha=0.3)
    plt.errorbar(otbl[refmagkey], otbl['zp'],
                 yerr=sqsum([otbl[inmagerkey], otbl[refmagerkey]]),
                 c='dodgerblue', ms=6, marker='o', ls='',
                 capsize=5, capthick=1,
                 label='NOT CLIPPED ({})'.format(len(otbl)), alpha=0.75)
    plt.scatter( xtbl[refmagkey], xtbl['zp'], color='tomato', s=50, marker='x', linewidth=1, alpha=1.0, label='CLIPPED ({})'.format(len(xtbl)) )
    plt.xlim(np.min(otbl[refmagkey])-0.05, np.max(otbl[refmagkey])+0.05)
    plt.ylim(zp-0.5, zp+0.5)
    #    SETTING
    plt.title(outname, {'fontsize': 16})
    plt.gca().invert_yaxis()
    plt.xlabel('REF.MAG.', {'color': 'black', 'fontsize': 20})
    plt.ylabel('ZERO POINT [AB]', {'color': 'black', 'fontsize': 20})
    plt.legend(loc='best', prop={'size': 14}, edgecolor=None)
    plt.tight_layout()
    plt.minorticks_on()
    plt.savefig(savepath+outname)
    plt.close()
    #    PRINT
    print('MAG TYP     : '+inmagkey)
    print('ZP          : '+str(round(zp, 3)))
    print('ZP ERR      : '+str(round(zper, 3)))
    print('STD.NUMB    : '+str(int(len(otbl))))
    print('REJ.NUMB    : '+str(int(len(xtbl))))
    return 0

def plotshow(inim, outname, numb_list, xim_list, yim_list, savepath, add=None, numb_addlist=None, xim_addlist=None, yim_addlist=None):
    '''
    From Gregory Peak's phot.py
    '''

    data, hdr   = fits.getdata(inim, header=True)
    wcs         = WCS(hdr)
    norm_zscale = ImageNormalize(data, interval=ZScaleInterval(), stretch=LinearStretch())
    fig         = plt.figure(figsize=(10,10)) 
    ax          = plt.subplot(projection=wcs)
    ax.set_title('{}'.format(outname))
    ax.imshow(data, cmap='gray', origin='lower', norm=norm_zscale) 

    # for labels
    ax.scatter([], [], color='dodgerblue', marker='o', alpha=0.7, s=20, label='ZP STARS (n={})'.format(len(xim_list)))
    ax.scatter([], [], color='gold', marker='x', alpha=0.7, s=20, label='CLIPPED (n={})'.format(len(xim_addlist)))
    for xx, yy in zip(xim_list, yim_list):
        ax.scatter(xx, yy, color='dodgerblue', marker='o', alpha=0.7, s=20)
    if add != None:
        for xx, yy in zip(xim_addlist, yim_addlist):
            ax.scatter(xx, yy, color='gold', marker='x', alpha=0.7, s=20)
    else:
        pass
    ax.legend()
    fig.savefig(savepath+outname, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
    plt.close()
    return 0

def add_colorbar(mappable, clabel, clim):

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(clabel)
    cbar.mappable.set_clim(clim[0], clim[1])
    plt.sca(last_axes)

def date2MJD(dateobs):

    from astropy.time import Time

    return Time(dateobs, format='isot', scale='utc').mjd

def MJD2date(mjd):
    
    from datetime import datetime, timedelta
    
    jd = mjd + 2400000.5
    delta = jd - 2440587.5
    timestamp = timedelta(days=delta)
    date = datetime.utcfromtimestamp(timestamp.total_seconds())
    return date.strftime("%Y-%m-%dT%H:%M:%S")

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

#%% ToOampcom.py

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
            catname     = f'{path_data}{serial}.{chip}.fits.cat'
            param       = os.path.join(path_cfg, 'kmtnet.param')
            cfg         = os.path.join(path_cfg, 'kmtnet.sex')
            conv        = os.path.join(path_cfg, 'kmtnet.conv')
            nnw         = os.path.join(path_cfg, 'kmtnet.nnw')
    
            os.system(f'sex {path_data}{serial}.{chip}.fits -c {cfg} -CATALOG_TYPE ASCII_HEAD -CATALOG_NAME {catname} -PARAMETERS_NAME {param} -FILTER_NAME {conv} -STARNNW_NAME {nnw} -DETECT_THRESH 50 -ANALYSIS_THRESH 50')
                      
            cat = Table.read(f'{path_data}{serial}.{chip}.fits.cat', format ='ascii')
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
        
        if (skykk <= 20) or (skymm <= 20) or (skytt <= 20) or (skynn <= 20):
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
    # os.system(f'rm {path_data}badccderror/0*')
    # os.system(f'rm {path_data}badseeing/0*')
    # os.system(f'rm {path_data}badtracking/0*')
    
    return 0
#%% ToOastrom.py
def astrom(path_data, path_cfg, path_cat, radius=1.0, thresh=5):
    """
    date = '240423_CTIO'
    path_data = f'/data4/kmtntoo/tutorial/data/raw/{date}/'
    path_cfg = '/data4/kmtntoo/tutorial/config/'
    path_cat = '/data4/kmtntoo/tutorial/catalog/'
    radius = 1.0
    thresh = 5
    """

    import os, sys
    from pathlib import Path
    import astropy.units as u
    from astropy.io import fits
    from astropy.table import Table
    from astropy.coordinates import Angle
    from astropy.coordinates import SkyCoord

    if not path_data.endswith('/'):
        path_data   = path_data + '/'
    
    fits_files = Path(path_data).glob('kmt*.fits')

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

    for i in range(len(info)):

        serial  = name[i][14:20]
        print(f'Astrometry process for frame:{serial}, thresh:{thresh} ({i+1}/{len(name)})')

        if name[i][3] == 'a': gap = 53 # SSO
        else: gap = 51
        
        chiparr = ['kk','mm','tt','nn']
        headnum = 0
        
        for k, chip in enumerate(chiparr):

            with fits.open(f"{path_data}{serial}.{chip}.fits", mode='update') as hd:
                header = hd[0].header
                header['CRVAL1'] = radd[i]  # Update RA
                header['CRVAL2'] = decdd[i]  # Update Dec
                hd.flush()  # Write changes to the file
            
            catname     = f'{path_data}{serial}.{chip}.cat'
            param       = os.path.join(path_cfg, 'kmtnet.param')
            cfg         = os.path.join(path_cfg, 'kmtnet.sex')
            conv        = os.path.join(path_cfg, 'kmtnet.conv')
            nnw         = os.path.join(path_cfg, 'kmtnet.nnw')
            
            sexcom  = f'sex {path_data}{serial}.{chip}.fits -c {cfg} -CATALOG_NAME {catname} -PARAMETERS_NAME {param} -FILTER_NAME {conv} -STARNNW_NAME {nnw} -HEADER_SUFFIX NONE -DETECT_THRESH {thresh} -ANALYSIS_THRESH {thresh} -SATUR_LEVEL 60000.0'
            
            if name[i][3] == 'a': ahead = f'{path_cfg}kmtnet_global_sso.{chip}.ahead' #Austrailia
            if name[i][3] == 's': ahead = f'{path_cfg}kmtnet_global.{chip}.ahead' #South Africa
            if name[i][3] == 'c': ahead = f'{path_cfg}kmtnet_global_ctio.{chip}.ahead' #Chile
            
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
            if os.path.exists(gaiacat) and centcoord.separation(kmtcoord).min().value < -0.1: # center matched with the grid in 0.1 deg
                gaialdac     = gaiacat.replace(".fits", "_ldac.fits")
                # if not os.path.exists(refldac):
                create_ldac_fits(gaiacat, gaialdac, center=centcoord, radius=radius)

                scampcom = f'scamp {catname} -c {os.path.join(path_cfg, "kmtnet.scamp")} -ASTREF_CATALOG FILE -ASTREFCAT_NAME {gaialdac} -POSITION_MAXERR 20.0 -CROSSID_RADIUS 5.0 -DISTORT_DEGREES 3 -PROJECTION_TYPE TPV -AHEADER_GLOBAL {ahead} -STABILITY_TYPE INSTRUMENT'
            else:
                scampcom = f'scamp {catname} -c {os.path.join(path_cfg, "kmtnet.scamp")} -ASTREF_CATALOG UCAC-4 -POSITION_MAXERR 20.0 -CROSSID_RADIUS 5.0 -DISTORT_DEGREES 3 -PROJECTION_TYPE TPV -AHEADER_GLOBAL {ahead} -STABILITY_TYPE INSTRUMENT'
            
            # Run SExtractor and SCAMP
            print(sexcom)
            os.system(sexcom)
            os.system(f'rm {path_data}{serial}.{chip}.head')
            print(scampcom)
            while len(glob.glob(f"{path_data}{serial}.{chip}.head")) == 0: os.system(scampcom)
            # SCAMP output header
            n       = open(f"{path_data}{serial}.{chip}.head")
            nl      = n.read()
            nline   = nl.count("\n")
            headnum = headnum + nline
            n.close()

        chiparr = ['kk','mm','tt','nn']
        skyarr  = [skykk[i], skymm[i], skytt[i], skynn[i]]
        fwhmarr = [fwhm1kk[i], fwhm1mm[i], fwhm1tt[i], fwhm1nn[i]]

        lenarr      = [0., 0., 0., 0.]
        ratioarr    = [0., 0., 0., 0.]
        rmsalignarr = [0., 0., 0., 0.]
        alignstdarr = [0., 0., 0., 0.]

        for k in range(len(chiparr)):
            chip = chiparr[k]
            tempsky = skyarr[k]
            tempfwhm = fwhmarr[k]
            
            f=open(f"{path_data}{serial}.{chip}.head",'r')
            lines=f.readlines()
            f.close()

            f=open(f"{path_data}{serial}.{chip}.head",'w')
            lines[1]=lines[1][0:37]+'\n'
            for line in lines[0:gap] : f.write(line)
            f.close()
            
            hdr     = fits.getheader(f"{path_data}{serial}.{chip}.fits")
            hdu     = hdr.fromtextfile(f"{path_data}{serial}.{chip}.head")
            hdu     = hdr[0:7] + hdu
            hdu.append(('filter',filt[i]), end=True)
            hdu.append(('object',obj[i]), end=True)
            hdu.append(('exptime',expt[i]), end=True)
            hdu.append(('date-obs',dateo[i]), end=True)
            hdu.append(('observat', observatory[i]), end=True)
            hdu.append(('centra', ra[i]), end=True)
            hdu.append(('centdec', dec[i]), end=True)
            hdu.append(('centsecz', secz[i]), end=True)
            hdu.append(('sky'+chip, tempsky), end=True)
            hdu.append(('fwhm1'+chip, tempfwhm), end=True)

            fits.PrimaryHDU(data=fits.getdata(f'{path_data}{serial}.{chip}.fits'), header=hdu).writeto(f'{path_data}{serial}.{chip}.fits', overwrite=True)

    os.system(f'chmod 777 {path_data}*')
    
    return 0

#%% ToOastromqa.py

def qatest(fname, configdir, gridcat, refcatdir, refcatname='GAIA EDR3', divnum=8, crreject=True, bleedreject=True) :
    """
    QATEST ver 1.3.2

    # Input Format : 
    os.chdir(f'/data4/kmtntoo/tutorial/pipe/')
    fname   = '/data4/kmtntoo/tutorial/data/raw/240423_CTIO/052188.kk.fits'
    configdir = '/data4/kmtntoo/tutorial/config/'
    gridcat = 'kmtnet_grid.fits'
    refcatdir = '/data4/kmtntoo/tutorial/catalog/'
    refcatname='gaiaxp'
    divnum=8
    crreject= True
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
    import sys
    from datetime import date as dt
    from astropy.table import Table
    from astropy.io import fits,ascii
    from astropy.stats import SigmaClip,sigma_clip
    import astropy.units as u
    import astropy.coordinates as coord
    from astropy.coordinates import SkyCoord,Angle,match_coordinates_sky
    from astroquery.vizier import Vizier
    import astroscrappy as cr
    import numpy as np
    from numpy import std,sqrt,mean,median
    import pandas as pd
    import warnings

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
                  ' '*int((63-len(fname))/2) + f'running qatest({fname})\n' + 
                  '='*79)

        def end() :
            print('> All done')

        def crmap():
            print('> Creating Cosmic ray reduction map...')

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


    # ====== NESTED FUNCTIONS =================================================

    def bleed_masking(datapath, skysigcut=1.5, convergence=5, bpidx_thres=3000, saturate=52000):
        
        import numpy as np
        from astropy.stats import sigma_clipped_stats
        from astropy.io import fits
        
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
    
    def crmap(fname, ction=False, bleedreject=False) :

        """
        generating a badpixel mask
        1. cosmic-ray
        2. crosstalk
        3. bleeding pattern
        """

        Msg.crmap()
        crmapname = fname.replace('.fits', '.crmap.fits')
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
            print("> Bleeding Pattern Masking ...")
            Xtalk   = np.add(Xtalk, bpMask)

        fits.PrimaryHDU(data=Xtalk.astype(np.int16), header=fits.getheader(fname)).writeto(crmapname, overwrite=True)
    # -------------------------------------------------------------------------

    def load_incat(fname, incatname, configdir) :
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

        os.system(f'sex {fname} {prompt_cfg} -CATALOG_NAME {fname}.cat -CATALOG_TYPE ASCII_HEAD -FLAG_IMAGE {fname.replace(".fits", ".crmap.fits")} -DETECT_THRESH 10 -ANALYSIS_THRESH 10')

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
            # return None # 2023.06.22
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
        csep = mrefcut['sep'].iloc[clip]

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
                                                  rmsalign < 1) 
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
        fastrom = 'good' if gbmap_row.count('bad') <= 2 else 'bad'
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

    def hdr_update(anlz_rprt,__version__,csep,refcatname,divnum,absfpath) :

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
                refcatname, 
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


    # ======= MAIN ============================================================

    Msg.start(fname)

    if os.path.isfile(fname) == True :

        # 1. cosmic-ray mask generation
        if crreject == True:
            if fits.getheader(fname)['OBSERVAT']=='CTIO' and os.path.basename(fname).split('.')[1] == 'nn':
                crmap(fname, ction=True, bleedreject=bleedreject)
            else:
                crmap(fname, ction=False, bleedreject=bleedreject)
        
        # 2. astrometry quality assurance
        # 2.1. SExtractor run
        data = load_incat(fname, incatname=f'{fname}.cat', configdir=configdir)
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
                    param_matching  = dict(intbl   = data,
                                           reftbl  = reftbl,
                                           inra    = data['ALPHA_J2000'], 
                                           indec   = data['DELTA_J2000'],
                                           refra   = reftbl['RA'], 
                                           refdec  = reftbl['DEC'],
                                           sep     = 2)
                    mref    = matching(**param_matching).to_pandas()
                    # 2.6. cliping unmatched catalog (cutnum: match radius threshold [arcsec], clipnum: sigma threshold)
                    mrefcut,csep    = cutclip(mref, cutnum=2/3600, clipnum=3)
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
            
            Cmd.rm(f'{fname}.cat')
        
    else : 
        Msg.err(fname, 'fpatherr')

    Msg.end()

#%% ToOimscaler.py

def zpscale(img, path_output, path_cfg, path_cat, path_plot, mode='1DLINEAR', zpscaled=30.0, pixscale=0.4, gain=0, figure=False, start=None, gridcat='/data4/tempdatabase/kmtnet_grid.cat'):
    
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
        img = '/data4/kmtntoo/data/20231208_CTIO/a018462.nn.fits'
        path_output = '/data4/kmtntoo/scaled/'
        path_cfg    = '/data4/kmtntoo/config/'
        path_cat    = '/data4/kmtntoo/cat/'
        path_plot   = '/data4/kmtntoo/result/plot/'
        mode        = '1DLINEAR'
        # mode        = '2DPOLYNOMIAL'
        zpscaled    = 30.0
        pixscale    = 0.4
        gain        = 0
        figure      = False
        start       = None
        gridcat     = '/data4/kmtntoo/config/astrometry/ToO_grid.cat'
    Returns:
        _type_: _description_
        
    What should be prepared in advance:
        ToOampcop.cat in the working directory including FWHM data
        SMSS or APASS reference catalog in pathcat/{survey}
    """
    
    # 0. Mode checker
    mode = mode.upper()
    if mode not in ['1DLINEAR', '2DPOLYNOMIAL']:
        raise ValueError("Mode must be either '1DLINEAR' or '2DPOLYNOMIAL'")
    
    if not path_output.endswith('/'):
        path_output   = path_output + '/'
        
    # 1. Basic analysis
    
    # 1.1. Image header
    file    = fits.open(img)
    serial  = os.path.basename(img).split('.')[0][1:]
    hdr     = fits.getheader(img)
    data    = fits.getdata(img)
    
    try:
        chip    = hdr['CCDNAME'].upper()
    except:
        chip    = os.path.basename(img).split('.')[-2][0].upper() # a000000.kk.fits
    band    = hdr['FILTER']
    obs     = hdr['OBSERVAT']
    
    xaxis   = hdr['NAXIS1']
    yaxis   = hdr['NAXIS2']
    
    obj     = hdr['OBJECT'].split('_')[0] # e.g. S190425z, TOO_0503, etc. 
    center  = SkyCoord(hdr['CENTRA'], hdr['CENTDEC'], unit=(u.hourangle, u.deg))
    
    # 1.2. Preprocess completeness check
    try:
        if hdr['QARESULT']==False: print(f'Bad QA in astrometry for {img}.'); return None
    except KeyError:
        print(f'No QARESULT for {img}. Run astrompro2 in advance.'); return None

    # 1.3. Coordinates check (ks4 predetermined tiles)
    ks4cat  = ascii.read(gridcat)
    ks4flds = SkyCoord(ks4cat['ra[deg]'], ks4cat['dec[deg]'], unit='deg')
    field   = str(ks4cat[center.separation(ks4flds).argmin()]['field_name1']).zfill(4)
    if center.separation(ks4flds).min().value > 1.0:
        print(f'Image pointing is out of {gridcat}. Check the coord in the header.'); return None
    radec   = ks4cat[ks4cat['field_name1']==int(field)]['field_name2'][0]

    # Basic analysis done. Checking an intermediate time.
    mid     = time.time()
    print('='*50)
    if start == None:
        print("{}_{}.{}_{}_{}_{} ({}sec elapsed)".format(serial, field, radec, band, chip, obs, round(mid)))
    else:
        print("{}_{}.{}_{}_{}_{} ({}sec elapsed)".format(serial, field, radec, band, chip, obs, round(mid-start)))
    print('='*50)
    
    # 2. Reference catalog query
    # This is basically prepared for KS4 fields.
    """    
    try:
        path_ref    = os.path.join(path_cat, 'gaiaxp')
        reftbl  = GAIAXP_query(field, path_ref)
        hdr['PHOTREF']  = 'GAIA XP'
    except FileNotFoundError:
    """
    if band == 'I':
        path_ref    = f'{path_cat}smss/'
        reftbl  = SMSSDR3_query(field, path_ref)
        hdr['PHOTREF']  = 'SMSS DR3'
    else:
        path_ref    = f'{path_cat}apass/'
        try:
            reftbl  = ascii.read(f'{path_ref}apass_{field}.{radec}.cat')
        except FileNotFoundError:
            xscale  = xaxis * pixscale # arcsec
            yscale  = yaxis * pixscale # arcsec
            frac    = 3 # >2*np.sqrt(2) due to dithering
            radius  = frac*np.mean([xscale, yscale])/3600 # searching radius in deg
            reftbl  = apass_query(center.ra.deg, center.dec.deg, radius)
            reftbl.write(f'{path_ref}apass_{field}.{radec}.cat'.format(path_ref, radec), format='ascii', overwrite=True)
        
        hdr['PHOTREF']  = 'APASS DR9'
    
    # 3. Photometry: SExtractor run
    
    # 3.1. FWHM check. Utilizing values from previous quality check --> this should be in the header (important)
    try:
        seeing = float(hdr[f'FWHM1{chip*2}'])
    except KeyError:
        # seeing  = [x for x in ascii.read(f'{os.path.dirname(img)}/ToOampcom.cat') if serial in x['name']][0][f'fwhm{img.split(".")[-2]}']
        seeing  = 2.0
    
    # 3.2. Configurations
    param       = os.path.join(path_cfg, 'default.param')
    cfg         = os.path.join(path_cfg, 'default.sex')
    conv        = os.path.join(path_cfg, 'default.conv')
    nnw         = os.path.join(path_cfg, 'default.nnw')
    
    catname     = f'{path_cat}{obs}_{radec}_{chip}_{band}_{serial}.cat'
    bkgname     = f'{path_cat}{obs}_{radec}_{chip}_{band}_{serial}.bkg'
    resname     = f'{path_cat}{obs}_{radec}_{chip}_{band}_{serial}.res'
    
    thres       = 3     # detection threshold
    bkgsize     = 256   # background size (global)

    # 3.3. Command line args
    inim_single = img
    prompt_cat  = f' -CATALOG_NAME {catname}'
    prompt_aper = f' -PHOT_APERTURES {14/pixscale}' # 14" same with APASS
    prompt_cfg  = f' -c {cfg} -PARAMETERS_NAME {param} -FILTER_NAME {conv} -STARNNW_NAME {nnw}'
    prompt_opt  = f' -GAIN {gain:.2f} -PIXEL_SCALE {pixscale:.2f} -SEEING_FWHM {seeing:.2f} -SATUR_LEVEL 55000'
    prompt_chk  = f' -CHECKIMAGE_TYPE BACKGROUND,-OBJECTS -CHECKIMAGE_NAME {bkgname},{resname}'
    prompt_bkg  = f' -BACK_SIZE {bkgsize} -DETECT_THRESH {thres}'
    prompt      = 'sex '+inim_single+prompt_cfg+prompt_aper+prompt_opt+prompt_cat+prompt_chk+prompt_bkg
    os.system(prompt)
    
    # 4. Output analysis 
    
    # 4.1. FWHM re-check
    intbl   = ascii.read(catname)
    stbl    = intbl[intbl['CLASS_STAR']>median(intbl['CLASS_STAR'])]
    stbl    = stbl[stbl['FLAGS']==0]
    stbl    = stbl[stbl['MAGERR_AUTO']<0.05]
    seeing  = np.median(stbl['FWHM_IMAGE'])*0.4
    hdr['FWHM'] = round(seeing,3)
    
    # 4.2. Electronic pattern noise
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
            thres += 1
            if 'QAALNRMS' not in hdr: rad   = 0.5
            else: rad = hdr['QAALNRMS']*thres
            param_matching  = dict( intbl   = amptbl,
                                    reftbl  = reftbl,
                                    inra    = amptbl['ALPHA_J2000'], 
                                    indec   = amptbl['DELTA_J2000'],
                                    refra   = reftbl['RAJ2000'], 
                                    refdec  = reftbl['DEJ2000'],
                                    sep     = rad)
            mtbl    = matching(**param_matching)
            if len(mtbl) > 50 or thres > 4 or 'QAALNRMS' not in hdr:
                break
        mtbl    = mtbl[mtbl['MAG_AUTO']!=99]

        # 5.3. Source cleaning
        cmtbl   = mtbl[mtbl['FLAGS']<4]
        # cmtbl   = mtbl[mtbl['FLAGS']==0]
        cmtbl   = cmtbl[~np.isnan(cmtbl[f'{band}mag'])]
        cmtbl   = cmtbl[cmtbl['CLASS_STAR']>np.median(amptbl['CLASS_STAR'])]
        
        # 5.4. Zeropoint for each sources (2d distribution)
        aperture= 'AUTO'
        magdif  = cmtbl[f'MAG_{aperture}'] - cmtbl[f'{band}mag']
        magerr  = sqsum([cmtbl[f'MAGERR_{aperture}'], cmtbl[f'e_{band}mag']])
      
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
            if not (s > -0.1) or len(cmtbl) < 3:
                try:
                    popt, pcov   = curve_fit(linfun, cmtbl['Y_IMAGE'], magdif)
                    # popt, pcov   = curve_fit(linfun, cmtbl['Y_IMAGE'], magdif-zp)
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
            
            # 5.5.3. Scaling factor array composite
            for j in range(yaxis):
                del_zp = zpscaled + linfun(j, m, b)
                fratio = round(10**(del_zp/(2.5)), 4)
                for k in np.arange(xps, xpe):
                    imscale[j][int(k)] = fratio
            
            # 5.5.3. Fitting the result plot 
            if figure: 
                plt.figure(figsize=(7,4))
                plt.errorbar(cmtbl['Y_IMAGE'], magdif, magerr, ms=6, marker='s', ls='', c='dodgerblue', capsize=4, capthick=1, alpha=0.5, zorder=0, label='Stars [17mag, 14mag]')
                # plt.errorbar(cmtbl['Y_IMAGE'], magdif-zp, magerr, ms=6, marker='s', ls='', c='dodgerblue', capsize=4, capthick=1, alpha=0.5, zorder=0, label='Stars [17mag, 14mag]')
                plt.plot(X, m*X+b,color='blue', zorder=1)#, label=r'Linear fit (${:.2e}y{:+.2f}$)'.format(popt[0], popt[1]))
                plt.xlim(0, xaxis)
                plt.legend(loc='upper right')
                plt.title(f'Zeropoint Tendency \n{radec}-field, {band}-band, {chip}-chip, {i+1}-amp\n Stars={len(cmtbl)}, $R^2$={s:.3f}')
                plt.xlabel('Y_IMAGE [PIXEL]')
                plt.ylabel(r'$m_{KS4} - m_{APASS}$ [ABmag]')
                plt.ylim(-30,-28)
                # plt.show()
                plt.savefig(f'{path_plot}{obs}_{radec}_{chip}_{band}_{serial}_{i+1}_zptend.png')
                plt.close()
                
        # 5.6. Zeropoint tendency fitting (2DPolynomial)
        elif mode == '2DPOLYNOMIAL':
            def poly_2D(xy, a, b, c, d, e, f): 
                x, y = xy
                return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

            # 5.6.1. Parameter space 
            X = cmtbl['X_IMAGE']
            Y = cmtbl['Y_IMAGE']
            Z = np.array(magdif)
            
            # 5.6.2. 2d curve-fit
            p0          = np.ones(6) # initial guess
            popt, pcov  = curve_fit(poly_2D, (X, Y), Z, p0)
            x_grid, y_grid = np.mgrid[0:xpe-xps, 0:yaxis]
            corray      = poly_2D((x_grid, y_grid), *popt)
            
            # 5.6.3. Scaling factor array composite
            for j in range(xps, xpe):
                for k in range(yaxis):
                    del_zp = zpscaled + corray[j%1152][k]
                    fratio = round(10**(del_zp/(2.5)), 4)
                    imscale[k][j]   = fratio
    
    # 6. Pattern noise reduction, background subtraction
    bkgdata         = fits.getdata(bkgname)
    flippattern     = np.flip(pattern, axis=1)
    noisedata       = np.concatenate((flippattern, flippattern, flippattern, flippattern, pattern, pattern, pattern, pattern), axis=1)
    if obs == 'SAAO':
        file[0].data    = imscale * (data - bkgdata - noisedata)
    else:
        file[0].data    = imscale * (data - bkgdata)
        
    # 7. Header update
    hdr['FIELD1']   = field
    hdr['FIELD2']   = radec
    hdr['BADAMP']   = str(badamp).zfill(8)[::-1]
    hdr.comments['BADAMP'] = 'Amps flag (0:okay, 1:fitting fail)'
    if badamp == 0:
        hdr['PHOTQA']   = True
    else:
        hdr['PHOTQA']   = False
    hdr['ZERO{}{}'.format(chip, chip)]  = zpscaled
    hdr.comments['ZERO{}{}'.format(chip, chip)] = 'ABmagnitude system, AUTO aperture'
    hdr['SATURATE']     = 60000 * np.median(imscale)
    hdr.comments['SATURATE']    = 'Saturation values after zero-point scaling'
    hdr['SCALEFIT']     = mode
    
    # 8. Ouput file save
    newname     = f'{obj}_{field}.{radec}.{band}.{hdr["DATE-OBS"].split("T")[0].replace("-","")}.{obs}.{serial}.{os.path.basename(img).split(".")[1]}.scaled.fits'
    fits.writeto(f'{path_output}{newname}', file[0].data, hdr, overwrite=True)

    os.system(f'rm {bkgname}')
    os.system(f'rm {resname}')
    os.system(f'chmod 777 {path_output}{newname}')

    return newname
    
#%% ToOimstackter.py

def stacking(filename_convention, path_input, path_output, path_cfg, path_ref, start=None, gridcat='/data8/tempdatabase/kmtnet_grid.cat'):

    """        
    date     = '20240423_SSO'
    filename_convention = r"(?P<field>\w+_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<band>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<serial>\d{6})\.(?P<chip>\w+)\.(?P<type>scaled|mask)\.fits"
    filename_convention = r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<band>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<serial>\d{6})\.(?P<chip>\w+)\.(?P<type>scaled|mask)\.fits"

    path_input  = f'/data4/kmtntoo/scaled/{date}/'
    path_output = f'/data4/kmtntoo/stack/{date}/'
    path_cfg    = '/data4/kmtntoo/config/'
    path_ref    = '/data8/KS4/database/stack/'
    start       = time.time()
    gridcat='/data8/KS4/config/kmtnet_grid.cat'
    """

    if not path_input.endswith('/'):
        path_input   = path_input + '/'
    if not path_output.endswith('/'):
        path_output   = path_output + '/'

    # os.chdir(path_input)
    
    regex = re.compile(filename_convention)
    
    all_files   = sorted(glob.glob(f'{path_input}*.fits'))
    sfits   = [file for file in all_files if regex.match(os.path.basename(file)) and regex.match(os.path.basename(file)).group("type") == "scaled"]
    cfits   = [file for file in all_files if regex.match(os.path.basename(file)) and regex.match(os.path.basename(file)).group("type") == "mask"]

    fields  = sorted(list(set([os.path.basename(f).split('.')[0].split('_')[1] for f in sfits]))) # \w+_0000
    # fields  = sorted(list(set([os.path.basename(f).split('.')[0] for f in sfits]))) # \w+_0000
    bands   = sorted(list(set([os.path.basename(b).split('.')[2] for b in sfits])))
    observs = sorted(list(set([os.path.basename(b).split('.')[4] for b in sfits])))
    total   = 0

    for observ in observs:

        for field in fields:
            
            ks4cat  = ascii.read(gridcat)
            radec   = ks4cat[ks4cat['field_name1']==int(field)]['field_name2'][0]
            
            for band in bands:

                # center coordinate (should be fixed for dual-mode photometry)
                try:
                    centerimgs      = [f for f in sorted(glob.glob(f'{path_ref}{field}.{radec}/ks4.*{band}*scaled.stack.fits')) if 'mask' not in f] # reference images
                    centerimg       = max(centerimgs, key=lambda f: int(re.compile(r'\d+(?=sec)').search(f).group())) # longest exposure time
                    centra, centdec = fits.getheader(centerimg)['CENTRA'], fits.getheader(centerimg)['CENTDEC']
                    
                    # # refimg coverage for masking
                    # refdata     = fits.getdata(centerimg)
                    # refmask     = np.where(refdata == 0,1,0)
                except:
                    print("The reference image not exist.")
                    fieldinfo   = ks4cat[ks4cat['field_name1']==int(field)]
                    centcoord   = SkyCoord(ra=round(fieldinfo['ra[deg]'][0],5)*u.degree, dec=round(fieldinfo['dec[deg]'][0],5)*u.degree)
                    centra, centdec = centcoord.to_string('hmsdms').replace('h',':').replace('m',':').replace('s','').replace('d',':').split(' ')
                    # return
            
                # os.chdir(path_input)
        
                # clean quality chip images
                allist  = []
                for im in [f for f in sfits if field in os.path.basename(f).split('.')[0] and os.path.basename(f).split('.')[2]==band and os.path.basename(f).split('.')[4]==observ]:
                # for im in sorted(glob.glob(f'{field}*.{band}.*scaled.fits')):
                    hdr     = fits.getheader(im)
                    if 'QARESULT' not in hdr or 'BADAMP' not in hdr:
                        print(f'{im}: Either astrompro2 or zpscalepro is incomplete.')
                    elif hdr['QARESULT']==True:
                        if hdr['BADAMP']=='00000000':
                            allist.append(im)
                        elif hdr['BADAMP']=='00000010':
                            if 'CTIO' in im or 'kmtc' in im:
                                allist.append(im)
                        elif hdr['BADAMP']=='00000001':
                            if 'SAAO' in im and 'nn' in im:
                                allist.append(im)
                        else:
                            print(f'{im} BADAMP={hdr["BADAMP"]}')
                    else:
                        print(f'{im} QARESULT={hdr["QARESULT"]}')

                # k,m,t,n full-frame chip sets
                klist = [i for i in allist if 'kk' in i]
                mlist = [i for i in allist if 'mm' in i]
                tlist = [i for i in allist if 'tt' in i]
                nlist = [i for i in allist if 'nn' in i]
                
                fullframes  = list({idn.split('.kk.scaled.')[0] for idn in klist} & {idn.split('.mm.scaled.')[0] for idn in mlist} & {idn.split('.nn.scaled.')[0] for idn in nlist} & {idn.split('.tt.scaled.')[0] for idn in tlist})
                dither  = len(fullframes)
                # checker
                if dither == 0: continue
                total  += dither
                
                # time check
                mid     = time.time()
                print('='*70)
                if start == None:
                    print(f'Stacking Proess: {field}-{band}-band: {mid:.2f}sec passed')
                else:
                    print(f'Stacking Proess: {field}-{band}-band: {mid-start:.2f}sec passed')
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
                obs     = [os.path.basename(img).split('.')[4] for img in imlist if 'kk' in img]
                dateobs = [fits.getheader(img)['DATE-OBS'] for img in imlist if 'kk' in img]
                meanobs = np.mean([date2MJD(date) for date in dateobs])
                meandate= MJD2date(meanobs)
                saturat = [fits.getheader(img)['SATURATE'] for img in imlist]
                fwhm    = np.mean([float(fits.getheader(img)[f'FWHM1{re.match(filename_convention, os.path.basename(img)).group("chip").upper()}']) for img in imlist])
                exptime = float(fits.getheader(imlist[0])['EXPTIME']) * dither
                # stacking with swarp
                stack  = f'{path_output}'+imlist[0].replace('.scaled', '.scaled.stack').replace(imlist[0].split('.')[0], f'ToO_{field}').replace(f'{".".join(imlist[0].split(".")[5:7])}', f'{round(exptime)}sec')
                weight = f'{path_output}'+imlist[0].replace('.scaled', '.scaled.weight').replace(imlist[0].split('.')[0], field).replace(f'{".".join(imlist[0].split(".")[5:7])}', f'{round(exptime)}sec')
                # os.system(f'swarp @{path_input}diths.list -c {path_cfg}ks4catalog.swarp -IMAGEOUT_NAME {stack} -CENTER {centra},{centdec} -IMAGE_SIZE 22000,22000') 
                """
                FIX THIS (2024.06.03) !!!
                """
                os.system(f'rm {path_output}*weight.fits')
                os.system(f'rm {path_input}diths.list')
                
                # header updates
                # with fits.open(stack, 'update') as f:
                #     for hdu in f:
                
                #         hdu.header['OBJECT']    = field
                #         hdu.header['FIELD1']    = os.path.basename(imlist[0]).split('.')[0].split('_')[1]
                #         hdu.header['FIELD2']    = os.path.basename(imlist[0]).split('.')[1]
                #         hdu.header['FILTER']    = band
                #         hdu.header['NUMDITH']   = dither
                #         hdu.header['NUMIMAGE']  = dither
                #         hdu.header['EXPTIME']   = exptime
                #         hdu.header['DATE-OBS']  = (meandate, 'Average DATE-OBS of images stacked')
                #         hdu.header['MEANMJD']   = (meanobs, 'Average DATE-OBS of images in MJD')
                #         hdu.header['FWHM']      = round(fwhm, 2)
                #         for i in range(dither):
                #             hdu.header[f'OBSERV{hex(i)[-1]}']    = obs[i]
                #             hdu.header[f'IMAGE{hex(i)[-1]}']     = [os.path.basename(k) for k in imlist if 'kk' in k][i].replace('.kk','')
                #             hdu.header[f'DATEOBS{hex(i)[-1]}']   = dateobs[i]
                #         if band == 'I':
                #             hdu.header['PHOTREF']       = 'SMSS DR3'
                #         else:
                #             hdu.header['PHOTREF']       = 'APASS DR9'
                #         hdu.header['MAGZERO']   = 30.0
                #         hdu.header['CENTRA']    = centra
                #         hdu.header['CENTDEC']   = centdec
                #         hdu.header['SATURATE']  = round(np.mean(saturat))
                #         hdu.header['UNDERSAT']  = round(np.min(saturat))
            
                # os.system(f'chmod 777 {stack}')
                
                
                # 3.5. WCS info update for masks
                crmaps  = [f for f in cfits if os.path.basename(f).split('.')[0].split('_')[1]==field and os.path.basename(f).split('.')[2]==band and os.path.basename(f).split('.')[4]==observ]
                if len(crmaps) != 0:
                    f = open(f'{path_input}mdiths.list', 'w')
                    for cr in crmaps:
                        im  = cr.replace('mask', 'scaled')
                        hdu = fits.PrimaryHDU(fits.getdata(cr), header=fits.getheader(im)+WCS(im).to_header())
                        hdu.writeto(cr, overwrite=True)
                        f.write(cr+'\n')
                    f.close()

                    # 3.7. SWarp for mask
                    mstack = f'{path_output}'+imlist[0].replace('.scaled', '.crmap.stack').replace(imlist[0].split('.')[0], f'ToO_{field}').replace(f'{".".join(imlist[0].split(".")[5:7])}', f'{round(exptime)}sec')
                    # weight = f'{path_output}'+imlist[0].replace('.scaled', '.crmap.weight').replace(imlist[0].split('.')[0], field).replace(f'{".".join(imlist[0].split(".")[5:7])}', f'{round(exptime)}sec')
                    os.system(f'swarp @{path_input}mdiths.list -c {path_cfg}mask.swarp -IMAGEOUT_NAME {mstack} -CENTER {centra},{centdec} -IMAGE_SIZE 22000,22000')
                    os.system(f'rm {path_output}*weight.fits')
                    os.system(f'rm {path_input}mdiths.list')
                else:
                    mstack = f'{path_cfg}ks4.empty.mask.fits'
                
                # 3.7.1. Empty ref coverage
                # try:
                #     # assert fits.getdata(mstack).shape == refmask.shape
                #     # fits.PrimaryHDU(data=np.add(fits.getdata(mstack), refmask).astype(np.int16), header=fits.getheader(mstack)).writeto(mstack, overwrite=True)
                #     fits.PrimaryHDU(data=np.add(fits.getdata(mstack), refmask, np.where(fits.getdata(stack) == 0,1,0)).astype(np.int16), header=fits.getheader(mstack)).writeto(mstack, overwrite=True)
                # except:
                #     pass
                
                # 3.8. header update for mask
                with fits.open(mstack, 'update') as f:
                    for hdu in f:
                
                        hdu.header['OBJECT']    = field
                        hdu.header['FIELD1']    = os.path.basename(imlist[0]).split('.')[0].split('_')[1]
                        hdu.header['FIELD2']    = os.path.basename(imlist[0]).split('.')[1]
                        hdu.header['FILTER']    = band
                        hdu.header['NUMDITH']   = dither
                        hdu.header['NUMIMAGE']  = dither
                        hdu.header['CENTRA']    = centra
                        hdu.header['CENTDEC']   = centdec

                # os.system(f'chmod 777 {mstack}')
    
    return total

#%%
def catalogmaker(cat, path_output, path_cat, flagcut=0, pixscale=0.4, clsstar=0.8, refmaglower=14, refmagupper=17, apertures=['APER', 'AUTO'], figure=False, path_plot='./', start=None):

    """
    Input: SExtractor output catalog from qa4stackpro
    Output: mag zero-point, FWHM, 5sigma depth and ref matched catalog
    cat = r"(?P<field>\w+_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<filter>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<exptime>\d+sec)\.(?P<type>scaled)\.stack\.fits\.cat"
    
    date        = '20230518_CTIO'
    os.chdir(f'/data4/kmtntoo/stack/{date}/')
    cat = '/data4/kmtntoo/stack/20230518_CTIO/S230518h_0403.122-72.R.20230519.CTIO.240sec.scaled.stack.fits.cat'
    path_output = f'/data4/kmtntoo/result/phot/{date}/'
    path_cat    = f'/data4/kmtntoo/cat/'
    flagcut     = 0
    pixscale    = 0.4
    clsstar     = 0.8
    refmaglower = 14
    refmagupper = 17
    apertures   = ['APER', 'AUTO']
    figure      = False
    path_plot   = './'
    start       = None
    """

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
    if band == 'I':
        path_ref    = f'{path_cat}smss/'
        reftbl      = SMSSDR3_query(radec, path_ref)
        hdr['PHOTREF']  = 'SMSS DR3'
    else:
        path_ref    = f'{path_cat}apass/'
        try:
            reftbl  = ascii.read(f'{path_ref}apass_{field}.{radec}.cat')
        except FileNotFoundError:
            reftbl  = apass_query(int(radec[:3]), int(radec[3:]), radius=np.sqrt(2))
            reftbl.write(f'{path_ref}apass_{field}.{radec}.cat'.format(path_ref, radec), format='ascii', overwrite=True)
        hdr['PHOTREF']  = 'APASS DR9'

    # catalog file read
    intbl   = ascii.read(cat)
    seeing  = np.median(intbl[intbl['CLASS_STAR']>clsstar]['FWHM_IMAGE'] * pixscale)
    
    # matching with the reference
    thres   = 0
    while 1:
        thres  += 1
        if 'ALNRMS' not in hdr: rad     = 0.5
        else: rad = hdr['ALNRMS']*thres
        param_matching  = dict(intbl    = intbl,
                                reftbl   = reftbl,
                                inra     = intbl['ALPHA_J2000'], 
                                indec    = intbl['DELTA_J2000'],
                                refra    = reftbl['RAJ2000'], 
                                refdec   = reftbl['DEJ2000'],
                                sep      = rad)
        mtbl    = matching(**param_matching)
        if len(mtbl) > 500 or thres >= 5 or 'ALNRMS' not in hdr:
            break

    # zero-point calculation
    for aperture in apertures:
        
        mtbl            = mtbl[mtbl[f'MAG_{aperture}']!=99]
        inmagkey        = 'MAG_{}'.format(aperture)
        inmagerkey      = 'MAGERR_{}'.format(aperture)
        refmagkey       = '{}mag'.format(band)
        refmagerkey     = 'e_{}mag'.format(band)

        refmagerupper   = 0.10
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
        # depth   = limitmag(5, zp, 15/0.4, skysig) # aperture = 15"
        depth   = limitmag(5, zp, 5/0.4, skysig) # aperture = 5"
    else:
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
    # sources_with_error_around_02 = point_sources[np.abs(point_sources["MAGERR_APER"] - mag_error_threshold) <= 0.05]
    # depth = np.mean(sources_with_error_around_02["MAG_APER"])

    if figure:
        plt.figure(figsize=(8, 12))
        gs = gridspec.GridSpec(15, 15)
        plt.rcParams.update({'font.size': 14})

        plt.suptitle('ToO & APASS Photometry Comparison\nField : {}, MAG_{}, {} band\nMagnitude Difference RMSE : {} ABmag\n5$\sigma$ Image Depth : {} ABmag'.format(field, aperture, band, round(rmse,3), depth))
        ax_1d   = plt.subplot(gs[9:15, :10])
        ax_hist = plt.subplot(gs[9:15, 10:])
        ax_2d   = plt.subplot(gs[0:8, :14])

        # 8.1. magnitude differences
        ax_1d.scatter(mtbl['MAG_{}'.format(aperture)]+zp, magdif, c='crimson', marker='o', alpha=0.1, label='Field Stars')
        ax_1d.errorbar(mtbl['MAG_{}'.format(aperture)]+zp, magdif, yerr=magerr, ms=6, ls='', c='crimson', marker='o', capsize=4, capthick=1, alpha=0.1)
        ax_1d.set(xlabel=r'$m_{ToO}$ [ABmag]', ylabel=r'$m_{ToO} - m_{APASS}$ [ABmag]')
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
        add_colorbar(plot, clabel=r'$m_{ToO} - m_{APASS}$ [mag]', clim=[-0.5, 0.5])
        ax_2d.legend(loc='upper right')
        ax_2d.set(xlabel='X axis [pixel]', ylabel='Y axis [pixel]')
        ax_2d.set_xlim(-1000, 24000)
        ax_2d.set_ylim(-1000, 24000)
        plt.savefig(f'{path_plot}{field}_{band}_{aperture}_phot.png')
        plt.close()

    with fits.open(img, 'update') as f:
        for hdu in f:
            hdu.header['FWHM']      = (round(seeing, 2), f'Median seeing of point sources [arcsec]]')
            hdu.header['MAGZERO']   = (round(zp, 3), 'Photometric zero-point for MAG_AUTO [ABmag]')
            hdu.header['ZEROERR']   = (round(zper, 3), 'Standard deviation of MAG_ZERO [ABmag]')
            hdu.header['ZPSTAR']    = (len(otbl), 'The number of stars for MAG_ZERO calculation')
            hdu.header['RMSPHOT']   = (rmse, 'RMSE of PHOTREF mag - KMTN mag (3sigma clipped)')
            hdu.header['DEPTH5']    = (depth, '5sigma depth in terms of 5" aperture')
            hdu.header['MATCHRAD']  = (rad, 'Matching radius with PHOTREF [arcsec]')
            hdu.header['CLSSTAR']   = (clsstar, 'CLASS_STAR cut used for analysis')

    return 0

#%% ToO Result Checker
import csv
import os
from astropy.io import fits
from astropy.wcs import WCS

def csv_to_ds9reg(csv_path, wcs_option=True):
    """
    Converts a CSV file with source locations into a DS9 region file.
    
    Args:
    - csv_path: Path to the input CSV file.
    - wcs_option: If True, try to use WCS for conversion if a corresponding FITS file exists.
    """
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
        regionfile.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        
        if wcs is not None:
            regionfile.write("fk5\n")  # Use fk5 for celestial coordinates
        else:
            regionfile.write("physical\n")  # Use physical for pixel coordinates
        
        # Write each source as a point
        for row in reader:
            if wcs is not None:
                # Convert pixel coordinates to RA and DEC if WCS is available
                ra, dec = wcs.pixel_to_world_values(float(row['X_IMAGE']), float(row['Y_IMAGE']))
                regionfile.write(f"point({ra},{dec}) # point=circle\n")
            else:
                # Use pixel coordinates directly
                x = row['X_IMAGE']
                y = row['Y_IMAGE']
                regionfile.write(f"point({x},{y}) # point=circle\n")
#%%
def BPM_update(img, path_cfg):
    """
    img = '/data4/kmtntoo/scaled/20240423_SAAO/S240422ed_2807.125-28.R.20240423.SAAO.011748.tt.scaled.fits'
    path_cfg = '/data4/kmtntoo/config/'
    """
    
    # Load the FITS file and read the header and data (if required)
    with fits.open(img) as f:
        hdr = f[0].header
        obs = hdr['OBSERVAT']
        chip = hdr['CCDNAME'].lower()
        badamp = hdr['BADAMP']
        
    msk = img.replace(".scaled.", ".crmap.")
    bpm = f'{path_cfg}{obs}_BPM.{chip}{chip}.fits'
    mbpm = msk.replace(".crmap.", ".mask.")

    if not (os.path.exists(msk) and os.path.exists(bpm)):
        print(f'CR mask or BPM does not exist for {img}.')
        return
    
    print(f'CR mask and BPM exist for {img}.')
    
    mdata = fits.getdata(msk)
    bpmdata = fits.getdata(bpm)
    naxis1 = hdr['NAXIS1']
    bound = np.arange(0, naxis1 + 1, naxis1 // 8)

    # Vectorize the masking process for bad amps
    bads = 0
    for i, amp in enumerate(badamp):
        if amp == '1':
            mdata[:, bound[i]:bound[i+1]] = 4  # Use broadcasting instead of np.full_like
            bads += 1
    
    mbpmdata = mdata + bpmdata
    fits.writeto(mbpm, mbpmdata.astype(np.float32), overwrite=True)
    # fits.writeto(mbpm, mbpmdata.astype(np.int16), overwrite=True)

    # Update header information on the original image
    with fits.open(img, 'update') as f:
        for hdu in f:
            if bads < 3:
                hdu.header['PHOTQA'] = (True, 'Badpixels and bad amps masked out')
            else:
                hdu.header['PHOTQA'] = (False, 'Badpixels and bad amps masked out')
            hdu.header['BADMAP'] = (os.path.basename(bpm), 'Badpixels map')
    
    return

#%%
import multiprocessing
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from itertools import repeat
from astropy.time import Time
from astroquery.imcce import Skybot
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from requests.exceptions import ConnectionError
from matplotlib.patches import Circle, PathPatch
from astropy.visualization import ZScaleInterval, MinMaxInterval
#============================================================
#    FUNCTION
#============================================================
def trim(inim, position, size, outim='trim.fits'):
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
    '''
    # path = os.path.dirname(inim)
    # image = os.path.basename(inim)
    # outim = f'hd{image}'
    # convim = f'hc{image}'
    # com = f'hotpants -c t -n i -iu 60000 -tu 6000000000 -tl -100000 -v 0 -inim {inim} -tmplim {refim} -outim {outim} -oci {convim}'
    if stamp is None:
        com = f'hotpants -c {convdir} -n i -iu 6000000000 -il -100000 -tu 6000000000 -tl -100000 -v 0 -inim {inim} -tmplim {refim} -imi {inmsk} -tmi {refmsk} -outim {outim} -oci {convim} -nrx {nrx} -nry {nry}'
    else:
        com = f'hotpants -c {convdir} -n i -iu 6000000000 -il -100000 -tu 6000000000 -tl -100000 -v 0 -inim {inim} -tmplim {refim} -imi {inmsk} -tmi {refmsk} -outim {outim} -oci {convim} -ssf {stamp} -nrx {nrx} -nry {nry}'
    print(com)
    os.system(com)
#------------------------------------------------------------
def wcsremap(inim, refim, outim, path_com='/data8/wcsremap/wcsremap-1.0.1/wcsremap'):
    import os
    com = f'{path_com} -template {refim} -source {inim} -outim {outim}'
    print(com)
    os.system(com)
    # return outim
#------------------------------------------------------------
def invert_image(inim, outim):
    data, hdr = fits.getdata(inim, header=True)
    # w = WCS(inim)
    invdata = data*(-1)
    fits.writeto(outim, invdata, header=hdr, overwrite=True)
#------------------------------------------------------------
def sexcom(inim, conf_sex, conf_param, conf_conv, conf_nnw, det_thres, detectiondual=None):
    outcat = inim.replace('fits', 'cat')
    if detectiondual == None:
        sexcom = f'sex {inim} -c {conf_sex} -CATALOG_NAME {outcat} -PARAMETERS_NAME {conf_param} -FILTER_NAME {conf_conv} -STARNNW_NAME {conf_nnw} -DETECT_THRESH {det_thres}'
    else:
        sexcom = f'sex {detectiondual},{inim} -c {conf_sex} -CATALOG_NAME {outcat} -PARAMETERS_NAME {conf_param} -FILTER_NAME {conf_conv} -STARNNW_NAME {conf_nnw} -DETECT_THRESH {det_thres}'
    # print(sexcom)
    # os.system(sexcom)
    return sexcom
#------------------------------------------------------------
def findloc(inim, path_table="/data8/kmtntoo/config/tables"):
    loctbl = ascii.read(f'{path_table}/obs.location.smnet.dat')
    #    code == 500 : default
    code = 500
    for i, obs in enumerate(loctbl['obs']):
        if obs in inim:
            code = loctbl['Code'][i].item()
            break
    return code
#------------------------------------------------------------
def generate_snapshot(trtbl, i, cutsize=2.0, pixscale=0.4):
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
        hdu.header['MAGAUTO']   = (trtbl['mag_auto'][i], "transient candidate MAG_AUTO")
        hdu.header['SNR']       = (trtbl['snr'][i], "transient candidate SNR")
        hdu.header['SEEING']    = (trtbl['FWHM_IMAGE'][i] * 0.4, "transient candidate FWHM")
        hdu.header['ELONG']     = (trtbl['ELONGATION'][i], "transient candidate elongation")
        hdu.header['CLSSTAR']   = (trtbl['CLASS_STAR'][i], "transient candidate CLASS_STAR")
        hdu.header['ASTEROID']  = (trtbl['flag_0'][i], "moving object matched within 5arcsec")
        try:
            hdu.header['IMAFLAG']   = (trtbl['IMAFLAGS_ISO'][i], "Mask image flags")
            hdu.header['BLEEDING']  = (trtbl['ratio_saturation_line'][i], "Bleeding pattern ratio")
        except: pass
        # Write the cutout to a new FITS file
        outim = f'{os.path.splitext(hdim)[0]}.{n:0>6}.{kind}{os.path.splitext(hdim)[1]}'
        outpng = f'{os.path.splitext(hdim)[0]}.{n:0>6}.{kind}.png'
        #    Save postage stamp *.png & *.fits
        hdu.writeto(outim, overwrite=True)
        # plot_snapshot(data, wcs, peeing, outpng, save=True)
#------------------------------------------------------------
def plot_snapshot(data, wcs, peeing, outpng, save=True):
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

    from astropy.visualization.stretch import LinearStretch
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
    
# %% main: subtraction

def subtraction(sciimg, path_ref, path_cat, path_refcat, path_output, path_config, div_col=1, div_row=1, pixscale=0.4, ncore=1, detect=1.5, cutsize=1.0, align=False):
    """
    Input: Photometric catalog from KMTNet_ToO.catalogmaker()
    Input: Stacked KMTNet image after astrometry & photometry (Science & Reference)
    Output: science, reference and subtraction snapshot images for transient candidates
    sciimg = r"(?P<field>\w+_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<filter>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<exptime>\d+sec)\.(?P<type>scaled)\.stack\.fits"

    Args:
    date        = '20240423_CTIO'
    os.chdir(f'/data4/kmtntoo/stack/{date}')
    simg        = 'ToO_0749.121-30.R.20240423.CTIO.240sec.scaled.stack.fits'
    sciimg      = f'/data4/kmtntoo/stack/{date}/{simg}'
    path_ref    = '/data8/KS4/database/stack/'
    path_cat    = f'/data4/kmtntoo/result/phot/{date}/'
    path_refcat = '/data8/KS4/catalog/EDR/'
    path_output = f'/data4/kmtntoo/subt/{date}/'
    path_config = '/data4/kmtntoo/config/'
    div_col     = 1
    div_row     = 1
    pixscale    = 0.4
    ncore       = 4
    detect      = 3
    cutsize     = 1.0
    align       = False
    """

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
        print(f"Science image: \n{sciimg}")

    # reference image
    pattern_ks4 = f'{path_ref}{field}.{radec}/ks4*{band}*.scaled.stack.fits'
    refimg = find_longest_exposure_image(pattern_ks4)

    if refimg is None:
        pattern_ps1 = f'{path_ref}{field}.{radec}/ps1*{band}*.stack.fits'
        refimg = find_longest_exposure_image(pattern_ps1)
        # if refimg is None:
        #     pattern_ks4 = f'{path_ref}{field}.{radec}/ks4.*.scaled.stack.fits' # other filters?
        #     refimg = find_longest_exposure_image(pattern_ks4)

    if refimg:
        print(f"Reference image: \n{os.path.basename(refimg)}")
    else:
        print(f'No suitable reference image found. Checked patterns: \n{pattern_ks4}\n{pattern_ps1}')

    if (sciimg is None) or (refimg is None):
        return 0

    # mask image
    ref_shape = fits.getdata(refimg).shape
    maskimg = sciimg.replace(".scaled.", ".mask.")  # Define the output filename
    if os.path.exists(maskimg):
        print('The mask image already exists.')
        # return 0 # when you do not want duplicated processes.
    fits.PrimaryHDU(data=(safe_load_fits(sciimg.replace(".scaled.", ".crmap."), shape=ref_shape)
                          +safe_load_fits(refimg.replace(".scaled.", ".crmap."), shape=ref_shape)
                          +safe_load_fits(refimg.replace(".scaled.", ".bpm."), shape=ref_shape)
                          +np.where(fits.getdata(refimg)==0,1,0)+np.where(fits.getdata(sciimg)==0,1,0)).astype(np.int16), 
                    header=fits.getheader(sciimg)).writeto(maskimg, overwrite=True)
    print(f"Mask image: \n{os.path.basename(maskimg)}")

    # science image photometric catalog --> HOTPANTs subtraction stamps
    # science image catalog
    scicat  = ascii.read(f'{path_cat}{os.path.basename(sciimg)}.zp.cat') # mandatory
    scicat  = scicat[scicat['FLAGS']==0]
    scicat  = scicat[scicat['CLASS_STAR'] > 0.8]
    scicat  = scicat[(scicat['MAG_AUTO'] > 14) & (scicat['MAG_AUTO'] < 20)]
    try:
        # reference image catalog
        refband = re.search(r'\.(B|V|R|I)\.', refimg).group(1)
        refcat  = Table(fits.open(glob.glob(f'{path_refcat}{field}.{radec}/ks4_{field}.{radec}_{refband}_*.zp.fits')[0])[1].data)
        # refcat  = ascii.read(glob.glob(f'{path_refcat}{field}.{radec}/ks4_{field}.{radec}*{refband}*.zp.cat')[0])
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
                                sep      = 0.4)
        mtbl    = matching(**param_matching)
        stamp   = copy.deepcopy(mtbl)
    except:
        print(f"Check if the catalog is in the {path_cat} directory.")
        stamp  = None

    # sci HDU
    scihdu = fits.open(sciimg)[0]
    sciwcs = WCS(scihdu.header)
    scidat = scihdu.data
    scihdr = scihdu.header
    try:
        magautozero = scihdr['MAGZERO']
    except KeyError:
        magautozero = 30
        
    # ref HDU
    refhdu = fits.open(refimg)[0]
    refwcs = WCS(refhdu.header)
    refdat = refhdu.data
    refhdr = refhdu.header
    
    # mask HDU
    maskhdu = fits.open(maskimg)[0]
    maskwcs = WCS(maskhdu.header)
    maskdat = maskhdu.data
    maskhdr = maskhdu.header
    
    # Trim Configuration
    x_indx_ranges = np.arange(div_col)
    y_indx_ranges = np.arange(div_row)
    num_divisions = div_col*div_row

    print(f"Image Split: {div_col} x {div_row} = {num_divisions} Images")
    xshape, yshape = scidat.shape
    x_trim_step = int(xshape/div_col)
    y_trim_step = int(yshape/div_row)
    
    # trimming: ~90sec for 4x4 divisions sci & ref & mask
    for xx in x_indx_ranges:
        for yy in y_indx_ranges:
            # start
            tt  = xx * div_col + yy
            print(f"[{tt}] Trim --> Sector:({xx},{yy})")
            # science image trimming
            trim_sciimg = os.path.join(path_output, os.path.basename(rename_convention(sciimg, "Calib").replace('fits', f"trim_{xx}x{yy}.fits")))
            trim_scidat = scidat[yy*y_trim_step:(yy+1)*y_trim_step, xx*x_trim_step:(xx+1)*x_trim_step]
            trim_scihdr = scihdr.copy()
            trim_scihdr['TRIM'] = int(tt)
            trim_sciwcs = sciwcs[yy*y_trim_step:(yy+1)*y_trim_step, xx*x_trim_step:(xx+1)*x_trim_step]
            trim_scihdu = fits.PrimaryHDU(data=trim_scidat, header=trim_scihdr)
            trim_scihdu.header.update(trim_sciwcs.to_header())
            if os.path.exists(trim_sciimg): os.remove(trim_sciimg)
            trim_scihdu.writeto(trim_sciimg, overwrite=True)
            # reference image trimming
            trim_refimg = os.path.join(path_output, os.path.basename(rename_convention(refimg, "REF").replace('fits', f"trim_{xx}x{yy}.fits")))
            trim_refdat = refdat[yy*y_trim_step:(yy+1)*y_trim_step, xx*x_trim_step:(xx+1)*x_trim_step]
            trim_refhdr = refhdr.copy()
            trim_refhdr['TRIM'] = int(tt)
            trim_refwcs = refwcs[yy*y_trim_step:(yy+1)*y_trim_step, xx*x_trim_step:(xx+1)*x_trim_step]
            trim_refhdu = fits.PrimaryHDU(data=trim_refdat, header=trim_refhdr)
            trim_refhdu.header.update(trim_refwcs.to_header())
            if os.path.exists(trim_refimg): os.remove(trim_refimg)
            trim_refhdu.writeto(trim_refimg, overwrite=True)
            # mask image trimming
            trim_maskimg = os.path.join(path_output, os.path.basename(rename_convention(sciimg, "MASK").replace('fits', f"trim_{xx}x{yy}.fits")))
            trim_maskdat = maskdat[yy*y_trim_step:(yy+1)*y_trim_step, xx*x_trim_step:(xx+1)*x_trim_step]
            trim_maskhdr = maskhdr.copy()
            trim_maskhdr['TRIM'] = int(tt)
            trim_maskwcs = maskwcs[yy*y_trim_step:(yy+1)*y_trim_step, xx*x_trim_step:(xx+1)*x_trim_step]
            trim_maskhdu = fits.PrimaryHDU(data=trim_maskdat, header=trim_maskhdr)
            trim_maskhdu.header.update(trim_maskwcs.to_header())
            if os.path.exists(trim_maskimg): os.remove(trim_maskimg)
            trim_maskhdu.writeto(trim_maskimg, overwrite=True)

    # collecting image sets
    # trim_science_images     = sorted(glob.glob(f"{path_output}{os.path.basename(rename_convention(sciimg, 'Calib').replace('.fits',''))}.trim_?x?.PSF.fits"))
    trim_science_images     = sorted(glob.glob(f"{path_output}{os.path.basename(rename_convention(sciimg, 'Calib').replace('.fits',''))}.trim_?x?.fits"))
    trim_reference_images   = sorted(glob.glob(f"{path_output}{os.path.basename(rename_convention(refimg, 'REF').replace('.fits', ''))}.trim_?x?.fits"))
    trim_mask_images        = sorted(glob.glob(f"{path_output}{os.path.basename(rename_convention(sciimg, 'MASK').replace('.fits',''))}.trim_?x?.fits"))

    # check the number
    print(f"Number of Trimmed Science images  : {len(trim_science_images)}")
    print(f"Number of Trimmed Reference images: {len(trim_reference_images)}")
    print(f"Number of Trimmed Mask images     : {len(trim_mask_images)}")
    if len(trim_science_images) == len(trim_reference_images):
        print(f"--> Same Number of Sci & Ref Images")
    else:
        print(f"--> N_sci != N_ref. Check the process")
        # return 0

    # subtraction with HOTPANTs
    numbers = np.arange(num_divisions)
    """ number = (i+1) x (j+1) - 1
    +-----+-----+-----+-----+
    |  3  |  7  |  11 |  15 |
    +-----+-----+-----+-----+
    |  2  |  6  |  10 |  14 |
    +-----+-----+-----+-----+
    |  1  |  5  |  9  |  13 |
    +-----+-----+-----+-----+
    |  0  |  4  |  8  |  12 |
    +-----+-----+-----+-----+
    =========================
    +-----+-----+-----+-----+
    | 0,3 | 1,3 | 2,3 | 3,3 |
    +-----+-----+-----+-----+
    | 0,2 | 1,2 | 2,2 | 3,2 |
    +-----+-----+-----+-----+
    | 0,1 | 1,1 | 2,1 | 3,1 |
    +-----+-----+-----+-----+
    | 0,0 | 1,0 | 2,0 | 3,0 |
    +-----+-----+-----+-----+
    """
    for number in numbers:
        
        #   Target image set
        trim_sciimg = trim_science_images[number]
        trim_refimg = trim_reference_images[number]
        trim_scimsk = trim_mask_images[number]
        trim_refmsk = trim_mask_images[number]
        #   Align
        if align:
            align_refimg = trim_refimg.replace("REF", "wrREF")
            if os.path.exists(align_refimg): os.system(f"rm {align_refimg}")
        else:
            align_refimg = trim_refimg
        #   Subtraction
        trim_subt_image = trim_sciimg.replace("Calib", "hdCalib")
        convolved_refimg= align_refimg.replace("REF", "hcREF")
        convolved_sciimg= trim_sciimg.replace("Calib", "hcCalib")
        #   FWHM check
        try:
            ref_seeing  = float(fits.getheader(align_refimg)['FWHM'])
            sci_seeing  = float(fits.getheader(trim_sciimg)['FWHM'])
            if ref_seeing > sci_seeing:
                hinim   = align_refimg
                hrefim  = trim_sciimg
                hconvim = convolved_sciimg
            else:
                hinim   = trim_sciimg
                hrefim  = align_refimg
                hconvim = convolved_refimg
        except KeyError:
            hinim   = trim_sciimg
            hrefim  = align_refimg
            hconvim = convolved_refimg
        #   Range
        if stamp is not None:   
            xx  = number // div_row
            yy  = number % div_col
            section = copy.deepcopy(stamp)
            section   = section[(section['X_IMAGE'] > xx*x_trim_step) & (section['X_IMAGE'] < (xx+1)*x_trim_step)]
            section   = section[(section['Y_IMAGE'] > yy*y_trim_step) & (section['Y_IMAGE'] < (yy+1)*y_trim_step)]
            stampname   = f"{path_output}{os.path.basename(trim_sciimg).replace('.fits', '.stamp')}"
            stampstars  = 0
            with open(stampname, "w") as f:
                for s in section:
                    f.write(f"{s['X_IMAGE']-xx*x_trim_step} {s['Y_IMAGE']-yy-y_trim_step} \n")
                    stampstars  += 1
            if stampstars > 20:
                hotpants(inim=hinim, refim=hrefim, outim=trim_subt_image, inmsk=trim_scimsk, refmsk=trim_refmsk, convim=hconvim, stamp=stampname)
            else:
                hotpants(inim=hinim, refim=hrefim, outim=trim_subt_image, inmsk=trim_scimsk, refmsk=trim_refmsk, convim=hconvim)
        else:
            #if os.path.exists(trim_subt_image) == False & (os.path.exists(convolved_refimg) == False):
            hotpants(inim=hinim, refim=hrefim, inmsk=trim_scimsk, refmsk=trim_refmsk, outim=trim_subt_image, convim=hconvim)
        #   Invert the subtraction in case REF-SCI
        if hinim == align_refimg and hrefim == trim_sciimg:
            invert_image(inim=trim_subt_image, outim=trim_subt_image)
            shutil.copy(align_refimg, align_refimg.replace("REF", "hcREF"))
    
    # transient search
    conv, nnw, param    = 'transient.kmtnet.conv', 'transient.kmtnet.nnw', 'transient.kmtnet.param'
    # sex  = f"{sciimg}.sex"
    # if os.path.isfile(sex):
    #     print(f"Science image configuration: \n{os.path.basename(sex)}")
    # else:
    sex     = 'transient.kmtnet.sex'
        # print(f'Check if SExtractor configuration exists: \n{sex}')
    conf_sex    = os.path.join(path_config, sex)
    conf_param  = os.path.join(path_config, param)
    conf_nnw    = os.path.join(path_config, nnw)
    conf_conv   = os.path.join(path_config, conv)

    # flagging
    frac    = 0.99      # flag3: edge detection
    sep     = 5.0       # flag0: matching radius [arcsec]
    fovval  = 1.0*60    # flag0: solar object searching radius[arcmin]
    ellipcut= 4         # flag5
    flagcut = 4         # flag6
    fwhmcut = [0.7,2.4] # flag7
    # backcut = 50        # flag8
    snrcut  = 5         # flag9
    satcut  = 0.7       # flag13: bleeding pattern detection criteria
    
    for number in numbers:
        #    Input Images
        trim_sciimg = trim_science_images[number]
        trim_refimg = trim_reference_images[number]
        trim_maskimg= trim_mask_images[number]
        print(f"[{number}] {os.path.basename(trim_sciimg)}")

        #    Middle-step images
        if align:
            align_refimg = trim_refimg.replace("REF", "wrREF")
            if os.path.exists(align_refimg): os.system(f"rm {align_refimg}")
        else:
            align_refimg = trim_refimg
        trim_subt_image = trim_sciimg.replace("Calib", "hdCalib")
        convolved_refimg= align_refimg.replace("REF", "hcREF")

        #------------------------------------------------------------
        #    Invert the images
        #------------------------------------------------------------
        inverse_trim_subt_image = trim_subt_image.replace("hd", "invhd")
        inverse_convolved_refimg= convolved_refimg.replace("hc", "invhc")
        
        invert_image(inim=trim_subt_image, outim=inverse_trim_subt_image)
        invert_image(inim=convolved_refimg, outim=inverse_convolved_refimg)
        #------------------------------------------------------------
        #    Photometry
        #------------------------------------------------------------
        #   mask
        try:
            fwhmcom     = f' -SEEING_FWHM {fits.getheader(trim_sciimg)["FWHM"]}'
        except:
            fwhmcom     = ''
        maskcom     = f' -FLAG_IMAGE {trim_maskimg} -FLAG_TYPE OR'
        #    Subtraction
        os.system(sexcom(trim_subt_image, conf_sex, conf_param, conf_conv, conf_nnw, det_thres=detect)+maskcom+fwhmcom)#, detectiondual=trim_sciimg))
        #    Convolved Reference
        os.system(sexcom(convolved_refimg, conf_sex, conf_param, conf_conv, conf_nnw, det_thres=detect)+maskcom+fwhmcom)#, detectiondual=trim_sciimg))
        #    Inverted Subtraction
        os.system(sexcom(inverse_trim_subt_image, os.path.join(path_config, "transient.sex"), conf_param, conf_conv, conf_nnw, det_thres=detect)+maskcom+fwhmcom)#, detectiondual=trim_sciimg))
        #    Inverted Reference
        os.system(sexcom(inverse_convolved_refimg, os.path.join(path_config, "transient.sex"), conf_param, conf_conv, conf_nnw, det_thres=detect)+maskcom+fwhmcom)#, detectiondual=trim_sciimg))
        
        #------------------------------------------------------------
        #    Output Catalog
        #------------------------------------------------------------
        #    Subtraction
        trim_subt_cat       = trim_subt_image.replace("fits", "cat")
        #    Convolved Reference
        convolved_ref_cat   = convolved_refimg.replace("fits", "cat")
        #    Inverted Subtraction
        inverse_subt_cat    = inverse_trim_subt_image.replace("fits", "cat")
        #    Inverted Reference
        inverse_convolved_ref_cat = inverse_convolved_refimg.replace("fits", "cat")
        #------------------------------------------------------------
        #    Read Catalog
        #------------------------------------------------------------
        xx  = number // div_row
        yy  = number % div_col
        scitbl  = copy.deepcopy(scicat)
        scitbl  = scitbl[scitbl["X_IMAGE"] > xx*x_trim_step]
        scitbl  = scitbl[scitbl["X_IMAGE"] < (xx+1)*x_trim_step]
        scitbl  = scitbl[scitbl["Y_IMAGE"] > yy*y_trim_step]
        scitbl  = scitbl[scitbl["Y_IMAGE"] < (yy+1)*y_trim_step]
        """
        for xx in x_indx_ranges:
            for yy in y_indx_ranges:
                # start
                tt  = xx * div_col + yy
                print(f"[{tt}] Trim --> Sector:({xx},{yy})")
                # science image trimming
                trim_sciimg = os.path.join(path_output, os.path.basename(rename_convention(sciimg, "Calib").replace('fits', f"trim_{xx}x{yy}.fits")))
                trim_scidat = scidat[yy*y_trim_step:(yy+1)*y_trim_step, xx*x_trim_step:(xx+1)*x_trim_step]
        """
        subtbl      = ascii.read(trim_subt_cat)
        reftbl      = ascii.read(convolved_ref_cat)
        invsubtbl   = ascii.read(inverse_subt_cat)
        invreftbl   = ascii.read(inverse_convolved_ref_cat)
        print(f"# Number of sources: {len(subtbl)}")
        #------------------------------------------------------------
        #    Mark the Flags
        #------------------------------------------------------------
        xcent   = int(x_trim_step/2)
        ycent   = int(y_trim_step/2)
        #------------------------------------------------------------
        subtbl = ascii.read(trim_subt_cat)
        subtbl['inim']  = trim_sciimg
        subtbl['hcim']  = convolved_refimg
        subtbl['hdim']  = trim_subt_image
        subtbl['mask']  = trim_maskimg
        subtbl['snr']   = 1/(subtbl['MAGERR_AUTO']+0.0001)
        # subtbl['snr'][np.isinf(subtbl['snr'])] = 9999
        #    Seeing
        subtbl.meta['SEEING']   = np.median(scitbl['FWHM_IMAGE']*0.4)
        # subtbl.meta['SEEING']   = np.sqrt(np.median(scitbl['FWHM_WORLD']*3600)**2 + np.median(refcat['FWHM_WORLD']*3600)**2)
        subtbl['ratio_seeing']  = subtbl['FWHM_WORLD']*3600/subtbl.meta['SEEING']
        #    Ellipticity
        subtbl.meta['ELLIPTICITY']  = np.median(scitbl['ELLIPTICITY'])
        # subtbl.meta['ELLIPTICITY']  = np.median(1-(1/scitbl['ELONGATION']))
        subtbl['ratio_ellipticity'] = subtbl['ELLIPTICITY']/subtbl.meta['ELLIPTICITY']
        #    Elongation
        subtbl.meta['ELONGATION']   = np.median(scitbl['ELONGATION'])
        subtbl['ratio_elongation']  = subtbl['ELONGATION']/subtbl.meta['ELONGATION']
        #    Magnitude Calibration (ZP=30)
        subtbl['mag_auto']  = subtbl['MAG_AUTO'] + magautozero

        w = WCS(trim_subt_image)
        #    Positional information
        c_cent = w.pixel_to_world(xcent, ycent)
        c_sub = SkyCoord(subtbl['ALPHA_J2000'], subtbl['DELTA_J2000'], unit=u.deg)

        flagnumbers = np.arange(14)
        #    Generate flag columns
        for num in flagnumbers:
            subtbl[f'flag_{num}'] = False
        epoch = Time(fits.getheader(trim_sciimg)['DATE-OBS'], format='isot')
        #    
        if "CTIO" in trim_sciimg:
            location="807"
        elif "SSO" in trim_sciimg:
            location="Q60"
        elif "SAAO" in trim_sciimg:
            location="M22"
        
        #------------------------------------------------------------
        #    flag 0
        #------------------------------------------------------------
        #    Skybot query
        # try: 
        #     sbtbl = Skybot.cone_search(c_cent, fovval*u.arcmin, epoch, location=location)
        #     c_sb = SkyCoord(sbtbl['RA'], sbtbl['DEC'])
        #     sbtbl['sep'] = c_cent.separation(c_sb).to(u.arcmin)
        #     #    Skybot matching
        #     indx_sb, sep_sb, _ = c_sub.match_to_catalog_sky(c_sb)
        #     subtbl['flag_0'][(sep_sb.arcsec<sep)] = True
        # except RuntimeError:
        #     print(f'No solar system object was found in the requested FOV ({fovval} arcmin)')
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
        #    flag 1
        #------------------------------------------------------------
        if len(invsubtbl)>0:
            #    Coordinate
            c_invhd = SkyCoord(invsubtbl['ALPHA_J2000'], invsubtbl['DELTA_J2000'], unit=u.deg)
            #    Matching with inverted images
            indx_invhd, sep_invhd, _ = c_sub.match_to_catalog_sky(c_invhd)
            subtbl['flag_1'][(sep_invhd.arcsec<(subtbl.meta['SEEING'])/2) | (subtbl['mag_auto'] - invsubtbl[indx_invhd]['MAG_AUTO'] - magautozero > -0.1)] = True
            # subtbl['flag_1'][(sep_invhd.arcsec<(subtbl.meta['SEEING'])/2) | (np.abs(subtbl['mag_auto'] - invsubtbl[indx_invhd]['MAG_AUTO'] - magautozero) < 1.0)] = True
        
        else:
            print('Inverted subtraction image has no source. ==> pass flag1')
            pass
        #------------------------------------------------------------
        #    flag 2
        #------------------------------------------------------------
        if len(invreftbl)>0:
            #    Coordinate
            c_invhc = SkyCoord(invreftbl['ALPHA_J2000'], invreftbl['DELTA_J2000'], unit=u.deg)
            #    Matching with inverted images
            indx_invhc, sep_invhc, _ = c_sub.match_to_catalog_sky(c_invhc)
            subtbl['flag_2'][(sep_invhc.arcsec<(subtbl.meta['SEEING'])/2) | (subtbl['mag_auto'] - invreftbl[indx_invhc]['MAG_AUTO'] - magautozero > -0.5)] = True
            # subtbl['flag_2'][(sep_invhc.arcsec<(subtbl.meta['SEEING'])/2) | (np.abs(subtbl['mag_auto'] - invreftbl[indx_invhc]['MAG_AUTO'] - magautozero) < 1.0)] = True
           
        else:
            print('Inverted reference image has no source. ==> pass flag2')
            pass
        
        #------------------------------------------------------------
        #    SEtractor criterion
        #------------------------------------------------------------
        #    flag 3
        #------------------------------------------------------------
        #    Sources @edge
        subtbl['flag_3'][
            ((subtbl['X_IMAGE'] < xcent - xcent*frac) |
            (subtbl['X_IMAGE']  > xcent + xcent*frac) |
            (subtbl['Y_IMAGE']  < ycent - ycent*frac) |
            (subtbl['Y_IMAGE']  > ycent + ycent*frac))
            ] = True
        #------------------------------------------------------------
        #    flag 4 --> skip
        #------------------------------------------------------------
        #    More than 5 sigma signal
        # subtbl['flag_4'][(subtbl['mag_aper']>hdr['ul5_1'])] = True
        #    Empirical criterion
        #------------------------------------------------------------
        #    flag 5
        #------------------------------------------------------------
        subtbl['flag_5'][(subtbl['ratio_ellipticity'] > ellipcut)] = True
        # subtbl['flag_5'][(subtbl['ELLIPTICITY'] > 0.5)] = True
        #------------------------------------------------------------
        #    flag 6
        #------------------------------------------------------------
        # subtbl['flag_6'][(subtbl['FLAGS'] > flagcut)] = True
        subtbl['flag_6'][(subtbl['FLAGS'] >= flagcut) |
                         (subtbl['IMAFLAGS_ISO'] > 0)
                         ] = True
        #------------------------------------------------------------
        #    flag 7
        #------------------------------------------------------------
        subtbl['flag_7'][
            # (subtbl['FWHM_WORLD']*3600>subtbl.meta['SEEING']*3.0) |
            # (subtbl['FWHM_WORLD']*3600<subtbl.meta['SEEING']*0.5)
            # (subtbl['FWHM_WORLD']*3600<subtbl.meta['SEEING']*0.8) |
            # (subtbl['FWHM_WORLD']*3600>subtbl.meta['SEEING']*2.0) 
            (subtbl['ratio_seeing']<fwhmcut[0]) |
            (subtbl['ratio_seeing']>fwhmcut[1])
            ] = True
        #------------------------------------------------------------
        #    flag 8
        #------------------------------------------------------------
        subtbl['flag_8'][
            (subtbl['BACKGROUND']<np.median(subtbl['BACKGROUND'])-2*np.std(subtbl['BACKGROUND'])) |
            (subtbl['BACKGROUND']>np.median(subtbl['BACKGROUND'])+2*np.std(subtbl['BACKGROUND']))
            ] = True
        #------------------------------------------------------------
        #    flag 9 --> skip
        #------------------------------------------------------------
        subtbl['flag_9'][
            (subtbl['snr']<snrcut)
        ] = True
        #------------------------------------------------------------
        #    flag 10+11
        #------------------------------------------------------------
        data = fits.getdata(trim_subt_image)
        peeing = subtbl.meta['SEEING']/pixscale
        skyval = np.median(subtbl['BACKGROUND'])
        skysig = np.std(subtbl['BACKGROUND'])

        nbadlist    = []
        ratiobadlist= []
        nnulllist   = []

        subtbl['n_bad'] = 0
        subtbl['ratio_bad'] = 0.0
        subtbl['n_null'] = 0
        
        #    Fraction
        f = 0.1
        for i, (tx, ty, bkg) in enumerate(zip(subtbl['X_IMAGE'], subtbl['Y_IMAGE'], subtbl['BACKGROUND'])):

            # tx, ty = subtbl['X_IMAGE'][i], subtbl['Y_IMAGE'][i]
            # bkg = subtbl['BACKGROUND'][i]
            #    Snapshot
            tsize = peeing
            y0, y1 = int(ty-tsize), int(ty+tsize)
            x0, x1 = int(tx-tsize), int(tx+tsize)
            cdata = data[y0:y1, x0:x1]
            # plt.close()
            # plt.imshow(cdata)
            crt = bkg - skysig*25
            cutline = cdata.size*f
            nbad = len(cdata[cdata<crt])
            try:
                ratiobad = nbad/cdata.size
            except:
                ratiobad = -99.0
            nnull = len(np.where(cdata == 1e-30)[0])
            #    Dipole
            if nbad > cutline:
                subtbl['flag_10'][i] = True
            #    HOTPANTS Null value
            if nnull != 0:
                subtbl['flag_11'][i] = True

            subtbl['n_bad'][i] = nbad
            subtbl['ratio_bad'][i] = ratiobad
            subtbl['n_null'][i] = nnull

        #------------------------------------------------------------
        #    flag 12
        #------------------------------------------------------------
        # x, y = fits.getdata(hcim).shape
        # w_ref = WCS(hcim)
        # xim, yim = w_ref.world_to_pixel(c_sub)
        # indx_nosci = np.where(
        #     (xim < 0) | (xim > x) | (yim < 0) | (yim > y)
        # )
        # subtbl['flag_12'][indx_nosci] = True
        x, y = fits.getdata(convolved_refimg).shape
        w_ref = WCS(convolved_refimg)
        xim, yim = w_ref.world_to_pixel(c_sub)
        subtbl['x_refim'] = xim
        subtbl['y_refim'] = yim
        indx_nosci = np.where(
            (xim < 0) | (xim > x) | (yim < 0) | (yim > y)
        )
        subtbl['flag_12'][indx_nosci] = True
        #------------------------------------------------------------
        #    flag 13: Bleeding pattern rejection
        #------------------------------------------------------------
        # subtbl['n_saturation_line'] = int(0)
        # subtbl['ratio_saturation_line'] = 0.0
        # npixsnap    = cutsize*60/pixscale
        # pad         = 10
        # y_to_check  = np.array([num for num in np.arange(5, npixsnap, pad) if num!=npixsnap/2])

        # for nn, (tx, ty) in enumerate(zip(subtbl['X_IMAGE'], subtbl['Y_IMAGE'])):
        #     tsize = int((fovval/pixscale)/2)
        #     y0, y1 = int(ty-tsize), int(ty+tsize)
        #     x0, x1 = int(tx-tsize), int(tx+tsize)
        #     _data = data[y0:y1, x0:x1]
        #     if _data.shape == (npixsnap, npixsnap):
        #         ct = 0
        #         xshp, yshp = _data.shape
        #         if (xshp > npixsnap-pad) & (yshp > npixsnap-pad) & (xshp == yshp):
        #             for ny, yy in enumerate(y_to_check):
        #                 center_mean_val = np.mean(_data[int(yy), int(xshp/2)-3:int(xshp/2)+3])
        #                 local_bkg = np.median(_data[int(yy), :])
        #                 ct += 1 if center_mean_val > local_bkg else 0
        #             # print(f"N={nn}: {ct}")
        #             subtbl['n_saturation_line'][nn] = ct
        #             subtbl['ratio_saturation_line'][nn] = ct/len(y_to_check)
        # subtbl['flag_13'][subtbl['ratio_saturation_line'] > satcut] = True
        subtbl['flag_13'] = False
        
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
        transient_cat   = trim_subt_cat.replace('cat', 'transient.cat')
        subtbl.write(transient_cat, format='ascii.tab', overwrite=True)
        print('-'*60)
        print(f'Filtered sources\t: {len(trtbl)} ({100*len(trtbl)/len(subtbl):1.3f})%')
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
        print(f"#\tSnapshot maker ({len(trtbl)})")
        if len(trtbl) > 0:
            if ncore == 1:
                for i in range(len(trtbl)):
                    generate_snapshot(trtbl, i, cutsize)
            #    Multi Thread
            else:
                with multiprocessing.Pool(processes=ncore) as pool:
                    results = pool.starmap(generate_snapshot, zip(repeat(trtbl), np.arange(len(trtbl)), repeat(cutsize)))
        else:
            print('No transient candidates.')
        
    print("All Done")

    return 0