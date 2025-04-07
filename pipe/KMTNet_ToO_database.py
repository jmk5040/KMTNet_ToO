#%% Import packages
import os
import glob
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
import astropy.units as u
import os, re, glob, time
from astropy.wcs import WCS
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.table import vstack
from astropy.io import ascii, fits
from matplotlib.patches import Rectangle
from astropy.coordinates import SkyCoord
from multiprocessing import Pool, cpu_count
from astropy.visualization import ZScaleInterval, LinearStretch
#%% functions
import warnings

def plot_snapshot(data, wcs, outpng, save=True):
    plt.close('all')
    plt.rc('font', family='serif')

    # Figure and axes setup
    fig = plt.figure(figsize=(1, 1))
    fig.set_size_inches(1. * data.shape[0] / data.shape[1], 1, forward=False)
    x = 720 / fig.dpi
    y = 720 / fig.dpi
    fig.set_figwidth(x)
    fig.set_figheight(y)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Data preparation
    data[np.isnan(data)] = 0.0
    transform = LinearStretch() + ZScaleInterval()
    bdata = transform(data)

    # Image display
    ax.imshow(bdata, cmap="gray", origin="lower")

    # Add rectangle
    rect_size = 10
    rectangle = Rectangle(
        (data.shape[1] / 2 - rect_size / 2, data.shape[0] / 2 - rect_size / 2),
        rect_size,
        rect_size,
        edgecolor='yellow',
        lw=1,
        facecolor='none',
        fill=False
    )
    ax.add_patch(rectangle)

    # Handle axis inversion based on WCS
    ra0, dec0 = wcs.all_pix2world(0, 0, 1)
    ra1, dec1 = wcs.all_pix2world(data.shape[0], data.shape[1], 1)
    if ra0 > ra1:
        ax.invert_xaxis()
    if dec0 < dec1:
        ax.invert_yaxis()

    # Save figure
    if save:
        # Suppress the specific warning about tight_layout
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.savefig(outpng, dpi=100)

def utc_to_mjd(utc_time):
    from datetime import datetime, timedelta
    # Reference date for MJD (JD = 2400000.5 corresponds to MJD = 0)
    mjd_reference = datetime(1858, 11, 17, 0, 0, 0)

    # Parse the UTC time
    utc_time_parsed = datetime.strptime(utc_time, "%Y-%m-%d %H:%M:%S")

    # Calculate the difference in days
    delta = utc_time_parsed - mjd_reference

    # Convert to fractional days
    mjd = delta.days + delta.seconds / 86400 + delta.microseconds / 86400e6

    return mjd

def process_image(real, snapdir='./snap/', outdir='./png/'):
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    for kind in ['new', 'ref', 'sub']:
        with fits.open(f"{snapdir}{os.path.basename(real['hdim'])[:-5]}.{real['NUMBER']:0>6}.{kind}.fits") as hdul:
            data = hdul[0].data
            wcs = WCS(hdul[0].header)
            outname = f"{outdir}{os.path.basename(real['hdim'])[:-5]}.{real['NUMBER']:0>6}.{kind}.png"
            plot_snapshot(data, wcs, outname, save=True)

def generate_snapshots(realcat, threads=4):
    with Pool(processes=threads) as pool:
        list(tqdm(pool.imap(process_image, [real for index, real in realcat.iterrows()]), total=len(realcat), desc='Generating snapshot images'))

def move_file(file, path_snap, path_ftr, path_trans):
    source_file = join(path_snap, file)
    target_directory = path_trans if '.new.' in file or '.ref.' in file or '.sub.' in file else path_ftr if '.fits' in file else path_trans
    try:
        shutil.move(source_file, target_directory)
    except:
        pass

def move_files(allfiles, path_snap, path_ftr, path_trans, threads=3):
    from functools import partial

    move_file_partial = partial(move_file, path_snap=path_snap, path_ftr=path_ftr, path_trans=path_trans)
    with Pool(processes=threads) as pool:
        list(tqdm(pool.imap(move_file_partial, allfiles), total=len(allfiles), desc="Processing files"))

#%%
# Setup paths and directories
path_base = '/data4/kmtntoo/tutorial/'
path_data = join(path_base, 'data/')
path_dest = '/data6/GECKO/'
path_subt = join(path_data, 'subt/')
path_stack= join(path_data, 'stack/')
#%%
dates   = [os.path.basename(date) for date in sorted(glob.glob(f'{path_subt}*_*'))]

print(f"List of Data Directories in {path_subt}:")
print("="*20)
for directory in dates:
    print(directory)
print("="*20)
#%%
event   = 'S250206dm'
date = input(f'Enter the directory name to process for {event}: ')

subtimgs= sorted(glob.glob(f'{path_subt}{date}/hdCalib*.stack.fits'))
ftrs    = list(set([os.path.basename(img).split('.')[4] for img in subtimgs]))
obses   = list(set([os.path.basename(img).split('.')[1].replace('KMTNet', 'KMTN') for img in subtimgs]))
#%% Database input directory structures
for ftr in ftrs:
    for obs in obses:
        
        path_event = join(path_dest, event)
        path_obs = join(path_event, obs)
        path_ftr = join(path_obs, ftr)
        path_trans = join(path_ftr, 'transients')
        path_snap = join(path_subt, date)

        os.makedirs(path_event, exist_ok=True)
        os.makedirs(path_obs, exist_ok=True)
        os.makedirs(path_ftr, exist_ok=True)
        os.makedirs(path_trans, exist_ok=True)
        
        os.chmod(path_event, 0o777)
        os.chmod(path_obs, 0o777)
        os.chmod(path_ftr, 0o777)
        os.chmod(path_trans, 0o777)


#%%
# Process files
allfiles = os.listdir(path_snap)
trancats = sorted(glob.glob(join(path_snap, '*transient.cat')))
rbclass = pd.read_csv(join(path_snap, 'snap/rbscore.csv'))
# rbclass = pd.read_csv(join(path_snap, 'rbscore_ri+ngi+gd.csv'))

os.chdir(path_snap)
for tran in trancats:

    trancat = pd.read_csv(tran, sep='\t')
    trancat['id'] = trancat['hdim'].apply(lambda x: os.path.basename(x).split(".fits")[0]) + '.' + trancat['NUMBER'].astype(str).str.zfill(6)
    rbcat = pd.merge(trancat, rbclass, on='id', how='left')
    dateobs     = fits.getheader(list(set(list(rbcat['inim'])))[0])['DATE-OBS']
    rbcat['TRANRA']     = rbcat['ALPHA_J2000']
    rbcat['TRANDEC']    = rbcat['DELTA_J2000']
    rbcat['DATE-OBS']   = dateobs
    rbcat.to_csv(f'{tran.replace("transient", "transientRB")}', index=False)
    realcat = rbcat[rbcat['prob'] >= 0.5]
    generate_snapshots(realcat)
try:
    os.system(f'cp {path_snap}/snap/rbscore.csv {path_snap}/snap/rbscore_{date}.csv')
except FileNotFoundError:
    pass
# Move files
#%%
# ftr     = os.path.basename(tran).split('.')[4]
# obs     = os.path.basename(tran).split('.')[1].replace('KMTNet', 'KMTN')
# allfiles1 = [f for f in os.listdir(path_snap) if os.path.isfile(os.path.join(path_snap, f))]
for ftr in ftrs:
    for obs in obses:
        print("Processing: ", f"{obs} {ftr}")
        path_obs    = join(path_event, obs)
        path_snap   = join(path_subt, date)
        path_ftr    = join(path_obs, ftr)
        path_trans  = join(path_ftr, 'transients')
        allfiles1 = sorted(glob.glob(f'{path_snap}/*{obs.replace("KMTN", "KMTNet")}*.{ftr}.*'))
        move_files(allfiles1, path_snap, path_ftr, path_trans)
        allfiles2 = [os.path.basename(x) for x in sorted(glob.glob(f'{path_snap}/png/*{obs.replace("KMTN", "KMTNet")}*.{ftr}.*.png'))]
        move_files(allfiles2, join(path_snap, 'png'), path_ftr, path_trans)
        allfiles3 = [x.replace('.png', '.fits') for x in allfiles2]
        move_files(allfiles3, join(path_snap, 'snap'), path_ftr, path_trans)
        print("Remaining Files: ", len(os.listdir(path_snap)), len(os.listdir(os.path.join(path_snap, 'png'))), len(os.listdir(os.path.join(path_snap, 'snap'))))

#%% Treasure Map
fields = """
222
223
224
292
293
294
295
296
297
298
299
359
360
361
362
363
364
365
366
422
423
424
425
""".split()
GWmjd = utc_to_mjd("2025-02-06 21:25:44")
dates   = [os.path.basename(date) for date in sorted(glob.glob(f'{path_subt}*_*'))]
for date in dates:
    stkimgs     = sorted(glob.glob(f'{path_stack}{date}/*.stack.fits'))
    for img in stkimgs:
        hdr = fits.getheader(img)
        try:
            if str(int(hdr['FIELD1'])) in fields:
                print(f"{os.path.basename(img)} {hdr['FIELD1']} {hdr['FILTER']} {hdr['OBSERV0']} {hdr['CENTRA']} {hdr['CENTDEC']} {hdr['DATE-OBS']} {utc_to_mjd(hdr['DATE-OBS'].replace('T',' '))-GWmjd} {hdr['DEPTH5']} {hdr['FWHM']}")
        except:
            pass
#%%
for obs in ['KMTN_CTIO', 'KMTN_SAAO']:
    for band in ['R', 'I']:
        hdimgs = sorted(glob.glob(f'{path_event}/{obs}/{band}/hdCalib.*.stack.fits'))
        for img in hdimgs:
            hdr = fits.getheader(img)
            print(f"{os.path.basename(img)} {hdr['FIELD1']} {hdr['FILTER']} {hdr['OBSERV0']} {hdr['CENTRA']} {hdr['CENTDEC']} {hdr['DATE-OBS']} {utc_to_mjd(hdr['DATE-OBS'].replace('T',' '))-GWmjd} {hdr['DEPTH5']} {hdr['FWHM']}")
#%%
for i, date in enumerate(dates[-6:]):
    print(date)
    when = date.split('_')[0]
    obs = 'KMTN_'+date.split('_')[1]
    bands = 'I R I R R I'.split()
    band = bands[i]
    hdimgs = sorted(glob.glob(f'{path_event}/{obs}/{band}/hdCalib.*{when}*.stack.fits'))
    for field in fields:
        for img in hdimgs:
            if f"_{field.zfill(4)}." in os.path.basename(img):
                hdr = fits.getheader(img)
                print(int(hdr['EXPTIME']))
                break
        else:
            print(0)
        



# %%
# from collections import Counter
# # requested fields (scTOO.cat && observation log)
# # update this field list with the date of interest
# rfields = """
# #2025
# #2025
# #2025
# #2025
# #2025
# #2025
# #2025
# #2025
# #2149
# #2149
# #2149
# #2149
# #2149
# #2149
# #2149
# #2149
# #2147
# #2147
# #2147
# #2147
# #2147
# #2147
# #2147
# #2147
# #2148
# #2148
# #2148
# #2148
# #2148
# #2148
# #2148
# #2148
# """.split()
# field_counts = Counter(field.strip('#') for field in rfields)

# # Display the dictionary
# print(field_counts)
# # %%
# # total field numbers of S240422ed localization area
# fields = """
# 2025
# 2147
# 2148
# 2149
# """.split()
# # %%
# # exposure times for each field
# # copy & paste to the spreadsheet
# for field in fields:
#     count = field_counts.get(field, 0)  # This gets the count from field_counts or returns 0 if not found
#     print(120*count)
# # %%
# import re
# from astropy.io import fits, ascii
# from astropy.table import Table

# simg        = 'S240915b_2149.115-50.R.20240915.CTIO.960sec.stack.fits'
# date        = '240915_CTIO'
# simg        = 'S240915b_2149.115-50.R.20240915.CTIO.960sec.stack.fits'
# sciimg      = f'/data4/kmtntoo/tutorial/data/stack/{date}/{simg}'
# path_ref    = '/data4/kmtntoo/tutorial/data/template/'
# path_cat    = f'/data4/kmtntoo/tutorial/data/stack/{date}/'
# path_refcat = '/data4/kmtntoo/tutorial/data/template/'
# path_output = f'/data4/kmtntoo/tutorial/data/subt/{date}/'
# path_config = '/data4/kmtntoo/tutorial/config/'
# scicat  = ascii.read(f'{path_cat}{os.path.basename(sciimg)}.zp.cat') # mandatory
# scicat  = scicat[scicat['FLAGS']==0]
# scicat  = scicat[scicat['CLASS_STAR'] > 0.8]
# scicat  = scicat[(scicat['MAG_AUTO'] > 14) & (scicat['MAG_AUTO'] < 20)]
# field   = sciimg.split('.')[0].split('_')[-1]      # 0000
# radec   = sciimg.split('.')[1]                     # 000-00
# band    = sciimg.split('.')[2]                     # B|V|R|I
# date    = sciimg.split('.')[3]                     # 20000000
# site    = sciimg.split('.')[4]                     # SAAO|SSO|CTIO
# exptime = int(sciimg.split('.')[5].split('sec')[0])
# # %%
# refimg = '/data4/kmtntoo/tutorial/data/template/2149.115-50/ks4.2149.115-50.R.720sec.reduced.scaled.stack.fits'
# refband = re.search(r'\.(B|V|R|I)\.', refimg).group(1)
# refcat  = Table(fits.open(glob.glob(f'{path_refcat}{field}.{radec}/ks4_{field}.{radec}*{refband}*.zp.fits')[0])[1].data)
# refcat  = refcat[refcat['FLAGS']==0]
# refcat  = refcat[refcat['CLASS_STAR'] > 0.9]
# refcat  = refcat[(refcat['MAG_AUTO'] > 14) & (refcat['MAG_AUTO'] < 20)]
# # %%import numpy as np
# import matplotlib.pyplot as plt
# from astropy.table import Table
# plt.rcParams.update({'font.size': 20})
# # Assuming scicat and refcat are the astropy tables with 'X_IMAGE', 'Y_IMAGE', 'FWHM_IMAGE'

# # Define image dimensions and number of bins
# x_range = [0, 22000]
# y_range = [0, 22000]
# num_bins = 4

# # Create bin edges
# x_bins = np.linspace(x_range[0], x_range[1], num_bins + 1)
# y_bins = np.linspace(y_range[0], y_range[1], num_bins + 1)

# # Create 2D arrays to store median FWHM values for both images
# median_fwhm_sci = np.zeros((num_bins, num_bins))
# median_fwhm_ref = np.zeros((num_bins, num_bins))
# median_fwhm_diff = np.zeros((num_bins, num_bins))

# # Function to calculate median FWHM for a given catalog
# def calculate_median_fwhm(catalog, x_bins, y_bins):
#     median_fwhm = np.zeros((num_bins, num_bins))
#     pixscale = 0.4
#     for i in range(num_bins):
#         for j in range(num_bins):
#             in_bin = (catalog['X_IMAGE'] >= x_bins[i]) & (catalog['X_IMAGE'] < x_bins[i + 1]) & \
#                      (catalog['Y_IMAGE'] >= y_bins[j]) & (catalog['Y_IMAGE'] < y_bins[j + 1])
#             median_fwhm[j, i] = np.median(catalog['FWHM_IMAGE'][in_bin])*pixscale
#     return median_fwhm

# # Calculate median FWHM for both scicat and refcat
# median_fwhm_sci = calculate_median_fwhm(scicat, x_bins, y_bins)
# median_fwhm_ref = calculate_median_fwhm(refcat, x_bins, y_bins)

# # Calculate the difference between the two images
# median_fwhm_diff = median_fwhm_sci - median_fwhm_ref

# # Define the common color scale for both scicat and refcat
# # vmin = min(np.min(median_fwhm_sci), np.min(median_fwhm_ref))
# # vmax = max(np.max(median_fwhm_sci), np.max(median_fwhm_ref))
# vmin, vmax = -2, 2
# # Plot the results
# fig, axs = plt.subplots(1, 3, figsize=(18, 7))
# plt.suptitle(f'{os.path.basename(sciimg)}')
# # Plot median FWHM for scicat
# img1 = axs[0].imshow(median_fwhm_sci, origin='lower', cmap='Blues', vmin=0, vmax=vmax,
#                      extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
# axs[0].set_title(f'scicat: Median FWHM = {np.median(scicat["FWHM_IMAGE"])*0.4:.2f}')
# axs[0].set_xlabel('X_IMAGE')
# axs[0].set_ylabel('Y_IMAGE')


# # Add text with the median FWHM for each bin
# for i in range(num_bins):
#     for j in range(num_bins):
#         x_pos = (x_bins[i] + x_bins[i + 1]) / 2
#         y_pos = (y_bins[j] + y_bins[j + 1]) / 2
#         axs[0].text(x_pos, y_pos, f'{median_fwhm_sci[j, i]:.2f}', color='white', ha='center', va='center')


# # Plot median FWHM for refcat
# img2 = axs[1].imshow(median_fwhm_ref, origin='lower', cmap='Reds', vmin=0, vmax=vmax,
#                      extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
# axs[1].set_title(f'refcat: Median FWHM = {np.median(refcat["FWHM_IMAGE"])*0.4:.2f}')
# axs[1].set_xlabel('X_IMAGE')
# axs[1].set_ylabel('Y_IMAGE')


# # Add text with the median FWHM for each bin
# for i in range(num_bins):
#     for j in range(num_bins):
#         x_pos = (x_bins[i] + x_bins[i + 1]) / 2
#         y_pos = (y_bins[j] + y_bins[j + 1]) / 2
#         axs[1].text(x_pos, y_pos, f'{median_fwhm_ref[j, i]:.2f}', color='white', ha='center', va='center')


# # Plot the difference between scicat and refcat
# img3 = axs[2].imshow(median_fwhm_diff, origin='lower', cmap='RdBu', vmin=vmin, vmax=vmax, extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
# axs[2].set_title('Difference: scicat - refcat')
# axs[2].set_xlabel('X_IMAGE')
# axs[2].set_ylabel('Y_IMAGE')

# # Add text with the difference in median FWHM for each bin
# for i in range(num_bins):
#     for j in range(num_bins):
#         x_pos = (x_bins[i] + x_bins[i + 1]) / 2
#         y_pos = (y_bins[j] + y_bins[j + 1]) / 2
#         axs[2].text(x_pos, y_pos, f'{median_fwhm_diff[j, i]:.2f}', color='black', ha='center', va='center')

# # Add a single horizontal colorbar for the FWHM values
# cbar = fig.colorbar(img3, orientation='vertical', fraction=0.0477, pad=0.0)
# cbar.set_label('Median FWHM [arcsec]')

# axs[0].set_xticks([])
# axs[0].set_yticks([])
# axs[1].set_xticks([])
# axs[1].set_yticks([])
# axs[2].set_xticks([])
# axs[2].set_yticks([])


# plt.tight_layout()
# plt.show()
# #%%
# total = 0
# for date in dates:
#     pointings = sorted(glob.glob(f'/data4/kmtntoo/tutorial/data/stack/{date}/S*.stack.fits'))
#     print(date, len(pointings))
#     total += len(pointings)
# # %%
# os.chdir('/data6/GECKO/S240915b/KMTN_CTIO/R/transients')
# # %%
# trancats = sorted(glob.glob('*RB.cat'))
# # %%
# transname = ['GECKO24l', 'GECKO24m', 'GECKO24n', 'GECKO24o']
# ras, decs = [109.723912, 110.4148614, 108.3119069,107.9809049], [-49.5148821, -49.2397919, -47.872987,-48.4370922]
# fields = ['2147', '2148', '2025', '2025']

# #%%

# total = 0
# for date in dates:
#     pointings = sorted(glob.glob(f'/data4/kmtntoo/tutorial/data/stack/{date}/S*.stack.fits'))
#     print(date, len(pointings))
#     total += len(pointings)

# for i, ra in enumerate(ras):
#     name    = transname[i]
#     dec = decs[i]
#     print(name)
#     field = fields[i]
#     filtered_rows = []
#     combined_table = Table()
#     for tran in trancats:  # trancats should be a list of filenames or catalog data
#             trancat = ascii.read(tran)
#             incoord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
#             refcoord = SkyCoord(trancat['ALPHA_J2000'], trancat['DELTA_J2000'], unit=(u.deg, u.deg))

#             # Matching the input coordinate to the catalog
#             indx, d2d, d3d = incoord.match_to_catalog_sky(refcoord)

#             # Check if the closest separation is less than 1 arcsecond
#             if d2d.to(u.arcsec).value[0] < 1:
#                 closest_object = trancat[indx]  # Get the closest object row
#                 filtered_rows.append(closest_object)  # Store it in the list
#         else:
#             pass
#     if len(filtered_rows)>0:
#         combined_table = vstack(filtered_rows)
#         print(combined_table)
#     else:
#         print(f"No objects with separation < 1 arcsecond from {name}.")
# #%%
# def read_header(filename):
#     """
#     Reads a .head file and extracts key-value pairs from it.
    
#     Args:
#         filename (str): Path to the .head file.
        
#     Returns:
#         dict: A dictionary containing key-value pairs from the header file.
#     """
#     header_dict = {}

#     # Open the .head file and read it line by line
#     with open(filename, 'r') as header_file:
#         for line in header_file:
#             # Check if the line contains a key-value pair
#             if '=' in line:
#                 # Split the line at the '=' sign to separate the key and value
#                 key, value_comment = line.split('=', 1)
#                 key = key.strip()  # Clean up any extra spaces
                
#                 # Separate the value from the comment, if present
#                 if '/' in value_comment:
#                     value, comment = value_comment.split('/', 1)
#                     value = value.strip()  # Clean up value
#                     comment = comment.strip()  # Clean up comment
#                 else:
#                     value = value_comment.strip()
#                     comment = None

#                 # Store the key-value pair in the dictionary
#                 header_dict[key] = value

#     return header_dict
# #%%
# for obs in ['KMTN_SAAO', 'KMTN_SSO', 'KMTN_CTIO']:
#     for ftr in ['R', 'I']:
#         base    = '/data6/GECKO/S240915b/'
#         path    = os.path.join(base, obs, ftr)
#         fitfiles  = sorted(glob.glob(f'{path}/*fits'))

#         dest    = '/data4/kmtntoo/tutorial/data/subt/'
#         for file in fitfiles:
#             hdr     = file.replace('.fits', '.head')
#             os.system(f'imhead {file} > {hdr}')
#             os.system(f'mv {hdr} {dest}')
# # %%
# transname = ['GECKO24l', 'GECKO24m', 'GECKO24n', 'GECKO24o']
# ras, decs = [109.723912, 110.4148614, 108.3119069,107.9809049], [-49.5148821, -49.2397919, -47.872987,-48.4370922]
# fields = ['2147', '2148', '2025', '2025']
# #%%
# os.chdir('/data4/kmtntoo/tutorial/data/subt/meta')
# # %%
# rbcats  = sorted(glob.glob('*RB.cat'))
# for rbcat in rbcats:
#     dateobs = rbcat.split('.')[3]
#     hdhdr   = sorted(glob.glob(f'hd*{dateobs}*head'))[0]
#     hdrdict = read_header(hdhdr)
#     field = hdrdict['FIELD1'].strip("'").strip()
#     newkey  = f"S240915b_{field}"

#     newname     = rbcat.replace(rbcat.split('.')[2], newkey)

#     shutil.move(rbcat, newname)
#     hdrs    = sorted(glob.glob(f'*{dateobs}*head'))
#     for hdr in hdrs:
#         newhdr = hdr.replace(hdr.split('.')[2], newkey)
#         shutil.move(hdr, newhdr)
# # %%
# for i, tran in enumerate(transname):
#     ra = ras[i]
#     dec = decs[i]
#     field = fields[i]
#     print(tran)
#     for ftr in ['R', 'I']:
#         rbcats  = sorted(glob.glob(f'*{field}*.{ftr}.*RB.cat'))
#         filtered_rows = []
#         combined_table = Table()
#         for rbcat in rbcats:
#             print(rbcat)
#             trancat = ascii.read(rbcat)
#             incoord = SkyCoord(ra, dec, unit=(u.deg, u.deg))
#             refcoord = SkyCoord(trancat['ALPHA_J2000'], trancat['DELTA_J2000'], unit=(u.deg, u.deg))

#                 # Matching the input coordinate to the catalog
#             indx, d2d, d3d = incoord.match_to_catalog_sky(refcoord)

#                 # Check if the closest separation is less than 1 arcsecond
#             if d2d.to(u.arcsec).value[0] < 1:
#                 closest_object = trancat[indx]  # Get the closest object row
#                 filtered_rows.append(closest_object)  # Store it in the list
#             else:
#                 pass
#         if len(filtered_rows)>0:
#             combined_table = vstack(filtered_rows)
#             combined_table.write(f'{tran}.{ftr}.cat', format='ascii', overwrite=True)
#         else:
#             print(f"No objects with separation < 1 arcsecond from {tran}.")
# # %%
# from astropy.time import Time

# for i, tran in enumerate(transname):
#     ra = ras[i]
#     dec = decs[i]
#     field = fields[i]
#     print(tran)
#     plt.figure(figsize=(5,2))
#     plt.title(f'{tran} {ftr}')
#     for ftr in ['R', 'I']:
#         if os.path.isfile(f'{tran}.{ftr}.cat'):
#             rbcat   = ascii.read(f'{tran}.{ftr}.cat')
#             date_obs = rbcat['DATE-OBS']
#             magnitudes    = rbcat['MAG_AUTO']
#             magnitude_errors = rbcat['MAGERR_AUTO']
#             flags   = rbcat['flag']
#             probs   = rbcat['prob']

#             rbtrue  = rbcat[rbcat['flag']=='False']

#             date_obst    = rbtrue['DATE-OBS']
#             magnitudest  = rbtrue['MAG_AUTO']
#             magnitude_errorst = rbtrue['MAGERR_AUTO']
#             # Convert 'DATE-OBS' to Astropy Time object
#             time_obs = Time(date_obs, format='isot', scale='utc')
#             time_obst = Time(date_obst, format='isot', scale='utc')
#             # Convert time to Julian Date or any other preferred time format
#             julian_dates = time_obs.jd
#             julian_datest = time_obst.jd

#             # Plot the light curve
#             if ftr =='R':
#                 color  = 'crimson'
#             else:
#                 color = 'brown'
#             plt.errorbar(julian_dates, magnitudes, yerr=magnitude_errors, fmt='o', linestyle='', capsize=3, label=f'{ftr}', c=color)
#             plt.errorbar(julian_datest, magnitudest, yerr=magnitude_errorst, fmt='x', linestyle='', capsize=3, c=color)
#             plt.xlim(2460568, 2460577)
#             plt.ylim(18, 22)
#             # Invert the y-axis since brighter stars have lower magnitudes
#             plt.gca().invert_yaxis()
#             plt.grid('--', alpha=0.2)
#             # Add labels and title
#             plt.xlabel('Julian Date')
#             plt.ylabel('Magnitude')
#             plt.legend()
# # %%
# imgs = sorted(glob.glob(f'{path_base}data/raw/{date}/reference_all/kmtc*fits'))
# cimgs = []
# cfield = []
# cseps = []
# iseps = []
# for img in imgs:
#     hdr = fits.getheader(img)
#     centcoord   = SkyCoord(hdr['RA'], hdr['DEC'], unit=(u.hourangle, u.deg))
#     field = ["9005", "9007", "9012", "9013", "9016", "9017"]
#     ra_deg = [10.80591667, 13.58845833, 20.03091667, 15.82554167, 20.38095833, 22.65970833]
#     dec_deg = [-25.55888889, -27.55888889, -31.55888889, -27.55888889, -29.55888889, -29.55888889]
#     # Create the table
#     kmtgrid = Table([field, ra_deg, dec_deg], names=['field', 'ra[deg]', 'dec[deg]'])
#     kmtcoord    = SkyCoord(kmtgrid['ra[deg]'], kmtgrid['dec[deg]'], unit='deg')
#     trgt_field  = kmtgrid[centcoord.separation(kmtcoord).argmin()]

#     if centcoord.separation(kmtcoord).min().value < 0.5:
#         cimgs.append(img)
#         cfield.append(trgt_field['field'])
#         cseps.append(centcoord.separation(kmtcoord).min().value)
#     else:
#         iseps.append(centcoord.separation(kmtcoord).min().value)
# ctbl = Table([cimgs, cfield, cseps], names=['image', 'field', 'separation'])
# for cimg in cimgs:
#     serial = cimg.split('.')[-2]
#     os.system(f'mv {path_base}data/raw/{date}/reference_all/*{serial}* {path_base}data/raw/{date}')
#%%
# os.chdir('/data4/kmtntoo/tutorial/data/subt/S250206dm/transients/Bronze/')
# # %%
# imgs = sorted(glob.glob(f'*.new.fits'))
# # %%
# def calculate_crosstalk_positions(x_position):
#     # Define the width of one section
#     section_width = 1152
#     # Calculate which section the bright star is in
#     star_section = (x_position // section_width) + (1 if x_position % section_width != 0 else 0)
#     # Determine the sections where crosstalk will appear based on the star’s section
#     if star_section % 2 == 0:  # Star is in an even section
#         crosstalk_sections = [2, 4, 6, 8]
#     else:  # Star is in an odd section
#         crosstalk_sections = [1, 3, 5, 7]
#     # Calculate the flipped x position within its section
#     distance_in_section = x_position - section_width * (star_section - 1)
#     flipped_x = section_width - distance_in_section
#     # Calculate the crosstalk positions
#     crosstalk_positions = []
#     flipped_sections = [5, 6, 7, 8] if star_section in [1, 2, 3, 4] else [1, 2, 3, 4]
#     for section in crosstalk_sections:
#         if section in flipped_sections:
#             crosstalk_x = (section - 1) * section_width + flipped_x
#         else:
#             crosstalk_x = (section - 1) * section_width + distance_in_section
#         crosstalk_positions.append(crosstalk_x)
#     return crosstalk_positions
# #%%
# import warnings
# from astropy.wcs import FITSFixedWarning

# warnings.filterwarnings("ignore", category=FITSFixedWarning)

# # Now, when you run your code that triggers the warning, it will be ignored.
# for img in imgs:
#     date = img.split('.')[3].split('-')[0].replace('2025', '25')
#     obs  = img.split('.')[1].split('_')[1]
#     exptime = int(img.split('.')[5])
#     band    = img.split('.')[4]
#     path_raw    = f'/data4/kmtntoo/tutorial/data/raw/{date}_{obs}/'
#     path_single = f'/data4/kmtntoo/tutorial/data/scaled/{date}_{obs}/'
#     path_cat    = f'/data4/kmtntoo/tutorial/data/stack/{date}_{obs}/'
#     hdr = fits.getheader(img)
#     tranra, trandec = hdr['TRANRA'], hdr['TRANDEC']
#     all_xtalk_coords = []
#     for i in range(int(exptime / 120 * 4)):
#         single = os.path.join(path_single, hdr[f'FILE{str(i+1).zfill(4)}'])
#         wcs = WCS(fits.getheader(single))
#         x, y  =  wcs.wcs_world2pix(tranra, trandec, 0)
#         if x > 0 and x < 9216 and y > 0 and y < 9232:
#             # print(single)
#             # print(x, y)
#             xtalk_positions = np.array([(px, y) for px in calculate_crosstalk_positions(x)])
#             xtalk_coords = np.array(wcs.wcs_pix2world(xtalk_positions, 0))
#             all_xtalk_coords.append(xtalk_coords)  
#     if all_xtalk_coords:  # Check if the list is not empty
#         stacked_xtalk_coords = np.vstack(all_xtalk_coords)
#         # print("Stacked Coordinates:")
#         # print(stacked_xtalk_coords)
#         xtalk_coords = SkyCoord(ra=stacked_xtalk_coords[:, 0],
#                                 dec=stacked_xtalk_coords[:, 1],
#                                 unit='deg')
#         catname = f"{hdr['OBJECT']}.{hdr['FIELD2']}.{band}.20{date}.{obs}.{exptime}sec.stack.fits.zp.cat"
#         fullcat = ascii.read(f'{path_cat}{catname}')
#         # Create a SkyCoord object for scicat using its RA and DEC columns.
#         fullcat_coords = SkyCoord(ra=fullcat['ALPHA_J2000'],
#                                 dec=fullcat['DELTA_J2000'],
#                                 unit='deg')

#         # For each xtalk coordinate, find the closest fullcat coordinate.
#         idx, d2d, _ = xtalk_coords.match_to_catalog_sky(fullcat_coords)

#         # Set a maximum separation threshold (for example, 1 arcsec).
#         max_sep = 1.0 * u.arcsec

#         # Create a mask for matches within the threshold.
#         mask = d2d < max_sep

#         # Extract the matched cross-talk coordinates and the corresponding fullcat entries.
#         matched_fullcat = fullcat[idx[mask]]
#         if len(matched_fullcat)>0 and (np.min(matched_fullcat['MAG_AUTO']) < 14):
#             print(f"Cross-talk contamination detected for {img}")
#             print("Cross-talk coordinates:")
#             print(xtalk_coords[mask])
#             print("Closest fullcat coordinates:")
#             print(matched_fullcat)
#         else:
#             print("No xtalk coordinates were found.")
            
#             try: print(f"{np.min(matched_fullcat['MAG_AUTO'])} at {np.min(matched_fullcat['sep'])*3600}")
#             except ValueError: pass
#     else:
#         print("No xtalk coordinates were found.")
# # %%
