# %%
"""
KMTNet Observation Request Preparation

SAAO (UTC+2): 
20.810444 -32.3793 1762

CTIO (UTC-4 (apr-aug) or UTC-3 (sep-mar)): 
-70.804 -30.1672 2167

SSO (UTC+10 (apr to sep) or UTC+11 (oct to mar))
149.062556 -31.271194 1143

e.g. UTC noon ==> SSO 10 pm, CTIO 8 am, SAAO 2 pm

evening obervations: CTIO ==> SSO ==> SAAO
dawn obervations: SAAO ==> CTIO ==> SSO
"""
#%% packages
import os
import glob
import shutil
import re, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.io import ascii
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.table import vstack
from astropy.coordinates import SkyCoord
from multiprocessing import Pool, cpu_count
#%% functions
# Function to determine the correct UTC offset for CTIO and SSO
def get_utc_offset(site, date):
    """Determine UTC offset based on date for CTIO and SSO"""
    year, month, _ = map(int, date[:4]), int(date[4:6]), int(date[6:8])
    
    if site == "CTIO":
        return -4 if 4 <= month <= 8 else -3  # UTC-4 (Apr-Aug), UTC-3 (Sep-Mar)
    elif site == "SSO":
        return 10 if 4 <= month <= 9 else 11  # UTC+10 (Apr-Sep), UTC+11 (Oct-Mar)
    return observatories[site]["utc_offset"]  # Default for SAAO

from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import os

def get_closest_tile(ra, dec, path_cfg='/data4/kmtntoo/tutorial/config/', gridcat='kmtnet_grid.fits'):
    """
    Find the closest predefined tile to a given celestial coordinate.

    Parameters:
    -----------
    ra : str or float
        Right Ascension (RA) of the target in either:
        - Sexagesimal format (e.g., "00:00:00")
        - Decimal degrees (float)
    dec : str or float
        Declination (Dec) of the target in either:
        - Sexagesimal format (e.g., "-30:00:00")
        - Decimal degrees (float)
    path_cfg : str, optional
        Path to the directory containing the grid catalog file (default: '/data4/kmtntoo/tutorial/config/').
    gridcat : str, optional
        Filename of the grid catalog (default: 'kmtnet_grid.fits').

    Returns:
    --------
    trgt_field : str
        The field name of the closest tile in the grid.
    sep : float
        The angular separation (in degrees) between the target and the closest tile.

    Notes:
    ------
    - If the catalog file is in FITS format, it will be read as FITS. Otherwise, it falls back to ASCII.
    - The function converts RA/Dec from sexagesimal format if necessary before performing the nearest-neighbor search.
    """

    # Convert RA/Dec input to SkyCoord
    coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')

    # Try loading the predefined grid catalog
    try:
        kmtgrid = Table.read(os.path.join(path_cfg, gridcat), format='fits')
    except:
        kmtgrid = Table.read(os.path.join(path_cfg, gridcat), format='ascii')

    # Convert grid coordinates to SkyCoord
    kmtcoord = SkyCoord(kmtgrid['ra[deg]'], kmtgrid['dec[deg]'], unit='deg', frame='icrs')

    # Find the closest tile
    closest_idx = coord.separation(kmtcoord).argmin()
    trgt_field = kmtgrid[closest_idx]['field_name1']
    sep = coord.separation(kmtcoord).min().value  # Separation in degrees

    return trgt_field, sep

def calculate_HA(RA, obs_date, obs_lon):
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u
    """
    Calculate the hour angle of a source.

    Parameters
    ----------
    RA : str
        Right Ascension of the source in "hh:mm:ss" format.
    obs_date : str
        Observation date and time in UTC (ISO format, e.g., "YYYY-MM-DDTHH:MM:SS").
    obs_lon : float
        Observatory longitude in degrees (positive for East, negative for West).

    Returns
    -------
    ha_hours : float
        Hour angle in hours, normalized to the range [-12, +12] hours.
    lst : astropy.units.Quantity
        Local Sidereal Time (LST) as an Astropy Quantity (in hours).
    """
    # Create a Time object for the observation date/time.
    time = Time(obs_date)
    
    # Compute the Local Sidereal Time (LST) in hours using the observatory's longitude.
    lst = time.sidereal_time('apparent', longitude=obs_lon * u.deg)
    
    # Parse the source RA into a SkyCoord object.
    source = SkyCoord(ra=RA, dec=0, unit=u.hourangle)
    
    # Calculate the hour angle as: HA = LST - RA.
    ha = lst - source.ra
    
    # Normalize the hour angle to be within the range [-12, 12] hours.
    ha_hours = ha.hour
    if ha_hours > 12:
        ha_hours -= 24
    elif ha_hours < -12:
        ha_hours += 24
    
    return ha_hours

calculate_HA('16:33:48.2', '2025-02-10T01:50:18.70', 20.810444)
calculate_HA('16:40:40.7', '2025-02-10T07:30:15.32', -70.804)
calculate_HA('16:33:48.2', '2025-02-10T07:25:15.32', -70.804)
from datetime import datetime, timedelta

def utc_to_mjd(utc_time):
    # Reference date for MJD (JD = 2400000.5 corresponds to MJD = 0)
    mjd_reference = datetime(1858, 11, 17, 0, 0, 0)

    # Parse the UTC time
    utc_time_parsed = datetime.strptime(utc_time, "%Y-%m-%d %H:%M:%S")

    # Calculate the difference in days
    delta = utc_time_parsed - mjd_reference

    # Convert to fractional days
    mjd = delta.days + delta.seconds / 86400 + delta.microseconds / 86400e6

    return mjd

# Example usage
utc_time = "2025-02-06 21:25:30"
mjd = utc_to_mjd(utc_time)
print(f"MJD: {mjd}")

from astropy.coordinates import SkyCoord
import astropy.units as u

def deg_to_sexagesimal(ra_deg, dec_deg, precision=2, sep=":"):
    """
    Convert RA, DEC in decimal degrees to a sexagesimal string using Astropy.

    Parameters:
        ra_deg (float): Right Ascension in decimal degrees.
        dec_deg (float): Declination in decimal degrees.
        precision (int): Number of decimal places for seconds.
        sep (str): Separator between hours, minutes, and seconds (and similarly for DEC).

    Returns:
        str: A string representation of the coordinates in sexagesimal format,
             for example: "16:02:26.15 -66:17:04.44"
    """
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
    return coord.to_string('hmsdms', sep=sep, precision=precision)

ra_deg, dec_deg =241.3327766, -65.7043893
sexagesimal_coords = deg_to_sexagesimal(ra_deg, dec_deg)
print("Sexagesimal Coordinates:", sexagesimal_coords)

#%% event info
event   = 'S250206dm'
status  = 'UPDATE'
#%% Path
path_base   = '/data4/kmtntoo/tutorial/'
path_cfg    = '/data4/kmtntoo/tutorial/config/'
path_data   = join(path_base, 'data/')
path_subt   = join(path_data, 'subt/')
path_ref    = '/data8/KS4/database/stack/'
path_scr    = '/data4/kmtntoo/tutorial/result/script/'
path_gecko  = f'/data7/GECKO/LVK_alert/{event}_{status}/'
#%% gridcat
gridcat     ='kmtnet_grid.fits'
kmtgrid     = Table.read(os.path.join(path_cfg, gridcat), format='fits')
# %% Target fields
skygrid = pd.read_csv(join(path_gecko, 'SkyGridCatalog_KMTNet_90.csv'))
fields = sorted(skygrid['id'])
fields = []
for script in sorted(glob.glob(f'{path_gecko}/scTOO*.cat'))[3:]:
    targets = ascii.read(script)
    for target in targets:
        if target['col12'] != '#--':
            fields.append(int(target['col12'][1:]))
fields = sorted(list(set(fields)))
# %% Check the fields
ks4fields = []
for field in fields:
    for kmt in kmtgrid:
        if int(field) == kmt['field_name1']:
            print(f"{field} {str(kmt['field_name1']).zfill(4)}.{kmt['field_name2']} {kmt['ra[deg]']} {kmt['dec[deg]']} {kmt['ra[hms]']} {kmt['dec[hms]']}")
            ks4fields.append(str(kmt['field_name1']).zfill(4) + '.' + kmt['field_name2'])
# %% Check if the reference image exists
band = 'I'
band = 'R'
for ks4field in ks4fields:
    if len(sorted(glob.glob(f'{path_ref}{ks4field}/ks4*{band}*.fits'))) != 0:
        print(ks4field, '1')
    else:
        print(ks4field, '0')
# %% Copy the reference image (unprocessed ones)
if False:
    band = 'I'
    reflist = []
    for ks4field in ks4fields:
        if len(sorted(glob.glob(f'{path_ref}{ks4field}/ks4*{band}*.fits'))) != 0:
            pass
        else:
            ks4ifield = re.sub(r'^0+(\d)', r'\1', ks4field)
            imgs = sorted(glob.glob(f'/data8/KS4/database/initial/{band}/{ks4ifield}/*.kk.fits'))
            if len(imgs) != 0:
                for img in imgs:
                    date = img.split('.')[-4]
                    serial = img.split('.')[-3]
                    if len(sorted(glob.glob(f'/data5/ks4/data/{date}/kmt*{serial}*.fits'))) != 0:
                        reflist.append(glob.glob(f'/data5/ks4/data/{date}/kmt*{serial}*.fits')[0])
    local_temp_dir = "/data5/ks4/temp/S250206dm_ref"
    os.makedirs(local_temp_dir, exist_ok=True)
    for file in reflist:
        shutil.copy(file, local_temp_dir)

    print("All files copied to:", local_temp_dir)

# sftp to skynet /data8/ or qso /data4/ for processing 
#%% if there are already covered fields
unwants = """
#1075
#1076
#1226
#1227
#1371
#1373
#1374
#1515
#1516
#1653
#1654
#1655
#1788
#1789
#1790
#1918
#1919
#1920
#1921
#2045
#2046
#2047
#2167
#2168
#2169
#224
#2285
#2286
#2397
#2398
#2504
#2505
#2506
#2820
#2821
#2822
#2987
#2988
#2989
#299
#3157
#3158
#3329
#3330
#3331
#3504
#3505
#3506
#366
#3680
#3681
#3682
#3859
#3860
#764
#765
#359
#359
#359
#359
#359
#292
#292
#292
#292
#360
#360
#360
#360
#293
#293
#293
#293
#422
#422
#422
#422
#361
#361
#361
#361
#294
#294
#294
#294
#294
#294
#222
#222
#222
#222
#295
#295
#295
#295
#362
#362
#362
#362
#423
#423
#423
#423
#223
#223
#223
#223
#--""".split()
unwants = sorted(list(set(unwants)))
# %% sorted scripts
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

# Define observatory locations
observatories = {
    "SAAO": {"lon": 20.810444, "lat": -32.3793, "alt": 1762, "utc_offset": 2},
    "CTIO": {"lon": -70.804, "lat": -30.1672, "alt": 2167, "utc_offset": None},  # Dynamic offset
    "SSO": {"lon": 149.062556, "lat": -31.271194, "alt": 1143, "utc_offset": None},  # Dynamic offset
}

# Select observatory
obs = "SSO" 
obs = "SAAO" 
obs = "CTIO" 
date = "20250211"
obstime = "06:00:00"

# Get site properties
obs_info = observatories[obs]
obs_info["utc_offset"] = get_utc_offset(obs, date)

# Define observatory location
site = EarthLocation.from_geodetic(
    lon=obs_info["lon"] * u.deg, lat=obs_info["lat"] * u.deg, height=obs_info["alt"] * u.m
)

# Load the catalog
script = f'{path_gecko}scTOO_{date}_{obs}.cat'

if not os.path.exists(script):
    raise FileNotFoundError(f"Observation script {script} not found.")

# Read the target table
targets = ascii.read(script)

# Convert RA/Dec to SkyCoord
target_coords = SkyCoord(ra=targets['col3'], dec=targets['col4'], unit=(u.hourangle, u.deg))
# Compute observation times (assume night-time observation)
obs_time = Time(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{obstime}", format='isot', scale='utc') # Specify time in UTC

lst = obs_time.sidereal_time('mean', longitude=site.lon)  # Local Sidereal Time at SAAO

# Compute Hour Angle (HA = LST - RA)
hour_angle = (lst - target_coords.ra).wrap_at(180*u.deg)

# Compute altitude (for filtering)
altaz_frame = AltAz(obstime=obs_time, location=site)
altitudes = target_coords.transform_to(altaz_frame).alt

# Sort by closest to meridian (HA ≈ 0)
sorted_indices = np.argsort(np.abs(hour_angle.deg))
sorted_targets = targets[sorted_indices]
sorted_tiles = list(dict.fromkeys(sorted_targets['col12']))

# Remove Unwanted Fields
if False:
    # unwants = ['#--'] # define here or above
    sorted_targets = sorted_targets[~np.isin(sorted_targets['col12'], unwants)]
    sorted_tiles = list(dict.fromkeys(sorted_targets['col12']))

# Write the sorted table to a new file
with open(script, 'r') as p:
    origin = p.readlines()  # Read all lines
with open(f'{path_scr}scTOO_{date}_{obs}.cat', 'w') as f:
    # Write the first four lines as headers, replacing "TOO" with "MMA"
    f.writelines([line.replace("TOO", "MMA") for line in origin[:4]])  

    # Iterate through sorted_tiles and find corresponding rows
    for tile in sorted_tiles:
        for i in range(4, len(origin), 4):  # Start from 4th line, iterate in steps of 4
            if origin[i].strip().endswith(tile):  # Check if the last column matches the tile
                # Replace "TOO" with "MMA" in the selected four-line block
                f.writelines([line.replace("TOO", "MMA") for line in origin[i:i+4]])
                break  # Move to the next tile after finding a match

# Test for StarALT input
if obs == 'CTIO': print('-70.804 -30.1672 2167')
elif obs == 'SSO': print('149.062556 -31.271194 1143')
else: print('20.810444 -32.3793 1762')
for i, target in enumerate(sorted_targets):
# for i, target in enumerate(targets):
    if i % 4 == 0:
        print(f"{target['col3']} {target['col4']}")

#%%
# %% AFTER OBSERVATION: Update the observation log

radecs = """
15:08:18	-69:59:59
15:09:05	-69:59:59
15:08:18	-69:52:59
15:09:05	-69:52:59
15:12:40	-67:59:59
15:13:23	-67:59:59
15:12:40	-67:52:59
15:13:23	-67:52:59
15:30:28	-69:59:59
15:31:14	-69:59:59
15:30:28	-69:52:59
15:31:14	-69:52:59
15:32:57	-67:59:59
15:33:40	-67:59:59
15:32:57	-67:52:59
15:33:40	-67:52:59
15:51:52	-71:59:59
15:52:43	-71:59:59
15:51:52	-71:52:59
15:52:43	-71:52:59
15:52:37	-69:59:59
15:53:24	-69:59:59
15:52:37	-69:52:59
15:53:24	-69:52:59
15:53:14	-67:59:59
15:53:57	-67:59:59
15:53:14	-67:52:59
15:53:57	-67:52:59
16:12:28	-65:59:59
16:13:07	-65:59:59
16:12:28	-65:52:59
16:13:07	-65:52:59
16:13:31	-67:59:59
16:14:14	-67:59:59
16:13:31	-67:52:59
16:14:14	-67:52:59
16:14:46	-69:59:59
16:15:33	-69:59:59
16:14:46	-69:52:59
16:15:33	-69:52:59
16:16:16	-71:59:59
16:17:08	-71:59:59
16:16:16	-71:52:59
16:17:08	-71:52:59
16:31:10	-65:59:59
16:31:49	-65:59:59
16:31:10	-65:52:59
16:31:49	-65:52:59
16:33:48	-67:59:59
16:34:31	-67:59:59
16:33:48	-67:52:59
16:34:31	-67:52:59
16:36:55	-69:59:59
16:37:42	-69:59:59
16:36:55	-69:52:59
16:37:42	-69:52:59
16:40:41	-71:59:59
16:41:32	-71:59:59
""".split()
radecs_pairs = list(zip(radecs[0::2], radecs[1::2]))
rfields = [f"#{get_closest_tile(ra, dec)[0]}" for ra, dec in radecs_pairs]
for rfield in rfields:
    print(rfield)

#%%

# rfields = sorted_tiles

from collections import Counter
# requested fields (scTOO.cat && observation log)
# update this field list with the date of interest
field_counts = Counter(field.strip('#') for field in rfields)

# Display the dictionary
print(field_counts)
# %%
# total field numbers of event localization area
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
764
765
1075
1076
1226
1227
1371
1373
1374
1515
1516
1653
1654
1655
1788
1789
1790
1918
1919
1920
1921
2045
2046
2047
2167
2168
2169
2285
2286
2397
2398
2504
2505
2506
2820
2821
2822
2987
2988
2989
3157
3158
3329
3330
3331
3504
3505
3506
3680
3681
3682
3859
3860
""".split()
# exposure times for each field
# copy & paste to the spreadsheet
for field in fields:
    count = field_counts.get(field, 0)  # This gets the count from field_counts or returns 0 if not found
    print(120*count)


#%%
from io import StringIO
tab1 = """
FIELD	RA	DEC	UPDATE	Reference	R1	R2	I1	R3	I2	I3
222	16:12:28.05	-66:00:00.00	1	1	0	0	0	480	480	0
223	16:31:10.13	-66:00:00.00	1	0	0	0	0	480	600	0
224	16:49:52.20	-66:00:00.00	0	0	0	0	0	0	0	0
292	15:12:40.57	-68:00:00.00	1	1	0	0	0	480	480	480
293	15:32:57.47	-68:00:00.00	1	1	0	0	0	480	480	480
294	15:53:14.37	-68:00:00.00	1	1	0	0	0	720	480	600
295	16:13:31.27	-68:00:00.00	1	1	0	0	0	480	720	0
296	16:33:48.17	-68:00:00.00	1	0	0	0	0	0	480	0
297	16:54:05.07	-68:00:00.00	1	0	0	0	0	0	0	480
298	17:14:21.97	-68:00:00.00	1	0	0	0	0	0	0	480
299	17:34:38.88	-68:00:00.00	0	0	0	0	0	0	0	0
359	15:08:18.46	-70:00:00.00	1	1	0	0	0	600	480	480
360	15:30:27.69	-70:00:00.00	1	1	0	0	0	480	480	480
361	15:52:36.93	-70:00:00.00	1	1	0	0	0	480	480	600
362	16:14:46.16	-70:00:00.00	1	1	0	0	0	480	480	0
363	16:36:55.39	-70:00:00.00	1	1	0	0	0	0	240	0
364	16:59:04.62	-70:00:00.00	1	0	0	0	0	0	0	480
365	17:21:13.85	-70:00:00.00	1	0	0	0	0	0	0	480
366	17:43:23.08	-70:00:00.00	0	0	0	0	0	0	0	0
422	15:51:51.87	-72:00:00.00	1	1	0	0	0	480	480	480
423	16:16:16.27	-72:00:00.00	1	1	0	0	0	480	480	0
424	16:40:40.68	-72:00:00.00	1	0	0	0	0	0	0	600
425	17:05:05.08	-72:00:00.00	1	0	0	0	0	0	0	600
764	10:19:44.81	-30:00:00.00	0	1	0	0	0	0	0	0
765	10:28:51.65	-30:00:00.00	0	1	0	480	480	0	0	0
1075	10:29:24.24	-34:00:00.00	0	1	0	600	600	0	0	0
1076	10:38:56.42	-34:00:00.00	0	1	480	0	480	0	0	0
1226	10:42:09.73	-36:00:00.00	0	1	0	480	480	0	0	0
1227	10:51:53.51	-36:00:00.00	0	1	0	0	0	0	0	0
1371	10:30:00.00	-38:00:00.00	0	1	0	0	0	0	0	0
1373	10:50:00.00	-38:00:00.00	0	1	240	0	480	0	0	0
1374	11:00:00.00	-38:00:00.00	0	1	0	0	0	0	0	0
1515	10:48:00.00	-40:00:00.00	0	1	480	0	480	0	0	0
1516	10:58:17.14	-40:00:00.00	0	1	0	480	480	0	0	0
1653	10:45:52.94	-42:00:00.00	0	1	0	480	480	0	0	0
1654	10:56:28.23	-42:00:00.00	0	1	0	480	480	0	0	0
1655	11:07:03.53	-42:00:00.00	0	1	0	0	480	0	0	0
1788	10:54:32.73	-44:00:00.00	0	1	0	480	480	0	0	0
1789	11:05:27.27	-44:00:00.00	0	1	600	0	480	0	0	0
1790	11:16:21.82	-44:00:00.00	0	1	480	0	480	0	0	0
1918	10:52:30.00	-46:00:00.00	0	1	0	480	480	0	0	0
1919	11:03:45.00	-46:00:00.00	0	1	480	0	480	0	0	0
1920	11:15:00.00	-46:00:00.00	0	1	480	0	480	0	0	0
1921	11:26:15.00	-46:00:00.00	0	1	0	0	480	0	0	0
2045	11:07:19.02	-48:00:00.00	0	1	600	0	480	0	0	0
2046	11:19:01.46	-48:00:00.00	0	1	480	0	480	0	0	0
2047	11:30:43.91	-48:00:00.00	0	1	480	0	480	0	0	0
2167	11:17:38.83	-50:00:00.00	0	1	0	0	480	0	0	0
2168	11:29:44.88	-50:00:00.00	0	1	480	0	480	0	0	0
2169	11:41:50.93	-50:00:00.00	0	1	0	0	0	0	0	0
2285	11:34:44.21	-52:00:00.00	0	1	480	0	0	0	0	0
2286	11:47:22.11	-52:00:00.00	0	1	0	0	0	0	0	0
2397	11:40:11.01	-54:00:00.00	0	1	0	0	0	0	0	0
2398	11:53:23.67	-54:00:00.00	0	1	0	0	0	0	0	0
2504	11:46:09.23	-56:00:00.00	0	1	0	0	0	0	0	0
2505	12:00:00.00	-56:00:00.00	0	1	0	0	0	0	0	0
2506	12:13:50.77	-56:00:00.00	0	1	0	0	0	0	0	0
2820	10:15:54.22	-28:43:24.26	0	0	0	480	0	0	0	0
2821	10:24:34.70	-28:43:24.26	0	0	480	0	480	0	0	0
2822	10:33:15.18	-28:43:24.26	0	0	0	480	480	0	0	0
2987	10:13:29.47	-26:48:30.64	0	0	0	0	600	0	0	0
2988	10:22:00.71	-26:48:30.64	0	0	480	0	480	0	0	0
2989	10:30:31.95	-26:48:30.64	0	0	0	600	0	0	0	0
3157	10:11:09.77	-24:53:37.02	0	0	0	480	480	0	0	0
3158	10:19:32.09	-24:53:37.02	0	0	480	0	480	0	0	0
3329	10:04:08.28	-22:58:43.40	0	0	0	480	480	0	0	0
3330	10:12:24.83	-22:58:43.40	0	0	0	480	480	0	0	0
3331	10:20:41.38	-22:58:43.40	0	0	0	480	480	0	0	0
3504	10:05:27.27	-21:03:49.79	0	0	0	480	480	0	0	0
3505	10:13:38.18	-21:03:49.79	0	0	480	0	480	0	0	0
3506	10:21:49.09	-21:03:49.79	0	0	0	0	0	0	0	0
3680	09:58:39.10	-19:08:56.17	0	0	0	480	480	0	0	0
3681	10:06:44.49	-19:08:56.17	0	0	480	0	480	0	0	0
3682	10:14:49.89	-19:08:56.17	0	0	0	480	480	0	0	0
3859	10:00:00.00	-17:14:02.55	0	0	480	0	480	0	0	0
3860	10:08:00.00	-17:14:02.55	0	0	0	480	480	0	0	0
"""
obslog = pd.read_csv(StringIO(tab1), sep="\t")
obslog = Table.from_pandas(obslog)
# %%
ctio_250207 = """
PROJID	IMGTYPE	OBJECT	RA	DEC	EXP	SECZ	ST	FILT	DATE-OBS	UT	MIDJD	FILENAME	Tile
MMA	OBJECT	S250206dm	9:42:18	-13:24:14	120	1.9	5:41:44	R	2025-02-07 1:13:27	2025-02-07 1:13:27	2460713.552	xkmtc.20250207.061570.fits	#4220
MMA	OBJECT	S250206dm	9:42:34	-13:24:14	120	1.87	5:44:45	R	2025-02-07 1:16:39	2025-02-07 1:16:40	2460713.554	xkmtc.20250207.061571.fits	#4220
MMA	OBJECT	S250206dm	9:42:18	-13:17:14	120	1.83	5:47:54	R	2025-02-07 1:19:35	2025-02-07 1:19:36	2460713.556	xkmtc.20250207.061572.fits	#4220
MMA	OBJECT	S250206dm	9:42:34	-13:17:14	120	1.81	5:50:52	R	2025-02-07 1:22:33	2025-02-07 1:22:33	2460713.558	xkmtc.20250207.061573.fits	#4220
MMA	OBJECT	S250206dm	9:51:34	-11:29:21	120	1.91	5:53:49	R	2025-02-07 1:25:42	2025-02-07 1:25:42	2460713.56	xkmtc.20250207.061574.fits	#4405
MMA	OBJECT	S250206dm	9:51:50	-11:29:21	120	1.88	5:56:59	R	2025-02-07 1:28:52	2025-02-07 1:28:52	2460713.562	xkmtc.20250207.061575.fits	#4405
MMA	OBJECT	S250206dm	9:51:34	-11:22:21	120	1.85	6:00:07	R	2025-02-07 1:31:47	2025-02-07 1:31:47	2460713.564	xkmtc.20250207.061576.fits	#4405
MMA	OBJECT	S250206dm	9:51:50	-11:22:21	120	1.82	6:03:05	R	2025-02-07 1:34:44	2025-02-07 1:34:44	2460713.566	xkmtc.20250207.061577.fits	#4405
MMA	OBJECT	S250206dm	9:37:32	-3:49:46	120	1.83	6:06:02	R	2025-02-07 1:37:40	2025-02-07 1:37:41	2460713.569	xkmtc.20250207.061578.fits	#5148
MMA	OBJECT	S250206dm	9:37:48	-3:49:46	120	1.8	6:08:59	R	2025-02-07 1:40:37	2025-02-07 1:40:37	2460713.571	xkmtc.20250207.061579.fits	#5148
MMA	OBJECT	S250206dm	9:37:32	-3:42:46	120	1.77	6:11:57	R	2025-02-07 1:43:35	2025-02-07 1:43:35	2460713.573	xkmtc.20250207.061580.fits	#5148
MMA	OBJECT	S250206dm	9:37:32	-3:49:46	120	1.74	6:14:54	R	2025-02-07 1:46:31	2025-02-07 1:46:31	2460713.575	xkmtc.20250207.061581.fits	#5148
MMA	OBJECT	S250206dm	9:37:48	-3:42:46	120	1.72	6:17:52	R	2025-02-07 1:49:42	2025-02-07 1:49:42	2460713.577	xkmtc.20250207.061582.fits	#5148
MMA	OBJECT	S250206dm	9:17:25	7:39:35	120	1.73	6:35:22	R	2025-02-07 2:06:56	2025-02-07 2:06:56	2460713.589	xkmtc.20250207.061583.fits	#6270
MMA	OBJECT	S250206dm	9:17:41	7:39:35	120	1.71	6:38:20	R	2025-02-07 2:09:53	2025-02-07 2:09:53	2460713.591	xkmtc.20250207.061584.fits	#6270
MMA	OBJECT	S250206dm	9:17:25	7:46:35	120	1.69	6:41:17	R	2025-02-07 2:12:49	2025-02-07 2:12:49	2460713.593	xkmtc.20250207.061585.fits	#6270
MMA	OBJECT	S250206dm	11:15:00	-45:59:59	120	1.71	6:45:33	R	2025-02-07 2:17:05	2025-02-07 2:17:05	2460713.596	xkmtc.20250207.061586.fits	#1920
MMA	OBJECT	S250206dm	11:15:23	-45:59:59	120	1.69	6:48:37	R	2025-02-07 2:20:08	2025-02-07 2:20:09	2460713.598	xkmtc.20250207.061587.fits	#1920
MMA	OBJECT	S250206dm	11:15:00	-45:52:59	120	1.67	6:51:34	R	2025-02-07 2:23:05	2025-02-07 2:23:05	2460713.6	xkmtc.20250207.061588.fits	#1920
MMA	OBJECT	S250206dm	11:15:23	-45:52:59	120	1.65	6:54:38	R	2025-02-07 2:26:09	2025-02-07 2:26:09	2460713.602	xkmtc.20250207.061589.fits	#1920
MMA	OBJECT	S250206dm	9:17:25	7:46:35	120	1.58	6:58:05	R	2025-02-07 2:29:34	2025-02-07 2:29:35	2460713.605	xkmtc.20250207.061590.fits	#6270
MMA	OBJECT	S250206dm	9:17:41	7:46:35	120	1.57	7:01:02	R	2025-02-07 2:32:32	2025-02-07 2:32:32	2460713.607	xkmtc.20250207.061591.fits	#6270
MMA	OBJECT	S250206dm	9:37:32	-5:44:40	120	1.39	7:04:00	R	2025-02-07 2:35:42	2025-02-07 2:35:42	2460713.609	xkmtc.20250207.061592.fits	#4961
MMA	OBJECT	S250206dm	9:37:49	-5:44:40	120	1.38	7:07:10	R	2025-02-07 2:38:51	2025-02-07 2:38:52	2460713.611	xkmtc.20250207.061593.fits	#4961
MMA	OBJECT	S250206dm	9:37:32	-5:37:40	120	1.36	7:10:21	R	2025-02-07 2:41:48	2025-02-07 2:41:49	2460713.613	xkmtc.20250207.061594.fits	#4961
MMA	OBJECT	S250206dm	9:37:49	-5:37:40	120	1.35	7:13:18	R	2025-02-07 2:44:58	2025-02-07 2:44:59	2460713.615	xkmtc.20250207.061595.fits	#4961
MMA	OBJECT	S250206dm	9:22:08	5:44:42	120	1.47	7:16:28	R	2025-02-07 2:47:55	2025-02-07 2:47:55	2460713.617	xkmtc.20250207.061596.fits	#6084
MMA	OBJECT	S250206dm	9:22:24	5:44:42	120	1.46	7:19:26	R	2025-02-07 2:50:52	2025-02-07 2:50:52	2460713.619	xkmtc.20250207.061597.fits	#6084
MMA	OBJECT	S250206dm	9:22:08	5:51:42	120	1.45	7:22:23	R	2025-02-07 2:54:01	2025-02-07 2:54:01	2460713.622	xkmtc.20250207.061598.fits	#6084
MMA	OBJECT	S250206dm	9:22:24	5:51:42	120	1.44	7:25:33	R	2025-02-07 2:56:58	2025-02-07 2:56:59	2460713.624	xkmtc.20250207.061599.fits	#6084
MMA	OBJECT	S250206dm	9:50:10	-13:24:14	120	1.26	7:28:31	R	2025-02-07 3:00:08	2025-02-07 3:00:08	2460713.626	xkmtc.20250207.061600.fits	#4221
MMA	OBJECT	S250206dm	9:50:26	-13:24:14	120	1.25	7:31:42	R	2025-02-07 3:03:06	2025-02-07 3:03:06	2460713.628	xkmtc.20250207.061601.fits	#4221
MMA	OBJECT	S250206dm	9:50:10	-13:17:14	120	1.24	7:34:38	R	2025-02-07 3:06:02	2025-02-07 3:06:02	2460713.63	xkmtc.20250207.061602.fits	#4221
MMA	OBJECT	S250206dm	9:50:26	-13:17:14	120	1.23	7:37:35	R	2025-02-07 3:08:58	2025-02-07 3:08:59	2460713.632	xkmtc.20250207.061603.fits	#4221
MMA	OBJECT	S250206dm	8:56:00	17:14:05	120	1.58	7:40:44	R	2025-02-07 3:12:07	2025-02-07 3:12:07	2460713.634	xkmtc.20250207.061604.fits	#7187
MMA	OBJECT	S250206dm	8:56:17	17:14:05	120	1.57	7:43:41	R	2025-02-07 3:15:03	2025-02-07 3:15:04	2460713.636	xkmtc.20250207.061605.fits	#7187
MMA	OBJECT	S250206dm	8:56:00	17:21:05	120	1.57	7:46:37	R	2025-02-07 3:17:59	2025-02-07 3:18:00	2460713.638	xkmtc.20250207.061606.fits	#7187
MMA	OBJECT	S250206dm	8:56:17	17:21:05	120	1.56	7:49:35	R	2025-02-07 3:20:56	2025-02-07 3:20:57	2460713.64	xkmtc.20250207.061607.fits	#7187
MMA	OBJECT	S250206dm	9:34:28	-1:54:53	120	1.26	7:52:33	R	2025-02-07 3:24:06	2025-02-07 3:24:06	2460713.642	xkmtc.20250207.061608.fits	#5335
MMA	OBJECT	S250206dm	9:34:44	-1:54:52	120	1.25	7:55:42	R	2025-02-07 3:27:03	2025-02-07 3:27:03	2460713.644	xkmtc.20250207.061609.fits	#5335
MMA	OBJECT	S250206dm	9:34:28	-1:47:52	120	1.25	7:58:39	R	2025-02-07 3:29:59	2025-02-07 3:30:00	2460713.647	xkmtc.20250207.061610.fits	#5335
MMA	OBJECT	S250206dm	9:34:44	-1:47:52	120	1.24	8:01:43	R	2025-02-07 3:33:03	2025-02-07 3:33:03	2460713.649	xkmtc.20250207.061611.fits	#5335
MMA	OBJECT	S250206dm	9:45:14	-5:44:39	120	1.21	8:04:40	R	2025-02-07 3:35:58	2025-02-07 3:35:59	2460713.651	xkmtc.20250207.061612.fits	#4962
MMA	OBJECT	S250206dm	9:45:31	-5:44:39	120	1.2	8:07:38	R	2025-02-07 3:38:56	2025-02-07 3:38:56	2460713.653	xkmtc.20250207.061613.fits	#4962
MMA	OBJECT	S250206dm	9:45:14	-5:37:39	120	1.2	8:10:35	R	2025-02-07 3:41:52	2025-02-07 3:41:53	2460713.655	xkmtc.20250207.061614.fits	#4962
MMA	OBJECT	S250206dm	9:45:31	-5:37:40	120	1.19	8:13:31	R	2025-02-07 3:44:49	2025-02-07 3:44:49	2460713.657	xkmtc.20250207.061615.fits	#4962
MMA	OBJECT	S250206dm	10:01:19	-15:19:08	120	1.14	8:16:28	R	2025-02-07 3:47:58	2025-02-07 3:47:58	2460713.659	xkmtc.20250207.061616.fits	#4040
MMA	OBJECT	S250206dm	10:01:36	-15:19:08	120	1.13	8:19:39	R	2025-02-07 3:51:08	2025-02-07 3:51:08	2460713.661	xkmtc.20250207.061617.fits	#4040
MMA	OBJECT	S250206dm	10:01:19	-15:12:08	120	1.13	8:22:56	R	2025-02-07 3:54:12	2025-02-07 3:54:12	2460713.663	xkmtc.20250207.061618.fits	#4040
MMA	OBJECT	S250206dm	10:01:36	-15:12:08	120	1.12	8:25:53	R	2025-02-07 3:57:08	2025-02-07 3:57:09	2460713.665	xkmtc.20250207.061619.fits	#4040
MMA	OBJECT	S250206dm	9:29:50	3:49:49	120	1.25	8:28:53	R	2025-02-07 4:00:08	2025-02-07 4:00:08	2460713.667	xkmtc.20250207.061620.fits	#5898
MMA	OBJECT	S250206dm	9:30:06	3:49:49	120	1.25	8:31:49	R	2025-02-07 4:03:03	2025-02-07 4:03:04	2460713.669	xkmtc.20250207.061621.fits	#5898
MMA	OBJECT	S250206dm	9:29:50	3:56:49	120	1.24	8:34:52	R	2025-02-07 4:06:05	2025-02-07 4:06:05	2460713.672	xkmtc.20250207.061622.fits	#5898
MMA	OBJECT	S250206dm	9:30:06	3:56:49	120	1.24	8:37:49	R	2025-02-07 4:09:02	2025-02-07 4:09:03	2460713.674	xkmtc.20250207.061623.fits	#5898
MMA	OBJECT	S250206dm	9:53:24	-15:19:07	120	1.08	8:40:53	R	2025-02-07 4:12:06	2025-02-07 4:12:07	2460713.676	xkmtc.20250207.061624.fits	#4039
MMA	OBJECT	S250206dm	9:53:41	-15:19:07	120	1.08	8:43:50	R	2025-02-07 4:15:03	2025-02-07 4:15:03	2460713.678	xkmtc.20250207.061625.fits	#4039
MMA	OBJECT	S250206dm	9:53:24	-15:12:07	120	1.08	8:46:47	R	2025-02-07 4:17:59	2025-02-07 4:18:00	2460713.68	xkmtc.20250207.061626.fits	#4039
MMA	OBJECT	S250206dm	9:53:41	-15:12:07	120	1.07	8:49:44	R	2025-02-07 4:20:56	2025-02-07 4:20:56	2460713.682	xkmtc.20250207.061627.fits	#4039
MMA	OBJECT	S250206dm	9:26:49	1:54:56	120	1.19	8:52:39	R	2025-02-07 4:23:51	2025-02-07 4:23:51	2460713.684	xkmtc.20250207.061628.fits	#5710
MMA	OBJECT	S250206dm	9:27:05	1:54:56	120	1.19	8:55:37	R	2025-02-07 4:27:01	2025-02-07 4:27:01	2460713.686	xkmtc.20250207.061629.fits	#5710
MMA	OBJECT	S250206dm	9:26:49	2:01:56	120	1.19	8:58:47	R	2025-02-07 4:29:57	2025-02-07 4:29:57	2460713.688	xkmtc.20250207.061630.fits	#5710
MMA	OBJECT	S250206dm	9:27:05	2:01:56	120	1.19	9:01:44	R	2025-02-07 4:32:53	2025-02-07 4:32:54	2460713.69	xkmtc.20250207.061631.fits	#5710
MMA	OBJECT	S250206dm	10:22:01	-26:48:29	120	1.05	9:05:07	R	2025-02-07 4:36:16	2025-02-07 4:36:16	2460713.693	xkmtc.20250207.061632.fits	#2988
MMA	OBJECT	S250206dm	9:27:05	2:01:56	120	1.18	9:08:32	R	2025-02-07 4:39:41	2025-02-07 4:39:41	2460713.695	xkmtc.20250207.061633.fits	#5710
MMA	OBJECT	S250206dm	10:22:19	-26:48:29	120	1.04	9:11:57	R	2025-02-07 4:43:05	2025-02-07 4:43:05	2460713.697	xkmtc.20250207.061634.fits	#2988
MMA	OBJECT	S250206dm	10:22:01	-26:41:30	120	1.04	9:14:53	R	2025-02-07 4:46:01	2025-02-07 4:46:01	2460713.699	xkmtc.20250207.061635.fits	#2988
MMA	OBJECT	S250206dm	10:22:19	-26:41:29	120	1.04	9:17:50	R	2025-02-07 4:48:56	2025-02-07 4:48:57	2460713.701	xkmtc.20250207.061636.fits	#2988
MMA	OBJECT	S250206dm	9:45:14	-3:49:45	120	1.12	9:21:07	R	2025-02-07 4:52:13	2025-02-07 4:52:14	2460713.704	xkmtc.20250207.061637.fits	#5149
MMA	OBJECT	S250206dm	9:45:31	-3:49:45	120	1.12	9:24:14	R	2025-02-07 4:55:20	2025-02-07 4:55:20	2460713.706	xkmtc.20250207.061638.fits	#5149
MMA	OBJECT	S250206dm	9:45:14	-3:42:45	120	1.12	9:27:10	R	2025-02-07 4:58:16	2025-02-07 4:58:16	2460713.708	xkmtc.20250207.061639.fits	#5149
MMA	OBJECT	S250206dm	9:45:31	-3:42:45	120	1.12	9:30:07	R	2025-02-07 5:01:25	2025-02-07 5:01:25	2460713.71	xkmtc.20250207.061640.fits	#5149
MMA	OBJECT	S250206dm	11:03:45	-45:59:59	120	1.1	9:34:28	R	2025-02-07 5:05:31	2025-02-07 5:05:32	2460713.713	xkmtc.20250207.061641.fits	#1919
MMA	OBJECT	S250206dm	11:04:08	-45:59:59	120	1.09	9:37:23	R	2025-02-07 5:08:27	2025-02-07 5:08:28	2460713.715	xkmtc.20250207.061642.fits	#1919
MMA	OBJECT	S250206dm	11:03:45	-45:52:59	120	1.09	9:40:20	R	2025-02-07 5:11:23	2025-02-07 5:11:24	2460713.717	xkmtc.20250207.061643.fits	#1919
MMA	OBJECT	S250206dm	11:04:08	-45:52:59	120	1.09	9:43:19	R	2025-02-07 5:14:22	2025-02-07 5:14:23	2460713.719	xkmtc.20250207.061644.fits	#1919
MMA	OBJECT	S250206dm	9:34:28	0:00:02	120	1.15	9:47:42	R	2025-02-07 5:18:45	2025-02-07 5:18:45	2460713.722	xkmtc.20250207.061645.fits	#5523
MMA	OBJECT	S250206dm	9:34:44	0:00:02	120	1.15	9:50:40	R	2025-02-07 5:21:41	2025-02-07 5:21:42	2460713.724	xkmtc.20250207.061646.fits	#5523
MMA	OBJECT	S250206dm	9:34:28	0:07:02	120	1.16	9:53:36	R	2025-02-07 5:24:37	2025-02-07 5:24:38	2460713.726	xkmtc.20250207.061647.fits	#5523
MMA	OBJECT	S250206dm	9:34:44	0:07:02	120	1.16	9:56:33	R	2025-02-07 5:27:33	2025-02-07 5:27:34	2460713.728	xkmtc.20250207.061648.fits	#5523
MMA	OBJECT	S250206dm	11:05:27	-43:59:59	120	1.06	10:01:00	R	2025-02-07 5:32:00	2025-02-07 5:32:00	2460713.731	xkmtc.20250207.061649.fits	#1789
MMA	OBJECT	S250206dm	11:05:49	-43:59:59	120	1.06	10:03:58	R	2025-02-07 5:34:58	2025-02-07 5:34:58	2460713.733	xkmtc.20250207.061650.fits	#1789
MMA	OBJECT	S250206dm	11:05:27	-43:52:59	120	1.06	10:06:57	R	2025-02-07 5:37:55	2025-02-07 5:37:56	2460713.735	xkmtc.20250207.061651.fits	#1789
MMA	OBJECT	S250206dm	11:05:49	-43:52:59	120	1.05	10:09:52	R	2025-02-07 5:40:51	2025-02-07 5:40:51	2460713.737	xkmtc.20250207.061652.fits	#1789
MMA	OBJECT	S250206dm	11:05:49	-43:52:59	120	1.05	10:14:40	R	2025-02-07 5:45:37	2025-02-07 5:45:37	2460713.741	xkmtc.20250207.061653.fits	#1789
MMA	OBJECT	S250206dm	9:34:28	1:54:56	120	1.2	10:19:20	R	2025-02-07 5:50:17	2025-02-07 5:50:17	2460713.744	xkmtc.20250207.061654.fits	#5711
MMA	OBJECT	S250206dm	9:34:44	1:54:56	120	1.2	10:22:17	R	2025-02-07 5:53:14	2025-02-07 5:53:14	2460713.746	xkmtc.20250207.061655.fits	#5711
MMA	OBJECT	S250206dm	9:34:28	2:01:56	120	1.2	10:25:17	R	2025-02-07 5:56:26	2025-02-07 5:56:27	2460713.748	xkmtc.20250207.061656.fits	#5711
MMA	OBJECT	S250206dm	9:34:44	2:01:56	120	1.21	10:28:29	R	2025-02-07 5:59:24	2025-02-07 5:59:25	2460713.75	xkmtc.20250207.061657.fits	#5711
MMA	OBJECT	S250206dm	11:19:01	-47:59:59	120	1.07	10:33:09	R	2025-02-07 6:04:04	2025-02-07 6:04:04	2460713.754	xkmtc.20250207.061658.fits	#2046
MMA	OBJECT	S250206dm	11:19:25	-47:59:58	120	1.07	10:36:06	R	2025-02-07 6:07:00	2025-02-07 6:07:00	2460713.756	xkmtc.20250207.061659.fits	#2046
MMA	OBJECT	S250206dm	11:19:02	-47:52:59	120	1.06	10:39:03	R	2025-02-07 6:09:57	2025-02-07 6:09:57	2460713.758	xkmtc.20250207.061660.fits	#2046
MMA	OBJECT	S250206dm	11:19:25	-47:52:59	120	1.06	10:42:03	R	2025-02-07 6:13:09	2025-02-07 6:13:09	2460713.76	xkmtc.20250207.061661.fits	#2046
MMA	OBJECT	S250206dm	9:04:00	17:14:05	120	1.66	10:48:17	R	2025-02-07 6:19:09	2025-02-07 6:19:10	2460713.764	xkmtc.20250207.061662.fits	#7188
MMA	OBJECT	S250206dm	9:04:17	17:14:05	120	1.67	10:51:14	R	2025-02-07 6:22:18	2025-02-07 6:22:19	2460713.766	xkmtc.20250207.061663.fits	#7188
MMA	OBJECT	S250206dm	9:04:00	17:21:05	120	1.69	10:54:24	R	2025-02-07 6:25:15	2025-02-07 6:25:15	2460713.768	xkmtc.20250207.061664.fits	#7188
MMA	OBJECT	S250206dm	9:04:17	17:21:05	120	1.71	10:57:25	R	2025-02-07 6:28:16	2025-02-07 6:28:16	2460713.77	xkmtc.20250207.061665.fits	#7188
MMA	OBJECT	S250206dm	11:29:45	-49:59:59	120	1.07	11:02:13	R	2025-02-07 6:33:03	2025-02-07 6:33:04	2460713.774	xkmtc.20250207.061666.fits	#2168
MMA	OBJECT	S250206dm	11:30:10	-49:59:59	120	1.07	11:05:10	R	2025-02-07 6:36:01	2025-02-07 6:36:01	2460713.776	xkmtc.20250207.061667.fits	#2168
MMA	OBJECT	S250206dm	11:29:45	-49:52:59	120	1.07	11:08:11	R	2025-02-07 6:38:59	2025-02-07 6:39:00	2460713.778	xkmtc.20250207.061668.fits	#2168
MMA	OBJECT	S250206dm	11:30:10	-49:52:59	120	1.07	11:11:08	R	2025-02-07 6:41:56	2025-02-07 6:41:56	2460713.78	xkmtc.20250207.061669.fits	#2168
MMA	OBJECT	S250206dm	10:24:35	-28:43:22	120	1.02	11:14:44	R	2025-02-07 6:45:32	2025-02-07 6:45:32	2460713.782	xkmtc.20250207.061670.fits	#2821
MMA	OBJECT	S250206dm	10:24:53	-28:43:22	120	1.02	11:17:51	R	2025-02-07 6:48:38	2025-02-07 6:48:38	2460713.784	xkmtc.20250207.061671.fits	#2821
MMA	OBJECT	S250206dm	10:24:35	-28:36:22	120	1.02	11:20:52	R	2025-02-07 6:51:38	2025-02-07 6:51:39	2460713.787	xkmtc.20250207.061672.fits	#2821
MMA	OBJECT	S250206dm	10:24:53	-28:36:22	120	1.02	11:23:54	R	2025-02-07 6:54:40	2025-02-07 6:54:41	2460713.789	xkmtc.20250207.061673.fits	#2821
MMA	OBJECT	S250206dm	11:16:22	-43:59:59	120	1.03	11:27:20	R	2025-02-07 6:58:06	2025-02-07 6:58:06	2460713.791	xkmtc.20250207.061674.fits	#1790
MMA	OBJECT	S250206dm	11:16:44	-43:59:59	120	1.03	11:30:17	R	2025-02-07 7:01:15	2025-02-07 7:01:16	2460713.793	xkmtc.20250207.061675.fits	#1790
MMA	OBJECT	S250206dm	11:16:22	-43:52:58	120	1.03	11:33:27	R	2025-02-07 7:04:12	2025-02-07 7:04:12	2460713.795	xkmtc.20250207.061676.fits	#1790
MMA	OBJECT	S250206dm	11:16:44	-43:52:58	120	1.03	11:36:24	R	2025-02-07 7:07:08	2025-02-07 7:07:09	2460713.797	xkmtc.20250207.061677.fits	#1790
MMA	OBJECT	S250206dm	11:34:44	-51:59:59	120	1.08	11:39:22	R	2025-02-07 7:10:18	2025-02-07 7:10:18	2460713.8	xkmtc.20250207.061678.fits	#2285
MMA	OBJECT	S250206dm	11:35:10	-51:59:59	120	1.08	11:42:34	R	2025-02-07 7:13:17	2025-02-07 7:13:18	2460713.802	xkmtc.20250207.061679.fits	#2285
MMA	OBJECT	S250206dm	11:34:44	-51:52:59	120	1.08	11:45:32	R	2025-02-07 7:16:14	2025-02-07 7:16:15	2460713.804	xkmtc.20250207.061680.fits	#2285
MMA	OBJECT	S250206dm	11:35:10	-51:52:59	120	1.08	11:48:30	R	2025-02-07 7:19:12	2025-02-07 7:19:13	2460713.806	xkmtc.20250207.061681.fits	#2285
MMA	OBJECT	S250206dm	10:48:00	-39:59:58	120	1.04	11:51:31	R	2025-02-07 7:22:12	2025-02-07 7:22:12	2460713.808	xkmtc.20250207.061682.fits	#1515
MMA	OBJECT	S250206dm	10:48:21	-39:59:58	120	1.04	11:54:27	R	2025-02-07 7:25:08	2025-02-07 7:25:08	2460713.81	xkmtc.20250207.061683.fits	#1515
MMA	OBJECT	S250206dm	10:48:00	-39:52:58	120	1.04	11:57:24	R	2025-02-07 7:28:04	2025-02-07 7:28:04	2460713.812	xkmtc.20250207.061684.fits	#1515
MMA	OBJECT	S250206dm	10:48:21	-39:52:59	120	1.05	12:00:23	R	2025-02-07 7:31:16	2025-02-07 7:31:16	2460713.814	xkmtc.20250207.061685.fits	#1515
MMA	OBJECT	S250206dm	11:07:19	-47:59:58	120	1.07	12:03:35	R	2025-02-07 7:34:15	2025-02-07 7:34:15	2460713.816	xkmtc.20250207.061686.fits	#2045
MMA	OBJECT	S250206dm	11:07:43	-47:59:59	120	1.07	12:06:33	R	2025-02-07 7:37:25	2025-02-07 7:37:26	2460713.818	xkmtc.20250207.061687.fits	#2045
MMA	OBJECT	S250206dm	11:07:19	-47:52:59	120	1.07	12:09:43	R	2025-02-07 7:40:22	2025-02-07 7:40:22	2460713.82	xkmtc.20250207.061688.fits	#2045
MMA	OBJECT	S250206dm	11:07:43	-47:52:59	120	1.08	12:12:39	R	2025-02-07 7:43:17	2025-02-07 7:43:18	2460713.822	xkmtc.20250207.061689.fits	#2045
MMA	OBJECT	S250206dm	10:38:56	-33:59:58	120	1.07	12:15:44	R	2025-02-07 7:46:21	2025-02-07 7:46:22	2460713.825	xkmtc.20250207.061690.fits	#1076
MMA	OBJECT	S250206dm	11:07:43	-47:52:59	120	1.08	12:18:40	R	2025-02-07 7:49:17	2025-02-07 7:49:18	2460713.827	xkmtc.20250207.061691.fits	#2045
MMA	OBJECT	S250206dm	10:39:16	-33:59:58	120	1.07	12:21:37	R	2025-02-07 7:52:13	2025-02-07 7:52:14	2460713.829	xkmtc.20250207.061692.fits	#1076
MMA	OBJECT	S250206dm	10:38:56	-33:52:59	120	1.08	12:24:32	R	2025-02-07 7:55:09	2025-02-07 7:55:09	2460713.831	xkmtc.20250207.061693.fits	#1076
MMA	OBJECT	S250206dm	10:39:16	-33:52:59	120	1.08	12:27:31	R	2025-02-07 7:58:06	2025-02-07 7:58:07	2460713.833	xkmtc.20250207.061694.fits	#1076
MMA	OBJECT	S250206dm	10:06:45	-19:08:55	120	1.2	12:30:28	R	2025-02-07 8:01:04	2025-02-07 8:01:04	2460713.835	xkmtc.20250207.061695.fits	#3681
MMA	OBJECT	S250206dm	10:07:01	-19:08:55	120	1.21	12:33:25	R	2025-02-07 8:04:00	2025-02-07 8:04:01	2460713.837	xkmtc.20250207.061696.fits	#3681
MMA	OBJECT	S250206dm	10:06:45	-19:01:55	120	1.22	12:36:23	R	2025-02-07 8:06:58	2025-02-07 8:06:58	2460713.839	xkmtc.20250207.061697.fits	#3681
MMA	OBJECT	S250206dm	10:07:01	-19:01:55	120	1.23	12:39:26	R	2025-02-07 8:10:00	2025-02-07 8:10:01	2460713.841	xkmtc.20250207.061698.fits	#3681
MMA	OBJECT	S250206dm	11:30:44	-47:59:59	120	1.08	12:42:39	R	2025-02-07 8:13:12	2025-02-07 8:13:12	2460713.843	xkmtc.20250207.061699.fits	#2047
MMA	OBJECT	S250206dm	11:31:08	-47:59:59	120	1.08	12:45:36	R	2025-02-07 8:16:09	2025-02-07 8:16:10	2460713.845	xkmtc.20250207.061700.fits	#2047
MMA	OBJECT	S250206dm	11:30:44	-47:52:59	120	1.09	12:48:34	R	2025-02-07 8:19:06	2025-02-07 8:19:06	2460713.847	xkmtc.20250207.061701.fits	#2047
MMA	OBJECT	S250206dm	11:31:08	-47:52:59	120	1.09	12:51:36	R	2025-02-07 8:22:08	2025-02-07 8:22:08	2460713.849	xkmtc.20250207.061702.fits	#2047
MMA	OBJECT	S250206dm	10:19:32	-24:53:36	120	1.2	12:54:37	R	2025-02-07 8:25:08	2025-02-07 8:25:09	2460713.851	xkmtc.20250207.061703.fits	#3158
MMA	OBJECT	S250206dm	10:19:50	-24:53:36	120	1.21	12:57:37	R	2025-02-07 8:28:07	2025-02-07 8:28:08	2460713.854	xkmtc.20250207.061704.fits	#3158
MMA	OBJECT	S250206dm	10:19:32	-24:46:36	120	1.22	13:00:33	R	2025-02-07 8:31:16	2025-02-07 8:31:17	2460713.856	xkmtc.20250207.061705.fits	#3158
MMA	OBJECT	S250206dm	10:19:50	-24:46:36	120	1.23	13:03:44	R	2025-02-07 8:34:15	2025-02-07 8:34:15	2460713.858	xkmtc.20250207.061706.fits	#3158
MMA	OBJECT	S250206dm	9:45:30	-15:19:08	120	1.48	13:06:41	R	2025-02-07 8:37:10	2025-02-07 8:37:11	2460713.86	xkmtc.20250207.061707.fits	#4038
MMA	OBJECT	S250206dm	9:45:46	-15:19:08	120	1.5	13:09:38	R	2025-02-07 8:40:07	2025-02-07 8:40:07	2460713.862	xkmtc.20250207.061708.fits	#4038
MMA	OBJECT	S250206dm	9:45:30	-15:19:08	120	1.52	13:12:40	R	2025-02-07 8:43:21	2025-02-07 8:43:21	2460713.864	xkmtc.20250207.061709.fits	#4038
MMA	OBJECT	S250206dm	9:45:30	-15:12:08	120	1.54	13:15:49	R	2025-02-07 8:46:17	2025-02-07 8:46:18	2460713.866	xkmtc.20250207.061710.fits	#4038
MMA	OBJECT	S250206dm	9:45:46	-15:12:08	120	1.56	13:18:46	R	2025-02-07 8:49:14	2025-02-07 8:49:14	2460713.868	xkmtc.20250207.061711.fits	#4038
MMA	OBJECT	S250206dm	10:00:00	-17:14:02	120	1.46	13:21:43	R	2025-02-07 8:52:10	2025-02-07 8:52:10	2460713.87	xkmtc.20250207.061712.fits	#3859
MMA	OBJECT	S250206dm	10:00:17	-17:14:02	120	1.48	13:24:40	R	2025-02-07 8:55:06	2025-02-07 8:55:06	2460713.872	xkmtc.20250207.061713.fits	#3859
MMA	OBJECT	S250206dm	10:00:00	-17:07:02	120	1.5	13:27:35	R	2025-02-07 8:58:02	2025-02-07 8:58:02	2460713.874	xkmtc.20250207.061714.fits	#3859
MMA	OBJECT	S250206dm	10:00:17	-17:07:02	120	1.51	13:30:35	R	2025-02-07 9:01:01	2025-02-07 9:01:01	2460713.876	xkmtc.20250207.061715.fits	#3859
MMA	OBJECT	S250206dm	9:58:02	-13:24:14	120	1.61	13:33:42	R	2025-02-07 9:04:19	2025-02-07 9:04:19	2460713.879	xkmtc.20250207.061716.fits	#4222
MMA	OBJECT	S250206dm	9:58:18	-13:24:14	120	1.63	13:36:51	R	2025-02-07 9:07:15	2025-02-07 9:07:16	2460713.881	xkmtc.20250207.061717.fits	#4222
MMA	OBJECT	S250206dm	9:58:02	-13:17:14	120	1.66	13:39:50	R	2025-02-07 9:10:27	2025-02-07 9:10:27	2460713.883	xkmtc.20250207.061718.fits	#4222
MMA	OBJECT	S250206dm	9:58:18	-13:17:14	120	1.68	13:43:01	R	2025-02-07 9:13:24	2025-02-07 9:13:25	2460713.885	xkmtc.20250207.061719.fits	#4222
MMA	OBJECT	S250206dm	10:13:38	-21:03:49	120	1.48	13:46:05	R	2025-02-07 9:16:28	2025-02-07 9:16:28	2460713.887	xkmtc.20250207.061720.fits	#3505
MMA	OBJECT	S250206dm	10:13:55	-21:03:49	120	1.5	13:49:02	R	2025-02-07 9:19:37	2025-02-07 9:19:37	2460713.889	xkmtc.20250207.061721.fits	#3505
MMA	OBJECT	S250206dm	10:13:38	-20:56:49	120	1.52	13:52:12	R	2025-02-07 9:22:34	2025-02-07 9:22:34	2460713.891	xkmtc.20250207.061722.fits	#3505
MMA	OBJECT	S250206dm	10:13:55	-20:56:49	120	1.54	13:55:08	R	2025-02-07 9:25:30	2025-02-07 9:25:30	2460713.893	xkmtc.20250207.061723.fits	#3505
MMA	OBJECT	S250206dm	10:50:00	-37:59:59	120	1.28	13:58:14	R	2025-02-07 9:28:34	2025-02-07 9:28:35	2460713.896	xkmtc.20250207.061724.fits	#1373
MMA	OBJECT	S250206dm	10:50:20	-37:59:59	120	1.29	14:01:16	R	2025-02-07 9:31:36	2025-02-07 9:31:37	2460713.898	xkmtc.20250207.061725.fits	#1373
"""
ctio_250207 = pd.read_csv(StringIO(ctio_250207), sep="\t")
ctio_250207 = Table.from_pandas(ctio_250207)

saao_250207 = """
PROJID	IMGTYPE	OBJECT	RA	DEC	EXP	SECZ	ST	FILT	DATE-OBS	UT	MIDJD	FILENAME	Tile
MMA	OBJECT	S250206dm	9:58:39	-19:08:56	120	1.9	5:45:43	R	2025-02-07 19:08:01	2025-02-07 19:08:01	2460714.298	xkmts.20250207.005956.fits	#3680
MMA	OBJECT	S250206dm	9:58:56	-19:08:56	120	1.86	5:49:09	R	2025-02-07 19:11:26	2025-02-07 19:11:27	2460714.3	xkmts.20250207.005957.fits	#3680
MMA	OBJECT	S250206dm	9:58:39	-19:01:56	120	1.83	5:52:05	R	2025-02-07 19:14:36	2025-02-07 19:14:36	2460714.303	xkmts.20250207.005958.fits	#3680
MMA	OBJECT	S250206dm	9:58:56	-19:01:56	120	1.8	5:55:16	R	2025-02-07 19:17:45	2025-02-07 19:17:45	2460714.305	xkmts.20250207.005959.fits	#3680
MMA	OBJECT	S250206dm	10:04:08	-22:58:43	120	1.75	5:58:25	R	2025-02-07 19:20:55	2025-02-07 19:20:55	2460714.307	xkmts.20250207.005960.fits	#3329
MMA	OBJECT	S250206dm	10:04:26	-22:58:43	120	1.72	6:01:37	R	2025-02-07 19:23:52	2025-02-07 19:23:53	2460714.309	xkmts.20250207.005961.fits	#3329
MMA	OBJECT	S250206dm	10:04:08	-22:51:43	120	1.69	6:04:35	R	2025-02-07 19:26:51	2025-02-07 19:26:51	2460714.311	xkmts.20250207.005962.fits	#3329
MMA	OBJECT	S250206dm	10:04:26	-22:51:43	120	1.67	6:07:33	R	2025-02-07 19:30:00	2025-02-07 19:30:00	2460714.313	xkmts.20250207.005963.fits	#3329
MMA	OBJECT	S250206dm	10:05:27	-21:03:50	120	1.68	6:10:41	R	2025-02-07 19:33:08	2025-02-07 19:33:08	2460714.315	xkmts.20250207.005964.fits	#3504
MMA	OBJECT	S250206dm	10:05:44	-21:03:50	120	1.66	6:13:51	R	2025-02-07 19:36:18	2025-02-07 19:36:18	2460714.318	xkmts.20250207.005965.fits	#3504
MMA	OBJECT	S250206dm	10:05:27	-20:56:50	120	1.63	6:17:02	R	2025-02-07 19:39:27	2025-02-07 19:39:28	2460714.32	xkmts.20250207.005966.fits	#3504
MMA	OBJECT	S250206dm	10:05:44	-20:56:50	120	1.61	6:20:11	R	2025-02-07 19:42:23	2025-02-07 19:42:24	2460714.322	xkmts.20250207.005967.fits	#3504
MMA	OBJECT	S250206dm	10:08:00	-17:14:03	120	1.67	6:23:24	R	2025-02-07 19:45:36	2025-02-07 19:45:37	2460714.324	xkmts.20250207.005968.fits	#3860
MMA	OBJECT	S250206dm	10:08:17	-17:14:03	120	1.64	6:26:21	R	2025-02-07 19:48:33	2025-02-07 19:48:34	2460714.326	xkmts.20250207.005969.fits	#3860
MMA	OBJECT	S250206dm	10:08:00	-17:07:03	120	1.62	6:29:20	R	2025-02-07 19:51:32	2025-02-07 19:51:32	2460714.328	xkmts.20250207.005970.fits	#3860
MMA	OBJECT	S250206dm	10:08:17	-17:07:03	120	1.6	6:32:18	R	2025-02-07 19:54:42	2025-02-07 19:54:42	2460714.33	xkmts.20250207.005971.fits	#3860
MMA	OBJECT	S250206dm	10:11:10	-24:53:37	120	1.49	6:35:32	R	2025-02-07 19:57:42	2025-02-07 19:57:42	2460714.332	xkmts.20250207.005972.fits	#3157
MMA	OBJECT	S250206dm	10:11:27	-24:53:37	120	1.48	6:38:30	R	2025-02-07 20:00:52	2025-02-07 20:00:52	2460714.335	xkmts.20250207.005973.fits	#3157
MMA	OBJECT	S250206dm	10:11:10	-24:46:37	120	1.46	6:41:40	R	2025-02-07 20:04:01	2025-02-07 20:04:01	2460714.337	xkmts.20250207.005974.fits	#3157
MMA	OBJECT	S250206dm	10:11:27	-24:46:37	120	1.44	6:44:49	R	2025-02-07 20:07:10	2025-02-07 20:07:10	2460714.339	xkmts.20250207.005975.fits	#3157
MMA	OBJECT	S250206dm	10:12:25	-22:58:43	120	1.45	6:47:59	R	2025-02-07 20:10:07	2025-02-07 20:10:07	2460714.341	xkmts.20250207.005976.fits	#3330
MMA	OBJECT	S250206dm	10:12:42	-22:58:43	120	1.43	6:50:55	R	2025-02-07 20:13:15	2025-02-07 20:13:16	2460714.343	xkmts.20250207.005977.fits	#3330
MMA	OBJECT	S250206dm	10:12:25	-22:51:43	120	1.42	6:54:04	R	2025-02-07 20:16:24	2025-02-07 20:16:25	2460714.345	xkmts.20250207.005978.fits	#3330
MMA	OBJECT	S250206dm	10:12:42	-22:51:43	120	1.4	6:57:14	R	2025-02-07 20:19:33	2025-02-07 20:19:34	2460714.348	xkmts.20250207.005979.fits	#3330
MMA	OBJECT	S250206dm	10:14:50	-19:08:56	120	1.44	7:00:25	R	2025-02-07 20:22:31	2025-02-07 20:22:31	2460714.35	xkmts.20250207.005980.fits	#3682
MMA	OBJECT	S250206dm	10:15:07	-19:08:56	120	1.42	7:03:22	R	2025-02-07 20:25:41	2025-02-07 20:25:41	2460714.352	xkmts.20250207.005981.fits	#3682
MMA	OBJECT	S250206dm	10:14:50	-19:01:56	120	1.41	7:06:31	R	2025-02-07 20:28:50	2025-02-07 20:28:50	2460714.354	xkmts.20250207.005982.fits	#3682
MMA	OBJECT	S250206dm	10:15:07	-19:01:56	120	1.39	7:09:42	R	2025-02-07 20:31:59	2025-02-07 20:31:59	2460714.356	xkmts.20250207.005983.fits	#3682
MMA	OBJECT	S250206dm	10:15:54	-28:43:24	120	1.3	7:12:51	R	2025-02-07 20:34:56	2025-02-07 20:34:56	2460714.358	xkmts.20250207.005984.fits	#2820
MMA	OBJECT	S250206dm	10:16:13	-28:43:24	120	1.29	7:15:49	R	2025-02-07 20:38:06	2025-02-07 20:38:06	2460714.36	xkmts.20250207.005985.fits	#2820
MMA	OBJECT	S250206dm	10:15:54	-28:36:24	120	1.28	7:18:59	R	2025-02-07 20:41:15	2025-02-07 20:41:15	2460714.363	xkmts.20250207.005986.fits	#2820
MMA	OBJECT	S250206dm	10:16:13	-28:36:24	120	1.27	7:22:09	R	2025-02-07 20:44:24	2025-02-07 20:44:24	2460714.365	xkmts.20250207.005987.fits	#2820
MMA	OBJECT	S250206dm	10:20:41	-22:58:43	120	1.31	7:25:19	R	2025-02-07 20:47:21	2025-02-07 20:47:21	2460714.367	xkmts.20250207.005988.fits	#3331
MMA	OBJECT	S250206dm	10:20:59	-22:58:43	120	1.3	7:28:17	R	2025-02-07 20:50:31	2025-02-07 20:50:32	2460714.369	xkmts.20250207.005989.fits	#3331
MMA	OBJECT	S250206dm	10:20:41	-22:51:43	120	1.29	7:31:29	R	2025-02-07 20:53:42	2025-02-07 20:53:42	2460714.371	xkmts.20250207.005990.fits	#3331
MMA	OBJECT	S250206dm	10:20:59	-22:51:43	120	1.27	7:34:53	R	2025-02-07 20:56:53	2025-02-07 20:56:53	2460714.374	xkmts.20250207.005991.fits	#3331
MMA	OBJECT	S250206dm	10:28:52	-30:00:00	120	1.25	7:38:55	R	2025-02-07 21:00:55	2025-02-07 21:00:55	2460714.376	xkmts.20250207.005992.fits	#765
MMA	OBJECT	S250206dm	10:29:10	-30:00:00	120	1.24	7:41:51	R	2025-02-07 21:04:04	2025-02-07 21:04:04	2460714.379	xkmts.20250207.005993.fits	#765
MMA	OBJECT	S250206dm	10:28:52	-29:53:00	120	1.23	7:45:09	R	2025-02-07 21:07:07	2025-02-07 21:07:08	2460714.381	xkmts.20250207.005994.fits	#765
MMA	OBJECT	S250206dm	10:29:10	-29:53:00	120	1.22	7:48:06	R	2025-02-07 21:10:18	2025-02-07 21:10:18	2460714.383	xkmts.20250207.005995.fits	#765
MMA	OBJECT	S250206dm	10:29:24	-34:00:00	120	1.2	7:51:17	R	2025-02-07 21:13:27	2025-02-07 21:13:28	2460714.385	xkmts.20250207.005996.fits	#1075
MMA	OBJECT	S250206dm	10:29:44	-34:00:00	120	1.19	7:54:28	R	2025-02-07 21:16:37	2025-02-07 21:16:37	2460714.387	xkmts.20250207.005997.fits	#1075
MMA	OBJECT	S250206dm	10:29:24	-33:53:00	120	1.18	7:57:37	R	2025-02-07 21:19:35	2025-02-07 21:19:35	2460714.389	xkmts.20250207.005998.fits	#1075
MMA	OBJECT	S250206dm	10:29:44	-33:53:00	120	1.18	8:00:36	R	2025-02-07 21:22:45	2025-02-07 21:22:45	2460714.391	xkmts.20250207.005999.fits	#1075
MMA	OBJECT	S250206dm	10:30:32	-26:48:31	120	1.19	8:04:08	R	2025-02-07 21:26:03	2025-02-07 21:26:04	2460714.394	xkmts.20250207.006000.fits	#2989
MMA	OBJECT	S250206dm	10:29:44	-33:53:00	120	1.16	8:08:49	R	2025-02-07 21:30:44	2025-02-07 21:30:45	2460714.397	xkmts.20250207.006001.fits	#1075
MMA	OBJECT	S250206dm	10:30:32	-26:48:31	120	1.17	8:11:48	R	2025-02-07 21:33:43	2025-02-07 21:33:43	2460714.399	xkmts.20250207.006002.fits	#2989
MMA	OBJECT	S250206dm	10:30:50	-26:48:31	120	1.16	8:14:47	R	2025-02-07 21:36:54	2025-02-07 21:36:54	2460714.401	xkmts.20250207.006003.fits	#2989
MMA	OBJECT	S250206dm	10:30:32	-26:41:31	120	1.15	8:17:57	R	2025-02-07 21:40:03	2025-02-07 21:40:04	2460714.404	xkmts.20250207.006004.fits	#2989
MMA	OBJECT	S250206dm	10:30:50	-26:41:31	120	1.15	8:21:09	R	2025-02-07 21:43:02	2025-02-07 21:43:02	2460714.406	xkmts.20250207.006005.fits	#2989
MMA	OBJECT	S250206dm	10:33:15	-28:43:24	120	1.14	8:24:06	R	2025-02-07 21:45:59	2025-02-07 21:45:59	2460714.408	xkmts.20250207.006006.fits	#2822
MMA	OBJECT	S250206dm	10:33:33	-28:43:24	120	1.13	8:27:05	R	2025-02-07 21:48:56	2025-02-07 21:48:56	2460714.41	xkmts.20250207.006007.fits	#2822
MMA	OBJECT	S250206dm	10:33:15	-28:36:24	120	1.13	8:30:02	R	2025-02-07 21:51:54	2025-02-07 21:51:54	2460714.412	xkmts.20250207.006008.fits	#2822
MMA	OBJECT	S250206dm	10:33:33	-28:36:24	120	1.12	8:33:02	R	2025-02-07 21:54:53	2025-02-07 21:54:53	2460714.414	xkmts.20250207.006009.fits	#2822
MMA	OBJECT	S250206dm	10:42:10	-36:00:00	120	1.12	8:36:03	R	2025-02-07 21:57:54	2025-02-07 21:57:54	2460714.416	xkmts.20250207.006010.fits	#1226
MMA	OBJECT	S250206dm	10:42:30	-36:00:00	120	1.12	8:39:03	R	2025-02-07 22:00:53	2025-02-07 22:00:53	2460714.418	xkmts.20250207.006011.fits	#1226
MMA	OBJECT	S250206dm	10:42:10	-35:53:00	120	1.11	8:42:02	R	2025-02-07 22:03:51	2025-02-07 22:03:51	2460714.42	xkmts.20250207.006012.fits	#1226
MMA	OBJECT	S250206dm	10:42:30	-35:53:00	120	1.1	8:45:02	R	2025-02-07 22:06:51	2025-02-07 22:06:51	2460714.422	xkmts.20250207.006013.fits	#1226
MMA	OBJECT	S250206dm	10:45:53	-42:00:00	120	1.11	8:47:59	R	2025-02-07 22:10:01	2025-02-07 22:10:01	2460714.424	xkmts.20250207.006014.fits	#1653
MMA	OBJECT	S250206dm	10:46:15	-42:00:00	120	1.11	8:51:12	R	2025-02-07 22:13:00	2025-02-07 22:13:00	2460714.426	xkmts.20250207.006015.fits	#1653
MMA	OBJECT	S250206dm	10:45:53	-41:53:00	120	1.1	8:54:09	R	2025-02-07 22:16:10	2025-02-07 22:16:10	2460714.429	xkmts.20250207.006016.fits	#1653
MMA	OBJECT	S250206dm	10:46:15	-41:53:00	120	1.1	8:57:21	R	2025-02-07 22:19:21	2025-02-07 22:19:21	2460714.431	xkmts.20250207.006017.fits	#1653
MMA	OBJECT	S250206dm	10:52:30	-46:00:00	120	1.11	9:00:31	R	2025-02-07 22:22:30	2025-02-07 22:22:31	2460714.433	xkmts.20250207.006018.fits	#1918
MMA	OBJECT	S250206dm	10:52:53	-46:00:00	120	1.11	9:03:42	R	2025-02-07 22:25:41	2025-02-07 22:25:41	2460714.435	xkmts.20250207.006019.fits	#1918
MMA	OBJECT	S250206dm	10:52:30	-45:53:00	120	1.1	9:06:53	R	2025-02-07 22:28:52	2025-02-07 22:28:52	2460714.437	xkmts.20250207.006020.fits	#1918
MMA	OBJECT	S250206dm	10:52:53	-45:53:00	120	1.1	9:10:05	R	2025-02-07 22:31:50	2025-02-07 22:31:50	2460714.439	xkmts.20250207.006021.fits	#1918
MMA	OBJECT	S250206dm	10:54:33	-44:00:00	120	1.09	9:13:03	R	2025-02-07 22:35:00	2025-02-07 22:35:00	2460714.442	xkmts.20250207.006022.fits	#1788
MMA	OBJECT	S250206dm	10:54:55	-44:00:00	120	1.09	9:16:14	R	2025-02-07 22:38:10	2025-02-07 22:38:10	2460714.444	xkmts.20250207.006023.fits	#1788
MMA	OBJECT	S250206dm	10:54:33	-43:53:00	120	1.08	9:19:24	R	2025-02-07 22:41:20	2025-02-07 22:41:20	2460714.446	xkmts.20250207.006024.fits	#1788
MMA	OBJECT	S250206dm	10:54:55	-43:53:00	120	1.08	9:22:34	R	2025-02-07 22:44:17	2025-02-07 22:44:17	2460714.448	xkmts.20250207.006025.fits	#1788
MMA	OBJECT	S250206dm	10:56:28	-42:00:00	120	1.07	9:25:31	R	2025-02-07 22:47:26	2025-02-07 22:47:26	2460714.45	xkmts.20250207.006026.fits	#1654
MMA	OBJECT	S250206dm	10:56:50	-42:00:00	120	1.07	9:28:40	R	2025-02-07 22:50:35	2025-02-07 22:50:36	2460714.452	xkmts.20250207.006027.fits	#1654
MMA	OBJECT	S250206dm	10:56:28	-41:53:00	120	1.06	9:31:51	R	2025-02-07 22:53:32	2025-02-07 22:53:32	2460714.455	xkmts.20250207.006028.fits	#1654
MMA	OBJECT	S250206dm	10:56:50	-41:53:00	120	1.06	9:34:51	R	2025-02-07 22:56:45	2025-02-07 22:56:45	2460714.457	xkmts.20250207.006029.fits	#1654
MMA	OBJECT	S250206dm	10:58:17	-40:00:00	120	1.05	9:38:00	R	2025-02-07 22:59:53	2025-02-07 22:59:54	2460714.459	xkmts.20250207.006030.fits	#1516
MMA	OBJECT	S250206dm	10:58:38	-40:00:00	120	1.05	9:42:40	R	2025-02-07 23:04:20	2025-02-07 23:04:20	2460714.462	xkmts.20250207.006031.fits	#1516
MMA	OBJECT	S250206dm	10:58:17	-39:53:00	120	1.05	9:46:44	R	2025-02-07 23:08:23	2025-02-07 23:08:23	2460714.465	xkmts.20250207.006032.fits	#1516
MMA	OBJECT	S250206dm	10:58:38	-39:53:00	120	1.04	9:49:41	R	2025-02-07 23:11:20	2025-02-07 23:11:21	2460714.467	xkmts.20250207.006033.fits	#1516"""
saao_250207 = pd.read_csv(StringIO(saao_250207), sep="\t")
saao_250207 = Table.from_pandas(saao_250207)

ctio_250208 = """
PROJID	IMGTYPE	OBJECT	RA	DEC	EXP	SECZ	ST	FILT	DATE-OBS	UT	MIDJD	FILENAME	Tile
MMA	OBJECT	S250206dm	9:58:39	-19:08:55	120	1.9	5:46:54	I	2025-02-08 1:14:40	2025-02-08 1:14:40	2460714.553	xkmtc.20250208.061766.fits	#3680
MMA	OBJECT	S250206dm	9:58:56	-19:08:55	120	1.87	5:49:52	I	2025-02-08 1:17:38	2025-02-08 1:17:38	2460714.555	xkmtc.20250208.061767.fits	#3680
MMA	OBJECT	S250206dm	9:58:39	-19:01:55	120	1.83	5:52:49	I	2025-02-08 1:20:34	2025-02-08 1:20:34	2460714.557	xkmtc.20250208.061768.fits	#3680
MMA	OBJECT	S250206dm	9:58:56	-19:01:55	120	1.81	5:55:46	I	2025-02-08 1:23:30	2025-02-08 1:23:31	2460714.559	xkmtc.20250208.061769.fits	#3680
MMA	OBJECT	S250206dm	10:00:00	-17:14:02	120	1.82	5:58:54	I	2025-02-08 1:26:38	2025-02-08 1:26:38	2460714.561	xkmtc.20250208.061770.fits	#3859
MMA	OBJECT	S250206dm	10:00:17	-17:14:02	120	1.79	6:01:52	I	2025-02-08 1:29:48	2025-02-08 1:29:48	2460714.563	xkmtc.20250208.061771.fits	#3859
MMA	OBJECT	S250206dm	10:00:00	-17:07:02	120	1.76	6:05:03	I	2025-02-08 1:32:46	2025-02-08 1:32:47	2460714.565	xkmtc.20250208.061772.fits	#3859
MMA	OBJECT	S250206dm	10:00:17	-17:07:02	120	1.73	6:08:02	I	2025-02-08 1:35:57	2025-02-08 1:35:58	2460714.567	xkmtc.20250208.061773.fits	#3859
MMA	OBJECT	S250206dm	10:04:08	-22:58:42	120	1.65	6:11:22	I	2025-02-08 1:39:03	2025-02-08 1:39:03	2460714.569	xkmtc.20250208.061774.fits	#3329
MMA	OBJECT	S250206dm	10:04:26	-22:58:42	120	1.62	6:14:31	I	2025-02-08 1:42:11	2025-02-08 1:42:12	2460714.572	xkmtc.20250208.061775.fits	#3329
MMA	OBJECT	S250206dm	10:04:08	-22:51:42	120	1.6	6:17:27	I	2025-02-08 1:45:07	2025-02-08 1:45:08	2460714.574	xkmtc.20250208.061776.fits	#3329
MMA	OBJECT	S250206dm	10:04:26	-22:51:42	120	1.58	6:20:24	I	2025-02-08 1:48:04	2025-02-08 1:48:05	2460714.576	xkmtc.20250208.061777.fits	#3329
MMA	OBJECT	S250206dm	10:05:27	-21:03:49	120	1.59	6:23:25	I	2025-02-08 1:51:05	2025-02-08 1:51:05	2460714.578	xkmtc.20250208.061778.fits	#3504
MMA	OBJECT	S250206dm	10:05:44	-21:03:49	120	1.57	6:26:23	I	2025-02-08 1:54:01	2025-02-08 1:54:02	2460714.58	xkmtc.20250208.061779.fits	#3504
MMA	OBJECT	S250206dm	10:05:27	-20:56:49	120	1.55	6:29:23	I	2025-02-08 1:57:01	2025-02-08 1:57:02	2460714.582	xkmtc.20250208.061780.fits	#3504
MMA	OBJECT	S250206dm	10:05:44	-20:56:49	120	1.53	6:32:21	I	2025-02-08 1:59:58	2025-02-08 1:59:59	2460714.584	xkmtc.20250208.061781.fits	#3504
MMA	OBJECT	S250206dm	10:06:44	-19:08:55	120	1.54	6:35:18	I	2025-02-08 2:02:55	2025-02-08 2:02:55	2460714.586	xkmtc.20250208.061782.fits	#3681
MMA	OBJECT	S250206dm	10:07:01	-19:08:55	120	1.52	6:38:17	I	2025-02-08 2:06:07	2025-02-08 2:06:08	2460714.588	xkmtc.20250208.061783.fits	#3681
MMA	OBJECT	S250206dm	10:06:44	-19:01:55	120	1.5	6:41:28	I	2025-02-08 2:09:04	2025-02-08 2:09:05	2460714.59	xkmtc.20250208.061784.fits	#3681
MMA	OBJECT	S250206dm	10:07:01	-19:01:55	120	1.48	6:44:25	I	2025-02-08 2:12:01	2025-02-08 2:12:01	2460714.592	xkmtc.20250208.061785.fits	#3681
MMA	OBJECT	S250206dm	10:08:00	-17:14:02	120	1.49	6:47:21	I	2025-02-08 2:14:57	2025-02-08 2:14:58	2460714.594	xkmtc.20250208.061786.fits	#3860
MMA	OBJECT	S250206dm	10:08:17	-17:14:02	120	1.47	6:50:30	I	2025-02-08 2:18:05	2025-02-08 2:18:06	2460714.597	xkmtc.20250208.061787.fits	#3860
MMA	OBJECT	S250206dm	10:08:00	-17:07:02	120	1.46	6:53:27	I	2025-02-08 2:21:01	2025-02-08 2:21:02	2460714.599	xkmtc.20250208.061788.fits	#3860
MMA	OBJECT	S250206dm	10:08:17	-17:07:02	120	1.44	6:56:25	I	2025-02-08 2:23:59	2025-02-08 2:23:59	2460714.601	xkmtc.20250208.061789.fits	#3860
MMA	OBJECT	S250206dm	10:11:10	-24:53:36	120	1.37	6:59:20	I	2025-02-08 2:26:54	2025-02-08 2:26:55	2460714.603	xkmtc.20250208.061790.fits	#3157
MMA	OBJECT	S250206dm	10:11:27	-24:53:36	120	1.36	7:02:18	I	2025-02-08 2:29:51	2025-02-08 2:29:52	2460714.605	xkmtc.20250208.061791.fits	#3157
MMA	OBJECT	S250206dm	10:11:10	-24:46:36	120	1.34	7:05:14	I	2025-02-08 2:32:47	2025-02-08 2:32:48	2460714.607	xkmtc.20250208.061792.fits	#3157
MMA	OBJECT	S250206dm	10:11:27	-24:46:36	120	1.33	7:08:11	I	2025-02-08 2:35:43	2025-02-08 2:35:44	2460714.609	xkmtc.20250208.061793.fits	#3157
MMA	OBJECT	S250206dm	10:12:25	-22:58:42	120	1.33	7:11:16	I	2025-02-08 2:38:47	2025-02-08 2:38:48	2460714.611	xkmtc.20250208.061794.fits	#3330
MMA	OBJECT	S250206dm	10:12:42	-22:58:42	120	1.32	7:14:12	I	2025-02-08 2:41:43	2025-02-08 2:41:44	2460714.613	xkmtc.20250208.061795.fits	#3330
MMA	OBJECT	S250206dm	10:12:25	-22:51:42	120	1.31	7:17:11	I	2025-02-08 2:44:41	2025-02-08 2:44:42	2460714.615	xkmtc.20250208.061796.fits	#3330
MMA	OBJECT	S250206dm	10:12:42	-22:51:42	120	1.3	7:20:08	I	2025-02-08 2:47:38	2025-02-08 2:47:38	2460714.617	xkmtc.20250208.061797.fits	#3330
MMA	OBJECT	S250206dm	10:13:29	-26:48:30	120	1.27	7:23:12	I	2025-02-08 2:50:55	2025-02-08 2:50:56	2460714.619	xkmtc.20250208.061798.fits	#2987
MMA	OBJECT	S250206dm	10:13:47	-26:48:30	120	1.26	7:26:25	I	2025-02-08 2:54:07	2025-02-08 2:54:07	2460714.622	xkmtc.20250208.061799.fits	#2987
MMA	OBJECT	S250206dm	10:13:29	-26:41:30	120	1.25	7:29:35	I	2025-02-08 2:57:05	2025-02-08 2:57:05	2460714.624	xkmtc.20250208.061800.fits	#2987
MMA	OBJECT	S250206dm	10:13:47	-26:48:30	120	1.24	7:32:34	I	2025-02-08 3:00:02	2025-02-08 3:00:02	2460714.626	xkmtc.20250208.061801.fits	#2987
MMA	OBJECT	S250206dm	10:13:47	-26:41:30	120	1.23	7:35:29	I	2025-02-08 3:02:57	2025-02-08 3:02:57	2460714.628	xkmtc.20250208.061802.fits	#2987
MMA	OBJECT	S250206dm	10:13:38	-21:03:49	120	1.24	7:38:25	I	2025-02-08 3:05:53	2025-02-08 3:05:53	2460714.63	xkmtc.20250208.061803.fits	#3505
MMA	OBJECT	S250206dm	10:13:55	-21:03:49	120	1.24	7:41:23	I	2025-02-08 3:08:50	2025-02-08 3:08:50	2460714.632	xkmtc.20250208.061804.fits	#3505
MMA	OBJECT	S250206dm	10:13:38	-20:56:49	120	1.23	7:44:20	I	2025-02-08 3:11:46	2025-02-08 3:11:47	2460714.634	xkmtc.20250208.061805.fits	#3505
MMA	OBJECT	S250206dm	10:13:55	-20:56:49	120	1.22	7:47:17	I	2025-02-08 3:14:43	2025-02-08 3:14:43	2460714.636	xkmtc.20250208.061806.fits	#3505
MMA	OBJECT	S250206dm	10:14:50	-19:08:55	120	1.22	7:50:14	I	2025-02-08 3:17:39	2025-02-08 3:17:40	2460714.638	xkmtc.20250208.061807.fits	#3682
MMA	OBJECT	S250206dm	10:15:07	-19:08:55	120	1.21	7:53:11	I	2025-02-08 3:20:36	2025-02-08 3:20:36	2460714.64	xkmtc.20250208.061808.fits	#3682
MMA	OBJECT	S250206dm	10:14:50	-19:01:55	120	1.2	7:56:07	I	2025-02-08 3:23:45	2025-02-08 3:23:45	2460714.642	xkmtc.20250208.061809.fits	#3682
MMA	OBJECT	S250206dm	10:15:07	-19:01:55	120	1.2	7:59:18	I	2025-02-08 3:26:43	2025-02-08 3:26:43	2460714.644	xkmtc.20250208.061810.fits	#3682
MMA	OBJECT	S250206dm	10:19:32	-24:53:36	120	1.17	8:02:16	I	2025-02-08 3:29:40	2025-02-08 3:29:40	2460714.646	xkmtc.20250208.061811.fits	#3158
MMA	OBJECT	S250206dm	10:19:50	-24:53:36	120	1.16	8:05:13	I	2025-02-08 3:32:36	2025-02-08 3:32:37	2460714.648	xkmtc.20250208.061812.fits	#3158
MMA	OBJECT	S250206dm	10:19:32	-24:46:36	120	1.16	8:08:13	I	2025-02-08 3:35:48	2025-02-08 3:35:49	2460714.651	xkmtc.20250208.061813.fits	#3158
MMA	OBJECT	S250206dm	10:19:50	-24:46:36	120	1.15	8:11:26	I	2025-02-08 3:39:01	2025-02-08 3:39:01	2460714.653	xkmtc.20250208.061814.fits	#3158
MMA	OBJECT	S250206dm	10:20:41	-22:58:42	120	1.15	8:14:42	I	2025-02-08 3:42:03	2025-02-08 3:42:04	2460714.655	xkmtc.20250208.061815.fits	#3331
MMA	OBJECT	S250206dm	10:20:59	-22:58:42	120	1.14	8:17:39	I	2025-02-08 3:45:01	2025-02-08 3:45:01	2460714.657	xkmtc.20250208.061816.fits	#3331
MMA	OBJECT	S250206dm	10:20:41	-22:51:42	120	1.14	8:20:48	I	2025-02-08 3:48:08	2025-02-08 3:48:08	2460714.659	xkmtc.20250208.061817.fits	#3331
MMA	OBJECT	S250206dm	10:20:59	-22:51:42	120	1.13	8:23:45	I	2025-02-08 3:51:05	2025-02-08 3:51:06	2460714.661	xkmtc.20250208.061818.fits	#3331
MMA	OBJECT	S250206dm	10:22:01	-26:48:30	120	1.11	8:26:42	I	2025-02-08 3:54:01	2025-02-08 3:54:01	2460714.663	xkmtc.20250208.061819.fits	#2988
MMA	OBJECT	S250206dm	10:22:19	-26:48:30	120	1.11	8:29:43	I	2025-02-08 3:57:01	2025-02-08 3:57:02	2460714.665	xkmtc.20250208.061820.fits	#2988
MMA	OBJECT	S250206dm	10:22:01	-26:41:30	120	1.1	8:32:50	I	2025-02-08 4:00:08	2025-02-08 4:00:09	2460714.667	xkmtc.20250208.061821.fits	#2988
MMA	OBJECT	S250206dm	10:22:19	-26:41:30	120	1.1	8:35:50	I	2025-02-08 4:03:08	2025-02-08 4:03:09	2460714.67	xkmtc.20250208.061822.fits	#2988
MMA	OBJECT	S250206dm	10:24:35	-28:43:23	120	1.09	8:38:47	I	2025-02-08 4:06:04	2025-02-08 4:06:05	2460714.672	xkmtc.20250208.061823.fits	#2821
MMA	OBJECT	S250206dm	10:24:53	-28:43:23	120	1.09	8:41:44	I	2025-02-08 4:09:01	2025-02-08 4:09:01	2460714.674	xkmtc.20250208.061824.fits	#2821
MMA	OBJECT	S250206dm	10:24:35	-28:36:23	120	1.08	8:44:46	I	2025-02-08 4:12:15	2025-02-08 4:12:15	2460714.676	xkmtc.20250208.061825.fits	#2821
MMA	OBJECT	S250206dm	10:24:53	-28:36:23	120	1.08	8:47:56	I	2025-02-08 4:15:12	2025-02-08 4:15:12	2460714.678	xkmtc.20250208.061826.fits	#2821
MMA	OBJECT	S250206dm	10:28:52	-29:59:59	120	1.08	8:50:52	I	2025-02-08 4:18:08	2025-02-08 4:18:09	2460714.68	xkmtc.20250208.061827.fits	#765
MMA	OBJECT	S250206dm	10:29:10	-29:59:59	120	1.07	8:53:49	I	2025-02-08 4:21:04	2025-02-08 4:21:05	2460714.682	xkmtc.20250208.061828.fits	#765
MMA	OBJECT	S250206dm	10:28:52	-29:52:59	120	1.07	8:56:46	I	2025-02-08 4:24:00	2025-02-08 4:24:01	2460714.684	xkmtc.20250208.061829.fits	#765
MMA	OBJECT	S250206dm	10:29:10	-29:52:59	120	1.06	8:59:43	I	2025-02-08 4:26:57	2025-02-08 4:26:57	2460714.686	xkmtc.20250208.061830.fits	#765
MMA	OBJECT	S250206dm	10:29:24	-33:59:59	120	1.06	9:02:39	I	2025-02-08 4:29:53	2025-02-08 4:29:54	2460714.688	xkmtc.20250208.061831.fits	#1075
MMA	OBJECT	S250206dm	10:29:43	-33:59:59	120	1.06	9:05:38	I	2025-02-08 4:33:04	2025-02-08 4:33:04	2460714.69	xkmtc.20250208.061832.fits	#1075
MMA	OBJECT	S250206dm	10:29:24	-33:52:59	120	1.05	9:08:48	I	2025-02-08 4:36:00	2025-02-08 4:36:01	2460714.692	xkmtc.20250208.061833.fits	#1075
MMA	OBJECT	S250206dm	10:29:43	-33:59:59	120	1.05	9:11:45	I	2025-02-08 4:38:57	2025-02-08 4:38:58	2460714.694	xkmtc.20250208.061834.fits	#1075
MMA	OBJECT	S250206dm	10:29:43	-33:52:59	120	1.05	9:14:43	I	2025-02-08 4:41:54	2025-02-08 4:41:55	2460714.696	xkmtc.20250208.061835.fits	#1075
MMA	OBJECT	S250206dm	10:33:15	-28:43:22	120	1.05	9:17:41	I	2025-02-08 4:45:05	2025-02-08 4:45:05	2460714.699	xkmtc.20250208.061836.fits	#2822
MMA	OBJECT	S250206dm	10:33:33	-28:43:22	120	1.04	9:20:50	I	2025-02-08 4:48:02	2025-02-08 4:48:02	2460714.701	xkmtc.20250208.061837.fits	#2822
MMA	OBJECT	S250206dm	10:33:15	-28:36:22	120	1.04	9:23:48	I	2025-02-08 4:50:58	2025-02-08 4:50:59	2460714.703	xkmtc.20250208.061838.fits	#2822
MMA	OBJECT	S250206dm	10:33:33	-28:36:22	120	1.04	9:26:45	I	2025-02-08 4:53:55	2025-02-08 4:53:55	2460714.705	xkmtc.20250208.061839.fits	#2822
MMA	OBJECT	S250206dm	10:38:56	-33:59:59	120	1.04	9:29:43	I	2025-02-08 4:56:52	2025-02-08 4:56:52	2460714.707	xkmtc.20250208.061840.fits	#1076
MMA	OBJECT	S250206dm	10:39:16	-33:59:58	120	1.04	9:32:43	I	2025-02-08 5:00:05	2025-02-08 5:00:05	2460714.709	xkmtc.20250208.061841.fits	#1076
MMA	OBJECT	S250206dm	10:38:56	-33:52:58	120	1.03	9:35:54	I	2025-02-08 5:03:02	2025-02-08 5:03:02	2460714.711	xkmtc.20250208.061842.fits	#1076
MMA	OBJECT	S250206dm	10:39:16	-33:52:58	120	1.03	9:38:50	I	2025-02-08 5:05:58	2025-02-08 5:05:58	2460714.713	xkmtc.20250208.061843.fits	#1076
MMA	OBJECT	S250206dm	10:42:10	-35:59:58	120	1.03	9:41:51	I	2025-02-08 5:08:58	2025-02-08 5:08:59	2460714.715	xkmtc.20250208.061844.fits	#1226
MMA	OBJECT	S250206dm	10:42:29	-35:59:59	120	1.03	9:44:48	I	2025-02-08 5:11:55	2025-02-08 5:11:55	2460714.717	xkmtc.20250208.061845.fits	#1226
MMA	OBJECT	S250206dm	10:42:10	-35:52:58	120	1.03	9:47:45	I	2025-02-08 5:14:51	2025-02-08 5:14:51	2460714.719	xkmtc.20250208.061846.fits	#1226
MMA	OBJECT	S250206dm	10:42:30	-35:52:58	120	1.03	9:50:42	I	2025-02-08 5:17:48	2025-02-08 5:17:48	2460714.721	xkmtc.20250208.061847.fits	#1226
MMA	OBJECT	S250206dm	10:45:53	-41:59:58	120	1.04	9:53:39	I	2025-02-08 5:20:57	2025-02-08 5:20:57	2460714.724	xkmtc.20250208.061848.fits	#1653
MMA	OBJECT	S250206dm	10:46:15	-41:59:59	120	1.04	9:56:51	I	2025-02-08 5:24:08	2025-02-08 5:24:09	2460714.726	xkmtc.20250208.061849.fits	#1653
MMA	OBJECT	S250206dm	10:45:53	-41:52:59	120	1.04	10:00:03	I	2025-02-08 5:27:20	2025-02-08 5:27:20	2460714.728	xkmtc.20250208.061850.fits	#1653
MMA	OBJECT	S250206dm	10:46:14	-41:52:59	120	1.04	10:03:18	I	2025-02-08 5:30:21	2025-02-08 5:30:22	2460714.73	xkmtc.20250208.061851.fits	#1653
MMA	OBJECT	S250206dm	10:48:00	-39:59:58	120	1.03	10:06:20	I	2025-02-08 5:33:34	2025-02-08 5:33:35	2460714.732	xkmtc.20250208.061852.fits	#1515
MMA	OBJECT	S250206dm	10:48:21	-39:59:58	120	1.03	10:09:27	I	2025-02-08 5:36:31	2025-02-08 5:36:32	2460714.734	xkmtc.20250208.061853.fits	#1515
MMA	OBJECT	S250206dm	10:48:00	-39:52:59	120	1.03	10:12:26	I	2025-02-08 5:39:29	2025-02-08 5:39:29	2460714.736	xkmtc.20250208.061854.fits	#1515
MMA	OBJECT	S250206dm	10:48:21	-39:52:58	120	1.02	10:15:24	I	2025-02-08 5:42:25	2025-02-08 5:42:25	2460714.738	xkmtc.20250208.061855.fits	#1515
MMA	OBJECT	S250206dm	10:50:00	-37:59:58	120	1.02	10:18:20	I	2025-02-08 5:45:21	2025-02-08 5:45:21	2460714.741	xkmtc.20250208.061856.fits	#1373
MMA	OBJECT	S250206dm	10:50:20	-37:59:58	120	1.02	10:21:24	I	2025-02-08 5:48:24	2025-02-08 5:48:24	2460714.743	xkmtc.20250208.061857.fits	#1373
MMA	OBJECT	S250206dm	10:50:00	-37:52:58	120	1.02	10:24:20	I	2025-02-08 5:51:20	2025-02-08 5:51:20	2460714.745	xkmtc.20250208.061858.fits	#1373
MMA	OBJECT	S250206dm	10:50:20	-37:52:58	120	1.01	10:27:17	I	2025-02-08 5:54:17	2025-02-08 5:54:17	2460714.747	xkmtc.20250208.061859.fits	#1373
MMA	OBJECT	S250206dm	10:52:30	-45:59:59	120	1.04	10:30:16	I	2025-02-08 5:57:16	2025-02-08 5:57:16	2460714.749	xkmtc.20250208.061860.fits	#1918
MMA	OBJECT	S250206dm	10:52:53	-45:59:58	120	1.04	10:33:14	I	2025-02-08 6:00:12	2025-02-08 6:00:13	2460714.751	xkmtc.20250208.061861.fits	#1918
MMA	OBJECT	S250206dm	10:52:30	-45:52:58	120	1.04	10:36:11	I	2025-02-08 6:03:09	2025-02-08 6:03:10	2460714.753	xkmtc.20250208.061862.fits	#1918
MMA	OBJECT	S250206dm	10:52:53	-45:52:59	120	1.04	10:39:08	I	2025-02-08 6:06:06	2025-02-08 6:06:06	2460714.755	xkmtc.20250208.061863.fits	#1918
MMA	OBJECT	S250206dm	10:54:33	-43:59:59	120	1.03	10:42:05	I	2025-02-08 6:09:02	2025-02-08 6:09:03	2460714.757	xkmtc.20250208.061864.fits	#1788
MMA	OBJECT	S250206dm	10:54:55	-43:59:58	120	1.03	10:45:06	I	2025-02-08 6:12:16	2025-02-08 6:12:16	2460714.759	xkmtc.20250208.061865.fits	#1788
MMA	OBJECT	S250206dm	10:54:33	-43:52:59	120	1.03	10:48:23	I	2025-02-08 6:15:19	2025-02-08 6:15:20	2460714.761	xkmtc.20250208.061866.fits	#1788
MMA	OBJECT	S250206dm	10:54:55	-43:52:58	120	1.03	10:51:23	I	2025-02-08 6:18:19	2025-02-08 6:18:19	2460714.763	xkmtc.20250208.061867.fits	#1788
MMA	OBJECT	S250206dm	10:56:28	-41:59:58	120	1.02	10:54:20	I	2025-02-08 6:21:17	2025-02-08 6:21:17	2460714.765	xkmtc.20250208.061868.fits	#1654
MMA	OBJECT	S250206dm	10:56:50	-41:59:59	120	1.02	10:57:21	I	2025-02-08 6:24:28	2025-02-08 6:24:29	2460714.768	xkmtc.20250208.061869.fits	#1654
MMA	OBJECT	S250206dm	10:56:28	-41:52:58	120	1.02	11:00:31	I	2025-02-08 6:27:25	2025-02-08 6:27:26	2460714.77	xkmtc.20250208.061870.fits	#1654
MMA	OBJECT	S250206dm	10:56:50	-41:52:58	120	1.02	11:03:35	I	2025-02-08 6:30:29	2025-02-08 6:30:29	2460714.772	xkmtc.20250208.061871.fits	#1654
MMA	OBJECT	S250206dm	10:58:17	-39:59:58	120	1.02	11:06:39	I	2025-02-08 6:33:32	2025-02-08 6:33:32	2460714.774	xkmtc.20250208.061872.fits	#1516
MMA	OBJECT	S250206dm	10:58:38	-39:59:58	120	1.02	11:09:40	I	2025-02-08 6:36:46	2025-02-08 6:36:46	2460714.776	xkmtc.20250208.061873.fits	#1516
MMA	OBJECT	S250206dm	10:58:17	-39:52:58	120	1.02	11:12:50	I	2025-02-08 6:39:43	2025-02-08 6:39:43	2460714.778	xkmtc.20250208.061874.fits	#1516
MMA	OBJECT	S250206dm	10:58:38	-39:52:58	120	1.02	11:15:51	I	2025-02-08 6:42:42	2025-02-08 6:42:42	2460714.78	xkmtc.20250208.061875.fits	#1516
MMA	OBJECT	S250206dm	11:03:45	-45:59:58	120	1.04	11:18:51	I	2025-02-08 6:45:43	2025-02-08 6:45:43	2460714.782	xkmtc.20250208.061876.fits	#1919
MMA	OBJECT	S250206dm	11:04:08	-45:59:59	120	1.04	11:21:54	I	2025-02-08 6:48:44	2025-02-08 6:48:45	2460714.785	xkmtc.20250208.061877.fits	#1919
MMA	OBJECT	S250206dm	11:03:45	-45:52:58	120	1.04	11:24:51	I	2025-02-08 6:51:54	2025-02-08 6:51:54	2460714.787	xkmtc.20250208.061878.fits	#1919
MMA	OBJECT	S250206dm	11:04:08	-45:52:58	120	1.04	11:28:09	I	2025-02-08 6:54:59	2025-02-08 6:54:59	2460714.789	xkmtc.20250208.061879.fits	#1919
MMA	OBJECT	S250206dm	11:05:27	-43:59:58	120	1.03	11:31:07	I	2025-02-08 6:57:56	2025-02-08 6:57:56	2460714.791	xkmtc.20250208.061880.fits	#1789
MMA	OBJECT	S250206dm	11:05:50	-43:59:58	120	1.04	11:34:08	I	2025-02-08 7:01:09	2025-02-08 7:01:09	2460714.793	xkmtc.20250208.061881.fits	#1789
MMA	OBJECT	S250206dm	11:05:27	-43:52:59	120	1.04	11:37:20	I	2025-02-08 7:04:08	2025-02-08 7:04:08	2460714.795	xkmtc.20250208.061882.fits	#1789
MMA	OBJECT	S250206dm	11:05:50	-43:52:58	120	1.04	11:40:18	I	2025-02-08 7:07:18	2025-02-08 7:07:18	2460714.797	xkmtc.20250208.061883.fits	#1789
MMA	OBJECT	S250206dm	11:07:04	-41:59:58	120	1.03	11:43:35	I	2025-02-08 7:10:22	2025-02-08 7:10:23	2460714.8	xkmtc.20250208.061884.fits	#1655
MMA	OBJECT	S250206dm	11:07:25	-41:59:58	120	1.03	11:46:36	I	2025-02-08 7:13:23	2025-02-08 7:13:24	2460714.802	xkmtc.20250208.061885.fits	#1655
MMA	OBJECT	S250206dm	11:07:04	-41:52:58	120	1.03	11:49:41	I	2025-02-08 7:16:27	2025-02-08 7:16:28	2460714.804	xkmtc.20250208.061886.fits	#1655
MMA	OBJECT	S250206dm	11:07:25	-41:52:58	120	1.03	11:52:40	I	2025-02-08 7:19:25	2025-02-08 7:19:26	2460714.806	xkmtc.20250208.061887.fits	#1655
MMA	OBJECT	S250206dm	11:07:19	-47:59:59	120	1.06	11:55:42	I	2025-02-08 7:22:28	2025-02-08 7:22:28	2460714.808	xkmtc.20250208.061888.fits	#2045
MMA	OBJECT	S250206dm	11:07:43	-47:59:59	120	1.07	11:58:40	I	2025-02-08 7:25:25	2025-02-08 7:25:25	2460714.81	xkmtc.20250208.061889.fits	#2045
MMA	OBJECT	S250206dm	11:07:19	-47:52:59	120	1.07	12:01:42	I	2025-02-08 7:28:27	2025-02-08 7:28:27	2460714.812	xkmtc.20250208.061890.fits	#2045
MMA	OBJECT	S250206dm	11:07:43	-47:52:59	120	1.07	12:04:46	I	2025-02-08 7:31:29	2025-02-08 7:31:30	2460714.814	xkmtc.20250208.061891.fits	#2045
MMA	OBJECT	S250206dm	11:15:00	-45:59:58	120	1.06	12:07:46	I	2025-02-08 7:34:29	2025-02-08 7:34:30	2460714.816	xkmtc.20250208.061892.fits	#1920
MMA	OBJECT	S250206dm	11:15:23	-45:59:58	120	1.06	12:10:50	I	2025-02-08 7:37:32	2025-02-08 7:37:33	2460714.818	xkmtc.20250208.061893.fits	#1920
MMA	OBJECT	S250206dm	11:15:00	-45:52:58	120	1.06	12:13:53	I	2025-02-08 7:40:36	2025-02-08 7:40:36	2460714.821	xkmtc.20250208.061894.fits	#1920
MMA	OBJECT	S250206dm	11:15:23	-45:52:58	120	1.06	12:17:00	I	2025-02-08 7:43:41	2025-02-08 7:43:42	2460714.823	xkmtc.20250208.061895.fits	#1920
MMA	OBJECT	S250206dm	11:16:22	-43:59:59	120	1.05	12:20:03	I	2025-02-08 7:46:45	2025-02-08 7:46:45	2460714.825	xkmtc.20250208.061896.fits	#1790
MMA	OBJECT	S250206dm	11:16:44	-43:59:59	120	1.06	12:23:02	I	2025-02-08 7:49:43	2025-02-08 7:49:43	2460714.827	xkmtc.20250208.061897.fits	#1790
MMA	OBJECT	S250206dm	11:16:22	-43:52:59	120	1.06	12:25:59	I	2025-02-08 7:52:39	2025-02-08 7:52:40	2460714.829	xkmtc.20250208.061898.fits	#1790
MMA	OBJECT	S250206dm	11:16:44	-43:52:59	120	1.06	12:28:56	I	2025-02-08 7:55:36	2025-02-08 7:55:36	2460714.831	xkmtc.20250208.061899.fits	#1790
MMA	OBJECT	S250206dm	11:17:39	-49:59:59	120	1.1	12:32:02	I	2025-02-08 7:58:54	2025-02-08 7:58:54	2460714.833	xkmtc.20250208.061900.fits	#2167
MMA	OBJECT	S250206dm	11:18:04	-49:59:59	120	1.1	12:35:11	I	2025-02-08 8:01:50	2025-02-08 8:01:50	2460714.835	xkmtc.20250208.061901.fits	#2167
MMA	OBJECT	S250206dm	11:17:39	-49:52:59	120	1.1	12:38:07	I	2025-02-08 8:04:58	2025-02-08 8:04:58	2460714.837	xkmtc.20250208.061902.fits	#2167
MMA	OBJECT	S250206dm	11:18:04	-49:52:59	120	1.1	12:41:17	I	2025-02-08 8:07:54	2025-02-08 8:07:55	2460714.84	xkmtc.20250208.061903.fits	#2167
MMA	OBJECT	S250206dm	11:19:02	-47:59:59	120	1.09	12:44:17	I	2025-02-08 8:11:06	2025-02-08 8:11:07	2460714.842	xkmtc.20250208.061904.fits	#2046
MMA	OBJECT	S250206dm	11:19:25	-47:59:59	120	1.1	12:47:27	I	2025-02-08 8:14:04	2025-02-08 8:14:05	2460714.844	xkmtc.20250208.061905.fits	#2046
MMA	OBJECT	S250206dm	11:19:02	-47:52:59	120	1.1	12:50:26	I	2025-02-08 8:17:02	2025-02-08 8:17:03	2460714.846	xkmtc.20250208.061906.fits	#2046
MMA	OBJECT	S250206dm	11:19:25	-47:52:59	120	1.1	12:53:25	I	2025-02-08 8:20:01	2025-02-08 8:20:01	2460714.848	xkmtc.20250208.061907.fits	#2046
MMA	OBJECT	S250206dm	11:26:15	-45:59:59	120	1.09	12:56:31	I	2025-02-08 8:23:07	2025-02-08 8:23:07	2460714.85	xkmtc.20250208.061908.fits	#1921
MMA	OBJECT	S250206dm	11:26:38	-45:59:58	120	1.09	12:59:29	I	2025-02-08 8:26:03	2025-02-08 8:26:04	2460714.852	xkmtc.20250208.061909.fits	#1921
MMA	OBJECT	S250206dm	11:26:15	-45:52:59	120	1.1	13:02:37	I	2025-02-08 8:29:12	2025-02-08 8:29:12	2460714.854	xkmtc.20250208.061910.fits	#1921
MMA	OBJECT	S250206dm	11:26:38	-45:52:59	120	1.1	13:05:34	I	2025-02-08 8:32:08	2025-02-08 8:32:08	2460714.856	xkmtc.20250208.061911.fits	#1921
MMA	OBJECT	S250206dm	11:29:45	-49:59:59	120	1.12	13:08:38	I	2025-02-08 8:35:11	2025-02-08 8:35:11	2460714.858	xkmtc.20250208.061912.fits	#2168
MMA	OBJECT	S250206dm	11:30:10	-49:59:59	120	1.13	13:11:36	I	2025-02-08 8:38:08	2025-02-08 8:38:09	2460714.861	xkmtc.20250208.061913.fits	#2168
MMA	OBJECT	S250206dm	11:29:45	-49:52:59	120	1.13	13:14:38	I	2025-02-08 8:41:10	2025-02-08 8:41:10	2460714.863	xkmtc.20250208.061914.fits	#2168
MMA	OBJECT	S250206dm	11:30:10	-49:52:59	120	1.13	13:17:39	I	2025-02-08 8:44:10	2025-02-08 8:44:11	2460714.865	xkmtc.20250208.061915.fits	#2168
MMA	OBJECT	S250206dm	11:30:44	-47:59:59	120	1.13	13:20:38	I	2025-02-08 8:47:22	2025-02-08 8:47:23	2460714.867	xkmtc.20250208.061916.fits	#2047
MMA	OBJECT	S250206dm	11:31:08	-47:59:59	120	1.13	13:23:58	I	2025-02-08 8:50:29	2025-02-08 8:50:29	2460714.869	xkmtc.20250208.061917.fits	#2047
MMA	OBJECT	S250206dm	11:30:44	-47:52:59	120	1.13	13:27:00	I	2025-02-08 8:53:30	2025-02-08 8:53:30	2460714.871	xkmtc.20250208.061918.fits	#2047
MMA	OBJECT	S250206dm	11:31:08	-47:52:59	120	1.14	13:30:00	I	2025-02-08 8:56:43	2025-02-08 8:56:43	2460714.873	xkmtc.20250208.061919.fits	#2047
"""
ctio_250208 = pd.read_csv(StringIO(ctio_250208), sep="\t")
ctio_250208 = Table.from_pandas(ctio_250208)

saao_250208 = """
PROJID	IMGTYPE	OBJECT	RA	DEC	EXP	SECZ	ST	FILT	DATE-OBS	UT	MIDJD	FILENAME	Tile
MMA	OBJECT	S250206dm	15:08:19	-70:00:00	120	1.56	11:09:51	R	2025-02-09 0:27:20	2025-02-09 0:27:19	2460715.52	xkmts.20250209.006183.fits	#359
MMA	OBJECT	S250206dm	15:08:19	-70:00:00	120	1.52	11:24:13	R	2025-02-09 0:41:39	2025-02-09 0:41:38	2460715.53	xkmts.20250209.006184.fits	#359
MMA	OBJECT	S250206dm	15:09:05	-70:00:00	120	1.51	11:27:12	R	2025-02-09 0:44:39	2025-02-09 0:44:37	2460715.532	xkmts.20250209.006185.fits	#359
MMA	OBJECT	S250206dm	15:08:19	-69:53:00	120	1.5	11:30:10	R	2025-02-09 0:47:36	2025-02-09 0:47:34	2460715.534	xkmts.20250209.006186.fits	#359
MMA	OBJECT	S250206dm	15:09:05	-69:53:00	120	1.5	11:33:09	R	2025-02-09 0:50:48	2025-02-09 0:50:46	2460715.536	xkmts.20250209.006187.fits	#359
MMA	OBJECT	S250206dm	15:12:41	-68:00:00	120	1.48	11:36:21	R	2025-02-09 0:53:58	2025-02-09 0:53:57	2460715.538	xkmts.20250209.006188.fits	#292
MMA	OBJECT	S250206dm	15:13:23	-68:00:00	120	1.47	11:39:33	R	2025-02-09 0:57:10	2025-02-09 0:57:08	2460715.54	xkmts.20250209.006189.fits	#292
MMA	OBJECT	S250206dm	15:12:41	-67:53:00	120	1.46	11:42:42	R	2025-02-09 1:00:19	2025-02-09 1:00:17	2460715.543	xkmts.20250209.006190.fits	#292
MMA	OBJECT	S250206dm	15:13:23	-67:53:00	120	1.46	11:45:52	R	2025-02-09 1:03:29	2025-02-09 1:03:27	2460715.545	xkmts.20250209.006191.fits	#292
MMA	OBJECT	S250206dm	15:30:28	-70:00:00	120	1.51	11:49:04	R	2025-02-09 1:06:39	2025-02-09 1:06:37	2460715.547	xkmts.20250209.006192.fits	#360
MMA	OBJECT	S250206dm	15:31:15	-70:00:00	120	1.51	11:52:22	R	2025-02-09 1:09:44	2025-02-09 1:09:42	2460715.549	xkmts.20250209.006193.fits	#360
MMA	OBJECT	S250206dm	15:30:28	-69:53:00	120	1.5	11:55:20	R	2025-02-09 1:12:55	2025-02-09 1:12:53	2460715.551	xkmts.20250209.006194.fits	#360
MMA	OBJECT	S250206dm	15:31:15	-69:53:00	120	1.49	11:58:32	R	2025-02-09 1:16:07	2025-02-09 1:16:05	2460715.554	xkmts.20250209.006195.fits	#360
MMA	OBJECT	S250206dm	15:32:58	-68:00:00	120	1.47	12:01:44	R	2025-02-09 1:19:19	2025-02-09 1:19:17	2460715.556	xkmts.20250209.006196.fits	#293
MMA	OBJECT	S250206dm	15:33:40	-68:00:00	120	1.46	12:04:55	R	2025-02-09 1:22:29	2025-02-09 1:22:27	2460715.558	xkmts.20250209.006197.fits	#293
MMA	OBJECT	S250206dm	15:32:58	-67:53:00	120	1.45	12:08:07	R	2025-02-09 1:25:39	2025-02-09 1:25:37	2460715.56	xkmts.20250209.006198.fits	#293
MMA	OBJECT	S250206dm	15:33:40	-67:53:00	120	1.45	12:11:16	R	2025-02-09 1:28:48	2025-02-09 1:28:46	2460715.562	xkmts.20250209.006199.fits	#293
MMA	OBJECT	S250206dm	15:51:52	-72:00:00	120	1.53	12:14:30	R	2025-02-09 1:32:01	2025-02-09 1:32:00	2460715.565	xkmts.20250209.006200.fits	#422
MMA	OBJECT	S250206dm	15:52:44	-72:00:00	120	1.52	12:17:41	R	2025-02-09 1:35:12	2025-02-09 1:35:10	2460715.567	xkmts.20250209.006201.fits	#422
MMA	OBJECT	S250206dm	15:51:52	-71:53:00	120	1.51	12:20:52	R	2025-02-09 1:38:22	2025-02-09 1:38:20	2460715.569	xkmts.20250209.006202.fits	#422
MMA	OBJECT	S250206dm	15:52:44	-71:53:00	120	1.51	12:24:02	R	2025-02-09 1:41:32	2025-02-09 1:41:31	2460715.571	xkmts.20250209.006203.fits	#422
MMA	OBJECT	S250206dm	15:52:37	-70:00:00	120	1.48	12:27:14	R	2025-02-09 1:44:30	2025-02-09 1:44:29	2460715.573	xkmts.20250209.006204.fits	#361
MMA	OBJECT	S250206dm	15:53:24	-70:00:00	120	1.47	12:30:10	R	2025-02-09 1:47:39	2025-02-09 1:47:38	2460715.575	xkmts.20250209.006205.fits	#361
MMA	OBJECT	S250206dm	15:52:37	-69:53:00	120	1.46	12:33:21	R	2025-02-09 1:50:49	2025-02-09 1:50:48	2460715.578	xkmts.20250209.006206.fits	#361
MMA	OBJECT	S250206dm	15:53:24	-69:53:00	120	1.46	12:36:31	R	2025-02-09 1:53:58	2025-02-09 1:53:57	2460715.58	xkmts.20250209.006207.fits	#361
MMA	OBJECT	S250206dm	15:53:14	-68:00:00	120	1.43	12:39:40	R	2025-02-09 1:57:08	2025-02-09 1:57:06	2460715.582	xkmts.20250209.006208.fits	#294
MMA	OBJECT	S250206dm	15:53:57	-68:00:00	120	1.42	12:42:58	R	2025-02-09 2:00:24	2025-02-09 2:00:22	2460715.584	xkmts.20250209.006209.fits	#294
MMA	OBJECT	S250206dm	15:53:14	-67:53:00	120	1.41	12:46:07	R	2025-02-09 2:03:21	2025-02-09 2:03:19	2460715.586	xkmts.20250209.006210.fits	#294
MMA	OBJECT	S250206dm	15:53:57	-67:53:00	120	1.41	12:49:27	R	2025-02-09 2:06:40	2025-02-09 2:06:38	2460715.589	xkmts.20250209.006211.fits	#294
MMA	OBJECT	S250206dm	15:53:14	-67:53:00	120	1.4	12:53:28	R	2025-02-09 2:10:40	2025-02-09 2:10:39	2460715.591	xkmts.20250209.006212.fits	#294
MMA	OBJECT	S250206dm	15:53:57	-67:53:00	120	1.39	12:56:26	R	2025-02-09 2:13:38	2025-02-09 2:13:36	2460715.593	xkmts.20250209.006213.fits	#294
MMA	OBJECT	S250206dm	16:12:28	-66:00:00	120	1.41	12:59:24	R	2025-02-09 2:16:49	2025-02-09 2:16:47	2460715.596	xkmts.20250209.006214.fits	#222
MMA	OBJECT	S250206dm	16:13:07	-66:00:00	120	1.4	13:02:39	R	2025-02-09 2:20:03	2025-02-09 2:20:01	2460715.598	xkmts.20250209.006215.fits	#222
MMA	OBJECT	S250206dm	16:12:28	-65:53:00	120	1.39	13:05:48	R	2025-02-09 2:23:12	2025-02-09 2:23:10	2460715.6	xkmts.20250209.006216.fits	#222
MMA	OBJECT	S250206dm	16:13:07	-65:53:00	120	1.39	13:08:59	R	2025-02-09 2:26:09	2025-02-09 2:26:07	2460715.602	xkmts.20250209.006217.fits	#222
MMA	OBJECT	S250206dm	16:13:31	-68:00:00	120	1.4	13:11:54	R	2025-02-09 2:29:17	2025-02-09 2:29:15	2460715.604	xkmts.20250209.006218.fits	#295
MMA	OBJECT	S250206dm	16:14:14	-68:00:00	120	1.4	13:15:05	R	2025-02-09 2:32:26	2025-02-09 2:32:25	2460715.607	xkmts.20250209.006219.fits	#295
MMA	OBJECT	S250206dm	16:13:31	-67:53:00	120	1.39	13:18:15	R	2025-02-09 2:35:36	2025-02-09 2:35:34	2460715.609	xkmts.20250209.006220.fits	#295
MMA	OBJECT	S250206dm	16:14:14	-67:53:00	120	1.39	13:21:25	R	2025-02-09 2:38:46	2025-02-09 2:38:45	2460715.611	xkmts.20250209.006221.fits	#295
MMA	OBJECT	S250206dm	16:14:46	-70:00:00	120	1.41	13:24:35	R	2025-02-09 2:41:55	2025-02-09 2:41:54	2460715.613	xkmts.20250209.006222.fits	#362
MMA	OBJECT	S250206dm	16:15:33	-70:00:00	120	1.4	13:27:45	R	2025-02-09 2:45:04	2025-02-09 2:45:03	2460715.615	xkmts.20250209.006223.fits	#362
MMA	OBJECT	S250206dm	16:14:46	-69:53:00	120	1.4	13:30:54	R	2025-02-09 2:48:14	2025-02-09 2:48:12	2460715.618	xkmts.20250209.006224.fits	#362
MMA	OBJECT	S250206dm	16:15:33	-69:53:00	120	1.39	13:34:05	R	2025-02-09 2:51:11	2025-02-09 2:51:09	2460715.62	xkmts.20250209.006225.fits	#362
MMA	OBJECT	S250206dm	16:16:16	-72:00:00	120	1.42	13:37:02	R	2025-02-09 2:54:20	2025-02-09 2:54:18	2460715.622	xkmts.20250209.006226.fits	#423
MMA	OBJECT	S250206dm	16:17:08	-72:00:00	120	1.42	13:40:11	R	2025-02-09 2:57:16	2025-02-09 2:57:15	2460715.624	xkmts.20250209.006227.fits	#423
MMA	OBJECT	S250206dm	16:16:16	-71:53:00	120	1.41	13:43:09	R	2025-02-09 3:00:26	2025-02-09 3:00:25	2460715.626	xkmts.20250209.006228.fits	#423
MMA	OBJECT	S250206dm	16:17:08	-71:53:00	120	1.41	13:46:19	R	2025-02-09 3:03:35	2025-02-09 3:03:34	2460715.628	xkmts.20250209.006229.fits	#423
MMA	OBJECT	S250206dm	16:31:10	-66:00:00	120	1.34	13:49:28	R	2025-02-09 3:06:44	2025-02-09 3:06:42	2460715.63	xkmts.20250209.006230.fits	#223
MMA	OBJECT	S250206dm	16:31:50	-66:00:00	120	1.34	13:52:38	R	2025-02-09 3:09:53	2025-02-09 3:09:51	2460715.633	xkmts.20250209.006231.fits	#223
MMA	OBJECT	S250206dm	16:31:10	-65:53:00	120	1.33	13:55:50	R	2025-02-09 3:12:53	2025-02-09 3:12:51	2460715.635	xkmts.20250209.006232.fits	#223
MMA	OBJECT	S250206dm	16:31:50	-65:53:00	120	1.33	13:58:48	R	2025-02-09 3:16:02	2025-02-09 3:16:00	2460715.637	xkmts.20250209.006233.fits	#223
"""
saao_250208 = pd.read_csv(StringIO(saao_250208), sep="\t")
saao_250208 = Table.from_pandas(saao_250208)

ctio_250209 = """
PROJID	IMGTYPE	OBJECT	RA	DEC	EXP	SECZ	ST	FILT	DATE-OBS	UT	MIDJD	FILENAME	Tile
MMA	OBJECT	S250206dm	15:08:18	-70:00:00	120	1.73	10:37:13	I	2025-02-09 6:00:15	2025-02-09 6:00:15	2460715.751	xkmtc.20250209.062106.fits	#359
MMA	OBJECT	S250206dm	15:09:05	-69:59:59	120	1.72	10:40:14	I	2025-02-09 6:03:16	2025-02-09 6:03:16	2460715.753	xkmtc.20250209.062107.fits	#359
MMA	OBJECT	S250206dm	15:08:18	-69:52:59	120	1.71	10:43:11	I	2025-02-09 6:06:25	2025-02-09 6:06:25	2460715.755	xkmtc.20250209.062108.fits	#359
MMA	OBJECT	S250206dm	15:09:05	-69:52:59	120	1.7	10:46:27	I	2025-02-09 6:09:28	2025-02-09 6:09:28	2460715.757	xkmtc.20250209.062109.fits	#359
MMA	OBJECT	S250206dm	15:12:40	-67:59:59	120	1.69	10:49:24	I	2025-02-09 6:12:24	2025-02-09 6:12:24	2460715.759	xkmtc.20250209.062110.fits	#292
MMA	OBJECT	S250206dm	15:13:23	-67:59:59	120	1.68	10:52:21	I	2025-02-09 6:15:21	2025-02-09 6:15:21	2460715.761	xkmtc.20250209.062111.fits	#292
MMA	OBJECT	S250206dm	15:12:40	-67:52:59	120	1.67	10:55:19	I	2025-02-09 6:18:18	2025-02-09 6:18:18	2460715.763	xkmtc.20250209.062112.fits	#292
MMA	OBJECT	S250206dm	15:13:23	-67:52:59	120	1.66	10:58:19	I	2025-02-09 6:21:19	2025-02-09 6:21:19	2460715.765	xkmtc.20250209.062113.fits	#292
MMA	OBJECT	S250206dm	15:30:28	-69:59:59	120	1.72	11:03:56	I	2025-02-09 6:26:53	2025-02-09 6:26:54	2460715.769	xkmtc.20250209.062114.fits	#360
MMA	OBJECT	S250206dm	15:31:14	-69:59:59	120	1.71	11:06:51	I	2025-02-09 6:30:01	2025-02-09 6:30:02	2460715.772	xkmtc.20250209.062115.fits	#360
MMA	OBJECT	S250206dm	15:30:28	-69:52:59	120	1.69	11:10:08	I	2025-02-09 6:33:05	2025-02-09 6:33:05	2460715.774	xkmtc.20250209.062116.fits	#360
MMA	OBJECT	S250206dm	15:31:14	-69:52:59	120	1.69	11:13:18	I	2025-02-09 6:36:14	2025-02-09 6:36:14	2460715.776	xkmtc.20250209.062117.fits	#360
MMA	OBJECT	S250206dm	15:32:57	-67:59:59	120	1.67	11:16:15	I	2025-02-09 6:39:11	2025-02-09 6:39:11	2460715.778	xkmtc.20250209.062118.fits	#293
MMA	OBJECT	S250206dm	15:33:40	-67:59:59	120	1.66	11:19:14	I	2025-02-09 6:42:10	2025-02-09 6:42:10	2460715.78	xkmtc.20250209.062119.fits	#293
MMA	OBJECT	S250206dm	15:32:57	-67:52:59	120	1.65	11:22:13	I	2025-02-09 6:45:08	2025-02-09 6:45:08	2460715.782	xkmtc.20250209.062120.fits	#293
MMA	OBJECT	S250206dm	15:33:40	-67:52:59	120	1.64	11:25:11	I	2025-02-09 6:48:05	2025-02-09 6:48:05	2460715.784	xkmtc.20250209.062121.fits	#293
MMA	OBJECT	S250206dm	15:51:52	-71:59:59	120	1.72	11:28:08	I	2025-02-09 6:51:02	2025-02-09 6:51:02	2460715.786	xkmtc.20250209.062122.fits	#422
MMA	OBJECT	S250206dm	15:52:43	-71:59:59	120	1.72	11:31:04	I	2025-02-09 6:53:57	2025-02-09 6:53:57	2460715.788	xkmtc.20250209.062123.fits	#422
MMA	OBJECT	S250206dm	15:51:52	-71:52:59	120	1.7	11:34:02	I	2025-02-09 6:57:08	2025-02-09 6:57:08	2460715.79	xkmtc.20250209.062124.fits	#422
MMA	OBJECT	S250206dm	15:52:43	-71:53:00	120	1.7	11:37:11	I	2025-02-09 7:00:04	2025-02-09 7:00:04	2460715.792	xkmtc.20250209.062125.fits	#422
MMA	OBJECT	S250206dm	15:52:37	-69:59:59	120	1.67	11:40:09	I	2025-02-09 7:03:01	2025-02-09 7:03:02	2460715.794	xkmtc.20250209.062126.fits	#361
MMA	OBJECT	S250206dm	15:53:24	-69:59:59	120	1.66	11:43:08	I	2025-02-09 7:06:12	2025-02-09 7:06:12	2460715.797	xkmtc.20250209.062127.fits	#361
MMA	OBJECT	S250206dm	15:52:37	-69:52:59	120	1.65	11:46:19	I	2025-02-09 7:09:23	2025-02-09 7:09:23	2460715.799	xkmtc.20250209.062128.fits	#361
MMA	OBJECT	S250206dm	15:53:24	-69:52:59	120	1.64	11:49:36	I	2025-02-09 7:12:26	2025-02-09 7:12:27	2460715.801	xkmtc.20250209.062129.fits	#361
MMA	OBJECT	S250206dm	15:53:14	-67:59:59	120	1.61	11:52:34	I	2025-02-09 7:15:24	2025-02-09 7:15:24	2460715.803	xkmtc.20250209.062130.fits	#294
MMA	OBJECT	S250206dm	15:53:57	-67:59:59	120	1.61	11:55:31	I	2025-02-09 7:18:21	2025-02-09 7:18:21	2460715.805	xkmtc.20250209.062131.fits	#294
MMA	OBJECT	S250206dm	15:53:14	-67:52:59	120	1.59	11:58:27	I	2025-02-09 7:21:16	2025-02-09 7:21:16	2460715.807	xkmtc.20250209.062132.fits	#294
MMA	OBJECT	S250206dm	15:53:57	-67:52:59	120	1.59	12:01:24	I	2025-02-09 7:24:12	2025-02-09 7:24:12	2460715.809	xkmtc.20250209.062133.fits	#294
MMA	OBJECT	S250206dm	16:12:28	-65:59:59	120	1.62	12:04:20	I	2025-02-09 7:27:08	2025-02-09 7:27:09	2460715.811	xkmtc.20250209.062134.fits	#222
MMA	OBJECT	S250206dm	16:13:07	-65:59:59	120	1.61	12:07:19	I	2025-02-09 7:30:19	2025-02-09 7:30:20	2460715.813	xkmtc.20250209.062135.fits	#222
MMA	OBJECT	S250206dm	16:12:28	-65:52:59	120	1.6	12:10:31	I	2025-02-09 7:33:18	2025-02-09 7:33:18	2460715.815	xkmtc.20250209.062136.fits	#222
MMA	OBJECT	S250206dm	16:13:07	-65:52:59	120	1.59	12:13:28	I	2025-02-09 7:36:14	2025-02-09 7:36:15	2460715.818	xkmtc.20250209.062137.fits	#222
MMA	OBJECT	S250206dm	16:13:31	-67:59:59	120	1.6	12:16:26	I	2025-02-09 7:39:13	2025-02-09 7:39:14	2460715.82	xkmtc.20250209.062138.fits	#295
MMA	OBJECT	S250206dm	16:14:14	-67:59:59	120	1.59	12:19:28	I	2025-02-09 7:42:13	2025-02-09 7:42:14	2460715.822	xkmtc.20250209.062139.fits	#295
MMA	OBJECT	S250206dm	16:13:31	-67:59:59	120	1.58	12:22:31	I	2025-02-09 7:45:16	2025-02-09 7:45:17	2460715.824	xkmtc.20250209.062140.fits	#295
MMA	OBJECT	S250206dm	16:14:14	-67:59:59	120	1.58	12:25:37	I	2025-02-09 7:48:22	2025-02-09 7:48:22	2460715.826	xkmtc.20250209.062141.fits	#295
MMA	OBJECT	S250206dm	16:13:31	-67:52:59	120	1.56	12:28:35	I	2025-02-09 7:51:32	2025-02-09 7:51:32	2460715.828	xkmtc.20250209.062142.fits	#295
MMA	OBJECT	S250206dm	16:14:14	-67:52:59	120	1.56	12:31:54	I	2025-02-09 7:54:38	2025-02-09 7:54:38	2460715.83	xkmtc.20250209.062143.fits	#295
MMA	OBJECT	S250206dm	16:14:46	-69:59:59	120	1.57	12:34:52	I	2025-02-09 7:57:34	2025-02-09 7:57:35	2460715.832	xkmtc.20250209.062144.fits	#362
MMA	OBJECT	S250206dm	16:15:33	-69:59:59	120	1.57	12:37:47	I	2025-02-09 8:00:30	2025-02-09 8:00:30	2460715.834	xkmtc.20250209.062145.fits	#362
MMA	OBJECT	S250206dm	16:14:46	-69:52:59	120	1.56	12:40:44	I	2025-02-09 8:03:25	2025-02-09 8:03:26	2460715.836	xkmtc.20250209.062146.fits	#362
MMA	OBJECT	S250206dm	16:15:33	-69:52:59	120	1.55	12:43:39	I	2025-02-09 8:06:21	2025-02-09 8:06:21	2460715.838	xkmtc.20250209.062147.fits	#362
MMA	OBJECT	S250206dm	16:16:16	-71:59:59	120	1.58	12:46:37	I	2025-02-09 8:09:31	2025-02-09 8:09:31	2460715.841	xkmtc.20250209.062148.fits	#423
MMA	OBJECT	S250206dm	16:17:08	-71:59:59	120	1.57	12:49:48	I	2025-02-09 8:12:40	2025-02-09 8:12:41	2460715.843	xkmtc.20250209.062149.fits	#423
MMA	OBJECT	S250206dm	16:16:16	-71:52:59	120	1.56	12:53:06	I	2025-02-09 8:15:45	2025-02-09 8:15:46	2460715.845	xkmtc.20250209.062150.fits	#423
MMA	OBJECT	S250206dm	16:17:08	-71:52:59	120	1.55	12:56:06	I	2025-02-09 8:18:46	2025-02-09 8:18:46	2460715.847	xkmtc.20250209.062151.fits	#423
MMA	OBJECT	S250206dm	16:31:10	-65:59:59	120	1.51	12:59:08	I	2025-02-09 8:21:47	2025-02-09 8:21:47	2460715.849	xkmtc.20250209.062152.fits	#223
MMA	OBJECT	S250206dm	16:31:49	-65:59:59	120	1.5	13:02:05	I	2025-02-09 8:24:44	2025-02-09 8:24:44	2460715.851	xkmtc.20250209.062153.fits	#223
MMA	OBJECT	S250206dm	16:31:10	-65:59:59	120	1.49	13:05:02	I	2025-02-09 8:27:53	2025-02-09 8:27:53	2460715.853	xkmtc.20250209.062154.fits	#223
MMA	OBJECT	S250206dm	16:31:10	-65:52:59	120	1.48	13:09:13	I	2025-02-09 8:31:50	2025-02-09 8:31:50	2460715.856	xkmtc.20250209.062155.fits	#223
MMA	OBJECT	S250206dm	16:31:49	-65:52:59	120	1.47	13:12:12	I	2025-02-09 8:34:48	2025-02-09 8:34:49	2460715.858	xkmtc.20250209.062156.fits	#223
MMA	OBJECT	S250206dm	16:33:48	-67:59:59	120	1.5	13:15:08	I	2025-02-09 8:37:45	2025-02-09 8:37:45	2460715.86	xkmtc.20250209.062157.fits	#296
MMA	OBJECT	S250206dm	16:34:31	-67:59:59	120	1.49	13:18:07	I	2025-02-09 8:40:55	2025-02-09 8:40:55	2460715.862	xkmtc.20250209.062158.fits	#296
MMA	OBJECT	S250206dm	16:33:48	-67:52:59	120	1.48	13:21:16	I	2025-02-09 8:43:51	2025-02-09 8:43:52	2460715.864	xkmtc.20250209.062159.fits	#296
MMA	OBJECT	S250206dm	16:34:31	-67:52:59	120	1.47	13:24:13	I	2025-02-09 8:46:48	2025-02-09 8:46:48	2460715.867	xkmtc.20250209.062160.fits	#296
MMA	OBJECT	S250206dm	16:36:55	-69:59:59	120	1.5	13:27:11	I	2025-02-09 8:49:46	2025-02-09 8:49:46	2460715.869	xkmtc.20250209.062161.fits	#363
MMA	OBJECT	S250206dm	16:37:42	-69:59:59	120	1.5	13:30:09	I	2025-02-09 8:52:42	2025-02-09 8:52:42	2460715.871	xkmtc.20250209.062162.fits	#363
"""
ctio_250209 = pd.read_csv(StringIO(ctio_250209), sep="\t")
ctio_250209 = Table.from_pandas(ctio_250209)

ctio_250210 = """
PROJID	IMGTYPE	OBJECT	RA	DEC	EXP	SECZ	ST	FILT	DATE-OBS	UT	MIDJD	FILENAME	Tile
MMA	OBJECT	S250206dm	15:08:18	-69:59:59	120	1.7	10:47:28	I	2025-02-10 6:06:32	2025-02-10 6:06:32	2460716.755	xkmtc.20250210.062337.fits	#359
MMA	OBJECT	S250206dm	15:09:05	-69:59:59	120	1.69	10:50:25	I	2025-02-10 6:09:30	2025-02-10 6:09:30	2460716.757	xkmtc.20250210.062338.fits	#359
MMA	OBJECT	S250206dm	15:08:18	-69:52:59	120	1.68	10:53:23	I	2025-02-10 6:12:39	2025-02-10 6:12:39	2460716.759	xkmtc.20250210.062339.fits	#359
MMA	OBJECT	S250206dm	15:09:05	-69:52:59	120	1.67	10:56:40	I	2025-02-10 6:15:44	2025-02-10 6:15:44	2460716.762	xkmtc.20250210.062340.fits	#359
MMA	OBJECT	S250206dm	15:12:40	-67:59:59	120	1.65	10:59:37	I	2025-02-10 6:18:39	2025-02-10 6:18:40	2460716.764	xkmtc.20250210.062341.fits	#292
MMA	OBJECT	S250206dm	15:13:23	-67:59:59	120	1.65	11:02:34	I	2025-02-10 6:21:37	2025-02-10 6:21:38	2460716.766	xkmtc.20250210.062342.fits	#292
MMA	OBJECT	S250206dm	15:12:40	-67:52:59	120	1.63	11:05:39	I	2025-02-10 6:24:41	2025-02-10 6:24:41	2460716.768	xkmtc.20250210.062343.fits	#292
MMA	OBJECT	S250206dm	15:13:23	-67:52:59	120	1.63	11:08:35	I	2025-02-10 6:27:49	2025-02-10 6:27:49	2460716.77	xkmtc.20250210.062344.fits	#292
MMA	OBJECT	S250206dm	15:30:28	-69:59:59	120	1.69	11:11:45	I	2025-02-10 6:30:59	2025-02-10 6:30:59	2460716.772	xkmtc.20250210.062345.fits	#360
MMA	OBJECT	S250206dm	15:31:14	-69:59:59	120	1.68	11:15:04	I	2025-02-10 6:34:04	2025-02-10 6:34:04	2460716.774	xkmtc.20250210.062346.fits	#360
MMA	OBJECT	S250206dm	15:30:28	-69:52:59	120	1.67	11:18:04	I	2025-02-10 6:37:16	2025-02-10 6:37:16	2460716.777	xkmtc.20250210.062347.fits	#360
MMA	OBJECT	S250206dm	15:31:14	-69:52:59	120	1.66	11:21:23	I	2025-02-10 6:40:21	2025-02-10 6:40:22	2460716.779	xkmtc.20250210.062348.fits	#360
MMA	OBJECT	S250206dm	15:32:57	-67:59:59	120	1.64	11:24:20	I	2025-02-10 6:43:19	2025-02-10 6:43:19	2460716.781	xkmtc.20250210.062349.fits	#293
MMA	OBJECT	S250206dm	15:33:40	-67:59:59	120	1.63	11:27:17	I	2025-02-10 6:46:15	2025-02-10 6:46:15	2460716.783	xkmtc.20250210.062350.fits	#293
MMA	OBJECT	S250206dm	15:32:57	-67:52:59	120	1.62	11:30:15	I	2025-02-10 6:49:26	2025-02-10 6:49:26	2460716.785	xkmtc.20250210.062351.fits	#293
MMA	OBJECT	S250206dm	15:33:40	-67:52:59	120	1.61	11:33:27	I	2025-02-10 6:52:37	2025-02-10 6:52:38	2460716.787	xkmtc.20250210.062352.fits	#293
MMA	OBJECT	S250206dm	15:51:52	-71:59:59	120	1.7	11:36:36	I	2025-02-10 6:55:33	2025-02-10 6:55:33	2460716.789	xkmtc.20250210.062353.fits	#422
MMA	OBJECT	S250206dm	15:52:43	-71:59:59	120	1.69	11:39:33	I	2025-02-10 6:58:30	2025-02-10 6:58:30	2460716.791	xkmtc.20250210.062354.fits	#422
MMA	OBJECT	S250206dm	15:51:52	-71:52:59	120	1.68	11:42:34	I	2025-02-10 7:01:43	2025-02-10 7:01:43	2460716.794	xkmtc.20250210.062355.fits	#422
MMA	OBJECT	S250206dm	15:52:43	-71:52:59	120	1.67	11:45:43	I	2025-02-10 7:04:52	2025-02-10 7:04:52	2460716.796	xkmtc.20250210.062356.fits	#422
MMA	OBJECT	S250206dm	15:52:37	-69:59:59	120	1.64	11:48:54	I	2025-02-10 7:08:02	2025-02-10 7:08:02	2460716.798	xkmtc.20250210.062357.fits	#361
MMA	OBJECT	S250206dm	15:53:24	-69:59:59	120	1.64	11:52:04	I	2025-02-10 7:10:59	2025-02-10 7:10:59	2460716.8	xkmtc.20250210.062358.fits	#361
MMA	OBJECT	S250206dm	15:52:37	-69:52:59	120	1.62	11:55:02	I	2025-02-10 7:13:56	2025-02-10 7:13:56	2460716.802	xkmtc.20250210.062359.fits	#361
MMA	OBJECT	S250206dm	15:53:24	-69:52:59	120	1.62	11:58:02	I	2025-02-10 7:17:08	2025-02-10 7:17:08	2460716.804	xkmtc.20250210.062360.fits	#361
MMA	OBJECT	S250206dm	15:53:14	-67:59:59	120	1.59	12:01:21	I	2025-02-10 7:20:13	2025-02-10 7:20:13	2460716.806	xkmtc.20250210.062361.fits	#294
MMA	OBJECT	S250206dm	15:53:24	-69:52:59	120	1.6	12:04:18	I	2025-02-10 7:23:10	2025-02-10 7:23:10	2460716.808	xkmtc.20250210.062362.fits	#361
MMA	OBJECT	S250206dm	15:53:57	-67:59:59	120	1.57	12:07:22	I	2025-02-10 7:26:14	2025-02-10 7:26:14	2460716.811	xkmtc.20250210.062363.fits	#294
MMA	OBJECT	S250206dm	15:53:14	-67:52:59	120	1.56	12:10:21	I	2025-02-10 7:29:12	2025-02-10 7:29:12	2460716.813	xkmtc.20250210.062364.fits	#294
MMA	OBJECT	S250206dm	15:53:57	-67:52:59	120	1.55	12:13:19	I	2025-02-10 7:32:09	2025-02-10 7:32:09	2460716.815	xkmtc.20250210.062365.fits	#294
MMA	OBJECT	S250206dm	15:53:14	-67:52:59	120	1.54	12:16:25	I	2025-02-10 7:35:16	2025-02-10 7:35:16	2460716.817	xkmtc.20250210.062366.fits	#294
MMA	OBJECT	S250206dm	16:40:41	-71:59:59	120	1.72	12:19:23	I	2025-02-10 7:38:25	2025-02-10 7:38:25	2460716.819	xkmtc.20250210.062367.fits	#424
MMA	OBJECT	S250206dm	16:41:32	-71:59:59	120	1.71	12:22:35	I	2025-02-10 7:41:37	2025-02-10 7:41:37	2460716.821	xkmtc.20250210.062368.fits	#424
MMA	OBJECT	S250206dm	16:40:41	-71:59:59	120	1.7	12:25:45	I	2025-02-10 7:44:34	2025-02-10 7:44:34	2460716.823	xkmtc.20250210.062369.fits	#424
MMA	OBJECT	S250206dm	16:40:41	-71:52:59	120	1.69	12:28:51	I	2025-02-10 7:47:39	2025-02-10 7:47:39	2460716.825	xkmtc.20250210.062370.fits	#424
MMA	OBJECT	S250206dm	16:41:32	-71:52:59	120	1.68	12:31:56	I	2025-02-10 7:50:44	2025-02-10 7:50:44	2460716.828	xkmtc.20250210.062371.fits	#424
MMA	OBJECT	S250206dm	16:54:05	-67:59:59	120	1.68	12:35:00	I	2025-02-10 7:53:59	2025-02-10 7:53:59	2460716.83	xkmtc.20250210.062372.fits	#297
MMA	OBJECT	S250206dm	16:54:48	-67:59:59	120	1.67	12:38:15	I	2025-02-10 7:57:01	2025-02-10 7:57:02	2460716.832	xkmtc.20250210.062373.fits	#297
MMA	OBJECT	S250206dm	16:54:05	-67:52:59	120	1.65	12:41:22	I	2025-02-10 8:00:09	2025-02-10 8:00:09	2460716.834	xkmtc.20250210.062374.fits	#297
MMA	OBJECT	S250206dm	16:54:48	-67:52:59	120	1.64	12:44:28	I	2025-02-10 8:03:27	2025-02-10 8:03:27	2460716.836	xkmtc.20250210.062375.fits	#297
MMA	OBJECT	S250206dm	16:59:04	-69:59:59	120	1.67	12:47:46	I	2025-02-10 8:06:31	2025-02-10 8:06:31	2460716.839	xkmtc.20250210.062376.fits	#364
MMA	OBJECT	S250206dm	16:59:51	-69:59:59	120	1.66	12:50:44	I	2025-02-10 8:09:29	2025-02-10 8:09:29	2460716.841	xkmtc.20250210.062377.fits	#364
MMA	OBJECT	S250206dm	16:59:04	-69:52:59	120	1.65	12:53:42	I	2025-02-10 8:12:26	2025-02-10 8:12:26	2460716.843	xkmtc.20250210.062378.fits	#364
MMA	OBJECT	S250206dm	16:59:51	-69:52:59	120	1.64	12:56:39	I	2025-02-10 8:15:22	2025-02-10 8:15:22	2460716.845	xkmtc.20250210.062379.fits	#364
MMA	OBJECT	S250206dm	17:05:05	-71:59:59	120	1.67	12:59:47	I	2025-02-10 8:18:31	2025-02-10 8:18:31	2460716.847	xkmtc.20250210.062380.fits	#425
MMA	OBJECT	S250206dm	17:05:57	-71:59:59	120	1.66	13:02:51	I	2025-02-10 8:21:34	2025-02-10 8:21:34	2460716.849	xkmtc.20250210.062381.fits	#425
MMA	OBJECT	S250206dm	17:05:05	-71:52:59	120	1.65	13:05:54	I	2025-02-10 8:24:48	2025-02-10 8:24:48	2460716.851	xkmtc.20250210.062382.fits	#425
MMA	OBJECT	S250206dm	17:05:57	-71:52:59	120	1.64	13:09:04	I	2025-02-10 8:27:45	2025-02-10 8:27:46	2460716.853	xkmtc.20250210.062383.fits	#425
MMA	OBJECT	S250206dm	17:14:22	-67:59:59	120	1.62	13:12:11	I	2025-02-10 8:31:04	2025-02-10 8:31:04	2460716.856	xkmtc.20250210.062384.fits	#298
MMA	OBJECT	S250206dm	17:05:57	-71:52:59	120	1.63	13:15:21	I	2025-02-10 8:34:02	2025-02-10 8:34:02	2460716.858	xkmtc.20250210.062385.fits	#425
MMA	OBJECT	S250206dm	17:15:05	-67:59:59	120	1.6	13:18:20	I	2025-02-10 8:37:12	2025-02-10 8:37:13	2460716.86	xkmtc.20250210.062386.fits	#298
MMA	OBJECT	S250206dm	17:14:22	-67:52:59	120	1.59	13:21:29	I	2025-02-10 8:40:09	2025-02-10 8:40:09	2460716.862	xkmtc.20250210.062387.fits	#298
MMA	OBJECT	S250206dm	17:15:05	-67:52:59	120	1.58	13:24:26	I	2025-02-10 8:43:18	2025-02-10 8:43:18	2460716.864	xkmtc.20250210.062388.fits	#298
MMA	OBJECT	S250206dm	17:21:14	-69:59:59	120	1.61	13:27:43	I	2025-02-10 8:46:21	2025-02-10 8:46:22	2460716.866	xkmtc.20250210.062389.fits	#365
MMA	OBJECT	S250206dm	17:22:00	-69:59:59	120	1.61	13:30:50	I	2025-02-10 8:49:28	2025-02-10 8:49:29	2460716.868	xkmtc.20250210.062390.fits	#365
MMA	OBJECT	S250206dm	17:21:14	-69:52:59	120	1.59	13:33:47	I	2025-02-10 8:52:25	2025-02-10 8:52:25	2460716.87	xkmtc.20250210.062391.fits	#365
MMA	OBJECT	S250206dm	17:22:01	-69:52:59	120	1.59	13:36:46	I	2025-02-10 8:55:23	2025-02-10 8:55:23	2460716.872	xkmtc.20250210.062392.fits	#365
"""
ctio_250210 = pd.read_csv(StringIO(ctio_250210), sep="\t")
ctio_250210 = Table.from_pandas(ctio_250210)

# %%
for obs in obslog:
    if obs['R1'] != 0:
        field = obs['FIELD']
        ra = obs['RA']
        dec = obs['DEC']
        exptime = obs['R1']
        dateobs = ctio_250207[ctio_250207['Tile']==f"#{field}"]['DATE-OBS'][0]
        print(f"{ra} {dec} CTIO R {exptime} {dateobs.replace(' ', 'T')}")

for obs in obslog:
    if obs['R2'] != 0:
        field = obs['FIELD']
        ra = obs['RA']
        dec = obs['DEC']
        exptime = obs['R2']
        dateobs = saao_250207[saao_250207['Tile']==f"#{field}"]['DATE-OBS'][0]
        print(f"{ra} {dec} SAAO R {exptime} {dateobs.replace(' ', 'T')}")

for obs in obslog:
    if obs['I1'] != 0:
        field = obs['FIELD']
        ra = obs['RA']
        dec = obs['DEC']
        exptime = obs['I1']
        dateobs = ctio_250208[ctio_250208['Tile']==f"#{field}"]['DATE-OBS'][0]
        print(f"{ra} {dec} CTIO I {exptime} {dateobs.replace(' ', 'T')}")

for obs in obslog:
    if obs['R3'] != 0:
        field = obs['FIELD']
        ra = obs['RA']
        dec = obs['DEC']
        exptime = obs['R3']
        dateobs = saao_250208[saao_250208['Tile']==f"#{field}"]['DATE-OBS'][0]
        print(f"{ra} {dec} SAAO R {exptime} {dateobs.replace(' ', 'T')}")

for obs in obslog:
    if obs['I2'] != 0:
        field = obs['FIELD']
        ra = obs['RA']
        dec = obs['DEC']
        exptime = obs['I2']
        dateobs = ctio_250209[ctio_250209['Tile']==f"#{field}"]['DATE-OBS'][0]
        print(f"{ra} {dec} CTIO I {exptime} {dateobs.replace(' ', 'T')}")

for obs in obslog:
    if obs['I3'] != 0:
        field = obs['FIELD']
        ra = obs['RA']
        dec = obs['DEC']
        exptime = obs['I3']
        dateobs = ctio_250210[ctio_250210['Tile']==f"#{field}"]['DATE-OBS'][0]
        print(f"{ra} {dec} CTIO I {exptime} {dateobs.replace(' ', 'T')}")

#%% Visualize the skymap
import urllib.request
from io import BytesIO
import gzip
import astropy.io.fits as fits

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

# ---------------------------
# Parameters and Data Loading
# ---------------------------
event = "S250206dm"
if False:
    volurl = f'https://gracedb.ligo.org/api/superevents/{event}/files/Bilby.fits.gz' # after update
    newlog  = obslog[obslog['UPDATE']==1]
    newlog  = newlog[newlog['R3']+newlog['I2']+newlog['I3']!=0]
else:
    volurl = f'https://gracedb.ligo.org/api/superevents/{event}/files/bayestar.fits.gz' # before update
    newlog  = obslog[obslog['UPDATE']==0]
    newlog  = newlog[newlog['R1']+newlog['R2']+newlog['I1']!=0]

# Download the gzipped FITS file.
with urllib.request.urlopen(volurl) as response:
    data_bytes = response.read()

# Decompress the downloaded data.
uncompressed_data = gzip.decompress(data_bytes)

# Open the decompressed FITS data.
hdulist = fits.open(BytesIO(uncompressed_data))
hp_map = hp.read_map(hdulist)

# ---------------------------
# Observation Log Coordinates
# ---------------------------
# For demonstration, create a dummy observation log table with RA/DEC in sexagesimal strings.
# Convert RA and DEC to decimal degrees.
coords = SkyCoord(newlog['RA'], newlog['DEC'], unit=(u.hourangle, u.deg))
ra_deg  = coords.ra.deg
dec_deg = coords.dec.deg

# ---------------------------
# Plotting with Healpy Mollweide View
# ---------------------------
# Display the HEALPix probability map in a true Mollweide projection.
hp.mollview(hp_map,
            title=event,
            unit='Probability',
            cmap='cylon',
            min=0, max=1e-4)

# Overlay the observation point centers.
# hp.projscatter(ra_deg, dec_deg,
#                lonlat=True,   # specify that the input coordinates are in degrees
#                marker='s',    # square marker
#                s=10,          # marker size in points^2
#                color='dodgerblue',
#             #    edgecolor='None',
#                label='KMTNet')

# Optionally annotate each point.
for ra in [-180, -90, 90, 180]:
    hp.projtext(ra, 0, f"{(ra+360)%360}°", lonlat=True,
                color='k', fontsize=10, ha='center', va='bottom')
for dec in [-90, -45, 0, 45, 90]:
    # hp.projtext places text at the given RA/DEC (when lonlat=True, units are degrees)
    hp.projtext(0, dec, f"{dec}°", lonlat=True,
                color='k', fontsize=10, ha='left', va='center')

# ---------------------------
# Draw Each Pointing as a 2x2 deg² Square
# ---------------------------
# For each pointing, define the corners of a square that covers 2°×2°.
# (Each square is centered on the pointing; hence we subtract and add 1 degree.)
for ra_val, dec_val in zip(ra_deg, dec_deg):
    ra_corners = [ra_val - 1, ra_val + 1, ra_val + 1, ra_val - 1, ra_val - 1]
    dec_corners = [dec_val - 1, dec_val - 1, dec_val + 1, dec_val + 1, dec_val - 1]
    hp.projplot(ra_corners, dec_corners, lonlat=True, color='k', lw=1)
hp.projscatter([], [], marker='s', s=10, edgecolor='k', facecolors='none', label='KMTNet 2°×2° FOV')
# ---------------------------
# Add Gridlines as Axes/Ticks
# ---------------------------
hp.graticule(dpar=15, dmer=30, color='gray')

plt.legend()
plt.show()

#%%
"""
print('scKS4_'+date+'_'+site+'.cat')
f = open('scKS4_'+date+'_'+site+'.cat',"w")
g = open('KS4_'+date+'_'+site+'_vis.cat',"w")

f.write('# KMTNet Observation script\n')
#	f.write('#   LABEL         RA          DEC     COPT  IMGTYP   OBJECT_NAME FILTER EXPTIME      UTOBS          UTTOL #Comments\n')
f.write('#  ProjectID     LABEL         RA          DEC     COPT  IMGTYP   OBJECT_NAME FILTER EXPTIME      UTOBS          UTTOL #Comments\n')
f.write('#------------ ----------- ----------- ---- -------- ------------ ------ ------- ------------------- ----- ---------\n')
f.write('\n')

for i in range(len(craarr)):
    fieldnumber = cfield1[i]
    tdra = craarr[i]
    tddec = cdecarr[i]

    if cdith[i] == 1: tdra = craarr[i] + ((4/60.) / cos(cdecarr[i] * 3.141592 / 180.))
    if cdith[i] == 2: tddec = cdecarr[i] + 7/60.
    if cdith[i] == 3: tdra = craarr[i] + ((4/60.) / cos(cdecarr[i] * 3.141592 / 180.))
    if cdith[i] == 3: tddec = cdecarr[i] + 7/60.
    '''
    if (cmaxdith[i] == 4) & (cdith[i] == 1): tdra = craarr[i] + ((4/60.) / cos(cdecarr[i] * 3.141592 / 180.))
    if (cmaxdith[i] == 4) & (cdith[i] == 2): tddec = cdecarr[i] + 7/60.
    if (cmaxdith[i] == 4) & (cdith[i] == 3): tdra = craarr[i] + ((4/60.) / cos(cdecarr[i] * 3.141592 / 180.))
    if (cmaxdith[i] == 4) & (cdith[i] == 3): tddec = cdecarr[i] + 7/60.

    if (cmaxdith[i] == 3) & (cdith[i] == 1): tdra = craarr[i] + ((4/60.) / cos(cdecarr[i] * 3.141592 / 180.))
    if (cmaxdith[i] == 3) & (cdith[i] == 1): tddec = cdecarr[i] + 7/60.
    if (cmaxdith[i] == 3) & (cdith[i] == 2): tdra = craarr[i] - ((4/60.) / cos(cdecarr[i] * 3.141592 / 180.))
    if (cmaxdith[i] == 3) & (cdith[i] == 2) & (craarr[i] == 0): tdra = craarr[i] + 360. - ((4/60.) / cos(cdecarr[i] * 3.141592 / 180.))
    if (cmaxdith[i] == 3) & (cdith[i] == 2): tddec = cdecarr[i] - 7/60.
    '''
    tra = coord.Angle(tdra, unit=u.deg)
    tra = tra.to_string(unit=u.hour, sep=':',precision=1, pad=True)
    tdec = coord.Angle(tddec, unit=u.deg)
    tdec = tdec.to_string(unit=u.degree, sep=':',precision=0, pad=True)

    f.write('{0:<12}{1:<15}{2:<13}{3:<13}{4:<14}{5:<15}{6:<6}{7:<29}{8:}\n'.format('KS4',fieldnumber+'('+cband[i]+')',tra,tdec,'0   OBJECT',fieldnumber,cband[i],'120                   -     0',' #'+str(i+1)))
#		f.write('{0:<15}{1:<13}{2:<13}{3:<14}{4:<15}{5:<6}{6:<29}{7:}\n'.format(fieldnumber+'('+cband[i]+')',tra,tdec,'0   OBJECT',fieldnumber,cband[i],'120                   -     0',' #'+str(i+1)))
#		f.write('{0:a-15} {1:a-13} {2:a-13} {3:a} {4:a13} {5:a7} {6:a} {7:a}\n'.format(fieldnumber+'('+cband[i]+')',tra,tdec,'0   OBJECT',fieldnumber,cband[i],'     120                   -     0',' #'+str(i+1)))
#		printf,1,fieldnumber+'('+band[i]+')',tra,tdec,'0   OBJECT',fieldnumber,band[i],'     120                   -     0',' #'+strcompress(string(i+1),/remove_all),format='(a-15,a-13,a-13,a,a13,a7,a,a)'
#		g.write('{0:} {1:} {2:}\n'.format(fieldnumber,tdra,tddec))
    if (i == 0) | (i == int(scrnum[j-1]*1/4)-1) | (i == int(scrnum[j-1]*2/4)-1) | (i == int(scrnum[j-1]*3/4)-1) | (i == int(scrnum[j-1]*4/4)-1): g.write('{0:} {1:} {2:}\n'.format(fieldnumber,tdra,tddec))
    h.write('{0:} {1:} {2:} {3:} {4:} {5:}\n'.format(cnum[i],tdra,tddec,fieldnumber,cband[i],cdith[i]))

f.close()
g.close()

"""
# #	rts_vis MAKER WITH astroplan MODULE
# #	2019.12.18	CREATED BY	Gregory S.H. Paek
# #============================================================
# #	MODULES
# #------------------------------------------------------------
# import os, glob
# import numpy as np
# import astropy.units as u
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings(action='ignore')
# #------------------------------------------------------------
# from astropy.time import Time
# from astropy.io import ascii
# from astropy.coordinates import SkyCoord, get_moon
# # from pytz import timezone
# import pytz
# #------------------------------------------------------------
# from astroplan import Observer
# from astropy.coordinates import EarthLocation
# from astroplan.plots import plot_airmass
# #============================================================
# #	FUNCTION
# #------------------------------------------------------------
# def ut2local(observer, time):
# 	isoform = observer.astropy_time_to_datetime(time).isoformat()
# 	hhmm = (isoform.split('T')[1])[0:5]
# 	return hhmm
# #------------------------------------------------------------
# def lct2utc(lct):
# 	utctuple = lct.utctimetuple()
# 	utc = '{}-{}-{} {}:{}:{}'.format(utctuple[0], utctuple[1], utctuple[2], utctuple[3], utctuple[4], utctuple[5])
# 	return utc
# #------------------------------------------------------------
# def callobserver(obs, obstbl, description=''):
# 	indx_obs = np.where( obs == obstbl['name'] )
# 	longitude, latitude, elevation = obstbl[indx_obs]['longitude(E+)'], obstbl[indx_obs]['latitude(N+)'], obstbl[indx_obs]['altitude']
# 	tz = (obstbl[indx_obs]['timezone']).item()
# 	location = EarthLocation.from_geodetic(longitude, latitude, elevation)

# 	observer = Observer(name=obs,
# 						location=location,
# 						# pressure=0.615 * u.bar,
# 						# relative_humidity=0.11,
# 						# temperature=0 * u.deg_C,
# 						timezone=tz,
# 						description=description)
# 						# description="SAO 1-m Telescope on Seoul National University, Korea")
# 	return observer
# #------------------------------------------------------------
# def rts_vis_maker(c, observer, y, m, d, path_out='.'):
# 	#------------------------------------------------------------
# 	#	TIME
# 	#------------------------------------------------------------
# 	# time = Time('2020-01-01 12:00:00')		# UTC
# 	pst = pytz.timezone(observer.timezone.zone)
# 	lct_input = pst.localize(datetime(y, m, d, 1, 0, 0))	#	LOCAL TIME
# 	try:
# 		lct = pst.localize(datetime(y, m, d+1, 1, 0, 0))	#	LOCAL TIME
# 	except:
# 		lct = pst.localize(datetime(y, m+1, 1, 1, 0, 0))
# 	time = Time(lct2utc(lct))
# 	dts = pst.localize(time.datetime).dst().total_seconds()/3600.
# 	if dts >= 0:
# 		dts = '+{}'.format(dts)
# 	else:
# 		dts = str(dts)
# 	#------------------------------------------------------------
# 	#	WRITE rts_vis
# 	#------------------------------------------------------------
# 	f = open('{}/rts_vis_{}_IMSNG_{}.txt'.format(path_out, ''.join(lct_input.isoformat().split('T')[0].split('-')), obs), 'w')
# 	header = False
# 	#------------------------------------------------------------
# 	#	SUN INFO.
# 	#------------------------------------------------------------
# 	sunset_tonight = observer.sun_set_time(time, horizon=-18*u.deg, which='next')
# 	sunrise_tonight = observer.sun_rise_time(time, horizon=-18*u.deg, which='previous')
# 	# night_time = sunset_tonight + ((sunrise_tonight-0.25) - (sunset_tonight+0.25))*np.linspace(0, 1, 1000)
# 	for i, target in enumerate(c):
# 		#------------------------------------------------------------
# 		#	ONE TARGET
# 		#------------------------------------------------------------
# 		try:
# 			c_rise = observer.target_rise_time(time, target, horizon=constraint_alt, which='previous')
# 			c_set = observer.target_set_time(time, target, horizon=constraint_alt, which='next')
# 			all_up_start = np.max([c_rise])
# 			all_up_end = np.min([c_set])
# 			start = np.max([sunset_tonight, all_up_start])
# 			end = np.min([sunrise_tonight, all_up_end])
# 			visible_time = start + (end - start)*np.linspace(0, 1, 100)
# 			#	POSITION
# 			altaz = observer.altaz(visible_time, target)
# 			alt = altaz.alt
# 			az = altaz.az
# 			night_time = sunset_tonight + ((sunrise_tonight) - (sunset_tonight))*np.linspace(0, 1, 1000)
# 			target_time = c_rise + (c_set - c_rise)*np.linspace(0, 1, 1000)
# 			altaz_target = observer.altaz(target_time, target)
# 			alt_target = altaz_target.alt
# 			az_target = altaz_target.az
# 			#------------------------------------------------------------
# 			#	MOON INFO.
# 			#------------------------------------------------------------
# 			moon_altaz = observer.moon_altaz(visible_time)
# 			moon_alt, moon_az = moon_altaz.alt, moon_altaz.az

# 			c_moon = SkyCoord(moon_az, moon_alt, frame='altaz')
# 			moon_illumi = round(np.max(observer.moon_illumination(night_time)), 2)
# 			c_target = SkyCoord(az, alt, frame='altaz')
# 			sep_moon = c_target.separation(c_moon)
# 			#------------------------------------------------------------
# 			# c_moon_radec = get_moon(visible_time, location=observer.location).transform_to('icrs')
# 			# sep_moon = target.separation(c_moon_radec)
# 			#------------------------------------------------------------
# 			#	ALT, MOON DIST. CONSTRAINT
# 			#------------------------------------------------------------
# 			if (np.max(sep_moon) >= constraint_moon) & (np.max(alt) >= constraint_alt):
# 				transit_ut = Time((altaz_target[alt_target == np.max(alt_target)]).obstime.iso.item())
# 				#------------------------------------------------------------
# 				name = intbl['obj'][i]
# 				ra, dec = intbl['ra'][i], intbl['dec'][i]
# 				#------------------------------------------------------------
# 				rise_local = ut2local(observer, c_rise)
# 				transit_local = ut2local(observer, transit_ut)
# 				set_local = ut2local(observer, c_set)
# 				moon_dist = int(np.max(sep_moon).value)
# 				priority = intbl['priority'][i]

# 				if header == False:
# 					f.write('#\tObservatory\t: {}\n'.format(observer.name))
# 					f.write('#\t{} UTC & Day Time Saving {}\n'.format('/'.join(time.value.split(' ')[0].split('-')), dts))
# 					f.write('#\t-18 deg sunset\t: {}\n'.format(ut2local(observer, sunset_tonight)))
# 					f.write('#\t-18 deg sunrise\t: {}\n'.format(ut2local(observer, sunrise_tonight)))
# 					# f.write('Moon ra, dec\t: \n')
# 					f.write('#\tMoon phase\t: {}\n'.format(moon_illumi))
# 					# f.write('#\tMoon seperation\t: {}\n'.format(np.max(sep_moon)))
# 					f.write('#\tMoon seperation limit\t>= {}\n'.format(constraint_moon))
# 					f.write('#\tAltitude limit\t>= {}\n'.format(constraint_alt))
# 					f.write('#{}\n'.format('-'*60))
# 					f.write('name ra dec rise(LT) transit(LT) set(LT) moon_dist(deg) priority\n')
# 					header = True
# 				line = '%-16s%-13s%-15s%-8s%-8s%-8s%-4s%-4s\n' % (name, ra, dec, rise_local, transit_local, set_local, moon_dist, priority)
# 				# print(line)
# 				f.write(line)
# 			else:
# 				# print('out of constraint', intbl[i]['obj'], 'moon.dist', np.max(sep_moon).value, 'alt', np.max(alt).value)
# 				pass
# 		except:
# 			# print('numpy.float64 item error', intbl[i]['obj'])
# 			pass
# 	f.close()
# #============================================================
# #	SETTING
# #------------------------------------------------------------
# #	PATH
# #------------------------------------------------------------
# path_targetlist = '/data7/RASA36/script/imsng_alltarget_nopriority.dat'
# # path_targetlist = '/data7/RASA36/script/231211_BOAO_targets.dat'
# path_obs 	= '/data7/RASA36/script/observatory.txt'
# path_out 	= '/data7/RASA36/script/rts_vis_2025'
# #path_out 	= '/data7/RASA36/script/rts_vis_231211'
# #------------------------------------------------------------

# #	TABLE
# obstbl = ascii.read(path_obs)
# intbl = ascii.read(path_targetlist)
# intbl = intbl[intbl['priority'] < 3.0]
# #------------------------------------------------------------
# #	CONSTRAINT
# constraint_alt = 30*u.deg		# [deg]
# constraint_moon = 40*u.deg		# [deg]
# #------------------------------------------------------------
# #	OBSERVATORY INFO
# obs = 'RASA36'
# # obs = 'BOAO'
# # obs = 'SAO'
# # obs = 'LOAO'
# # obs = 'McD'
# observer = callobserver(obs, obstbl)
# #------------------------------------------------------------
# racol = intbl['ra']
# decol = intbl['dec']
# c = SkyCoord(racol, decol, unit=(u.hourangle, u.deg))

# # obslist = ['SAO']
# # obslist = ['LOAO']
# # obslist = ['McD']

# # y, m, d = 2020, 1, 1
# y, m, d = 2025, 1, 1
# # for obs in obslist:

# for m_step in np.arange(0, 12, 1):
# 	m_input = m + m_step
# 	for d_step in np.arange(0, 31, 1):
# 		d_input = d + d_step
# 		try:
# 			param_rts = dict(	c = c,
# 								observer=observer,
# 								y=y, m=m_input, d=d_input,
# 								path_out=path_out)
# 			print('{}\t:{}/{}/{}'.format(obs, y, m_input, d_input))
# 			rts_vis_maker(**param_rts)
# 		except:
# 			print('{}/{}\tDONE'.format(y, m_input))

# # datelist = [(2020, 1, 31), (2020, 2, 29), (2020, 3, 31), (2020, 4, 30), (2020, 5, 31), (2020, 6, 30), (2020, 7, 31), (2020, 8, 31), (2020, 9, 30), (2020, 10, 31), (2020, 11, 30), (2020, 12, 31), ]
# datelist = [(2025, 1, 31), (2025, 2, 28), (2025, 3, 31), (2025, 4, 30), (2025, 5, 31), (2025, 6, 30), (2025, 7, 31), (2025, 8, 31), (2025, 9, 30), (2025, 10, 31), (2025, 11, 30), (2025, 12, 31), ]
# for date in datelist:
# 	y, m, d = date[0], date[1], date[2]

# 	param_rts = dict(	c = c,
# 						observer=observer,
# 						y=y, m=m, d=d,
# 						path_out=path_out)
# 	print('{}\t:{}/{}/{}'.format(obs, y, m, d))
# 	rts_vis_maker(**param_rts)

# # path_out    = '/data7/RASA36/script/'
# # ys, ms, ds      = 2023, 3, 24
# # param_rts = dict(	c = c,
# # 					observer=observer,
# # 					y=ys, m=ms, d=ds,
# # 					path_out=path_out)
# # print('{}\t:{}/{}/{}'.format(obs, ys, ms, ds))
# # rts_vis_maker(**param_rts)
# #%% BOAO 231211 ALLBRICQS Targets
# # import pandas as pd
# # from io import StringIO

# # coords = """
# # ra	dec
# # 25.0324	43.889
# # 34.9536	48.4683
# # 160.1677	67.1917
# # 5.6257	34.5267
# # 212.5899	75.5651
# # 13.1842	47.3663
# # """
# # tbl = pd.read_csv(StringIO(coords), sep="\t")

# # # Convert and print the coordinates
# # for index, row in tbl.iterrows():
# #     # Right Ascension
# #     ra_decimal = row['ra'] / 15
# #     ra_hours = int(ra_decimal)
# #     ra_minutes = int((ra_decimal - ra_hours) * 60)
# #     ra_seconds = ((ra_decimal - ra_hours) * 60 - ra_minutes) * 60

# #     # Declination
# #     dec_degrees = int(row['dec'])
# #     dec_minutes = int(abs(row['dec'] - dec_degrees) * 60)
# #     dec_seconds = (abs(row['dec'] - dec_degrees) * 60 - dec_minutes) * 60
# #     print(f"{ra_hours:02d}:{ra_minutes:02d}:{ra_seconds:05.2f} {dec_degrees:+03d}:{dec_minutes:02d}:{dec_seconds:05.2f}")
# %%
fields = """
0222.243-66
0223.248-66
0292.228-68
0293.233-68
0294.238-68
0295.243-68
0296.248-68
0297.254-68
0298.259-68
0360.233-70
0361.238-70
0362.244-70
0363.249-70
0364.255-70
0365.260-70
0423.244-72
0424.250-72
0425.256-72
""".split()
# %%
band = 'R'
for field in fields:
    # os.makedirs(f'/data4/kmtntoo/tutorial/data/template/{field}', exist_ok=True)
    imgs = sorted(glob.glob(f'/data8/KS4/database/stack/{field}/ks4*{band}*.cat'))
    print(f'{field} : {len(imgs)}')
    if len(imgs) != 0:
        for img in imgs:
            shutil.copy(img, f'/data4/kmtntoo/tutorial/data/template/{field}/')
# %% Notice
"""
===============================
2024년 5월 MMA 관측 관련 공지
안녕하세요,

관측시간 사용과 대체시간 보상에 대한 정보를 보여주는 웹페이지를 구성하고 있는 중입니다(형태나 정보는 추후 바뀔 수 있습니다.). 아직 완벽히 구축이 되지 않았지만, 현재 관측이 진행되고 있기 때문에 빠른 이해가 필요하다고 판단하여, MMA 연구팀에게 먼저 설명 드리겠습니다. 

TOO 관측은 이번 달에 이루어지는데, 대체 보상하는 시점은 다음달입니다. 따라서, 이번 달의 관측요청은 다음 달 배정 받은 관측시간의 총합과 일정을 먼저 확인한 다음 요청해야 합니다. 이와 관련한  정보는 https://kmtnet.kasi.re.kr/kmtnet-monitor/21/ 에서 찾을 수 있습니다. 

웹페이지의 중간쯤을 보시면 TOO 관측 보상이란 부분이 있습니다. 이곳에는 오늘 기준 지난 달과 이번달에 수행한 TOO 관측 현황을 2개로 나누어서 보여주고 있습니다. 한편, 아래 블럭에는 이번 달과 다음 달의 관측일정을 달력 형태로 보여주고 있습니다. 최근 TOO관측이 활발히 이루어지고 있는 MMA 관측의 경우에는 핑크색으로 관측일정을 나타내고 그 안에 관측시간을 표시해 놓았습니다. 

이 두가지 정보를 이용하여 남아공에서의 TOO (MMA) 관측 현황을 예로 들어 설명하겠습니다. 

4월달에 MMA관측 요청(대체보상해야 할 과제가 있는 경우만 고려함)은 총 3회 (DEEPS, KSP, DEEPS)있었습니다.  달력 모양의 5월달 관측일정 배정을 보면 약 3시간 씩 MMA로 총 4회가 배정되어 있습니다. 따라서, 순차적으로 관측시간을 대체 보상하면, 5월 10일 관측 블럭 하나만 남게 됩니다. 즉, 4월에 총 4번의 TOO 관측을 요청할 수 있었는데, 3번을 사용하여 하나만 남긴 상황입니다. 이에, 5월 10일에 배정된 MMA 관측 시간에는 TOO 타겟이 있다면 TOO 관측을 하시거나, KS4 타겟을 관측하시면 됩니다. 5월달을 이어서 설명하겠습니다. 이번 달에는 5월 1일과 2일 MMA관측을 요청하여 사용하였는데, 달력모양의 6월달 관측일정을 보면 MMA 관측으로 총 6일의 블럭이 배정되어있습니다. 즉, 2번의 관측 요청이 있었기 때문에, 5월달 관측요청은 아직 4회 더 남았습니다. 

TOO 관측은 예에서 보인 방법을 이용하여 남은 관측 횟수를 미리 파악한 후에 요청해 주시기를 부탁드립니다. 한편, 5월1일 KSP시간에 사용한 MMA는 약 3.5 시간입니다. 그러나, 순차적 보상을 적용하면, KSP 과제에 보상해주는 시간은 6월 1일 1.9 시간으로 관측 시간에 있어 차이가 발생합니다. 관측시간 대체 보상 시 이런 부분들까지 완벽히 보상이 되면 좋겠으나, 이 차이는 대체 보상시간관리의 효율성 등을 이유로 제가 월 별로 판단하여 그리 크지 않다면 DIR시간이나 ENG등의 시간 등을 이용하여 보상하려고 합니다. 즉, 관측시간 블럭 위주로 대체 보상시간을 운영할 계획입니다. 그러나, 대체되는 시간차가 2시간 정도로 크게 발생하면, 이에 대해서는 관측 블럭 뿐아니라, 관측시간도 고려하겠습니다. 

이충욱 드림
===============================
2025년 2월 MMA 관측
CTIO
2월 가용 3회 14:47
3월 가용 2회 06:51
사용 3회 19:22
==> 2/10, 2/11, 2/12, 2/13 SITE 시간 이용하여 MMA 관측 가능 (14일 부터는 DEEPS 관측)
SSO
2월 가용 2회 09:48
3월 가용 2회 06:55
사용 0회 00:00
SAAO
2월 가용 3회 14:34
3월 가용 2회 06:48
사용 2회 10:57 + (2/9 관측 요청됨, 1회 03:00 ==> 날씨 흐려서 무효)
==> 2/10 부터 SITE 시간, 이 중 KS4 관측 시간은 MMA에 사용 가능.
===============================
"""
# %%
"""
import ligo.skymap.plot  

volurl = f'https://gracedb.ligo.org/api/superevents/{event}/files/bayestar.fits.gz'
ra_deg = 240. # S250206dm
dec_deg = -65.
center = SkyCoord(ra_deg*u.deg, dec_deg*u.deg)

fig = plt.figure(figsize=(10, 10), dpi=100)
ax = plt.axes(
    [0.05, 0.05, 0.9, 0.9],
    projection='astro globe',
    center=center)
ax.grid(alpha=0.7)

coords = SkyCoord(obslog['RA'], obslog['DEC'], unit=(u.hourangle, u.deg))

# Create the table
kmt_points = Table([coords.ra.deg, coords.dec.deg], names=('RA', 'DEC'))

ax.scatter(
    kmt_points['RA'], 
    kmt_points['DEC'],
    transform=ax.get_transform('world'),
    marker='s',
    s=90,  # size of the marker
    color='dodgerblue',
    edgecolors='k',  # no edge color for markers
    label='KMTNet',
    zorder =999)
ra_axis     = ax.coords['ra']
image_main = ax.imshow_hpx(volurl, cmap='cylon', vmin=0, vmax=1e-4, alpha=1)
plt.title(f"{event}", fontsize=25)

"""