#%% KMTNet ToO Pipeline
import subprocess
from astropy.io import fits
from datetime import datetime
from astropy.table import Table, vstack
import time, os, sys, glob, re, copy, shutil
pipeline_directory  = '/data4/kmtntoo/pipe/'
os.chdir(pipeline_directory)
import KMTNet_ToO as pipe
import transient_search_routine as gpsubt 
from astrom_qa import astrom_qa # for stacked images
from KMTNet_ToO_astqa import qatest # for single chip images

def ToO_pipeline(date):
    
    # start of the process
    start = time.time()
    print(f'KMTNet ToO Pipeline Starts for {date}.')
    
    # process managements
    ampcompro   = False
    astrompro   = False
    astrompro2  = False
    zpscalepro  = False
    stackingpro = False
    qa4stackpro = False
    catalogpro  = False
    subtpro     = False
    rbclasspro  = True

    # path defines
    path_base   = '/data4/kmtntoo/'
    path_data   = f'{path_base}data/'
    path_scale  = f'{path_base}scaled/'
    path_stack  = f'{path_base}stack/'
    path_subt   = f'{path_base}subt/'
    path_cfg    = f'{path_base}config/'
    path_cat    = f'{path_base}cat/'
    path_bin    = f'{path_base}trash/'

    path_res    = f'{path_base}result/'
    path_plot   = f'{path_res}plot/'
    path_phot   = f'{path_res}phot/'
    path_log    = f'{path_res}log/'
    path_reg    = f'{path_res}z_region/'
    path_ks4    = '/data8/KS4/database/stack/'
    path_ks4edr = '/data8/KS4/catalog/EDR/'

    # make output directories
    path_output1= f'{path_data}{date}/' # where a*fits chip images will be located
    os.makedirs(path_output1, exist_ok=True)
    os.chmod(path_output1, 0o777)
    
    path_output2 = f'{path_scale}{date}/' # where ToO*scaled.fits will be located
    os.makedirs(path_output2, exist_ok=True)
    os.chmod(path_output2, 0o777)

    path_output3 = f'{path_stack}{date}/' # where ToO*stack.fits will be located
    os.makedirs(path_output3, exist_ok=True)
    os.chmod(path_output3, 0o777)

    path_output4 = f'{path_phot}{date}/' # where ToO*zp.cat will be located
    os.makedirs(path_output4, exist_ok=True)
    os.chmod(path_output4, 0o777)

    path_output5 = f'{path_subt}{date}/' # where conv.res.ToO*NxN.fits will be located
    os.makedirs(path_output5, exist_ok=True)
    os.chmod(path_output5, 0o777)

    # log file setting
    log         = Table([['xxxxxxxxpro'], [9999], [0.0]], names=['process', 'frames', 'time'])
    LOGname     = f"{path_log}ToOprocess_{date}_{datetime.fromtimestamp(start).strftime('%Y-%m-%d_%H:%M:%S')}.log"
    
    # pipelines
    if ampcompro:

        pipe.ampcom(path_output1, path_cfg)
        log1            = copy.deepcopy(log)
        log1['process'] = 'ampcompro'
        log1['frames']  = len(glob.glob(f'{path_output1}/kmt*.*.*.fits')) # raw images
        log1['time']    = round(time.time()-start, 2)
        
        try:
            LOG     = vstack([LOG, log1])
        except NameError:
            LOG     = log1
        LOG.write(LOGname, format='ascii', overwrite=True)
        
    endampcom = time.time()
    time.sleep(0.1)
    print(f'Amp combine process done. {endampcom-start:.2f}sec')

    if astrompro:

        pipe.astrom(date, path_output1, path_cfg, radius=2.3, ithresh=5)
        log2            = copy.deepcopy(log)
        log2['process'] = 'astrompro'
        log2['frames']  = len(Table.read(f'{path_output1}info1.txt', format ='ascii'))
        log2['time']    = round(time.time()-start, 2)
        
        try:
            LOG     = vstack([LOG, log2])
        except NameError:
            LOG     = log2
        LOG.write(LOGname, format='ascii', overwrite=True)

    endastrom = time.time()
    time.sleep(0.1)
    print(f'The 1st astrometry done. {endastrom-start:.2f}sec')

    if astrompro2:
        
        regex = re.compile(r"a(?P<serial>\d{6})\.(?P<chip>kk|mm|tt|nn)\.fits")
        all_files   = sorted(glob.glob(f'{path_output1}*.fits'))
        afits   = [file for file in all_files if regex.match(os.path.basename(file))]
        for img in afits:
            if 'QARESULT' not in fits.open(img)[0].header:
                qatest(img, crreject=True)
                os.system(f'chmod 777 {path_output1}*crmap.fits')
        
        log3            = copy.deepcopy(log)
        log3['process'] = 'astrompro2'
        log3['frames']  = len(afits)
        log3['time']    = round(time.time()-start, 2)
        
        try:
            LOG     = vstack([LOG, log3])
        except NameError:
            LOG     = log3
        LOG.write(LOGname, format='ascii', overwrite=True)

    endastrom2 = time.time()
    time.sleep(0.1)
    print(f'Astrometry QA done. {endastrom2-start:.2f}sec')

    if zpscalepro:
        """
        reference catalog should be located:
        {path_cat}smss/
        {path_cat}apass/
        """
        regex = re.compile(r"a(?P<serial>\d{6})\.(?P<chip>kk|mm|tt|nn)\.fits")
        all_files   = sorted(glob.glob(f'{path_output1}*.fits')) # ToOampcom.cat should be located
        afits   = [file for file in all_files if regex.match(os.path.basename(file))]
        
        for img in afits:
            outname = pipe.zpscale(img, path_output2, path_cfg, path_cat, path_plot, zpscaled=30.0, figure=False, start=start, gridcat='/data8/tempdatabase/kmtnet_grid.cat')
            if outname != None and os.path.exists(img.replace('.fits', '.crmap.fits')):
                os.rename(img.replace('.fits', '.crmap.fits'), os.path.join(path_output2, outname.replace('.scaled', '.crmap')))
        log4            = copy.deepcopy(log)
        log4['process'] = 'zpscalepro'
        log4['frames']  = len(afits)
        log4['time']    = round(time.time()-start, 2)
        
        try:
            LOG     = vstack([LOG, log4])
        except NameError:
            LOG     = log4
        LOG.write(LOGname, format='ascii', overwrite=True)

    endzpscale     = time.time()
    time.sleep(0.1)
    print(f'ZP scaling process done. {endzpscale-start:.2f}')

    if stackingpro:

        pattern = r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<band>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<serial>\d{6})\.(?P<chip>\w+)\.(?P<type>scaled|crmap)\.fits"
        total   = pipe.stacking(pattern, path_output2, path_output3, path_cfg, path_ks4, start=start, gridcat='/data4/kmtntoo/config/astrometry/ToO_grid.cat')
        log5            = copy.deepcopy(log)
        log5['process'] = 'stackingpro'
        log5['frames']  = total
        log5['time']    = round(time.time()-start, 2)
        
        try:
            LOG     = vstack([LOG, log5])
        except NameError:
            LOG     = log5
        LOG.write(LOGname, format='ascii', overwrite=True)

    endstack    = time.time()
    time.sleep(0.1)
    print(f'Image stacking process done. {endstack-start:.2f}')    

    if qa4stackpro:
        
        regex = re.compile(r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<filter>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<exptime>\d+sec)\.(?P<type>scaled)\.stack\.fits")
        all_files   = sorted(glob.glob(f'{path_output3}*.fits'))
        stackimgs   = [file for file in all_files if regex.match(os.path.basename(file))]

        for simg in stackimgs:
            # if 'ALNRMS' not in fits.open(simg)[0].header:
            astrom_qa(simg)

        log6            = copy.deepcopy(log)
        log6['process'] = 'qa4stackpro'
        log6['frames']  = len(stackimgs)
        log6['time']    = round(time.time()-start, 2)
        
        try:
            LOG     = vstack([LOG, log6])
        except NameError:
            LOG     = log6
        LOG.write(LOGname, format='ascii', overwrite=True)
            
    endqa = time.time()
    time.sleep(0.1)
    print(f'QA for stacked images process done. {endqa-start:.2f}sec')

    if catalogpro:

        regex = re.compile(r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<filter>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<exptime>\d+sec)\.(?P<type>scaled)\.stack\.fits\.cat")
        all_cats= sorted(glob.glob(f'{path_output3}*.cat'))
        cats    = [file for file in all_cats if regex.match(os.path.basename(file))]
        for cat in cats:
            pipe.catalogmaker(cat, path_output=path_output4, path_cat=path_cat, figure=False, start=start, path_plot=path_plot)

        log7            = copy.deepcopy(log)
        log7['process'] = 'catalogpro'
        log7['frames']  = len(cats)
        log7['time']    = round(time.time()-start, 2)
        
        try:
            LOG     = vstack([LOG, log7])
        except NameError:
            LOG     = log7
        LOG.write(LOGname, format='ascii', overwrite=True)
            
    endcatalog = time.time()
    time.sleep(0.1)
    print(f"Catalog making process done. {endcatalog-start:.2f} sec")

    if subtpro:

        # subtraction with hotpants
        regex = re.compile(r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<filter>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<exptime>\d+sec)\.(?P<type>scaled)\.stack\.fits")
        all_files   = sorted(glob.glob(f'{path_output3}*.fits'))
        stackimgs   = [file for file in all_files if regex.match(os.path.basename(file))]
        for simg in stackimgs:
            gpsubt.subtraction(simg, path_ref=path_ks4, path_cat=path_output4, path_refcat=path_ks4edr, path_output=path_output5, path_config=path_cfg, detect=3, div_col=1, div_row=1)
        
        log8            = copy.deepcopy(log)
        log8['process'] = 'subtpro'
        log8['frames']  = len(sorted(glob.glob(f"{path_output5}*.new.*")))
        log8['time']    = round(time.time()-start, 2)
        
        try:
            LOG     = vstack([LOG, log8])
        except NameError:
            LOG     = log8
        LOG.write(LOGname, format='ascii', overwrite=True)
        
    endsubt     = time.time()
    time.sleep(0.1)
    print(f'Image subtraction process done. {endsubt-start:.2f}')
    
    if rbclasspro:
        # ML R/B classifications for the snapshot images
        command = ['python', 'Cowork-GW_universe-issue-3/inference.py', '--dir_fits', path_output5, '--dir_ckpt', 'Cowork-GW_universe-issue-3/ckpt']
        result = subprocess.run(command, capture_output=True, text=True, cwd='/data4/kmtntoo/pipe/')

        # Check result
        if result.returncode == 0:
            print("inference.py executed successfully.")
            print(result.stdout)
        else:
            print("inference.py execution failed.")
            print(result.stderr)

        shutil.copy(f'{path_output5}rbclass.csv', f'rbscore/rbscore_{date}.csv')

        log9            = copy.deepcopy(log)
        log9['process'] = 'rbclasspro'
        log9['frames']  = len(sorted(glob.glob(f"{path_output5}*.new.*")))
        log9['time']    = round(time.time()-start, 2)
        
        try:
            LOG     = vstack([LOG, log9])
        except NameError:
            LOG     = log9
        LOG.write(LOGname, format='ascii', overwrite=True)

    # end of process (LOG saving)
    try:
        LOG.write(LOGname, format='ascii', overwrite=True)
        print(f"KMTNet Single Image Reduction Pipeline for {date} Has Done. \nLog file location: {LOGname}")
    except NameError:
        pass
    
    return

#%% KMTNet ToO WatchDog
import argparse
import concurrent.futures
from collections import defaultdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class TooWatcher(FileSystemEventHandler):

    # basics
    def __init__(self, watch_directory, ncores):
        self.watch_directory    = watch_directory
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=ncores)  # adjust the max_workers as needed

    # creation checking sequences
    def on_created(self, event, max_wait_time=600, wait_interval=5):
        
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

        # file path check
        file_path = event.src_path
        file_name = os.path.basename(file_path)
        if not os.path.isfile(file_path): return # this event is not a file upload
        
        # observatory check (SAAO, SSO, CTIO)
        try:
            prefix = re.search(r"(kmts|kmta|kmtc)", file_name).group(0)
        except AttributeError:
            prefix = 'kmta'
        location = locations.get(prefix)

        # process time check
        start_time = time.time()
        
        # file completeness & move file
        while True:
            
            try:
                file_size = os.path.getsize(file_path)
            except FileNotFoundError: # temporary file name get changed after completion.
                match   = re.search(pattern, file_path) # assuming file name is involved in the temp name (e.g. .kmts.20230308.061242.fits.7uUd6N.)
                if match:
                    file_name   = match.group()
                    file_path   = os.path.join(self.watch_directory, file_name)
                    file_size   = os.path.getsize(file_path)
                else:
                    print(f"Lost File {file_path}")
                    return

            if file_size >= file_sizes[location]:
                time.sleep(1)
                # Check if the file name matches the expected pattern
                if not re.match(pattern, file_name):
                    print(f"Invalid file name {file_name}. Skipping.")
                    return
                print(f'New upload file detection: {os.path.basename(file_path)}')
                upload_directory    = self.move_to_directory(file_path, location, root_path=self.watch_directory)
                self.executor.submit(ToO_pipeline, os.path.basename(upload_directory))
                break
            elif time.time() - start_time > max_wait_time:
                print(f"Timeout reached for {file_name}. Skipping.")
                break

            time.sleep(wait_interval)

    def move_to_directory(self, file_path, location, root_path='./'):
        
        date    = os.path.basename(file_path).split('.')[1]
        serial  = os.path.basename(file_path).split('.')[2]
        new_directory = os.path.join(root_path, f'{date}_{location}_{serial}')
        os.makedirs(new_directory, exist_ok=True)
        
        new_file_path = os.path.join(new_directory, os.path.basename(file_path))
        os.rename(file_path, new_file_path)
        
        return new_directory

    def close_executor(self):
        self.executor.shutdown()

#%% main program
if __name__ == "__main__":
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="KMTNet_ToO_watchdog.py")
    parser.add_argument("object_name", nargs='?', default=None, help="Provide an optional date name (python KMTNet_ToO_watchdog.py yyyymmdd_SITE)")
    args = parser.parse_args()
    
    watch_directory    = '/data4/kmtntoo/tutorial/data/raw/'
    ncores             = 10

    # in case CLA provided
    if args.object_name is not None:
        ToO_pipeline(args.object_name)
    # regular watchdog
    else:
        print('KMTNet ToO Data WatchDog Activated: Looking for kmtx.00000000.000000.fits')
        observer = Observer()
        event_handler = TooWatcher(watch_directory, ncores)
        observer.schedule(event_handler, watch_directory, recursive=False)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()

        observer.join()
        event_handler.close_pool()
