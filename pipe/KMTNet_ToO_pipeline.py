#%% path defines
import time, os, sys, glob, re, copy, shutil, subprocess
path_base   = '/data4/kmtntoo/tutorial/'

path_data   = os.path.join(path_base, 'data/')
path_cfg    = os.path.join(path_base, 'config/')
path_cat    = os.path.join(path_base, 'catalog/')

path_raw    = os.path.join(path_data, 'raw')
path_scale  = os.path.join(path_data, 'scaled/')
path_stack  = os.path.join(path_data, 'stack/')
path_subt   = os.path.join(path_data, 'subt/')

path_res    = os.path.join(path_base, 'result/')
path_plot   = os.path.join(path_res, 'plot/')
path_phot   = os.path.join(path_res, 'phot/')
path_log    = os.path.join(path_res, 'log/')
#%% KMTNet ToO Pipeline
from astropy.io import fits
from datetime import datetime
from astropy.table import Table, vstack
import KMTNet_ToO_functions as pipe

def ToO_pipeline(date):
    
    # start of the process
    start = time.time()
    print(f'KMTNet ToO Pipeline Starts for {date}.')
    
    # process managements
    ampcompro   = False
    astrompro   = False
    astromqapro = True
    zpscalepro  = False
    bpmaskpro   = False
    stackingpro = False
    qa4stackpro = False
    catalogpro  = False
    subtpro     = False
    rbclasspro  = False

    process_status = {
        'ampcompro': ampcompro, 'astrompro': astrompro, 'astromqapro': astromqapro, 
        'zpscalepro': zpscalepro, 'bpmaskpro': bpmaskpro, 'stackingpro': stackingpro, 
        'qa4stackpro': qa4stackpro, 'catalogpro': catalogpro, 'subtpro': subtpro, 'rbclasspro': rbclasspro
    }

    # Print process status in a compact form
    for process, status in process_status.items():
        print(f"{process}:\t {'ON' if status else 'OFF'}")

    # make output directories
    path_output1= os.path.join(path_raw, f'{date}/') # where a*fits chip images will be located
    os.makedirs(path_output1, exist_ok=True)
    os.chmod(path_output1, 0o777)
    
    path_output2 = os.path.join(path_scale, f'{date}/') # where ToO*scaled.fits will be located
    os.makedirs(path_output2, exist_ok=True)
    os.chmod(path_output2, 0o777)

    path_output3 = os.path.join(path_stack, f'{date}/') # where ToO*stack.fits will be located
    os.makedirs(path_output3, exist_ok=True)
    os.chmod(path_output3, 0o777)

    path_output4 = os.path.join(path_phot, f'{date}/') # where ToO*zp.cat will be located
    os.makedirs(path_output4, exist_ok=True)
    os.chmod(path_output4, 0o777)

    path_output5 = os.path.join(path_subt, f'{date}/') # where conv.res.ToO*NxN.fits will be located
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

        pipe.astrom(path_output1, path_cfg, path_cat, radius=0.73, thresh=5)
        log2            = copy.deepcopy(log)
        log2['process'] = 'astrompro'
        log2['frames']  = len(Table.read(f'{path_output1}ToOastrom.txt', format ='ascii'))
        log2['time']    = round(time.time()-start, 2)
        
        try:
            LOG     = vstack([LOG, log2])
        except NameError:
            LOG     = log2
        LOG.write(LOGname, format='ascii', overwrite=True)

    endastrom = time.time()
    time.sleep(0.1)
    print(f'The 1st astrometry done. {endastrom-start:.2f}sec')

    if astromqapro:
        
        regex = re.compile(r"(?P<serial>\d{6})\.(?P<chip>kk|mm|tt|nn)\.fits")
        all_files   = sorted(glob.glob(f'{path_output1}*.fits'))
        afits   = [file for file in all_files if regex.match(os.path.basename(file))]
        for img in afits:
            # if 'QARESULT' not in fits.open(img)[0].header:
            pipe.qatest(img, configdir=path_cfg, refcatdir=path_cat, refcatname='gaiaxp', gridcat='kmtnet_grid.fits')
            os.system(f'chmod 777 {path_output1}*crmap.fits')
        
        log3            = copy.deepcopy(log)
        log3['process'] = 'astromqapro'
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
            # outname = pipe.zpscale(img, path_output2, path_cfg, path_cat, path_plot, zpscaled=30.0, figure=False, start=start, gridcat='/data4/kmtntoo/config/astrometry/ToO_grid.cat')
            outname = pipe.zpscale(img, path_output2, path_cfg, path_cat, path_plot, zpscaled=30.0, figure=False, start=start, gridcat='/data8/KS4/config/kmtnet_grid.cat')
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

    if bpmaskpro:

        regex = re.compile(r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<band>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<serial>\d{6})\.(?P<chip>\w+)\.(?P<type>scaled)\.fits")
        all_files = sorted(glob.glob(f'{path_output2}*.fits'))
        afits   = [file for file in all_files if regex.match(os.path.basename(file))]

        for img in afits:
            pipe.BPM_update(img, path_cfg)
        
        logm            = copy.deepcopy(log)
        logm['process'] = 'bpmaskpro'
        logm['frames']  = len(afits)
        logm['time']    = round(time.time()-start, 2)
        
        try:
            LOG     = vstack([LOG, logm])
        except NameError:
            LOG     = logm
        LOG.write(LOGname, format='ascii', overwrite=True)

    endbpmask    = time.time()
    time.sleep(0.1)
    print(f'BP mask process done. {endbpmask-start:.2f}')       

    if stackingpro:

        pattern = r"(?P<field>.*?_\d{4})\.(?P<radec>\d{3}-\d{2})\.(?P<band>[BVRI])\.(?P<date>\d{8})\.(?P<site>\w+)\.(?P<serial>\d{6})\.(?P<chip>\w+)\.(?P<type>scaled|mask)\.fits"
        total   = pipe.stacking(pattern, path_output2, path_output3, path_cfg, path_ks4, start=start, gridcat='/data8/KS4/config/kmtnet_grid.cat')
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
                # os.system('rm default*')
                # os.system('rm kmtn*')

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
            gpsubt.subtraction(simg, path_ref=path_ks4, path_cat=path_output4, path_refcat=path_ks4edr, path_output=path_output5, path_config=path_cfg, detect=3)
        
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
        result = subprocess.run(command, capture_output=True, text=True)

        # Check result
        if result.returncode == 0:
            print("inference.py executed successfully.")
            print(result.stdout)
        else:
            print("inference.py execution failed.")
            print(result.stderr)

        log9            = copy.deepcopy(log)
        log9['process'] = 'rbclasspro'
        log9['frames']  = len(sorted(glob.glob(f"{path_output5}*.new.*")))
        log9['time']    = round(time.time()-start, 2)
        
        try:
            LOG     = vstack([LOG, log9])
        except NameError:
            LOG     = log9
        LOG.write(LOGname, format='ascii', overwrite=True)

    endrb   = time.time()
    time.sleep(0.1)
    print(f'Real/Bogus classification process done. {endrb-start:.2f}')

    # end of process (LOG saving)
    try:
        LOG.write(LOGname, format='ascii', overwrite=True)
        print(f"KMTNet Reduction Pipeline for {date} Has Done. \nLog file location: {LOGname}")
    except NameError:
        pass
    
    return

#%% KMTNet ToO WatchDog
import shutil
import argparse
import multiprocessing
from collections import defaultdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class TooWatcher(FileSystemEventHandler):

    # basics
    def __init__(self, watch_directory, ncores):
        self.watch_directory    = watch_directory
        self.ncores             = ncores
        self.pool               = multiprocessing.Pool(processes=ncores)

    # creation checking sequences
    def on_created(self, event, max_wait_time=900, wait_interval=5):
        
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
        last_move_time = time.time()
        
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
                # self.move_to_directory(file_path, location, self.watch_directory, self.uploads_directory)
                last_move_time  = time.time()
                break
            elif time.time() - start_time > max_wait_time:
                print(f"Timeout reached for {file_name}. Skipping.")
                break

            time.sleep(wait_interval)

        # file set completeness & pipeline run
        while True:
            if time.time() - last_move_time > max_wait_time:
                print(f'{max_wait_time}sec have passed since the last image upload. \nKMTNet reduction pipeline will be running (n_core={self.ncores}).')
                
                pattern = r'kmt[asc]\.\d{8}\.\d{6}\.fits'
                alluploads = [os.path.basename(entry.path) for entry in os.scandir(watch_directory) if entry.is_file() and re.compile(pattern).match(os.path.basename(entry.path))]

                # classify the upload files
                for (date, site), files in self.group_files_by_date_and_site(alluploads).items():
                    process_directory   = f'{self.watch_directory}{date}_{locations.get(site)}'
                    # files to directory (yyyymmdd_SITE)
                    os.makedirs(process_directory, exist_ok=True)
                    for file in files:
                        shutil.move(os.path.join(self.watch_directory, file), os.path.join(process_directory, file))
                    # run the pipeline
                    if self.ncores == 1:
                        ToO_pipeline(os.path.basename(process_directory))
                    else:
                        self.pool.apply_async(ToO_pipeline, args=(os.path.basename(process_directory),)) # working directories are defined inside the function
                break
            elif self.is_new_file_generated(self.watch_directory, last_move_time):
                break
            else:
                time.sleep(wait_interval)

    # new "additional" file detections
    def is_new_file_generated(self, watch_directory, last_check_time):
        for file_name in os.listdir(watch_directory):
            file_path = os.path.join(watch_directory, file_name)
            if os.path.isfile(file_path):
                pattern = r'kmt[asc]\.\d{8}\.\d{6}\.fits'
                if re.match(pattern, file_name):
                    file_creation_time = os.path.getctime(file_path)
                    if file_creation_time > last_check_time:
                        # print('New file detected.')
                        return True
        return False

    # grouping newly uploaded file sets by dates and sites
    def group_files_by_date_and_site(self, file_list):

        # Define a regex pattern to extract the site and date information
        file_pattern = re.compile(r'(?P<site>kmt[asc])\.(?P<date>\d{8})\.\d{6}\.fits')

        # Create a defaultdict to store the grouped files
        grouped_files = defaultdict(list)

        # Iterate through the file list and group them by date and site
        for file in file_list:
            match = file_pattern.match(file)
            if match:
                site = match.group('site')
                date = match.group('date')
                grouped_files[(date, site)].append(file)
        
        return grouped_files
    
    def wait_for_file_transfer(file_path, check_interval=5, stable_time=10):
        
        previous_size = -1
        stable_count = 0

        while True:
            try:
                current_size = os.path.getsize(file_path)
            except FileNotFoundError:
                time.sleep(check_interval)
                continue

            if current_size == previous_size:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count * check_interval >= stable_time:
                break

            previous_size = current_size
            time.sleep(check_interval)
            
    def close_pool(self):
        self.pool.close()
        self.pool.join()

#%% main program
if __name__ == "__main__":
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="KMTNet_ToO_pipeline.py")
    parser.parse_args()
    
    watch_directory     = path_raw
    ncores              = 1
    data_dirs   = [d for d in os.listdir(watch_directory) if os.path.isdir(os.path.join(watch_directory, d))]

    print(f"List of Data Directories in {watch_directory}:")
    print("="*20)
    for directory in data_dirs:
        print(directory)
    print("="*20)
    
    user_input = input('Enter the directory name to process, or type ‘AUTO’ to start automatic monitoring of new uploads: ')

    if user_input == "AUTO":
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
    
    else:
        if os.path.isdir(os.path.join(watch_directory, user_input)):
            ToO_pipeline(user_input)
        else:
            print(f'Check if {os.path.join(watch_directory, user_input)} exists.')
    # regular watchdog
