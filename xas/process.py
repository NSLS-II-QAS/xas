from .bin import bin
from .file_io import (load_dataset_from_files, create_file_header, validate_file_exists, validate_path_exists,
                      save_interpolated_df_as_file, save_binned_df_as_file, find_e0)

from xas.db_io import load_apb_dataset_from_db, translate_apb_dataset, load_apb_trig_dataset_from_db, \
    load_xs3_dataset_from_db
from .interpolate import interpolate

from .xas_logger import get_logger
from .xs3 import load_data_with_xs3

from datetime import datetime


def process_interpolate_bin(doc, db, draw_func_interp = None, draw_func_binnned = None):

    logger = get_logger()

    if 'experiment' in db[doc['run_start']].start.keys():
        uid = doc['run_start']
        experiment = db[uid].start['experiment']
        if experiment.startswith('fly'):
            path_to_file = db[uid].start['interp_filename']
            e0 = find_e0(db, uid)
            comments = create_file_header(db, uid)
            validate_path_exists(db, uid)

            path_to_file = validate_file_exists(path_to_file, file_type='interp')
            print(f'>>>Path to file {path_to_file}')
            # try:
            if experiment ==  'fly_energy_scan':
                raw_df = load_dataset_from_files(db, uid)
                key_base = 'i0'
            elif  experiment ==  'fly_energy_scan_apb':
                apb_df, energy_df, energy_offset = load_apb_dataset_from_db(db, uid)
                raw_df= translate_apb_dataset(apb_df, energy_df, energy_offset)
                key_base = 'i0'
            elif experiment == 'fly_energy_scan_xs3':
                apb_df, energy_df, energy_offset = load_apb_dataset_from_db(db, uid)
                raw_df = translate_apb_dataset(apb_df, energy_df, energy_offset)

                apb_trig_timestamps = load_apb_trig_dataset_from_db(db, uid)
                xs3_dict = load_xs3_dataset_from_db(db, uid, apb_trig_timestamps)

                raw_df = {**raw_df, **xs3_dict}
                key_base = 'CHAN1ROI1'
            logger.info(f'Loading file successful for UID {uid}/{path_to_file}')
            # except:
            #     logger.info(f'Loading file failed for UID {uid}/{path_to_file}')
            try:
                interpolated_df = interpolate(raw_df, key_base = key_base)
                logger.info(f'Interpolation successful for {path_to_file}')
                save_interpolated_df_as_file(path_to_file, interpolated_df, comments)
            except:
                logger.info(f'Interpolation failed for {path_to_file}')

            try:
                if e0 > 0:
                    print('Inside xas process try draw (e0 > 0) start time: ', datetime.now())
                    binned_df = bin(interpolated_df, e0)
                    logger.info(f'Binning successful for {path_to_file}')
                    if  experiment ==  'fly_energy_scan_apb' or experiment ==  'fly_energy_scan_xs3':
                        save_binned_df_as_file(path_to_file, binned_df, comments, reorder=True)
                    else:
                        save_binned_df_as_file(path_to_file, binned_df, comments, reorder=False)
                    if draw_func_interp is not None:
                        draw_func_interp(interpolated_df)

                else:
                    print('Energy E0 is not defined')
            except Exception as e:
                logger.info(f'Binning failed for {path_to_file}')
                print(e)
                pass
        elif experiment.startswith('diffraction'):
            pass
                
        

def process_interpolate_only(doc, db):
    if 'experiment' in db[doc['run_start']].start.keys():
        if db[doc['run_start']].start['experiment'] == 'fly_energy_scan':
            raw_df = load_dataset_from_files(db, doc['run_start'])
            interpolated_df = interpolate(raw_df)
            return interpolated_df
