# from .bin import bin
# from .file_io import (load_dataset_from_files, create_file_header, validate_file_exists, validate_path_exists,
#                       save_interpolated_df_as_file, save_binned_df_as_file, find_e0)
#
# from xas.db_io import load_apb_dataset_from_db, translate_apb_dataset, load_apb_trig_dataset_from_db, \
#     load_xs3_dataset_from_db
# from .interpolate import interpolate

# (load_dataset_from_files, create_file_header, validate_file_exists, validate_path_exists,
#                   save_interpolated_df_as_file, save_binned_df_as_file, find_e0)

from xas.process import load_apb_dataset_from_db, translate_apb_dataset, load_apb_trig_dataset_from_db, \
    load_xs3_dataset_from_db, interpolate, bin


def average_roi_channels(dataframe=None):
    if dataframe is not None:
        for j in range(1,5):
            dat = 0
            for i in range(1,5):
                dat += getattr(dataframe, 'CHAN' + str(j) + 'ROI' + str(i))
            dataframe['ROI' + str(j) + 'AVG'] = dat/4
        print('Done with averaging')
    return dataframe

apb_df, energy_df, energy_offset = load_apb_dataset_from_db(db, db[-1].start['uid'])
raw_df = translate_apb_dataset(apb_df, energy_df, energy_offset)

apb_trig_timestamps = load_apb_trig_dataset_from_db(db, db[-1].start['uid'])
xs3_dict = load_xs3_dataset_from_db(db, db[-1].start['uid'], apb_trig_timestamps)

raw_df = {**raw_df, **xs3_dict}
key_base = 'CHAN1ROI1'

interpolated_df = interpolate(raw_df, key_base = key_base)
binned_df = bin(interpolated_df, 7709)