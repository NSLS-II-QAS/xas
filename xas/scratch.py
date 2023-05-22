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
    load_xs3_dataset_from_db, interpolate, rebin


def average_roi_channels(dataframe=None):
    if dataframe is not None:
        col1 = dataframe.columns.tolist()[:-1]
        for j in range(1,5):
            dat = 0
            for i in range(1,5):
                dat += getattr(dataframe, 'CHAN' + str(i) + 'ROI' + str(j))
            dataframe['ROI' + str(j) + 'AVG'] = dat/4
            col1.append('ROI' + str(j) + 'AVG')
        col1.append('energy')
        dataframe = dataframe[col1]
        print('Done with averaging')
    return dataframe

apb_df, energy_df, energy_offset = load_apb_dataset_from_db(db, db[-1].start['uid'])
raw_df = translate_apb_dataset(apb_df, energy_df, energy_offset)

apb_trig_timestamps = load_apb_trig_dataset_from_db(db, db[-1].start['uid'])
xs3_dict = load_xs3_dataset_from_db(db, db[-1].start['uid'], apb_trig_timestamps)

raw_df = {**raw_df, **xs3_dict}
key_base = 'CHAN1ROI1'

key_base = 'i0'

interpolated_df = interpolate(raw_df, key_base = key_base)
binned_df = bin(interpolated_df, 8333)

path = '/nsls2/data/qas-new/legacy/processed/2023/2/000000/'

energy_points = np.arange(8250,9400,5)

def step_scan():
    path = '/nsls2/data/qas-new/legacy/processed/2023/2/000000/'

    energy_points = np.arange(8200, 9400, 1)
    ch1 = []
    ch2 = []
    ch3 = []
    ch4 = []
    energy_pt = []
    i0 = []
    iff = []
    for energy in energy_points:
        mono1.energy.set(energy).wait()
        energy_pt.append(mono1.energy.user_readback.get())
        yield from bp.count([apb_ave], num=1)
        _i0 = db[-1].table()['apb_ave_ch1'][1]
        _iff = db[-1].table()['apb_ave_ch4'][1]
        # i0.append(apb_ave.ch1.get())
        # iff.append(apb_ave.ch4.get())

        i0.append(_i0)
        iff.append(_iff)
        yield from xs_count(0.250)
        print(xs.channel1.read()['xs_channel1_rois_roi01_value']['value'])
        ch1.append(xs.channel1.read()['xs_channel1_rois_roi01_value']['value'])
        ch2.append(xs.channel2.read()['xs_channel2_rois_roi01_value']['value'])
        ch3.append(xs.channel3.read()['xs_channel3_rois_roi01_value']['value'])
        ch4.append(xs.channel4.read()['xs_channel4_rois_roi01_value']['value'])

    ch1 = np.array(ch1)
    ch2 = np.array(ch2)
    ch3 = np.array(ch3)
    ch4 = np.array(ch4)

    avg = (ch1 + ch2 + ch3 + ch4)/4
    np.savetxt(path+'NiK_sdd_250ms_step2.dat', np.column_stack((energy_pt, i0, iff, ch1, ch2, ch3, ch4, avg)),
               header = 'energy i0 iff ch1 ch2 ch3 ch4')




def test():
    yield from xs_count(0.05)
    print(xs.channel1.read()['xs_channel1_rois_roi01_value']['value'])


for key in dataset.keys():
   # print(f'Dataset length >>>>> {len(dataset.get(key).iloc[:, 0])}')
   #  print(f'Timestamps length >>>>> {len(timestamps)}')
    if len(dataset.get(key).iloc[:, 0]) > 5 * len(timestamps):
        time = [np.mean(array) for array in np.array_split(dataset.get(key).iloc[:, 0].values, len(timestamps))]
        #print(f'Times {time}')
        val = [np.mean(array) for array in np.array_split(dataset.get(key).iloc[:, 1].values, len(timestamps))]
        #print(f'Values {val}')
        interpolated_dataset[key] = np.array([timestamps, np.interp(timestamps, time, val)]).transpose()
    else:
        interpolated_dataset[key] = np.array([timestamps, np.interp(timestamps, dataset.get(key).iloc[: ,0].values,
                                                                    dataset.get(key).iloc[:,1])]).transpose()