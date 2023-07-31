# from .bin import bin
# from .file_io import (load_dataset_from_files, create_file_header, validate_file_exists, validate_path_exists,
#                       save_interpolated_df_as_file, save_binned_df_as_file, find_e0)
#
from xas.db_io import load_apb_dataset_from_db, translate_apb_dataset, load_apb_trig_dataset_from_db, \
    load_xs3_dataset_from_db
from .interpolate import interpolate

# (load_dataset_from_files, create_file_header, validate_file_exists, validate_path_exists,
#                   save_interpolated_df_as_file, save_binned_df_as_file, find_e0)

from xas.process import load_apb_dataset_from_db, translate_apb_dataset, load_apb_trig_dataset_from_db, \
    load_xs3_dataset_from_db, interpolate, rebin

from xray import encoder2energy
def my_plan(detectors, motor, motor_positions, exposures, delay=0, md={}):
    @bpp.stage_decorator(list(detectors) + [motor])
    @bpp.run_decorator(md=md)
    def inner_plan():
       for pos, exp in zip(motor_positions, exposures):
        yield from bps.mv(motor, pos, detectors[0].settings.acquire_time, exp)
        yield from bps.trigger_and_read(list(detectors) + [motor])
        yield from bps.sleep(delay)
    return (yield from inner_plan())


RE(my_plan([xs, apb_ave], mono1.energy, np.arange(10000, 10100+1, 5).tolist(), [1, 2, 3, 4, 5], delay=0))


def my_plan(detectors, motor, motor_positions, exposures, delay=0):
    for dev in list(detectors) + [motor]:
        yield from bps.stage(dev)
    yield from bps.open_run()
    for pos, exp in zip(motor_positions, exposures):
        yield from bps.mv(motor, pos, detectors[0].settings.acquire_time, exp)
    yield from bps.trigger_and_read(list(detectors) + [motor])
    yield from bps.sleep(delay)
    yield from bps.close_run()




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


from xas.process import load_apb_dataset_from_db, translate_apb_dataset, load_apb_trig_dataset_from_db, \
    load_xs3_dataset_from_db, interpolate, rebin


apb_df, energy_df, energy_offset = load_apb_dataset_from_db(db, db[-1].start['uid'])
raw_df = translate_apb_dataset(apb_df, energy_df, energy_offset)

apb_trig_timestamps = load_apb_trig_dataset_from_db(db, db[-1].start['uid'])
xs3_dict = load_xs3_dataset_from_db(db, db[-1].start['uid'], apb_trig_timestamps)

raw_df = {**raw_df, **xs3_dict}
key_base = 'CHAN1ROI1'

# key_base = 'i0'

interpolated_df = interpolate(raw_df, key_base = key_base)
binned_df = bin(interpolated_df, 7112)

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


# RE(bp.list_scan([xs, apb_ave], mono1.energy, pt))

uid = 'cde7f69e-0014-4c4a-a7dd-9d3667d34f1b'

_data = db[uid].table()

en = _data['mono1_energy_user_setpoint']
i0 = _data['apb_ave_ch1']
it = _data['apb_ave_ch2']
ir = _data['apb_ave_ch3']
iff = _data['apb_ave_ch4']

roi1 = _data['xs_channel1_rois_roi01_value'] + _data['xs_channel2_rois_roi01_value'] + \
    _data['xs_channel3_rois_roi01_value'] + _data['xs_channel4_rois_roi01_value']
# roi2 =

uid = 'a00a2376-9b75-43fb-b4b9-5419cce9c17e'

en = np.arange(9600, 9740, 0.2)
pt = en.tolist()

uid = '14ebe1a8-6798-4628-a530-060ebb8b182e'

uid ='64b5b873-fb2a-4e74-8ddd-fe85168f8fb3'


uid = '65c43956-800f-4c45-868d-440140000a61'

uid = 'cba2473a-94e8-4d53-b085-66f1396a9924'

uid = '7c095e50-f3ad-4e3c-a5c9-ab3494c1d515'
def make_data_table_sdd(uid=None):
    data = {}
    _data = db[uid].table()

    data['en'] = _data['mono1_energy_user_setpoint']
    data['i0'] = _data['apb_ave_ch1']
    data['it'] = _data['apb_ave_ch2']
    data['ir'] = _data['apb_ave_ch3']
    data['iff'] = _data['apb_ave_ch4']

    data['roi1'] = (_data['xs_channel1_rois_roi01_value'] + _data['xs_channel2_rois_roi01_value'] +
                    _data['xs_channel3_rois_roi01_value'] + _data['xs_channel4_rois_roi01_value'])/4

    data['roi4'] = (_data['xs_channel1_rois_roi04_value'] + _data['xs_channel2_rois_roi04_value'] +
                    _data['xs_channel3_rois_roi04_value'] + _data['xs_channel4_rois_roi04_value'])/4

    return data



data = make_data_table_sdd(uid)

path = '/nsls2/data/qas-new/legacy/processed/2023/2/000000/'

np.savetxt((path+'cu_test.dat', np.column_stack((data['en'],
                                                 data['i0'],
                                                 data['it'],
                                                 data['ir'],
                                                 data['iff'],
                                                 data['roi1'],
                                                 data['roi4']), header = 'energy i0 iff ch1 ch2 ch3 ch4')))

fig, ax = plt.subplots(2,1)
ax[0].plot(data['en'], data['roi1']/data['i0'], '-r', label='roi1')
ax[0].plot(data['en'], data['roi2']/data['i0'], '-b', label='roi2')
ax[1].plot(data['en'], data['iff']/data['i0'], 'g', label='PIPS')
ax[0].legend()
ax[1].legend()


plt.figure()

plt.plot(raw_df['aux1']['timestamp'], raw_df['aux1']['adc'])

plt.plot(raw_df['aux1']['timestamp'], raw_df['aux1']['adc']/max(raw_df['aux1']['adc']))
plt.plot(raw_df['energy']['timestamp'], raw_df['energy']['encoder']/max(raw_df['energy']['encoder']))

plt.plot(raw_df['CHAN1ROI1']['timestamp'], raw_df['CHAN1ROI1']['CHAN1ROI1'] )

class custom_xs(Xspress3Detector):
    cnt_time = Cpt(EpicsSignal, 'C1_SCA0:Value_RBV')
    def __init__(self, prefix, *, configuration_attrs=None, read_attrs=None, **kwargs):
        super().__init__(prefix, configuration_attrs=configuration_attrs,
                         read_attrs=read_attrs, **kwargs)

xs2 = custom_xs('XF:07BMB-ES{Xsp:1}:', name='xs2')


th = np.arange(-9.03414, -9.50497, -0.00025)
offset = -0.14



def theta2energy(theta, offset = 0):
    return -12400 / (2 * 3.1356 * np.sin(np.deg2rad((theta) - float(offset))))


th = np.arange(-9.03414, -9.50497, -0.00025)
#en = 11800-12400

th = np.arange(-11.86877, -10.60915, -0.00025)

exp = np.ones(len(th))*2

uid = '6ac53d70-e851-433c-bbed-b10761f232de'
path = '/nsls2/data/qas-new/legacy/processed/2023/2/000000/'

def convert_step_scan_data(uid=None, filename = None):
    t = db[uid].table()
    i0 = t['apb_ave_ch1_mean']
    it = t['apb_ave_ch2_mean']
    ir = t['apb_ave_ch3_mean']
    iff = t ['apb_ave_ch4_mean']

    xs_ch1_roi1 = t['xs_channel1_rois_roi01_value']
    xs_ch2_roi1 = t['xs_channel2_rois_roi01_value']
    xs_ch3_roi1 = t['xs_channel3_rois_roi01_value']
    xs_ch4_roi1 = t['xs_channel4_rois_roi01_value']

    xs_avg = t['xs_channel1_rois_roi01_value'] + \
        t['xs_channel2_rois_roi01_value'] + \
        t['xs_channel3_rois_roi01_value'] + \
        t['xs_channel4_rois_roi01_value']

    xs_avg = xs_avg/4

    # time = t['time']

    energy = theta2energy(t['mono1_bragg'], offset=0.13996789)
    energy_setpoint = theta2energy(t['mono1_bragg_user_setpoint'], offset=0.13996789)

    # return energy, energy_setpoint, i0, it, iff, xs_avg, xs_ch1_roi1, xs_ch2_roi1, xs_ch3_roi1, xs_ch4_roi1, time

    np.savetxt(path + filename, np.column_stack((energy,
                                                 energy_setpoint,
                                                 i0,
                                                 it,
                                                 ir,
                                                 iff,
                                                 xs_avg,
                                                 xs_ch1_roi1,
                                                 xs_ch2_roi1,
                                                 xs_ch3_roi1,
                                                 xs_ch4_roi1)),
                                                 header='energy en_setpoint i0 it ir iff xs_avg ch1 ch2 ch3 ch4')






['time',
 'xs_settings_acquire_time',
 'xs_channel1_rois_roi01_value',
 'xs_channel1_rois_roi02_value',
 'xs_channel1_rois_roi03_value',
 'xs_channel1_rois_roi04_value',
 'xs_channel1_rois_roi05_value',
 'xs_channel1_rois_roi06_value',
 'xs_channel2_rois_roi01_value',
 'xs_channel2_rois_roi02_value',
 'xs_channel2_rois_roi03_value',
 'xs_channel2_rois_roi04_value',
 'xs_channel2_rois_roi05_value',
 'xs_channel2_rois_roi06_value',
 'xs_channel3_rois_roi01_value',
 'xs_channel3_rois_roi02_value',
 'xs_channel3_rois_roi03_value',
 'xs_channel3_rois_roi04_value',
 'xs_channel3_rois_roi05_value',
 'xs_channel3_rois_roi06_value',
 'xs_channel4_rois_roi01_value',
 'xs_channel4_rois_roi02_value',
 'xs_channel4_rois_roi03_value',
 'xs_channel4_rois_roi04_value',
 'xs_channel4_rois_roi05_value',
 'xs_channel4_rois_roi06_value',
 'xs_channel5_rois_roi01_value',
 'xs_channel5_rois_roi02_value',
 'xs_channel5_rois_roi03_value',
 'xs_channel5_rois_roi04_value',
 'xs_channel5_rois_roi05_value',
 'xs_channel5_rois_roi06_value',
 'xs_channel1',
 'xs_channel2',
 'xs_channel3',
 'xs_channel4',
 'xs_channel5',
 'xs_channel6',
 'mono1_bragg',
 'mono1_bragg_user_setpoint',
 'apb_ave_ch1',
 'apb_ave_ch2',
 'apb_ave_ch3',
 'apb_ave_ch4',
 'apb_ave_ch5',
 'apb_ave_ch6',
 'apb_ave_ch7',
 'apb_ave_ch8',
 'apb_ave_vi0',
 'apb_ave_vit',
 'apb_ave_vir',
 'apb_ave_vip',
 'apb_ave_ch1_adc_gain',
 'apb_ave_ch2_adc_gain',
 'apb_ave_ch3_adc_gain',
 'apb_ave_ch4_adc_gain',
 'apb_ave_ch5_adc_gain',
 'apb_ave_ch6_adc_gain',
 'apb_ave_ch7_adc_gain',
 'apb_ave_ch8_adc_gain',
 'apb_ave_ch1_adc_offset',
 'apb_ave_ch2_adc_offset',
 'apb_ave_ch3_adc_offset',
 'apb_ave_ch4_adc_offset',
 'apb_ave_ch5_adc_offset',
 'apb_ave_ch6_adc_offset',
 'apb_ave_ch7_adc_offset',
 'apb_ave_ch8_adc_offset',
 'apb_ave_pulse1_status',
 'apb_ave_pulse2_status',
 'apb_ave_pulse3_status',
 'apb_ave_pulse4_status',
 'apb_ave_pulse1_stream_status',
 'apb_ave_pulse2_stream_status',
 'apb_ave_pulse3_stream_status',
 'apb_ave_pulse4_stream_status',
 'apb_ave_pulse1_file_status',
 'apb_ave_pulse2_file_status',
 'apb_ave_pulse3_file_status',
 'apb_ave_pulse4_file_status',
 'apb_ave_pulse1_stream_count',
 'apb_ave_pulse2_stream_count',
 'apb_ave_pulse3_stream_count',
 'apb_ave_pulse4_stream_count',
 'apb_ave_pulse1_max_count',
 'apb_ave_pulse2_max_count',
 'apb_ave_pulse3_max_count',
 'apb_ave_pulse4_max_count',
 'apb_ave_pulse1_op_mode_sp',
 'apb_ave_pulse2_op_mode_sp',
 'apb_ave_pulse3_op_mode_sp',
 'apb_ave_pulse4_op_mode_sp',
 'apb_ave_pulse1_stream_mode_sp',
 'apb_ave_pulse2_stream_mode_sp',
 'apb_ave_pulse3_stream_mode_sp',
 'apb_ave_pulse4_stream_mode_sp',
 'apb_ave_pulse1_frequency_sp',
 'apb_ave_pulse2_frequency_sp',
 'apb_ave_pulse3_frequency_sp',
 'apb_ave_pulse4_frequency_sp',
 'apb_ave_pulse1_dutycycle_sp',
 'apb_ave_pulse2_dutycycle_sp',
 'apb_ave_pulse3_dutycycle_sp',
 'apb_ave_pulse4_dutycycle_sp',
 'apb_ave_pulse1_delay_sp',
 'apb_ave_pulse2_delay_sp',
 'apb_ave_pulse3_delay_sp',
 'apb_ave_pulse4_delay_sp',
 'apb_ave_data_rate',
 'apb_ave_divide',
 'apb_ave_sample_len',
 'apb_ave_wf_len',
 'apb_ave_stream_samples',
 'apb_ave_trig_source',
 'apb_ave_filename_bin',
 'apb_ave_filebin_status',
 'apb_ave_filename_txt',
 'apb_ave_filetxt_status',
 'apb_ave_ch1_mean',
 'apb_ave_ch2_mean',
 'apb_ave_ch3_mean',
 'apb_ave_ch4_mean',
 'apb_ave_ch5_mean',
 'apb_ave_ch6_mean',
 'apb_ave_ch7_mean',
 'apb_ave_ch8_mean',
 'apb_ave_time_wf',
 'apb_ave_ch1_wf',
 'apb_ave_ch2_wf',
 'apb_ave_ch3_wf',
 'apb_ave_ch4_wf',
 'apb_ave_ch5_wf',
 'apb_ave_ch6_wf',
 'apb_ave_ch7_wf',
 'apb_ave_ch8_wf']


plt.figure()
# plt.plot(raw_df['CHAN1ROI1']['timestamp'], raw_df['CHAN1ROI1']['CHAN1ROI1'])
plt.errorbar(raw_df['CHAN1ROI1']['timestamp'],raw_df['CHAN1ROI1']['CHAN1ROI1'], yerr= np.sqrt(raw_df['CHAN1ROI1']['CHAN1ROI1']), capsize=5)
plt.plot(raw_df['iff']['timestamp'], raw_df['iff']['adc']*90000)



plt.figure()
# plt.errorbar(raw_df['iff']['timestamp'],raw_df['iff']['adc'], capsize=5)
plt.plot(raw_df['i0']['timestamp'], raw_df['i0']['adc'], label='i0')
plt.plot(raw_df_fly['i0']['timestamp'], raw_df_fly['i0']['adc'], label='i0')
plt.plot(raw_df['it']['timestamp'], raw_df['it']['adc'], label='it')
plt.plot(raw_df['ir']['timestamp'], raw_df['ir']['adc'], label='ir')
plt.plot(raw_df['iff']['timestamp'], raw_df['iff']['adc'], label='iff')
plt.legend()

plt.figure()
plt.plot(raw_df['CHAN1ROI1']['timestamp'], raw_df['CHAN1ROI1']['CHAN1ROI1'])


uid = '4ec32c98-0404-4859-93d7-cf405f380fef'