# from .bin import bin
# from .file_io import (load_dataset_from_files, create_file_header, validate_file_exists, validate_path_exists,
#                       save_interpolated_df_as_file, save_binned_df_as_file, find_e0)
#
import numpy as np

from xas.db_io import load_apb_dataset_from_db, translate_apb_dataset, load_apb_trig_dataset_from_db, \
    load_xs3_dataset_from_db
from .interpolate import interpolate

# (load_dataset_from_files, create_file_header, validate_file_exists, validate_path_exists,
#                   save_interpolated_df_as_file, save_binned_df_as_file, find_e0)

from xas.process import load_apb_dataset_from_db, translate_apb_dataset, load_apb_trig_dataset_from_db, \
    load_xs3_dataset_from_db, interpolate, rebin

from xray import encoder2energy


from xas.process import load_apb_dataset_from_db, translate_apb_dataset, load_apb_trig_dataset_from_db, load_xs3_dataset_from_db, interpolate, issrebin


apb_df, energy_df, energy_offset = load_apb_dataset_from_db(db, db[-1].start['uid'])
raw_df = translate_apb_dataset(apb_df, energy_df, energy_offset)

apb_trig_timestamps = load_apb_trig_dataset_from_db(db, db[-1].start['uid'])
xs3_dict = load_xs3_dataset_from_db(db, db[-1].start['uid'], apb_trig_timestamps)

raw_df = {**raw_df, **xs3_dict}
key_base = 'CHAN1ROI1'

# key_base = 'i0'

interpolated_df = interpolate(raw_df, key_base = key_base)
binned_df = rebin(interpolated_df, 52)


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
    print('value', xs.channel1.read()['xs_channel1_rois_roi01_value']['value'])
    yield from xs_count(1)

def process_constant_energy(number_of_points=10):
    A = []
    for i in range(number_of_points,0, -1):
        hdr = db[-i]
        t = hdr.table()
        A.append(t.xs_channel1_rois_roi02_value)

    return A


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

    energy = theta2energy(t['mono1_bragg'], offset=0.1363215717444159)
    energy_setpoint = theta2energy(t['mono1_bragg_user_setpoint'], offset=0.1363215717444159)

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

## Cu K scan with sdd using step scan with mono1.bragg

th1 = -12.8451
th2 = -11.8736

th = np.arange(th1, th2, 0.001)
exp = np.ones(len(th))*0.02

theta2energy(th, offset=-0.13769446878876723)

RE(my_plan([xs, apb_ave], mono1.bragg, th.tolist(), exp.tolist(), delay=0))


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

    energy = theta2energy(t['mono1_bragg'], offset=0.1363215717444159)
    energy_setpoint = theta2energy(t['mono1_bragg_user_setpoint'], offset=0.1363215717444159)

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

#### Debug APB/ENC timestamps


## Fe K scan with sdd using steps with mono1.bragg

th1 = -16.51343
th2 = -15.14739


th1 = -16.76558
th2 = -14.01550

th = np.arange(th1, th2, 0.001)
exp = np.ones(len(th))*1

RE(my_plan([xs, apb_ave], mono1.bragg, th.tolist(), exp.tolist(), delay=0))


uid = '728deb21-1a41-43fd-8150-fa77021543fa'
uid = '612cde35-93fd-48c1-b499-3d751793a8b0'

def theta2energy(theta, offset = 0):
    return -12400 / (2 * 3.1356 * np.sin(np.deg2rad((theta) - float(offset))))


path = '/nsls2/data/qas-new/legacy/processed/2023/3/000000/'




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

    energy = theta2energy(t['mono1_bragg'], offset=0.1363215717444159)
    energy_setpoint = theta2energy(t['mono1_bragg_user_setpoint'], offset=0.1363215717444159)

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



def extract_data(hdrs):
    plt.figure()

    for i, hdr in enumerate(hdrs):
        enc_data = hdr.table(stream_name="pb1_enc1", fill=True)
        apb_data = hdr.table(stream_name="apb_stream", fill=True)

        # APB:
        apb_times = np.array(apb_data["apb_stream"][1]["timestamp"])
        plt.plot([0, apb_times[-1]-apb_times[0]] , [i, i], label="APB times")

        # plt.axvline(apb_times[0], ymin=i-0.5, ymax=i+0.5)
        # plt.axvline(apb_times[-1], ymin=i-0.5, ymax=i+0.5)

        # ENC:
        enc_times = np.array(enc_data["pb1_enc1"][1]["timestamp"])
        plt.plot([0, enc_times[-1]-enc_times[0]] , [i+0.5, i+0.5], label="Enc. PB times")

        # plt.axvline(enc_times[-1], ymin=i+0.5-0.5, ymax=i+0.5+0.5)

    plt.legend()

def split_dataset(dataset, n_periods):
    splitted_data =[]
    full_length = len(raw_df['energy']['timestamp'])
    single_period = int(full_length/20)
    for period in range(n_periods):
        single_scan = {}
        start_index = period*single_period
        stop_index = (1+period)*single_period-1
        if stop_index > full_length:
            stop_index = full_length
        single_scan['energy'] = dataset['energy'].iloc[start_index:stop_index]

        start_time=  raw_df['energy']['timestamp'].iloc[start_index]
        stop_time = raw_df['energy']['timestamp'].iloc[stop_index]
        #print(start_time)
        #print(stop_time)
        for key in dataset.keys():
            if key != 'energy':
                signal_start_index=df.iloc[(datasedt[key]['timestamp']-start_time).abs().argsort()[:1]]



        splitted_data.append(single_scan)



    return splitted_data

split_dataset(raw_df, 20)


def interpolate_series(dataset,key_base = 'i0'):
    interpolated_dataset = {}
    min_timestamp = max([dataset.get(key).iloc[0, 0] for key in dataset])
    max_timestamp = min([dataset.get(key).iloc[len(dataset.get(key)) - 1, 0] for key in
                         dataset if len(dataset.get(key).iloc[:, 0]) > 5])

    try:
        if key_base not in dataset.keys():
            raise ValueError('Could not find "{}" in the loaded scan. Pick another key_base'
                             ' for the interpolation.'.format(key_base))
    except ValueError as err:
        print(err.args[0], '\nAborted...')
        return

    timestamps = dataset[key_base].iloc[:,0]

    condition = timestamps < min_timestamp
    timestamps = timestamps[np.sum(condition):]

    condition = timestamps > max_timestamp
    timestamps = timestamps[: len(timestamps) - np.sum(condition)]

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
            # print ('>>>> else')

    intepolated_dataframe = pd.DataFrame(np.vstack((timestamps, np.array([interpolated_dataset[array][:, 1] for
                                                                            array in interpolated_dataset]))).transpose())
    keys = ['timestamp']
    keys.extend(interpolated_dataset.keys())
    intepolated_dataframe.columns = keys
    return intepolated_dataframe.sort_values('energy')


def check_device_staged_status():
    devices = {'apb': apb,
               'apb_ave': apb_ave,
               'apb_stream': apb_stream,
               'xs': xs,
               'pb1.enc1': pb1.enc1,
               'xs_stream': xs_stream,
               'apb_trigger': apb_trigger}

    for name, device in devices.items():
        status = device._staged
        print(f"{name} status {status}")


def unstage_staged_devices():
    devices = {'apb': apb,
               'apb_ave': apb_ave,
               'apb_stream': apb_stream,
               'xs': xs,
               'pb1.enc1': pb1.enc1,
               'xs_stream': xs_stream,
               'apb_trigger': apb_trigger}

    for name, device in devices.items():
        status = device._staged.value
        print(f"{name} staged status: {status}")
        if status == 'yes':
            device.unstage()
            status = device._staged.value
            print(f"{name} staged status: {status}")



plt.figure()

plt.plot(raw_df['energy']['timestamp'], raw_df['energy']['encoder'])

plt.plot(raw_df['ir']['timestamp'], raw_df['ir']['adc'])

plt.plot(raw_df['i0']['timestamp'], np.log(raw_df['i0']['adc']/raw_df['ir']['adc']))



def plot_interpolated_df(uid):
    apb_df, energy_df, energy_offset = load_apb_dataset_from_db(db, db[uid].start['uid'])
    raw_df = translate_apb_dataset(apb_df, energy_df, energy_offset)

    key_base = 'i0'

    interpolated_df = interpolate(raw_df, key_base=key_base)

    plt.figure()
    plt.plot(interpolated_df['energy'], np.log(interpolated_df['i0']/interpolated_df['it']))
    plt.plot(interpolated_df['energy'], np.log(interpolated_df['it'] / interpolated_df['ir']))
    plt.plot(interpolated_df['energy'], interpolated_df['iff'] / interpolated_df['i0'])


def plot_raw_df(uid):
    apb_df, energy_df, energy_offset = load_apb_dataset_from_db(db, db[uid].start['uid'])
    raw_df = translate_apb_dataset(apb_df, energy_df, energy_offset)

    plt.figure()
    plt.plot(raw_df['i0']['timestamp'], np.log(raw_df['i0']['adc'] / raw_df['it']['adc']))
    plt.plot(raw_df['i0']['timestamp'], np.log(raw_df['it']['adc'] / raw_df['ir']['adc']))
    plt.plot(raw_df['i0']['timestamp'], raw_df['iff']['adc'] / raw_df['i0']['adc'])


uid = 'f1833231-037c-4067-8b4e-9f5f4d9128af'

def plot_constant_energy(uid=None):
    apb_trig_timestamps = load_apb_trig_dataset_from_db(db, db[-1].start['uid'])
    xs3_dict = load_xs3_dataset_from_db(db, db[-1].start['uid'], apb_trig_timestamps)

    plt.figure()
    plt.errorbar(xs3_dict['CHAN1ROI2']['timestamp'],
                 xs3_dict['CHAN1ROI2']['CHAN1ROI2']/max(xs3_dict['CHAN1ROI2']['CHAN1ROI2']),
                 yerr=np.sqrt(xs3_dict['CHAN1ROI2']['CHAN1ROI2']))




class QASXspress3Detector(XspressTrigger, Xspress3Detector):
    roi_data = Cpt(PluginBase, 'ROIDATA:')
    channel1 = Cpt(Xspress3Channel, 'C1_', channel_num=1, read_attrs=['rois'])
    channel2 = Cpt(Xspress3Channel, 'C2_', channel_num=2, read_attrs=['rois'])
    channel3 = Cpt(Xspress3Channel, 'C3_', channel_num=3, read_attrs=['rois'])
    channel4 = Cpt(Xspress3Channel, 'C4_', channel_num=4, read_attrs=['rois'])
    channel5 = Cpt(Xspress3Channel, 'C5_', channel_num=5, read_attrs=['rois'])
    channel6 = Cpt(Xspress3Channel, 'C6_', channel_num=6, read_attrs=['rois'])
    # create_dir = Cpt(EpicsSignal, 'HDF5:FileCreateDir')

    mca1_sum = Cpt(EpicsSignal, 'ARRSUM1:ArrayData')
    mca2_sum = Cpt(EpicsSignal, 'ARRSUM2:ArrayData')
    mca3_sum = Cpt(EpicsSignal, 'ARRSUM3:ArrayData')
    mca4_sum = Cpt(EpicsSignal, 'ARRSUM4:ArrayData')
    mca5_sum = Cpt(EpicsSignal, 'ARRSUM5:ArrayData')
    mca6_sum = Cpt(EpicsSignal, 'ARRSUM6:ArrayData')

    mca1 = Cpt(EpicsSignal, 'ARR1:ArrayData')
    mca2 = Cpt(EpicsSignal, 'ARR2:ArrayData')
    mca3 = Cpt(EpicsSignal, 'ARR3:ArrayData')
    mca4 = Cpt(EpicsSignal, 'ARR4:ArrayData')
    mca5 = Cpt(EpicsSignal, 'ARR5:ArrayData')
    mca6 = Cpt(EpicsSignal, 'ARR6:ArrayData')

    cnt_time = Cpt(EpicsSignal, 'C1_SCA0:Value_RBV')

    # channel6 = Cpt(Xspress3Channel, 'C6_', channel_num=6)

    #TODO change folder to xspress3
    hdf5 = Cpt(Xspress3FileStoreFlyable, 'HDF5:',
               read_path_template='/nsls2/data/qas-new/legacy/raw/x3m/%Y/%m/%d/',
               root='/nsls2/data/qas-new/legacy/raw/',
               write_path_template='/nsls2/data/qas-new/legacy/raw/x3m/%Y/%m/%d/',
               )

    def __init__(self, prefix, *, configuration_attrs=None, read_attrs=None,
                 **kwargs):
        if configuration_attrs is None:
            configuration_attrs = ['external_trig', 'total_points',
                                   'spectra_per_point', 'settings',
                                   'rewindable']
        if read_attrs is None:
            read_attrs = ['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'hdf5', 'settings.acquire_time']
        super().__init__(prefix, configuration_attrs=configuration_attrs,
                         read_attrs=read_attrs, **kwargs)
        # self.set_channels_for_hdf5()
        # self.create_dir.put(-3)

        self._asset_docs_cache = deque()
        self._datum_counter = None

        self.channel1.rois.roi01.configuration_attrs.append('bin_low')

    # Step-scan interface methods.
    def stage(self):
        if self.spectra_per_point.get() != 1:
            raise NotImplementedError(
                "multi spectra per point not supported yet")

        ret = super().stage()
        self._datum_counter = itertools.count()
        return ret
    #
    def trigger(self):

        self._status = DeviceStatus(self)
        self.settings.erase.put(1)
        # self.settings.erase.put(1)    # this was
        self._acquisition_signal.put(1, wait=False)
        trigger_time = ttime.time()

        for sn in self.read_attrs:
            if sn.startswith('channel') and '.' not in sn:
                ch = getattr(self, sn)
                self.generate_datum(ch.name, trigger_time)

        self._abs_trigger_count += 1
        return self._status
    #
    def unstage(self):
        self.settings.trigger_mode.put(1)  # 'Software'
        super().unstage()
        self._datum_counter = None

    def stop(self):
        ret = super().stop()
        self.hdf5.stop()
        return ret
    #
    # # Fly-able interface methods.
    # def kickoff(self):
    #     # TODO: implement the kickoff method for the flying mode once the hardware is ready.
    #     raise NotImplementedError()
    #
    # def complete(self, *args, **kwargs):
    #     for resource in self.hdf5._asset_docs_cache:
    #         self._asset_docs_cache.append(('resource', resource[1]))
    #
    #     self._datum_ids = []
    #
    #     num_frames = self.hdf5.num_captured.get()
    #
    #     # print(f'\n!!! num_frames: {num_frames}\n')
    #
    #     for frame_num in range(num_frames):
    #         datum_id = '{}/{}'.format(self.hdf5._resource_uid, next(self._datum_counter))
    #         datum = {'resource': self.hdf5._resource_uid,
    #                  'datum_kwargs': {'frame': frame_num},
    #                  'datum_id': datum_id}
    #         self._asset_docs_cache.append(('datum', datum))
    #         self._datum_ids.append(datum_id)
    #
    #     return NullStatus()
    #
    # def collect(self):
    #     # TODO: try to separate it from the xspress3 class
    #     collected_frames = self.settings.array_counter.get()
    #
    #     # This is a hack around the issue with .NORD (number of elements to #
    #     # read) that does not match .NELM (number of elements to that the array
    #     # will hold)
    #     dpb_sec_nelm_count = int(dpb_sec_nelm.get())
    #     dpb_nsec_nelm_count = int(dpb_nsec_nelm.get())
    #     dpb_sec_values = np.array(dpb_sec.get(count=dpb_sec_nelm_count),
    #                               dtype='float128')[:collected_frames * 2: 2]
    #     dpb_nsec_values = np.array(dpb_nsec.get(count=dpb_nsec_nelm_count),
    #                                dtype='float128')[:collected_frames * 2: 2]
    #
    #     di_timestamps = dpb_sec_values + dpb_nsec_values * 1e-9
    #
    #     len_di_timestamps = len(di_timestamps)
    #     len_datum_ids = len(self._datum_ids)
    #
    #     if len_di_timestamps != len_datum_ids:
    #         warnings.warn(f'The length of "di_timestamps" ({len_di_timestamps}) '
    #                       f'does not match the length of "self._datum_ids" ({len_datum_ids})')
    #
    #     num_frames = min(len_di_timestamps, len_datum_ids)
    #     num_frames = len_datum_ids
    #     for frame_num in range(num_frames):
    #         datum_id = self._datum_ids[frame_num]
    #         # ts = di_timestamps[frame_num]
    #         ts = di_timestamps
    #
    #         data = {self.name: datum_id}
    #         # TODO: fix the lost precision as pymongo complained about np.float128.
    #         ts = float(ts)
    #
    #         # print(f'data: {data}\nlen_di_timestamps: {len_di_timestamps}\nlen_datum_ids: {len_di_timestamps}')
    #
    #         yield {'data': data,
    #                'timestamps': {key: ts for key in data},
    #                'time': ts,  # TODO: use the proper timestamps from the mono start and stop times
    #                'filled': {key: False for key in data}}
    #
    # # The collect_asset_docs(...) method was removed as it exists on the hdf5 component and should be used there.
    #
    # def set_channels_for_hdf5(self, channels=(1, 2, 3, 4, 5, 6)):
    #     """
    #     Configure which channels' data should be saved in the resulted hdf5 file.
    #     Parameters
    #     ----------
    #     channels: tuple, optional
    #         the channels to save the data for
    #     """
    #     # The number of channel
    #     for n in channels:
    #         getattr(self, f'channel{n}').rois.read_attrs = ['roi{:02}'.format(j) for j in [1, 2, 3, 4, 5, 6]]
    #     self.hdf5.num_extra_dims.put(0)
    #     # self.settings.num_channels.put(len(channels))
    #     self.settings.num_channels.put(6)
    #
    # # Currently only using four channels. Uncomment these to enable more
    # # channels:
    # # channel5 = C(Xspress3Channel, 'C5_', channel_num=5)
    # channel6 = Cpt(Xspress3Channel, 'C6_', channel_num=6)
    # # channel7 = C(Xspress3Channel, 'C7_', channel_num=7)
    # # channel8 = C(Xspress3Channel, 'C8_', channel_num=8)


xs = QASXspress3Detector('XF:07BMB-ES{Xsp:1}:', name='xs')


from scipy.interpolate import interp1d

def interpolate1(dataset,key_base = 'i0'):
    interpolated_dataset = {}
    min_timestamp = max([dataset.get(key).iloc[0, 0] for key in dataset])
    max_timestamp = min([dataset.get(key).iloc[len(dataset.get(key)) - 1, 0] for key in
                         dataset if len(dataset.get(key).iloc[:, 0]) > 5])

    try:
        if key_base not in dataset.keys():
            raise ValueError('Could not find "{}" in the loaded scan. Pick another key_base'
                             ' for the interpolation.'.format(key_base))
    except ValueError as err:
        print(err.args[0], '\nAborted...')
        return

    timestamps = dataset[key_base].iloc[:,0]

    condition = timestamps < min_timestamp
    timestamps = timestamps[np.sum(condition):]

    condition = timestamps > max_timestamp
    timestamps = np.array(timestamps[: len(timestamps) - np.sum(condition)])

    for key in dataset.keys():
        _time = dataset.get(key).iloc[:,0].values #array for timestamp
        _value = dataset.get(key).iloc[:,1].values # array for values e.g. i0, i1, CHAN1ROI1 etc.
        if len(_time) > 5 * len(timestamps):
            _time = [_time[0]] + [np.mean(array) for array in np.array_split(_time, len(timestamps))] + [_time[-1]]
            _value = [_value[0]] + [np.mean(array) for array in np.array_split(_value, len(timestamps))] + [_value[-1]]

        interpolator_func = interp1d(_time, np.array([v for v in _value]), axis=0)
        interpolated_value = interpolator_func(timestamps)

        if len(interpolated_value.shape) == 1:
            interpolated_dataset[key] = interpolated_value
        else:
            interpolated_dataset[key] = [v for v in interpolated_value]

    interpolated_dataframe = pd.DataFrame(interpolated_dataset)

    return interpolated_dataframe.sort_values('energy')

x = xview_gui

uids = []
for item in x.listFiles_bin.selectedItems():
    fname = os.path.join(x.workingFolder, item.text())
    df, header = load_binned_df_from_file(fname)
    print(fname)
    # uid_idx1 = header.find('Scan.uid:') + 10
    # uid_idx2 = header.find('\n', header.find('Scan:uid:'))
    # uid = header[uid_idx1: uid_idx2]
    # uids.append(uid)


def load_binned_df_from_file(filename):
    ''' Load interp file and return'''

    if not os.path.exists(filename):
        raise IOError(f'The file {filename} does not exist.')
    header = read_header(filename)

    keys = header[header.rfind('#'):][1:-1].split()
    df = pd.read_csv(filename, delim_whitespace=True, comment='#', names=keys, index_col=False)

    energy_key = None
    for col in df.columns:
        if ('energy' in col.lower()):# or ('e' in col.lower()):
            energy_key = col
            break

    if energy_key:
        df = df.rename(columns={energy_key: 'energy'})
        df = df.sort_values('energy')

    header = convert_header_to_dict(header)
    return df, header

def read_header(filename):
    header = ''
    line = '#'
    with open(filename) as myfile:
        while line[0] == '#':
            line = next(myfile)
            header += line
    return header[:-len(line)]

def convert_header_to_dict(header):
    lines = header.split('\n')
    lines = [line for line in lines if len(line) > 0]
    lines = [line.replace('# ', '') for line in lines]

    head = {}
    for line in lines[:-2]:
        buf = line.split(":")
        head[buf[0].strip()] = buf[1].strip()

    return head


fname = {}
for item in x.listFiles_bin.selectedItems():
    name = os.path.join(x.workingFolder, item.text())
    df, header = load_binned_df_from_file(name)
    print(name)
    fname[item.text()] = {}
    fname[item.text()]['UID'] = header['UID']
    fname[item.text()]['E0'] = float(header['E0'])



def save_data(dic, path=None):

    for key, val in dic.items():

        uid = val['UID']
        e0 = val['E0']

        apb_df, energy_df, energy_offset = load_apb_dataset_from_db(db, db[uid].start['uid'])
        raw_df = translate_apb_dataset(apb_df, energy_df, energy_offset)

        apb_trig_timestamps = load_apb_trig_dataset_from_db(db, db[uid].start['uid'])
        xs3_dict = load_xs3_dataset_from_db_new(db, db[uid].start['uid'], apb_trig_timestamps)

        raw_df = {**raw_df, **xs3_dict}
        key_base = 'CHAN1ROI1'

        interpolated_df = interpolate_new(raw_df, key_base=key_base)

        first_column = interpolated_df.pop('energy')

        interpolated_df.insert(0, 'energy', first_column)

        binned_df = energy_rebinning(interpolated_df, e0=e0)

        header = ""

        for k in binned_df.keys():
            header += k + ' '

        print(key)

        np.savetxt(path + key[:-4] + '.raw', np.column_stack([binned_df[k] for k in binned_df.keys()]),
                   header=header)

def energy_rebinning(dataframe,
                     e0=None,
                     edge1=-30,
                     edge2=50,
                     xanes_grid=0.5,
                     pre_grid=2,
                     k_grid=0.05,
                     emin=None,
                     emax=None):
    df = dataframe

    emin = df['energy'].iloc[0]
    emax = df['energy'].iloc[-1]

    print(f"{emin =}")
    print(f"{emax =}")

    k_max = np.sqrt(0.262467 * (emax - e0))
    k_min = np.sqrt(0.262467 * (e0 + edge2 - e0))
    k_range = np.arange(k_min - 0.001, k_max + 0.01, k_grid)
    energy_k_range = (k_range ** 2 / 0.262467) + e0
    energy_xanes_range = np.arange(e0 + edge1, e0 + edge2, xanes_grid)
    energy_pre_edge_range = np.arange(emin, e0 + edge1, pre_grid)

    energy_all = np.append(energy_pre_edge_range, energy_xanes_range)
    energy_all = np.append(energy_all, energy_k_range)

    energy_all = np.concatenate(([emin], energy_all, [emax]))

    print(f"{energy_all[0]= }")
    print(f"{energy_all[-1]= }")

    binned_dataset = {}
    binned_dataset['energy'] = energy_all
    for key in df.columns[1:]:
        interpolator_func = interp1d(df['energy'], [val for val in df[key]], axis=0)
        binned_dataset[key] = interpolator_func(energy_all)

    return binned_dataset



uid_23 = '01f07f06-da72-44cf-9237-6d607acb6520'
uid_24 = '0f5773dd-e1ef-4a88-9f4e-c85c22242eed'


def import_fuction():
    from xas.process import load_apb_dataset_from_db, translate_apb_dataset, load_apb_trig_dataset_from_db, load_xs3_dataset_from_db, interpolate, rebin
    from larch.xafs import preedge
    from larch.symboltable import Group


def binned_data(uid):
    apb_df, energy_df, energy_offset = load_apb_dataset_from_db(db, db[uid].start['uid'])
    raw_df = translate_apb_dataset(apb_df, energy_df, energy_offset)

    key_base = 'i0'

    interpolated_df = interpolate(raw_df, key_base=key_base)

    _dic = preedge(interpolated_df.energy, np.log(interpolated_df.it/interpolated_df.ir), e0=8979)

    dic = {}
    dic['energy'] = interpolated_df.energy
    dic['norm'] = _dic['norm']

    df = pd.DataFrame(dic)

    return energy_rebinning(df, e0=8979)


def read_voltage_and_set_condition():
    voltage = apb.ch7.value
    if voltage > 2000:
        condition = True
    else:
        condition = False
    return condition

def fly_scan_with_hardware_trigger(name: str, comment: str, n_cycles: int = 1, delay: float = 0, hutch_c: bool = False, shutter=shutter_fs, **kwargs):
    print(f'Hutch C is {hutch_c}')
    sys.stdout = kwargs.pop('stdout', sys.stdout)
    uids = []

    yield from bps.mv(shutter, "Open")

    condition1 = read_voltage_and_set_condition()
    condition2 = read_voltage_and_set_condition()
    count = 1
    while count < int(n_cycles):
        if condition2 - condition1 == 1:

            name_n = '{} {:04d}'.format(name, count + 1)
            yield from prep_traj_plan()
            print(f'Trajectory preparation complete at {print_now()}')
            if hutch_c:
                uid = (yield from execute_trajectory_apb_c(name_n, comment=comment))
            else:
                uid = (yield from execute_trajectory_apb(name_n, comment=comment))
            uids.append(uid)
            print(f'Trajectory is complete {print_now()}')
            yield from bps.sleep(float(delay))
            count += 1

            condition1 = condition2
            condition2 = read_voltage_and_set_condition()
        else:
            condition1 = condition2
            condition2 = read_voltage_and_set_condition()
def fly_scan_with_apb_with_controlled_loop(name: str, comment: str, n_cycles: int = 1, delay: float = 0, hutch_c: bool = False, shutter=shutter_fs, **kwargs):
    '''
    Trajectory Scan - Runs the monochromator along the trajectory that is previously loaded in the controller N times
    Parameters
    ----------
    name : str
        Name of the scan - it will be stored in the metadata
    n_cycles : int (default = 1)
        Number of times to run the scan automatically
    delay : float (default = 0)
        Delay in seconds between scans
    Returns
    -------
    uid : list(str)
        Lists containing the unique ids of the scans
    '''
    print(f'Hutch C is {hutch_c}')
    sys.stdout = kwargs.pop('stdout', sys.stdout)
    uids = []

    yield from bps.mv(shutter, "Open")

    print('Begin Timer')
    measure_time1 = time.time()

    name_n = '{} {:04d}'.format(name, 0 + 1)
    yield from prep_traj_plan()
    print(f'Trajectory preparation complete at {print_now()}')
    if hutch_c:
        uid = (yield from execute_trajectory_apb_c(name_n, comment=comment))
    else:
        uid = (yield from execute_trajectory_apb(name_n, comment=comment))
    uids.append(uid)
    print(f'Trajectory is complete {print_now()}')
    yield from bps.sleep(float(delay))

    measure_time3 = time.time()
    measure_time2 = time.time()
    count = 1
    while count < int(n_cycles):
        if measure_time2 - measure_time1 >= 60:
            print('>>>>>>>>>>>>>>>>>>>>> 60 seconds >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            measure_time1 = measure_time2
            name_n = '{} {:04d}'.format(name, count + 1)
            yield from prep_traj_plan()
            print(f'Trajectory preparation complete at {print_now()}')
            if hutch_c:
                uid = (yield from execute_trajectory_apb_c(name_n, comment=comment))
            else:
                uid = (yield from execute_trajectory_apb(name_n, comment=comment))
            uids.append(uid)
            print(f'Trajectory is complete {print_now()}')
            yield from bps.sleep(float(delay))

            measure_time2 = time.time()
            count += 1
        else:
            if measure_time2 - measure_time3> 1:
                print(f'Time elapsed: {measure_time2 - measure_time1:.0f}s after the scan')
                measure_time3 = time.time()
            measure_time2 = time.time()





uids = ['78c523ef-b8a1-498b-bf21-ccf86e6b5516',
        '75bcad4f-5340-4ead-b904-ec5ac001fff2',
        'f2e871e4-95e2-4683-9a7a-05e34c96a731',
        'c65f073a-4546-4842-974f-714934954aab',
        '85bd00fd-2cb6-477b-ae95-6ed55628d045',
        ]


path = '/home/xf07bm/Documents/Akhil_Tayal/Osc_traj_data/'
def save_interpolated_data(uids, path=None):

    for uid in uids:

        hdr = db[uid]


        apb_df, energy_df, energy_offset = load_apb_dataset_from_db(db, db[uid].start['uid'])
        raw_df = translate_apb_dataset(apb_df, energy_df, energy_offset)

        key_base = 'i0'

        interpolated_df = interpolate(raw_df, key_base=key_base)

        interpolated_df.to_csv(path + hdr.start['name'] + '.txt', sep='\t')


from PyQt5.QtCore import QThread



class FlyableEpicsMotor(Device): # device is needed to have Device status
    '''
    This class mimics hhm behavior that is used in the standard HHM ISS flyer
    '''

    def __init__(self, motor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motor = motor
        self.traj_dict = None
        self.flying_status = None

    def set_trajectory(self, traj_dict):
        # traj_dict = {'positions': [point1, point2, point3, point4],
        #              'durations': [t1_2, t2_3, t3_4]}
        self.traj_dict = traj_dict

    def prepare(self):
        return self.motor.move(self.traj_dict['positions'][0], wait=False)

    def kickoff(self):
        self.flying_status = DeviceStatus(self)

        self.thread = QThread()

        self.moveToThread(self.thread)
        self.thread.started.connect(self.execute_motion)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

        #
        #
        #
        # thread = threading.Thread(target=self.execute_motion, daemon=True)
        # thread.start()
        return self.flying_status

    def execute_motion(self):
        self.data = []
        def write_data_callback(value, timestamp, **kwargs):
            self.data.append([timestamp, value])
        cid = self.motor.user_readback.subscribe(write_data_callback)

        pre_fly_velocity = self.motor.velocity.get()
        for prev_position, next_position, duration in zip(self.traj_dict['positions'][:-1],
                                                          self.traj_dict['positions'][1:],
                                                          self.traj_dict['durations']):
            velocity = abs(next_position - prev_position) / duration
            self.motor.velocity.set(velocity).wait()
            self.motor.move(next_position).wait()
        self.flying_status.set_finished()

        self.motor.velocity.set(pre_fly_velocity).wait()
        self.motor.user_readback.unsubscribe(cid)

    def complete(self):
        self.flying_status = None
        self.traj_dict = None

    @property
    def current_trajectory_duration(self):
        return sum(self.traj_dict['durations'])


def combine_status_list(status_list):
    st_all = status_list[0]
    for st in status_list[1:]:
        # st_all = st_all and st
        st_all = st_all & st
    return st_all
def test_flying_epics_motor():
    flying_motor_cr_main_roll =  FlyableEpicsMotor(sample_stage1.x, name='flying_samplex')


    roll_center = 0
    roll_delta1 = -10
    roll_delta2 = 10
    traj_dict_main = {'positions': [roll_center - roll_delta1,
                                    roll_center - roll_delta2,
                                    roll_center + roll_delta2,
                                    roll_center + roll_delta1], 'durations': [5, 10, 5]}
    flying_motor_cr_main_roll.set_trajectory(traj_dict_main)

    prepare_st1 = flying_motor_cr_main_roll.prepare()


    combine_status_list([prepare_st1]).wait()

    st1 = flying_motor_cr_main_roll.kickoff()


    combine_status_list([st1]).wait()

    data1 = np.array(flying_motor_cr_main_roll.data)

    plt.figure(1, clear=True)
    plt.plot(data1[:, 0] - data1[0, 0], data1[:, 1], '.-')



def optimize_gains_plan(n_tries=3, trajectory_filename=None, mono_angle_offset=None):
    # sys.stdout = kwargs.pop('stdout', sys.stdout)

    detectors = [apb_ave]
    channels =  [apb_ave.ch1,  apb_ave.ch2,  apb_ave.ch3,  apb_ave.ch4]

    if trajectory_filename is not None:
        yield from prepare_trajectory_plan(trajectory_filename, offset=mono_angle_offset)
        # trajectory_stack.set_trajectory(trajectory_filename, offset=mono_angle_offset)

    threshold_hi = 7000 #mV
    threshold_lo = 50 #mV

    e_min, e_max = trajectory_manager.read_trajectory_limits()
    scan_positions = np.arange(e_max + 50, e_min - 50, -200).tolist()

    yield from actuate_photon_shutter_plan('Open')
    yield from shutter.open_plan()

    for jj in range(n_tries):

        plan = bp.list_scan(detectors, hhm.energy, scan_positions)
        yield from plan
        table = db[-1].table()

        all_gains_are_good = True

        for channel in channels:
            current_gain = channel.amp.get_gain()[0]
            if channel.polarity == 'neg':
                trace_extreme = table[channel.name].min()
            else:
                trace_extreme = table[channel.name].max()

            trace_extreme = trace_extreme / 1000

            print_to_gui(f'Extreme value {trace_extreme} for detector {channel.name}')
            if abs(trace_extreme) > threshold_hi:
                print_to_gui(f'Decreasing gain for detector {channel.name}')
                yield from channel.amp.set_gain_plan(current_gain - 1, False)
                all_gains_are_good = False
            elif abs(trace_extreme) <= threshold_hi and abs(trace_extreme) > threshold_lo:
                print_to_gui(f'Correct gain for detector {channel.name}')
            elif abs(trace_extreme) <= threshold_lo:
                print(f'Increasing gain for detector {channel.name}')
                yield from channel.amp.set_gain_plan(current_gain + 1, False)
                all_gains_are_good = False

        if all_gains_are_good:
            print(f'Gains are correct. Taking offsets..')
            break

    yield from shutter.close_plan()
    yield from get_offsets_plan()



class Mono2(Device):
    _default_configuration_attrs = ('bragg', 'energy', 'pico', 'diag')
    _default_read_attrs = ('bragg', 'energy', 'pico', 'diag')
    "Monochromator"
    ip = '10.68.50.104'
    traj_filepath = '/home/xf07bm/trajectory/'
    bragg = Cpt(EpicsMotor, 'Mono:1-Ax:Scan}Mtr')
    energy = Cpt(EpicsMotor, 'Mono:1-Ax:E}Mtr')
    pico = Cpt(EpicsMotor, 'Mono:1-Ax:Pico}Mtr')
    diag = Cpt(EpicsMotor, 'Mono:1-Ax:Diag}Mtr')

    main_motor_res = Cpt(EpicsSignal, 'Mono:1-Ax:Scan}Mtr.MRES')

    # The following are related to trajectory motion
    lut_number = Cpt(EpicsSignal, 'MC:03}LUT-Set')
    lut_number_rbv = Cpt(EpicsSignal, 'MC:03}LUT-Read')
    lut_start_transfer = Cpt(EpicsSignal, 'MC:03}TransferLUT')
    lut_transfering = Cpt(EpicsSignal, 'MC:03}TransferLUT-Read')
    trajectory_loading = Cpt(EpicsSignal, 'MC:03}TrajLoading')
    traj_mode = Cpt(EpicsSignal, 'MC:03}TrajFlag1-Set')
    traj_mode_rbv = Cpt(EpicsSignal, 'MC:03}TrajFlag1-Read')
    enable_ty = Cpt(EpicsSignal, 'MC:03}TrajFlag2-Set')
    enable_ty_rbv = Cpt(EpicsSignal, 'MC:03}TrajFlag2-Read')
    cycle_limit = Cpt(EpicsSignal, 'MC:03}TrajRows-Set')
    cycle_limit_rbv = Cpt(EpicsSignal, 'MC:03}TrajRows-Read')
    enable_loop = Cpt(EpicsSignal, 'MC:03}TrajLoopFlag-Set')
    enable_loop_rbv = Cpt(EpicsSignal, 'MC:03}TrajLoopFlag')

    prepare_trajectory = Cpt(EpicsSignal, 'MC:03}PrepareTraj')
    trajectory_ready = Cpt(EpicsSignal, 'MC:03}TrajInitPlc-Read')
    start_trajectory = Cpt(EpicsSignal, 'MC:03}StartTraj')
    stop_trajectory = Cpt(EpicsSignal, 'MC:03}StopTraj')
    trajectory_running = Cpt(EpicsSignal,'MC:03}TrajRunning', write_pv='MC:03}TrajRunning-Set')
    trajectory_progress = Cpt(EpicsSignal,'MC:03}TrajProgress')
    trajectory_name = Cpt(EpicsSignal, 'MC:03}TrajFilename')

    traj1 = Cpt(MonoTrajDesc, 'MC:03}Traj:1')
    traj2 = Cpt(MonoTrajDesc, 'MC:03}Traj:2')
    traj3 = Cpt(MonoTrajDesc, 'MC:03}Traj:3')
    traj4 = Cpt(MonoTrajDesc, 'MC:03}Traj:4')
    traj5 = Cpt(MonoTrajDesc, 'MC:03}Traj:5')
    traj6 = Cpt(MonoTrajDesc, 'MC:03}Traj:6')
    traj7 = Cpt(MonoTrajDesc, 'MC:03}Traj:7')
    traj8 = Cpt(MonoTrajDesc, 'MC:03}Traj:8')
    traj9 = Cpt(MonoTrajDesc, 'MC:03}Traj:9')

    # trajectory_type = None

    angle_offset = Cpt(EpicsSignal, 'Mono:1-Ax:E}Offset', limits=True)

    def __init__(self, *args, enc = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pulses_per_deg = 1/self.main_motor_res.get()
        self.enc = enc
        self._trajectory_type = None


    @property
    def trajectory_type(self):
        return self._trajectory_type

    @trajectory_type.setter
    def trajectory_type(self, value):
        self._trajectory_type = value




mono2 = Mono2('XF:07BMA-OP{', enc = pb1.enc1, name='mono2')
mono2.energy.kind = 'hinted'
mono2.bragg.kind = 'hinted'

