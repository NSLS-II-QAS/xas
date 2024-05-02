import numpy as np
import pandas as pd

from scipy.interpolate import interp1d



def interpolate(dataset,key_base = 'i0'):
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



def interpolate_new(dataset, key_base = 'i0'):
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

