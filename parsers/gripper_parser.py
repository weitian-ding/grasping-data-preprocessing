from collections import namedtuple

import numpy as np
import pandas as pd

import logging

# an encapsulation of parsed gripper file
ParsedGripperFile = namedtuple('ParsedGripperFile', 'timestamps motor_records grip_type desc is_grip_success')


# extracts timestamps, motor_records, grip_type, description, success_status from gripper file
def parse_gripper_file(filepath):
    timestamps = []
    motor_records = []
    grip_desc = ''
    status_desc = ''
    is_grip_success = False
    time_delta = None

    with open(filepath) as f:
        lines = f.readlines()

        for l in lines:
            if l.startswith('Time Difference'):
                # record time delta
                time_delta = int(l.split()[-1])
                time_delta = pd.Timedelta(microseconds=time_delta)

            elif l.startswith('T:'):
                # record grip type
                grip_type = int(l[2:].split('-')[0])
                grip_desc = l

            elif l.startswith('S:'):
                # record whether grip is successful
                # TODO double check failure cases
                is_grip_success = 'success' in l
                status_desc = l

            else:
                # try to parse a data record
                try:
                    ts, motor_record = _parse_gripper_single_record(l)

                    timestamps.append(ts)
                    motor_records.append(motor_record)

                except ValueError:
                    logging.warning('gripper file %s line %s malformed or not understood', filepath, l)

    # too few motor records to produce a spline interpolation
    if (len(motor_records) < 2):
        raise ValueError('too few motor recordings for interpolation')

    if timestamps is None:
        raise ValueError('time difference is not found in gripper file %s', filepath)

    motor_records = np.stack(motor_records, axis=0)  # a nx4 matrix of n motor records
    # interpolates 0s for 4 columns and re-stack to produces a 4xn matrix
    motor_records = np.stack([_interpolate_zeros(motor_records[:, ci]) for ci in range(0, 4)],
                             axis=0)
    motor_records = np.transpose(motor_records)   # transpose to produce nx4 matrix

    # synchronize time with polaris
    timestamps = [(t + time_delta) for t in timestamps]
    timestamps = np.array(timestamps)

    desc = '{} {}'.format(grip_desc, status_desc)

    return ParsedGripperFile(timestamps, motor_records, grip_type, desc, is_grip_success)


def _interpolate_zeros(series):
    """Fills zeros in series by lienar interpolation
    fills zeros in series via linear interpolation using adjacent non-zero elements
       if there are zeros at head or tail, then the first non-zero and last non-zero value is extended
       to fill the zeros at head or tail respectively

    :param series: an iterable
    :return: an iterable with zeros filled
    """
    last_non_zero = None
    conts_zero_counts = 0  # counts of continuous zeros
    interpolated = []  # a new series with zeros interpolated and non-zero elements copied

    # the for loop iterates the series, and fills interpolated
    for v in series:
        if not np.equal(v, 0):
            # this occurs only if series[0] == 0
            # this ensures the heading zeros are filled with the first non zero element in the series
            if last_non_zero is None:
                last_non_zero = v

            # interpolates between the previous non zero element, last_non_zero, and current element, v
            # zeros_interpolated includes both last_non_zero and v
            zeros_interpolated = np.linspace(last_non_zero, v, conts_zero_counts + 2)

            # extend the interpolated list, last_non_zero is excluded since it was included in previous extend
            interpolated.extend(zeros_interpolated[1:])

            conts_zero_counts = 0
            last_non_zero = v
        else:
            conts_zero_counts += 1

    if last_non_zero is None:
        raise ValueError('all elements in series is None')

    # fill trailing zeros by the last non zero element in the series
    if conts_zero_counts > 0:
        interpolated.extend([last_non_zero] * conts_zero_counts)

    return np.array(interpolated)


# parse a single data record in gripper file
def _parse_gripper_single_record(line):
    tokens = [t.strip() for t in line.split(',')]
    ts_str, motor_params = (tokens[0], tokens[1:])
    ts = pd.Timestamp(ts_str)
    motor_params = np.array([float(x) for x in motor_params], dtype=np.float64)

    if not motor_params.shape == (4,):
        raise ValueError('gripper file line %s is malformed' % line)

    return ts, motor_params