import pandas as pd
import numpy as np

import logging

SKIP_ROWS = 6


# extracts timestamp, polaris tool1 params and polaris tool2 params from polaris file
def parse_polaris_file(filepath):
    timestamps = []
    tool1_records = []
    tool2_records = []

    with open(filepath) as f:
        lines = f.readlines()
        for l in lines[SKIP_ROWS:]:  # skip the file headers
            # try to parse a line as a polaris data record
            try:
                ts, tool1_record, tool2_record = _parse_polaris_single_record(l)

                timestamps.append(ts)
                tool1_records.append(tool1_record)
                tool2_records.append(tool2_record)

            except:
                logging.warning('polaris file %s line %s malformed or not understood', filepath, l)
                continue

    return np.array(timestamps), \
           np.stack(tool1_records, axis=0), \
           np.stack(tool2_records, axis=0)


def _parse_polaris_single_record(line):
    tokens = [t.strip() for t in line.split(',')]
    ts_str, tool1_params, tool2_params = (tokens[1], tokens[3:10], tokens[11:18])
    ts = pd.Timestamp(pd.to_datetime(ts_str, format='%Y-%m-%d-%H-%M-%S.%f'))
    tool1_params = np.array([float(x) for x in tool1_params], dtype=np.float64)
    tool2_params = np.array([float(x) for x in tool2_params], dtype=np.float64)

    if not tool1_params.shape == (7,) or not tool2_params.shape == (7,):
        raise ValueError('polaris file line %s is malformed' % line)

    return ts, tool1_params, tool2_params