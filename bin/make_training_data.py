import argparse
from os import path

import pandas as pd
from tqdm import tqdm

# append current directory to sys path
import sys
sys.path.insert(0, '.')
from polaris_motor_data_extraction.data_extractors import PolarisMotorDataExtractor


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Make a training set from grasp data')
    parser.add_argument('--data-folderpath', action='store', type=str, required=True)
    parser.add_argument('--output-folderpath', action='store', type=str, required=True)
    parser.add_argument('--extractor', action='store', type=str, default='min_extractor')

    args = parser.parse_args()

    # read index
    index_df = pd.read_csv(path.join(args.data_folderpath, 'index.csv'))

    polaris_motor_data_extractor = PolarisMotorDataExtractor.factory(args.extractor)

    parsed_records = []
    for r in tqdm(index_df.to_dict('records')):
        # extract motor and polaris data
        gripper_data_filepath = path.join(args.data_folderpath, r['gripper_data_filepath'])
        parsed_record = polaris_motor_data_extractor(gripper_data_filepath)

        # remove gripper_data_filepath from the record since it was extracted
        parsed_record.pop('gripper_data_filepath', None)

        # merge motor and polaris data with original record
        parsed_record.update(r)

        parsed_records.append(parsed_record)

    pd.DataFrame(parsed_records).to_csv(path.join(args.output_folderpath, 'grasp_data.csv'), index=None)