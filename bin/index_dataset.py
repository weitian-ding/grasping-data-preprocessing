import argparse
import logging
import os
from functools import reduce, partial
from glob import glob
from os import path
from shutil import rmtree

from scipy.interpolate import interp1d
from tqdm import tqdm

import pandas as pd
import numpy as np

# append current directory to sys path
import sys
sys.path.insert(0, '.')

from parsers import polaris_coord_transform
from parsers.daily_origins_parser import parse_daily_origin
from parsers.gripper_parser import parse_gripper_file
from parsers.polaris_parser import parse_polaris_file


def _extract_session_id_from_gripper_filepath(fp):
    """Extracts session id from the directory name of gripper file
    if directory contains -2, then session id is 1, otherwise 0
    """
    if '-2' in path.dirname(fp):
        return 1
    return 0


# extracts grasp id from file path, filename is splitted by split
def _grasp_id_from_filepath(split, fp):
    return int(path.basename(fp).split(split)[0])


# list all files (gripper files, polaris files, image files, etc) and join them by grasp id
def _group_files_by_grasp_id(input_folderpath):
    # filetype_name => (extension, a function to extract grasp id from filename)
    filetypes = {'gripper_filepath': ('*displacement', partial(_grasp_id_from_filepath, '-')),
                 'polaris_filepath': ('*.txt', partial(_grasp_id_from_filepath, '-')),
                 'rs_depth_image_filepath': ('*RS_depth.npy', partial(_grasp_id_from_filepath, '_')),
                 'rs_color_image_filepath': ('*RS_color.npy', partial(_grasp_id_from_filepath, '_')),
                 'zed_depth_image_filepath': ('*ZED_depth.npy', partial(_grasp_id_from_filepath, '_')),
                 'zed_color_image_filepath': ('*ZED_color.npy', partial(_grasp_id_from_filepath, '_'))}
    filepath_dfs = []

    for filetype, (ext, extract_grasp_id) in filetypes.items():
        # find all filepaths with the same extension
        filepaths = []
        for dir in os.walk(input_folderpath):  # recursively walks all directories
            # find all files with ext in the current directory
            filepath_wildcard =  path.join(dir[0], ext)
            filepaths.extend([path.relpath(f, input_folderpath) for f in glob(filepath_wildcard)])

        filepath_df = pd.DataFrame({filetype: filepaths})
        filepath_df['grasp_id'] = filepath_df[filetype].map(extract_grasp_id)

        filepath_dfs.append(filepath_df)

    # join all filetypes by grasp id
    joined = reduce(lambda df1, df2: pd.merge(df1, df2, on='grasp_id'), filepath_dfs)

    # sort by grasp id
    joined = joined.sort_values(by=['grasp_id'])

    return joined


# join gripper records and polaris records by timestamp
def _merge_polaris_gripper(polaris_ts, polaris_records, gripper_motor_interps):
    gripper_motor_records = np.zeros(shape=(len(polaris_ts), 4))

    for i, ts in enumerate(polaris_ts):
        ts_value = ts.value
        gripper_motor_record = np.array([f(ts_value) for f in gripper_motor_interps])

        gripper_motor_records[i] = gripper_motor_record

    return pd.DataFrame({
        'timestamp': polaris_ts,
        'gripper_motor_1': gripper_motor_records[:, 0],
        'gripper_motor_2': gripper_motor_records[:, 1],
        'gripper_motor_3': gripper_motor_records[:, 2],
        'gripper_motor_4': gripper_motor_records[:, 3],
        'polaris_x': polaris_records[:, 0],
        'polaris_y': polaris_records[:, 1],
        'polaris_z': polaris_records[:, 2],
        'polaris_rx': polaris_records[:, 3],
        'polaris_ry': polaris_records[:, 4],
        'polaris_rz': polaris_records[:, 5]
    })


# produces spline interpolations with gripper records, i.e. f[i](timestamp) => gripper_motor_params[i]
def _gripper_motor_records_to_interps(gripper_ts, gripper_motor_records):
    # a list to hold 4 spline interpolations for 4 gripper motor parameters
    gripper_motor_params_interps = []

    for ci in range(0, gripper_motor_records.shape[1]):
        # convert timestamp to real number values, i.e. nanoseconds
        ts_values = np.array([t.value for t in gripper_ts])
        motor_params_ci = gripper_motor_records[:, ci]
        interp = interp1d(ts_values, motor_params_ci, fill_value='extrapolate', kind='cubic')

        gripper_motor_params_interps.append(interp)

    return gripper_motor_params_interps


# save gripper data, which contains both gripper motor and polaris records, to disk
def _save_gripper_data(grasp_id, gripper_df, output_folderpath):
    gripper_data_filepath = path.join(output_folderpath, 'gripper_data', '%s.csv' % grasp_id)
    if not path.exists(path.dirname(gripper_data_filepath)):
        os.mkdir(path.dirname(gripper_data_filepath))
    gripper_df.to_csv(gripper_data_filepath, index=False)

    return gripper_data_filepath


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Index gripper data')
    parser.add_argument('--input-folderpath', action='store', type=str, required=True)
    parser.add_argument('--output-folderpath', action='store', type=str, required=True)
    parser.add_argument('--update-origin', action='store_true', default=False)
    parser.add_argument('--daily-origin-filepath', action='store', type=str,
                        default='Data Collection - DailyOrigin.csv')
    parser.add_argument('--transformation-constants-filepath', action='store', type=str,
                        default='transformation.constants')
    parser.add_argument('--log-filename', action='store', type=str, default='log.txt')
    parser.add_argument('--limit-processing', action='store', type=int, default=None)

    args = parser.parse_args()

    # setupt file logging
    logging.basicConfig(filename=args.log_filename, filemode='w', level=logging.DEBUG)

    # prepare polaris coordinate transformer
    print('prepare polaris coords transformations...')
    polaris_coord_transform_constants = polaris_coord_transform.ndi_transformation(args.transformation_constants_filepath)
    polaris_coord_transformer = polaris_coord_transform.Transformer(polaris_coord_transform_constants)

    # prepare daily origins for coordinate transformation
    if args.update_origin:
        daily_origins = parse_daily_origin(args.daily_origin_filepath)

    print('list all grasp data files...')
    filepaths_df = _group_files_by_grasp_id(args.input_folderpath)
    print(filepaths_df.head())

    # make output folder, remove if it already exists
    if path.exists(args.output_folderpath):
        logging.warning('removing output folder %s', args.output_folderpath)
        rmtree(args.output_folderpath)
    os.makedirs(args.output_folderpath)

    print('start processing gripper data...')
    processing_counts = 0
    processed_grasps = []

    for r in tqdm(filepaths_df.to_dict('records')):

        # only processing a subset, useful for debugging
        if not args.limit_processing is None and processing_counts > args.limit_processing:
            break

        gripper_fp = path.join(args.input_folderpath, r['gripper_filepath'])
        polaris_fp = path.join(args.input_folderpath, r['polaris_filepath'])

        processing_counts += 1

        try:
            parsed_gripper_file = parse_gripper_file(gripper_fp)
            parsed_polaris_file = parse_polaris_file(polaris_fp)

            # update daily origin
            if args.update_origin:
                # use the first timestamp in polaris recording as the date of a grasp session
                session_date = pd.Timestamp(parsed_gripper_file.timestamps[0].date())
                session_id = _extract_session_id_from_gripper_filepath(gripper_fp)
                polaris_coord_transformer.object_origin = daily_origins\
                    .lookup_origin_by_session_date_and_id(session_date, session_id)

            # transform polaris coordinates
            polaris_records = [polaris_coord_transformer.transform_single_example(t1, t2)
                               for t1, t2 in zip(parsed_polaris_file.tool1_params, parsed_polaris_file.tool2_params)]
            polaris_records = np.array(polaris_records)

            gripper_motor_interps = _gripper_motor_records_to_interps(parsed_gripper_file.timestamps,
                                                                      parsed_gripper_file.motor_records)

            polaris_gripper_merged_df = _merge_polaris_gripper(parsed_polaris_file.timestamps,
                                                               polaris_records,
                                                               gripper_motor_interps)

            gripper_data_filepath = _save_gripper_data(grasp_id=r['grasp_id'],
                                                       gripper_df=polaris_gripper_merged_df,
                                                       output_folderpath=args.output_folderpath)

        except ValueError as e:
            logging.warning('%s processing record %s, probably something wrong in coordinate transformation', e, r)
            continue
        except Exception as e:
            logging.warning('unhandled exception %s processing record %s', e, r)
            continue

        # writes into index only if file processing is successful
        processed_grasp = {
            'id': r['grasp_id'],
            'gripper_data_filepath': path.relpath(gripper_data_filepath, args.output_folderpath),
            'rs_depth_image_filepath': r['rs_depth_image_filepath'],
            'rs_color_image_filepath': r['rs_color_image_filepath'],
            'zed_depth_image_filepath': r['zed_depth_image_filepath'],
            'zed_color_image_filepath': r['zed_color_image_filepath'],
            'grip_type': parsed_gripper_file.grip_type,
            'is_success': parsed_gripper_file.is_grip_success,
            'description': parsed_gripper_file.desc
        }

        processed_grasps.append(processed_grasp)

    # save the index to disk
    pd.DataFrame(processed_grasps).to_csv(path.join(args.output_folderpath, 'index.csv'), index=None)

    print('processing finished, attempted processing {} grasps, successed in {} grasps'
          .format(processing_counts, len(processed_grasps)))
