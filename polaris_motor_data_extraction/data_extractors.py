import pandas as pd
import numpy as np


class PolarisMotorDataExtractor:
    """RecordExtractor is an abstract class for extracting polaris and motor data from gripper data file,
          you can implement new extractors by extending this abstract class
    """
    _registered_extractor = {}

    def __call__(self, gripper_data_filepath):
        gripper_df = pd.read_csv(gripper_data_filepath)
        return self.call(gripper_df)

    # a wrapper for __call__
    def call(self, gripper_df):
        raise NotImplementedError

    @staticmethod
    def register_extractor(name, class_ref):
        """Register a new record extractor

        :param name: extractor name
        :param class_ref: class reference
        """
        PolarisMotorDataExtractor._registered_extractor[name] = class_ref

    @staticmethod
    def factory(name, **kwargs):
        """Factory method for record extractors

        :param name: registered extractor name
        :return: an instance of requested extractor
        """
        try:
            extractor = PolarisMotorDataExtractor._registered_extractor[name](**kwargs)
        except KeyError:
            raise ValueError('{} extractor not implemented'.format(name))

        return extractor


class MinExtractor(PolarisMotorDataExtractor):
    """Extracts the motor record with min motor_2 value, polaris record with min z value
    """

    def call(self, gripper_df):
        # extract motor record with min motor_2
        motor_df = gripper_df[['gripper_motor_1', 'gripper_motor_2', 'gripper_motor_3', 'gripper_motor_4']]
        min_motor2_ind = np.argmin(motor_df['gripper_motor_2'])
        motor_record = motor_df.iloc[min_motor2_ind]

        # extract polaris record with min z
        polaris_df = gripper_df[['polaris_x', 'polaris_y', 'polaris_z', 'polaris_rx', 'polaris_ry',	'polaris_rz']]
        min_z_ind = np.argmin(polaris_df['polaris_z'])
        polaris_record = polaris_df.iloc[min_z_ind]

        merged = dict(motor_record)
        merged.update(dict(polaris_record))

        return merged


# register extractors
PolarisMotorDataExtractor.register_extractor('min_extractor', MinExtractor)